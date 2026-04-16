"""
data_cleaning.py — Traffic Data Preprocessing
===============================================
Cleans and normalizes raw traffic data extracted from SUMO via TraCI.

Handles:
  - Missing values / NaN filling
  - Outlier clipping
  - Min-max or z-score normalization
  - Sliding window creation for STGCN input sequences

Input format:
    Raw arrays from SumoEnvironment.get_history_buffer():
    {
        'vehicle_counts': (T, N),   # T timesteps, N approach roads
        'queue_lengths':  (T, N),
        'waiting_times':  (T, N),
    }

Output:
    Normalized tensors ready for STGCN training.
"""

import logging
from typing import Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalization Helpers
# ---------------------------------------------------------------------------

class MinMaxScaler:
    """Per-feature Min-Max scaler, fitted on training data."""

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self.min_vals: Optional[np.ndarray] = None
        self.max_vals: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> "MinMaxScaler":
        """
        Fit scaler on data of shape (T, N).
        """
        self.min_vals = data.min(axis=0, keepdims=True)  # (1, N)
        self.max_vals = data.max(axis=0, keepdims=True)  # (1, N)
        logger.debug(f"MinMaxScaler fitted: min={self.min_vals}, max={self.max_vals}")
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.min_vals is None:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        range_vals = self.max_vals - self.min_vals
        range_vals = np.where(range_vals == 0, 1.0, range_vals)  # avoid div-by-zero
        lo, hi = self.feature_range
        normalized = (data - self.min_vals) / range_vals
        return (normalized * (hi - lo) + lo).astype(np.float32)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.min_vals is None:
            raise RuntimeError("Scaler not fitted.")
        lo, hi = self.feature_range
        range_vals = self.max_vals - self.min_vals
        range_vals = np.where(range_vals == 0, 1.0, range_vals)
        return ((data - lo) / (hi - lo)) * range_vals + self.min_vals


class ZScoreScaler:
    """Per-feature Z-Score (standardization) scaler."""

    def __init__(self):
        self.mean_vals: Optional[np.ndarray] = None
        self.std_vals: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> "ZScoreScaler":
        self.mean_vals = data.mean(axis=0, keepdims=True)
        self.std_vals = data.std(axis=0, keepdims=True)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean_vals is None:
            raise RuntimeError("Scaler not fitted.")
        std = np.where(self.std_vals == 0, 1.0, self.std_vals)
        return ((data - self.mean_vals) / std).astype(np.float32)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean_vals is None:
            raise RuntimeError("Scaler not fitted.")
        std = np.where(self.std_vals == 0, 1.0, self.std_vals)
        return (data * std) + self.mean_vals


# ---------------------------------------------------------------------------
# Data Cleaning
# ---------------------------------------------------------------------------

def fill_missing(data: np.ndarray) -> np.ndarray:
    """
    Fill NaN values using forward fill then backward fill.

    Parameters
    ----------
    data : np.ndarray, shape (T, N)

    Returns
    -------
    np.ndarray, shape (T, N)
    """
    result = data.copy()
    for col in range(result.shape[1]):
        col_data = result[:, col]
        # Forward fill
        mask = np.isnan(col_data)
        if mask.any():
            indices = np.where(~mask, np.arange(len(col_data)), 0)
            np.maximum.accumulate(indices, out=indices)
            col_data = col_data[indices]
        # Backward fill (remaining NaNs at start)
        mask = np.isnan(col_data)
        if mask.any():
            col_data = np.where(mask, 0.0, col_data)  # fill remaining with 0
        result[:, col] = col_data
    return result


def clip_outliers(data: np.ndarray, config: dict) -> np.ndarray:
    """
    Clip values to configured physical maxima.

    Parameters
    ----------
    data : np.ndarray, shape (T, N)
    config : dict — preprocessing section of config.yaml

    Returns
    -------
    np.ndarray, shape (T, N)
    """
    pre_cfg = config.get("preprocessing", {})
    max_count = pre_cfg.get("max_vehicle_count", 50)
    return np.clip(data, 0.0, max_count)


def clean_traffic_data(
    raw: Dict[str, np.ndarray],
    config: dict,
) -> Dict[str, np.ndarray]:
    """
    Clean raw traffic arrays from SUMO history buffer.

    Parameters
    ----------
    raw : dict with keys 'vehicle_counts', 'queue_lengths', 'waiting_times'
    config : project config dict

    Returns
    -------
    cleaned : dict — same keys, cleaned float32 arrays (T, N)
    """
    cleaned = {}
    pre_cfg = config.get("preprocessing", {})
    max_queue = pre_cfg.get("max_queue_length", 30)
    max_wait = pre_cfg.get("max_waiting_time", 120)
    max_count = pre_cfg.get("max_vehicle_count", 50)

    clips = {
        "vehicle_counts": (0.0, max_count),
        "queue_lengths": (0.0, max_queue),
        "waiting_times": (0.0, max_wait),
    }

    for key, arr in raw.items():
        arr = np.array(arr, dtype=np.float32)
        arr = fill_missing(arr)
        lo, hi = clips.get(key, (0.0, 1e6))
        arr = np.clip(arr, lo, hi)
        cleaned[key] = arr

    logger.info(
        f"Cleaned traffic data: "
        f"T={cleaned['vehicle_counts'].shape[0]} timesteps, "
        f"N={cleaned['vehicle_counts'].shape[1]} roads."
    )
    return cleaned


# ---------------------------------------------------------------------------
# Sliding Window Construction
# ---------------------------------------------------------------------------

def make_sequences(
    data: np.ndarray,
    seq_len: int,
    pred_len: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for time-series learning.

    Parameters
    ----------
    data : np.ndarray, shape (T, N)
        Time series — T timesteps, N nodes.
    seq_len : int
        Input sequence length (look-back window).
    pred_len : int
        Number of future steps to predict.

    Returns
    -------
    X : np.ndarray, shape (samples, seq_len, N, 1)
        Input sequences (channel dim added for STGCN).
    y : np.ndarray, shape (samples, N)
        Target values (pred_len=1: next step vehicle count).
    """
    T, N = data.shape
    X_list, y_list = [], []

    for t in range(T - seq_len - pred_len + 1):
        x_window = data[t : t + seq_len]           # (seq_len, N)
        y_window = data[t + seq_len : t + seq_len + pred_len]  # (pred_len, N)
        X_list.append(x_window[:, :, np.newaxis])  # (seq_len, N, 1)
        y_list.append(y_window.mean(axis=0))       # (N,) — avg over pred window

    if not X_list:
        raise ValueError(
            f"Not enough data for sequences: T={T}, seq_len={seq_len}, pred_len={pred_len}"
        )

    X = np.stack(X_list, axis=0).astype(np.float32)  # (samples, seq_len, N, 1)
    y = np.stack(y_list, axis=0).astype(np.float32)  # (samples, N)
    logger.debug(f"Created sequences: X={X.shape}, y={y.shape}")
    return X, y


def prepare_stgcn_data(
    history: Dict[str, np.ndarray],
    config: dict,
    scaler: Optional[MinMaxScaler] = None,
    fit_scaler: bool = True,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Full pipeline: clean → normalize → make sequences.

    Uses vehicle_counts as the primary signal for STGCN.

    Parameters
    ----------
    history : raw history buffer from SumoEnvironment
    config : project config dict
    scaler : existing scaler (for val/test), or None to create new
    fit_scaler : if True, fit scaler on this data

    Returns
    -------
    X : (samples, seq_len, N, 1)
    y : (samples, N)
    scaler : fitted MinMaxScaler
    """
    cleaned = clean_traffic_data(history, config)
    data = cleaned["vehicle_counts"]  # (T, N)

    stgcn_cfg = config.get("stgcn", {})
    seq_len = stgcn_cfg.get("seq_len", 12)
    pred_len = stgcn_cfg.get("pred_len", 1)

    if scaler is None:
        scaler = MinMaxScaler()

    if fit_scaler:
        data_norm = scaler.fit_transform(data)
    else:
        data_norm = scaler.transform(data)

    X, y = make_sequences(data_norm, seq_len, pred_len)
    logger.info(f"STGCN data prepared: X={X.shape}, y={y.shape}")
    return X, y, scaler


if __name__ == "__main__":
    import yaml
    logging.basicConfig(level=logging.DEBUG)

    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Simulate
    T, N = 200, 4
    dummy_history = {
        "vehicle_counts": np.random.randint(0, 30, (T, N)).astype(np.float32),
        "queue_lengths":  np.random.randint(0, 15, (T, N)).astype(np.float32),
        "waiting_times":  np.random.uniform(0, 60, (T, N)).astype(np.float32),
    }

    X, y, scaler = prepare_stgcn_data(dummy_history, cfg)
    print(f"X shape: {X.shape}")  # (samples, 12, 4, 1)
    print(f"y shape: {y.shape}")  # (samples, 4)
