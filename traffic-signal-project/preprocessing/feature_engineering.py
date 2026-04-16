"""
feature_engineering.py — Traffic Feature Engineering
======================================================
Adds extra features to raw traffic data:
  - Time-of-day encoding (sin/cos cyclical)
  - Rolling statistics (mean, std)
  - Rate-of-change (delta)

These features can augment the STGCN input or be used
directly by baseline ML models (linear regression).
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Time Encoding
# ---------------------------------------------------------------------------

def encode_time_of_day(
    timesteps: np.ndarray,
    period: int = 3600,
) -> np.ndarray:
    """
    Cyclical encoding of time-of-day using sin + cos.

    Parameters
    ----------
    timesteps : np.ndarray, shape (T,)
        Simulation step indices (seconds from start).
    period : int
        Full cycle in seconds (default 3600 = 1 hour looping).

    Returns
    -------
    time_features : np.ndarray, shape (T, 2)
        Columns: [sin_time, cos_time]
    """
    angle = 2.0 * np.pi * (timesteps % period) / period
    sin_t = np.sin(angle)
    cos_t = np.cos(angle)
    return np.stack([sin_t, cos_t], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Rolling Statistics
# ---------------------------------------------------------------------------

def rolling_mean(data: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Compute rolling mean over time axis.

    Parameters
    ----------
    data : np.ndarray, shape (T, N)
    window : int

    Returns
    -------
    np.ndarray, shape (T, N) — NaN-padded at start
    """
    T, N = data.shape
    df = pd.DataFrame(data)
    rolled = df.rolling(window=window, min_periods=1).mean()
    return rolled.values.astype(np.float32)


def rolling_std(data: np.ndarray, window: int = 5) -> np.ndarray:
    """Compute rolling standard deviation over time axis."""
    df = pd.DataFrame(data)
    rolled = df.rolling(window=window, min_periods=1).std().fillna(0.0)
    return rolled.values.astype(np.float32)


# ---------------------------------------------------------------------------
# Rate of Change
# ---------------------------------------------------------------------------

def compute_delta(data: np.ndarray) -> np.ndarray:
    """
    Compute first-order difference (rate-of-change).

    Parameters
    ----------
    data : np.ndarray, shape (T, N)

    Returns
    -------
    np.ndarray, shape (T, N)  — first row padded with 0
    """
    delta = np.diff(data, axis=0, prepend=data[:1, :])
    return delta.astype(np.float32)


# ---------------------------------------------------------------------------
# Full Feature Matrix (for baseline models)
# ---------------------------------------------------------------------------

def build_feature_matrix(
    vehicle_counts: np.ndarray,
    queue_lengths: np.ndarray,
    waiting_times: Optional[np.ndarray] = None,
    timesteps: Optional[np.ndarray] = None,
    rolling_window: int = 5,
) -> np.ndarray:
    """
    Build a flat feature matrix for baseline (sklearn) models.

    Features per timestep:
        - vehicle_counts          (N,)
        - queue_lengths           (N,)
        - rolling_mean(counts)    (N,)
        - rolling_std(counts)     (N,)
        - delta(counts)           (N,)
        - sin_time, cos_time      (2,)  [optional]
        - waiting_times           (N,)  [optional]

    Parameters
    ----------
    vehicle_counts : (T, N)
    queue_lengths  : (T, N)
    waiting_times  : (T, N) optional
    timesteps      : (T,) optional — simulation step indices
    rolling_window : int

    Returns
    -------
    features : np.ndarray, shape (T, num_features)
    """
    T, N = vehicle_counts.shape
    parts = [
        vehicle_counts,
        queue_lengths,
        rolling_mean(vehicle_counts, rolling_window),
        rolling_std(vehicle_counts, rolling_window),
        compute_delta(vehicle_counts),
    ]

    if waiting_times is not None:
        parts.append(waiting_times)

    if timesteps is not None:
        time_feats = encode_time_of_day(timesteps)  # (T, 2)
        parts.append(time_feats)

    features = np.concatenate(parts, axis=1)
    logger.debug(f"Feature matrix built: shape={features.shape}")
    return features.astype(np.float32)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    T, N = 200, 4

    counts = np.random.randint(0, 30, (T, N)).astype(np.float32)
    queues = np.random.randint(0, 15, (T, N)).astype(np.float32)
    waits  = np.random.uniform(0, 60, (T, N)).astype(np.float32)
    steps  = np.arange(T, dtype=np.float32)

    feats = build_feature_matrix(counts, queues, waits, steps)
    print(f"Feature matrix shape: {feats.shape}")
    # Expected: (200, 4*5 + 4 + 2) = (200, 26)
