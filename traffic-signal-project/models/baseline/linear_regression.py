"""
linear_regression.py — Linear Regression Baseline
====================================================
A simple baseline that predicts future vehicle counts per road
using sklearn LinearRegression on engineered time-series features.

Used as a benchmark against STGCN in evaluation/comparison.py.
"""

import os
import logging
import pickle
from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)

CHECKPOINT_PATH = "models/baseline/lr_model.pkl"


class LinearRegressionBaseline:
    """
    Linear Regression traffic predictor.

    Flattens a (seq_len, N) window into a feature vector
    and trains one Ridge regressor per approach road.

    Parameters
    ----------
    alpha : float — Ridge regularization strength
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.models = []        # one per road
        self.num_nodes = 0
        self.seq_len = 0
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionBaseline":
        """
        Fit one Ridge model per road.

        Parameters
        ----------
        X : (samples, seq_len, N, 1)  — same shape as STGCN input
        y : (samples, N)              — targets
        """
        samples, seq_len, N, C = X.shape
        self.num_nodes = N
        self.seq_len = seq_len

        # Flatten: (samples, seq_len * N * C)
        X_flat = X.reshape(samples, -1)

        self.models = []
        for node in range(N):
            model = Ridge(alpha=self.alpha)
            model.fit(X_flat, y[:, node])
            self.models.append(model)

        self._fitted = True
        logger.info(f"Fitted {N} Ridge models (seq_len={seq_len}, features={X_flat.shape[1]})")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (samples, seq_len, N, 1) or (1, seq_len, N, 1) for single prediction

        Returns
        -------
        predictions : (samples, N)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        samples = X.shape[0]
        X_flat = X.reshape(samples, -1)

        preds = np.stack(
            [m.predict(X_flat) for m in self.models], axis=1
        )  # (samples, N)
        return preds.astype(np.float32)

    def evaluate(
        self, X: np.ndarray, y: np.ndarray
    ) -> dict:
        """Return MSE and MAE metrics."""
        preds = self.predict(X)
        mse = mean_squared_error(y, preds)
        mae = mean_absolute_error(y, preds)
        logger.info(f"[LinearRegression] MSE={mse:.4f}, MAE={mae:.4f}")
        return {"mse": mse, "mae": mae, "rmse": np.sqrt(mse)}

    def save(self, path: str = CHECKPOINT_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str = CHECKPOINT_PATH) -> "LinearRegressionBaseline":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    T, N, seq_len = 200, 4, 12

    # Dummy data
    raw = np.random.rand(T, N).astype(np.float32)
    X_list, y_list = [], []
    for t in range(T - seq_len - 1):
        X_list.append(raw[t:t+seq_len, :, np.newaxis])
        y_list.append(raw[t+seq_len])
    X = np.stack(X_list)  # (samples, seq_len, N, 1)
    y = np.stack(y_list)  # (samples, N)

    split = int(0.8 * len(X))
    model = LinearRegressionBaseline()
    model.fit(X[:split], y[:split])
    metrics = model.evaluate(X[split:], y[split:])
    print(f"Test metrics: {metrics}")
