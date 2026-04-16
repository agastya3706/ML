"""
lstm_model.py — LSTM Baseline
================================
A single-layer LSTM that predicts future vehicle counts per road.
Serves as a temporal-only baseline (no spatial graph structure)
to compare against STGCN which models both space and time.

Architecture:
    Input (batch, seq_len, N) → LSTM → FC → Output (batch, N)
"""

import os
import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

CHECKPOINT_PATH = "models/baseline/lstm_best.pth"


class LSTMBaseline(nn.Module):
    """
    LSTM-based traffic predictor.

    Parameters
    ----------
    num_nodes : int     — number of approach roads (N)
    hidden_size : int   — LSTM hidden state dimension
    num_layers : int    — stacked LSTM layers
    dropout : float
    """

    def __init__(
        self,
        num_nodes: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_nodes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_nodes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, num_nodes)

        Returns
        -------
        out : (batch, num_nodes)
        """
        lstm_out, _ = self.lstm(x)       # (batch, seq_len, hidden_size)
        last_out = lstm_out[:, -1, :]    # (batch, hidden_size) — last timestep
        out = self.fc(last_out)          # (batch, num_nodes)
        return out

    def predict(self, X: np.ndarray, device: torch.device) -> np.ndarray:
        """
        Parameters
        ----------
        X : (samples, seq_len, N, 1) — same as STGCN input format

        Returns
        -------
        predictions : (samples, N)
        """
        self.eval()
        # Squeeze channel dim: (samples, seq_len, N)
        X_sq = X.squeeze(-1)
        X_t = torch.FloatTensor(X_sq).to(device)
        with torch.no_grad():
            out = self.forward(X_t)
        return out.cpu().numpy()


# ---------------------------------------------------------------------------
# Training Utilities
# ---------------------------------------------------------------------------

def train_lstm_baseline(
    X: np.ndarray,
    y: np.ndarray,
    config: dict,
    device: Optional[torch.device] = None,
) -> LSTMBaseline:
    """
    Train LSTM baseline.

    Parameters
    ----------
    X : (samples, seq_len, N, 1)
    y : (samples, N)
    config : project config dict

    Returns
    -------
    model : trained LSTMBaseline
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_nodes = config.get("intersection", {}).get("num_roads", 4)
    epochs = 30
    batch_size = config.get("stgcn", {}).get("batch_size", 32)
    lr = 0.001

    # Squeeze channel dim for LSTM: (samples, seq_len, N)
    X_sq = X.squeeze(-1)

    split = int(0.8 * len(X_sq))
    X_train, X_val = X_sq[:split], X_sq[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = LSTMBaseline(num_nodes=num_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_ds)

        if epoch % 10 == 0:
            logger.info(f"LSTM Epoch [{epoch}/{epochs}] Val Loss: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    # Load best
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    logger.info(f"LSTM baseline trained. Best val loss: {best_val:.6f}")
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    T, N, seq_len = 300, 4, 12
    raw = np.random.rand(T, N).astype(np.float32)
    X_list, y_list = [], []
    for t in range(T - seq_len - 1):
        X_list.append(raw[t:t+seq_len, :, np.newaxis])
        y_list.append(raw[t+seq_len])
    X = np.stack(X_list)
    y = np.stack(y_list)

    import yaml
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = train_lstm_baseline(X, y, cfg)
    preds = model.predict(X[-10:], torch.device("cpu"))
    print(f"LSTM predictions shape: {preds.shape}")
