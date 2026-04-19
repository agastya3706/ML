"""
train_stgcn.py — STGCN Training Script
========================================
Trains the STGCN prediction model on traffic data collected from SUMO.

Pipeline:
    1. Run SUMO warm-up to collect history buffer
    2. Clean + normalize data
    3. Make sliding window sequences
    4. Train STGCN with MSELoss + Adam
    5. Save best model checkpoint

Usage:
    python models/stgcn/train_stgcn.py
    or via:
    python main.py --mode train_stgcn
"""

import os
import sys
import logging
import time
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TrafficDataset(Dataset):
    """
    Dataset wrapping STGCN input/output pairs.

    Parameters
    ----------
    X : np.ndarray, shape (samples, seq_len, N, 1)
    y : np.ndarray, shape (samples, N)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(X_batch)           # (batch, N)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one validation epoch. Returns average loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


def train_stgcn(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    config: dict,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Full STGCN training pipeline.

    Parameters
    ----------
    model : STGCN instance
    X : (samples, seq_len, N, 1) — input sequences
    y : (samples, N) — targets
    config : project config dict
    device : torch.device (auto-detected if None)

    Returns
    -------
    history : dict with 'train_loss' and 'val_loss' lists
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training STGCN on device: {device}")

    stgcn_cfg = config.get("stgcn", {})
    epochs = stgcn_cfg.get("epochs", 50)
    batch_size = stgcn_cfg.get("batch_size", 32)
    lr = stgcn_cfg.get("learning_rate", 0.001)
    checkpoint_path = stgcn_cfg.get("checkpoint_path", "models/stgcn/best_model.pth")

    # Create dataset and split 80/20
    dataset = TrafficDataset(X, y)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0

    logger.info(f"Training: {len(train_ds)} samples | Validation: {len(val_ds)} samples")
    logger.info(f"Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")

    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - t0

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"Epoch [{epoch:3d}/{epochs}] "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                },
                checkpoint_path,
            )
            logger.debug(f"  ✓ Saved best model at epoch {epoch} (val_loss={val_loss:.6f})")

    logger.info(
        f"\nTraining complete. Best model: epoch {best_epoch}, "
        f"val_loss={best_val_loss:.6f}"
    )
    logger.info(f"Checkpoint saved to: {checkpoint_path}")
    return history


def load_stgcn_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Load a trained STGCN from checkpoint.

    Parameters
    ----------
    model : uninitialised STGCN (same architecture)
    checkpoint_path : str — path to .pth file
    device : torch.device

    Returns
    -------
    model : STGCN with loaded weights, in eval mode
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info(
        f"Loaded STGCN checkpoint from epoch {ckpt.get('epoch', '?')} "
        f"(val_loss={ckpt.get('val_loss', '?'):.6f})"
    )
    return model


# ---------------------------------------------------------------------------
# Environment Factory — auto-detect SUMO, fall back to synthetic
# ---------------------------------------------------------------------------

def _make_env(config: dict):
    """
    Return a traffic environment. Uses SUMO if installed, otherwise
    falls back to the built-in synthetic simulator — same API either way.
    """
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        try:
            from simulation.sumo_env import SumoEnvironment
            logger.info("SUMO detected — using SumoEnvironment (headless).")
            return SumoEnvironment(config, use_gui=False)
        except Exception as e:
            logger.warning(f"SUMO import failed ({e}). Falling back to synthetic env.")

    from simulation.synthetic_env import SyntheticTrafficEnv
    logger.info("SUMO not found — using SyntheticTrafficEnv (Python fallback).")
    return SyntheticTrafficEnv(config)


# ---------------------------------------------------------------------------
# Standalone runner (called by main.py --mode train_stgcn)
# ---------------------------------------------------------------------------

def run_stgcn_training(config: dict) -> None:
    """
    Full standalone STGCN training run.
    Requires SUMO to collect data OR uses cached data if available.
    """
    import yaml
    sys.path.insert(0, ".")

    from preprocessing.graph_builder import build_graph
    from preprocessing.data_cleaning import prepare_stgcn_data
    from models.stgcn.stgcn_model import build_stgcn

    logger.info("=" * 60)
    logger.info("STGCN Training — Starting data collection")
    logger.info("=" * 60)

    # --------------- Data Collection ---------------
    cache_path = "data/processed/stgcn_history.npz"
    os.makedirs("data/processed", exist_ok=True)

    if os.path.exists(cache_path):
        logger.info(f"Loading cached data from {cache_path}")
        npz = np.load(cache_path)
        history = {
            "vehicle_counts": npz["vehicle_counts"],
            "queue_lengths": npz["queue_lengths"],
            "waiting_times": npz["waiting_times"],
        }
    else:
        env = _make_env(config)
        logger.info(f"No cache found — collecting data from {type(env).__name__}...")
        try:
            env.start()
        except Exception as start_err:
            # SUMO may crash on broken network XML — fall back to synthetic env
            logger.warning(
                f"env.start() failed ({start_err}). "
                "Retrying with SyntheticTrafficEnv..."
            )
            try:
                env.close()
            except Exception:
                pass
            from simulation.synthetic_env import SyntheticTrafficEnv
            env = SyntheticTrafficEnv(config)
            env.start()

        warmup_steps = config.get("stgcn", {}).get("history_buffer_size", 500)
        for step in range(warmup_steps):
            env.get_state()   # populates internal history buffer
            still_running = env.step(1)
            if not still_running:
                break

        history = env.get_history_buffer()
        env.close()

        # Cache for future runs
        np.savez(
            cache_path,
            vehicle_counts=history["vehicle_counts"],
            queue_lengths=history["queue_lengths"],
            waiting_times=history["waiting_times"],
        )
        logger.info(f"History data cached to {cache_path}")

    T = history["vehicle_counts"].shape[0]
    logger.info(f"Total data collected: {T} timesteps")

    if T < config.get("stgcn", {}).get("seq_len", 12) + 20:
        raise ValueError(
            f"Not enough data for training: T={T}. "
            f"Run SUMO longer (increase history_buffer_size in config)."
        )

    # --------------- Prepare Sequences ---------------
    X, y, scaler = prepare_stgcn_data(history, config)
    logger.info(f"Sequences ready: X={X.shape}, y={y.shape}")

    # --------------- Build & Train Model ---------------
    graph = build_graph(config)
    model = build_stgcn(config, graph["cheb_polys"])

    history_log = train_stgcn(model, X, y, config)

    # --------------- Plot Training Curve ---------------
    try:
        import matplotlib.pyplot as plt
        os.makedirs("evaluation/results", exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history_log["train_loss"], label="Train Loss")
        ax.plot(history_log["val_loss"], label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("STGCN Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("evaluation/results/stgcn_training_loss.png", dpi=150)
        plt.close()
        logger.info("Training loss plot saved to evaluation/results/stgcn_training_loss.png")
    except Exception as e:
        logger.warning(f"Could not save training plot: {e}")


if __name__ == "__main__":
    import yaml
    logging.basicConfig(level=logging.INFO)

    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    run_stgcn_training(cfg)
