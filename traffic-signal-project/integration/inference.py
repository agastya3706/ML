"""
inference.py — Single-Step Inference
======================================
Load trained STGCN + DQN models and run inference on a given state.
Used for real-time deployment or testing without full SUMO simulation.

Usage:
    from integration.inference import TrafficInference
    infer = TrafficInference(config)
    action = infer.get_action(vehicle_counts, queue_lengths)
"""

import os
import logging
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TrafficInference:
    """
    Lightweight inference interface for deployment.

    Loads STGCN + DQN from checkpoints and provides
    a simple get_action() call.

    Parameters
    ----------
    config : dict — project config
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        stgcn_cfg = config.get("stgcn", {})
        self.seq_len = stgcn_cfg.get("seq_len", 12)
        self.num_nodes = config.get("intersection", {}).get("num_roads", 4)

        pre_cfg = config.get("preprocessing", {})
        self.max_count = pre_cfg.get("max_vehicle_count", 50)
        self.max_queue = pre_cfg.get("max_queue_length", 30)

        self._count_buffer = np.zeros((self.seq_len, self.num_nodes), dtype=np.float32)
        self.stgcn = None
        self.agent = None

        self._load_models()

    def _load_models(self) -> None:
        """Load STGCN and DQN from checkpoints."""
        from preprocessing.graph_builder import build_graph
        from models.stgcn.stgcn_model import build_stgcn
        from models.stgcn.train_stgcn import load_stgcn_checkpoint
        from rl_agents.agent import DQNAgent

        # Load STGCN
        stgcn_ckpt = self.config.get("stgcn", {}).get("checkpoint_path", "")
        if stgcn_ckpt and os.path.exists(stgcn_ckpt):
            graph = build_graph(self.config)
            model = build_stgcn(self.config, graph["cheb_polys"])
            self.stgcn = load_stgcn_checkpoint(model, stgcn_ckpt, self.device)
            logger.info("STGCN loaded for inference.")
        else:
            logger.warning("STGCN checkpoint not found — predictions will be zeros.")

        # Load DQN
        dqn_ckpt = self.config.get("rl", {}).get("checkpoint_path", "")
        if dqn_ckpt and os.path.exists(dqn_ckpt):
            agent = DQNAgent(self.config, device=self.device)
            agent.load(dqn_ckpt)
            self.agent = agent
            logger.info("DQN agent loaded for inference.")
        else:
            logger.warning("DQN checkpoint not found — using random actions.")

    def update_history(self, vehicle_counts: np.ndarray) -> None:
        """
        Update the rolling count buffer with the latest observation.

        Parameters
        ----------
        vehicle_counts : (N,) current vehicle counts
        """
        normalized = np.clip(vehicle_counts / (self.max_count + 1e-8), 0.0, 1.0)
        self._count_buffer = np.roll(self._count_buffer, shift=-1, axis=0)
        self._count_buffer[-1] = normalized

    def predict_traffic(self) -> np.ndarray:
        """
        Run STGCN prediction on current history buffer.

        Returns
        -------
        predicted : (N,) predicted vehicle counts (normalized)
        """
        if self.stgcn is None:
            return np.zeros(self.num_nodes, dtype=np.float32)
        x = self._count_buffer[:, :, np.newaxis][np.newaxis]  # (1, T, N, 1)
        return self.stgcn.predict(x, self.device)

    def get_action(
        self,
        vehicle_counts: np.ndarray,
        queue_lengths: np.ndarray,
    ) -> Tuple[int, np.ndarray]:
        """
        Get the DQN action for current traffic state.

        Parameters
        ----------
        vehicle_counts : (N,) current vehicle counts
        queue_lengths  : (N,) current queue lengths

        Returns
        -------
        action : int (0-3)
        state  : (12,) state vector used
        """
        self.update_history(vehicle_counts)
        predicted = self.predict_traffic()

        counts_norm = np.clip(vehicle_counts / (self.max_count + 1e-8), 0.0, 1.0)
        queues_norm = np.clip(queue_lengths / (self.max_queue + 1e-8), 0.0, 1.0)
        state = np.concatenate([counts_norm, queues_norm, predicted]).astype(np.float32)

        if self.agent is not None:
            action = self.agent.act_greedy(state)
        else:
            import random
            action = random.randint(0, 3)

        return action, state

    @property
    def action_meanings(self):
        return {
            0: "NS Green (normal 30s)",
            1: "EW Green (normal 30s)",
            2: "NS Green (extended 45s)",
            3: "EW Green (extended 45s)",
        }


if __name__ == "__main__":
    import yaml, sys
    sys.path.insert(0, ".")
    logging.basicConfig(level=logging.INFO)

    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    infer = TrafficInference(cfg)

    # Simulate 5 inference calls
    for step in range(5):
        counts = np.random.randint(0, 20, 4).astype(np.float32)
        queues = np.random.randint(0, 10, 4).astype(np.float32)
        action, state = infer.get_action(counts, queues)
        print(
            f"Step {step+1}: counts={counts}, queues={queues} -> "
            f"action={action} ({infer.action_meanings[action]})"
        )
