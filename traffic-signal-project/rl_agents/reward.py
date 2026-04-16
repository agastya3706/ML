"""
reward.py — Reward Calculator for RL Agent
============================================
Defines the reward signal for the traffic signal control agent.

Reward philosophy:
    The agent should MINIMIZE congestion.
    Congestion = long queues + high waiting times.
    → Reward = negative of congestion metrics.

Primary formula:
    reward = -(w_q * sum(queue_lengths) + w_w * sum(waiting_times))

Normalized to [-1, 0] range for stable RL training.
"""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RewardCalculator:
    """
    Congestion-based reward calculator.

    Parameters
    ----------
    config : dict — project config (preprocessing section for max values)
    w_queue : float — weight for queue length component
    w_wait : float — weight for waiting time component
    normalize : bool — if True, normalize reward to [-1, 0]
    """

    def __init__(
        self,
        config: dict,
        w_queue: float = 0.6,
        w_wait: float = 0.4,
        normalize: bool = True,
    ):
        self.w_queue = w_queue
        self.w_wait = w_wait
        self.normalize = normalize

        pre = config.get("preprocessing", {})
        self.max_queue = pre.get("max_queue_length", 30)
        self.max_wait = pre.get("max_waiting_time", 120)
        num_roads = config.get("intersection", {}).get("num_roads", 4)

        # Maximum possible raw penalty (for normalization)
        self.max_penalty = (
            w_queue * num_roads * self.max_queue
            + w_wait * num_roads * self.max_wait
        )

    def compute(
        self,
        queue_lengths: np.ndarray,
        waiting_times: np.ndarray,
        prev_queue: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute the reward for one timestep.

        Parameters
        ----------
        queue_lengths : np.ndarray, shape (N,)
            Current halting vehicle count per approach road.
        waiting_times : np.ndarray, shape (N,)
            Current waiting time per approach lane (seconds).
        prev_queue : np.ndarray or None
            Previous queue lengths (for improvement bonus, optional).

        Returns
        -------
        reward : float
            Value in [-1, 0] if normalize=True, else raw negative value.
        """
        queue_penalty = self.w_queue * float(queue_lengths.sum())
        wait_penalty = self.w_wait * float(waiting_times.sum())
        raw_penalty = queue_penalty + wait_penalty

        # Optional: bonus for reducing queues
        improvement_bonus = 0.0
        if prev_queue is not None:
            reduction = float((prev_queue - queue_lengths).sum())
            improvement_bonus = 0.1 * max(reduction, 0.0)  # small positive bonus

        raw_reward = -raw_penalty + improvement_bonus

        if self.normalize:
            # Clamp to [-max_penalty, 0] then normalize to [-1, 0]
            clamped = max(-self.max_penalty, raw_reward)
            reward = clamped / (self.max_penalty + 1e-8)
        else:
            reward = raw_reward

        return float(reward)

    def compute_from_state(self, state_dict: Dict[str, np.ndarray]) -> float:
        """
        Compute reward directly from a SUMO state dictionary.

        Parameters
        ----------
        state_dict : dict with 'queue_lengths' and 'waiting_times' keys

        Returns
        -------
        reward : float
        """
        return self.compute(
            queue_lengths=state_dict["queue_lengths"],
            waiting_times=state_dict["waiting_times"],
        )

    def info(self, queue_lengths: np.ndarray, waiting_times: np.ndarray) -> dict:
        """Return detailed reward breakdown for logging."""
        return {
            "queue_sum": float(queue_lengths.sum()),
            "wait_sum": float(waiting_times.sum()),
            "queue_per_road": queue_lengths.tolist(),
            "wait_per_road": waiting_times.tolist(),
            "reward": self.compute(queue_lengths, waiting_times),
        }


if __name__ == "__main__":
    import yaml
    logging.basicConfig(level=logging.DEBUG)

    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    calc = RewardCalculator(cfg)

    # Best case: empty intersection
    q_best = np.zeros(4)
    w_best = np.zeros(4)
    print(f"Best reward (empty): {calc.compute(q_best, w_best):.4f}")  # Should be 0.0

    # Moderate congestion
    q_med = np.array([5, 3, 8, 2])
    w_med = np.array([20, 10, 30, 5])
    print(f"Moderate congestion: {calc.compute(q_med, w_med):.4f}")

    # Worst case: maximum congestion
    q_worst = np.full(4, 30)
    w_worst = np.full(4, 120)
    print(f"Worst reward (max congestion): {calc.compute(q_worst, w_worst):.4f}")  # Should be -1.0
