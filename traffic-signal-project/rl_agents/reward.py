"""
reward.py — Reward Calculator for RL Agent
============================================
Defines the reward signal for the traffic signal control agent.

Reward philosophy:
    The agent must MINIMIZE congestion on ALL approaches simultaneously.
    - A single pile-up on one side is almost as bad as total congestion
    - Clearing the worst road gives a big bonus
    - Balanced queues across all roads is the goal

Design:
    reward = -(w1*mean_queue + w2*max_queue + w3*imbalance + w4*wait) + improvement_bonus
    Normalized to [-1, 0] for stable RL training.
"""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RewardCalculator:
    """
    Multi-objective congestion reward calculator.

    Heavily penalises:
      1. Mean queue length across all roads
      2. The single worst (piled) road — catches asymmetric congestion
      3. Queue imbalance across roads (std dev) — drives balanced service
      4. Mean waiting time across all roads

    Rewards:
      5. Active queue reduction — bonus when roads actually clear

    Parameters
    ----------
    config : dict — project config
    normalize : bool — clamp reward to [-1, 0]
    """

    def __init__(self, config: dict, normalize: bool = True):
        self.normalize = normalize

        pre  = config.get("preprocessing", {})
        self.max_queue = pre.get("max_queue_length", 20)   # realistic cap
        self.max_wait  = pre.get("max_waiting_time", 60)   # realistic cap
        self.num_roads = config.get("intersection", {}).get("num_roads", 4)

    def compute(
        self,
        queue_lengths: np.ndarray,
        waiting_times: np.ndarray,
        prev_queue: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute shaped reward for one timestep.

        Components:
            1. Mean queue penalty     (weight 0.20) — all roads must be clear
            2. Max queue penalty      (weight 0.35) — punish the worst pile-up heavily
            3. Imbalance penalty      (weight 0.20) — penalise uneven loading
            4. Wait time penalty      (weight 0.25) — drivers shouldn't wait long
            5. Improvement bonus      (up to +0.20) — reward active clearing

        Returns
        -------
        reward : float in [-1, 0+] if normalize=True
        """
        q = np.asarray(queue_lengths, dtype=np.float32)
        w = np.asarray(waiting_times, dtype=np.float32)

        # 1. Mean queue — scaled to [0, 1]
        mean_q_norm = q.mean() / (self.max_queue + 1e-8)

        # 2. Worst road (max) — highest weight: agent MUST service the piled road
        max_q_norm = q.max() / (self.max_queue + 1e-8)

        # 3. Imbalance — std dev, scaled to [0, 1]
        #    High std dev = one road is piling while others are clear → bad
        imbalance_norm = q.std() / (self.max_queue + 1e-8)

        # 4. Waiting time — scaled to [0, 1]
        mean_w_norm = w.mean() / (self.max_wait + 1e-8)

        # Weighted penalty — strongest on the worst road, then imbalance
        raw_penalty = (
            0.20 * mean_q_norm      # overall congestion
            + 0.35 * max_q_norm     # worst-road pile-up (dominant signal)
            + 0.20 * imbalance_norm # uneven loading penalty
            + 0.25 * mean_w_norm    # waiting time
        )

        # 5. Improvement bonus — reward clearing any road's queue
        improvement_bonus = 0.0
        if prev_queue is not None:
            pq = np.asarray(prev_queue, dtype=np.float32)
            reduction = float((pq - q).clip(min=0).sum())
            # Scale: clearing 1 vehicle from max_queue road = 0.05 bonus
            improvement_bonus = 0.20 * reduction / (self.num_roads * self.max_queue + 1e-8)

        raw_reward = -raw_penalty + improvement_bonus

        if self.normalize:
            reward = float(np.clip(raw_reward, -1.0, 0.10))
        else:
            reward = float(raw_reward)

        return reward

    def compute_from_state(self, state_dict: Dict[str, np.ndarray]) -> float:
        """
        Compute reward directly from a SUMO state dictionary.
        """
        return self.compute(
            queue_lengths=state_dict["queue_lengths"],
            waiting_times=state_dict["waiting_times"],
        )

    def info(self, queue_lengths: np.ndarray, waiting_times: np.ndarray) -> dict:
        """Return detailed reward breakdown for logging."""
        q = np.asarray(queue_lengths, dtype=np.float32)
        w = np.asarray(waiting_times, dtype=np.float32)
        return {
            "queue_sum":     float(q.sum()),
            "queue_mean":    float(q.mean()),
            "queue_max":     float(q.max()),
            "queue_std":     float(q.std()),
            "wait_mean":     float(w.mean()),
            "queue_per_road": q.tolist(),
            "wait_per_road":  w.tolist(),
            "reward":         self.compute(q, w),
        }


if __name__ == "__main__":
    import yaml
    logging.basicConfig(level=logging.DEBUG)

    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    calc = RewardCalculator(cfg)

    # Best case: empty intersection
    print(f"Best reward (empty):          {calc.compute(np.zeros(4), np.zeros(4)):.4f}")

    # Balanced moderate load
    q_bal = np.array([2.0, 2.0, 2.0, 2.0])
    w_bal = np.array([8.0, 8.0, 8.0, 8.0])
    print(f"Balanced moderate:            {calc.compute(q_bal, w_bal):.4f}")

    # Imbalanced: one road piling (the bad case we want to punish)
    q_imb = np.array([15.0, 1.0, 1.0, 1.0])
    w_imb = np.array([50.0, 5.0, 5.0, 5.0])
    print(f"One road piling (imbalanced): {calc.compute(q_imb, w_imb):.4f}")

    # Worst case: all roads maxed
    q_worst = np.full(4, 20.0)
    w_worst = np.full(4, 60.0)
    print(f"Worst (full congestion):      {calc.compute(q_worst, w_worst):.4f}")
