"""
environment.py — RL Training Environment (Gym-Style)
======================================================
Wraps SumoEnvironment + STGCN + RewardCalculator into a clean
OpenAI Gym-like interface for the DQN agent.

State vector (12-dim):
    [vehicle_counts(4), queue_lengths(4), predicted_traffic(4)]
     normalized to [0, 1]

Action space:
    Discrete(4): which road group gets extended green
    0 = NS green normal
    1 = EW green normal
    2 = NS green extended
    3 = EW green extended

Episode:
    Starts with SUMO reset.
    Ends when max_steps reached.
"""

import logging
from typing import Optional, Tuple, Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TrafficEnvironment:
    """
    Gym-style environment wrapping SUMO + STGCN + Reward.

    Parameters
    ----------
    config : dict — project config
    sumo_env : SumoEnvironment — must be created before passing
    stgcn_model : STGCN or None — if None, predicted_traffic = zeros
    device : torch.device
    use_gui : bool — override SUMO GUI mode
    """

    def __init__(
        self,
        config: dict,
        sumo_env,
        stgcn_model=None,
        device: Optional[torch.device] = None,
        use_gui: Optional[bool] = None,
    ):
        from rl_agents.reward import RewardCalculator

        self.config = config
        self.sumo_env = sumo_env
        self.stgcn_model = stgcn_model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        rl_cfg = config.get("rl", {})
        self.state_dim = rl_cfg.get("state_dim", 12)
        self.action_dim = rl_cfg.get("action_dim", 4)

        stgcn_cfg = config.get("stgcn", {})
        self.seq_len = stgcn_cfg.get("seq_len", 12)
        self.num_nodes = config.get("intersection", {}).get("num_roads", 4)

        pre_cfg = config.get("preprocessing", {})
        self.max_count = pre_cfg.get("max_vehicle_count", 50)
        self.max_queue = pre_cfg.get("max_queue_length", 30)

        # Signal step: how many SUMO steps per RL action
        self.signal_step = config.get("intersection", {}).get(
            "min_green", 10
        )  # apply action for min_green seconds

        self.reward_calc = RewardCalculator(config)

        # Rolling buffer for STGCN input
        self._count_history = np.zeros((self.seq_len, self.num_nodes), dtype=np.float32)
        self._prev_queues: Optional[np.ndarray] = None
        self._step = 0
        self._total_reward = 0.0

    # -----------------------------------------------------------------------
    # Gym Interface
    # -----------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """
        Reset environment for a new episode.

        Returns
        -------
        state : np.ndarray, shape (state_dim,)
        """
        self.sumo_env.reset()
        self._count_history = np.zeros((self.seq_len, self.num_nodes), dtype=np.float32)
        self._prev_queues = None
        self._step = 0
        self._total_reward = 0.0

        # Warm up: advance `seq_len` steps to fill history buffer
        for _ in range(self.seq_len):
            sumo_state = self.sumo_env.get_state()
            counts = sumo_state["vehicle_counts"]
            self._update_history(counts)
            self.sumo_env.step(1)

        state = self._build_state()
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Apply action, advance simulation, observe next state and reward.

        Parameters
        ----------
        action : int — from DQN agent

        Returns
        -------
        next_state : np.ndarray, shape (state_dim,)
        reward : float
        done : bool
        info : dict
        """
        # Apply action to SUMO traffic light
        self.sumo_env.apply_action(action, green_duration=self.signal_step)

        # Advance simulation for `signal_step` seconds
        still_running = self.sumo_env.step(steps=self.signal_step)

        # Observe new state
        sumo_state = self.sumo_env.get_state()
        counts = sumo_state["vehicle_counts"]
        self._update_history(counts)

        # Calculate reward
        reward_info = self.sumo_env.get_reward_info()
        reward = self.reward_calc.compute(
            queue_lengths=reward_info["queue_lengths"],
            waiting_times=reward_info["waiting_times"],
            prev_queue=self._prev_queues,
        )
        self._prev_queues = reward_info["queue_lengths"].copy()
        self._total_reward += reward

        # Build next state
        next_state = self._build_state()
        done = not still_running
        self._step += 1

        info = {
            "step": self._step,
            "sumo_step": sumo_state["step"],
            "queue_lengths": reward_info["queue_lengths"],
            "waiting_times": reward_info["waiting_times"],
            "vehicle_counts": counts,
            "action": action,
            "reward": reward,
            "total_reward": self._total_reward,
            "done": done,
        }

        return next_state, reward, done, info

    # -----------------------------------------------------------------------
    # State Construction
    # -----------------------------------------------------------------------

    def _update_history(self, vehicle_counts: np.ndarray) -> None:
        """Roll the history buffer and append latest counts."""
        normalized = np.clip(vehicle_counts / (self.max_count + 1e-8), 0.0, 1.0)
        self._count_history = np.roll(self._count_history, shift=-1, axis=0)
        self._count_history[-1] = normalized

    def _get_stgcn_prediction(self) -> np.ndarray:
        """
        Get STGCN prediction for the next timestep.
        Returns zeros if model not available or history not full.
        """
        if self.stgcn_model is None:
            return np.zeros(self.num_nodes, dtype=np.float32)

        try:
            # Shape: (1, seq_len, N, 1)
            x = self._count_history[:, :, np.newaxis][np.newaxis]  # (1, seq, N, 1)
            pred = self.stgcn_model.predict(x, self.device)          # (N,)
            return np.clip(pred, 0.0, 1.0)
        except Exception as e:
            logger.warning(f"STGCN prediction failed: {e}")
            return np.zeros(self.num_nodes, dtype=np.float32)

    def _build_state(self) -> np.ndarray:
        """
        Build the 12-dim state vector.

        Returns
        -------
        state : np.ndarray, shape (12,)
            [vehicle_counts(4)/max, queue_lengths(4)/max, predicted(4)]
        """
        sumo_state = self.sumo_env.get_state()

        counts_norm = np.clip(
            sumo_state["vehicle_counts"] / (self.max_count + 1e-8), 0.0, 1.0
        )
        queues_norm = np.clip(
            sumo_state["queue_lengths"] / (self.max_queue + 1e-8), 0.0, 1.0
        )
        predicted = self._get_stgcn_prediction()

        state = np.concatenate([counts_norm, queues_norm, predicted]).astype(np.float32)
        assert state.shape == (self.state_dim,), (
            f"State shape mismatch: {state.shape} vs {self.state_dim}"
        )
        return state

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def total_reward(self) -> float:
        return self._total_reward

    @property
    def current_step(self) -> int:
        return self._step

    def close(self) -> None:
        """Clean up SUMO subprocess."""
        self.sumo_env.close()

    def __repr__(self) -> str:
        return (
            f"TrafficEnvironment("
            f"state_dim={self.state_dim}, "
            f"actions={self.action_dim}, "
            f"step={self._step}, "
            f"stgcn={'yes' if self.stgcn_model else 'no'})"
        )
