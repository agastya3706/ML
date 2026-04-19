"""
agent.py — Deep Q-Network (DQN) Agent
=======================================
Custom DQN implementation from scratch (no Stable Baselines).

Components:
    - Policy network + Target network (3-layer MLP)
    - Experience replay buffer (circular deque)
    - Epsilon-greedy exploration
    - Periodic target network hard update

State:  12-dim vector [vehicle_counts(4), queue_lengths(4), predicted_traffic(4)]
Action: 4 discrete choices (which road gets extended green time:
        0=NS-normal, 1=EW-normal, 2=NS-extended, 3=EW-extended)
"""

import os
import logging
import random
from collections import deque
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Q-Network (MLP)
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    3-layer MLP that approximates Q(s, a) for all actions simultaneously.

    Parameters
    ----------
    state_dim : int
    action_dim : int
    hidden_dims : tuple of hidden layer sizes
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
    ):
        super().__init__()

        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
            ]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, state_dim)

        Returns
        -------
        q_values : (batch, action_dim)
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Circular experience replay buffer.

    Stores transitions: (state, action, reward, next_state, done)
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample a random minibatch."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Deep Q-Network Agent for traffic signal control.

    Parameters
    ----------
    config : dict — project config dict
    device : torch.device (auto-detect if None)
    """

    def __init__(self, config: dict, device: Optional[torch.device] = None):
        rl_cfg = config.get("rl", {})

        self.state_dim = rl_cfg.get("state_dim", 12)
        self.action_dim = rl_cfg.get("action_dim", 4)
        self.gamma = rl_cfg.get("gamma", 0.95)
        self.epsilon = rl_cfg.get("epsilon_start", 1.0)
        self.epsilon_decay = rl_cfg.get("epsilon_decay", 0.995)
        self.epsilon_min = rl_cfg.get("epsilon_min", 0.01)
        self.batch_size = rl_cfg.get("batch_size", 64)
        self.target_update_freq = rl_cfg.get("target_update_freq", 10)
        self.checkpoint_path = rl_cfg.get("checkpoint_path", "rl_agents/best_dqn.pth")
        lr = rl_cfg.get("learning_rate", 0.001)
        memory_size = rl_cfg.get("memory_size", 10000)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Policy and target networks
        self.policy_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # target net is never trained directly

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(memory_size)

        self._step_count = 0      # total optimization steps
        self._episode_count = 0   # episodes completed

        logger.info(
            f"DQNAgent initialized: state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, device={self.device}"
        )

    # -----------------------------------------------------------------------
    # Action Selection
    # -----------------------------------------------------------------------

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Parameters
        ----------
        state : np.ndarray, shape (state_dim,)
        training : bool — if False, use greedy policy (no exploration)

        Returns
        -------
        action : int in [0, action_dim)
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_t)      # (1, action_dim)
        self.policy_net.train()
        return int(q_values.argmax(dim=1).item())

    def act_greedy(self, state: np.ndarray) -> int:
        """Always greedy (evaluation mode)."""
        return self.act(state, training=False)

    # -----------------------------------------------------------------------
    # Experience Storage
    # -----------------------------------------------------------------------

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    # -----------------------------------------------------------------------
    # Learning (Q-Update)
    # -----------------------------------------------------------------------

    def learn(self) -> Optional[float]:
        """
        Sample a minibatch from replay buffer and perform one Q-learning update.

        Returns
        -------
        loss : float or None (if buffer not large enough yet)
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q values: Q(s, a) for taken actions
        q_current = self.policy_net(states_t)                    # (batch, action_dim)
        q_taken = q_current.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (batch,)

        # Target Q values: r + γ * max_a Q_target(s', a)
        with torch.no_grad():
            q_next = self.target_net(next_states_t).max(dim=1).values  # (batch,)
            q_target = rewards_t + self.gamma * q_next * (1.0 - dones_t)

        # Huber loss (smooth L1 — more robust than MSE for RL)
        loss = F.smooth_l1_loss(q_taken, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._step_count += 1
        return loss.item()

    # -----------------------------------------------------------------------
    # Episode End Callbacks
    # -----------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self) -> None:
        """Hard copy policy → target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        logger.debug(f"Target network updated at episode {self._episode_count}")

    def end_episode(self) -> None:
        """Call at end of each episode: decay epsilon + maybe update target."""
        self._episode_count += 1
        self.decay_epsilon()
        if self._episode_count % self.target_update_freq == 0:
            self.update_target_network()

    # -----------------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------------

    def save(self, path: Optional[str] = None, episode: int = 0, reward: float = 0.0) -> None:
        """Save agent checkpoint."""
        import time
        path = path or self.checkpoint_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state_dict = {
            "episode": episode,
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self._step_count,
            "best_reward": reward,
        }
        
        # Retry loop for Windows file locking issues (e.g., OneDrive syncing)
        max_retries = 5
        for attempt in range(max_retries):
            try:
                torch.save(state_dict, path)
                logger.debug(f"DQN checkpoint saved: {path} (episode={episode})")
                break
            except (RuntimeError, OSError) as e:
                # Catch error 32 / sharing violation
                if attempt < max_retries - 1:
                    logger.debug(f"Save failed ({e}). Retrying in 0.5s... ({attempt + 1}/{max_retries})")
                    time.sleep(0.5)
                else:
                    logger.error(f"Failed to save checkpoint after {max_retries} attempts: {e}")
                    raise

    def load(self, path: Optional[str] = None) -> None:
        """Load agent from checkpoint."""
        path = path or self.checkpoint_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"DQN checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_min)
        self._step_count = ckpt.get("step_count", 0)
        logger.info(
            f"DQN checkpoint loaded: {path}\n"
            f"  Episode: {ckpt.get('episode', '?')}, "
            f"  Best reward: {ckpt.get('best_reward', '?'):.4f}"
        )

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.policy_net.parameters())

    def __repr__(self) -> str:
        return (
            f"DQNAgent("
            f"state={self.state_dim}, actions={self.action_dim}, "
            f"epsilon={self.epsilon:.3f}, "
            f"buffer={len(self.replay_buffer)}/{self.replay_buffer.buffer.maxlen}, "
            f"steps={self._step_count})"
        )


if __name__ == "__main__":
    import yaml
    logging.basicConfig(level=logging.DEBUG)

    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    agent = DQNAgent(cfg)
    print(f"Agent: {agent}")
    print(f"Parameters: {agent.num_parameters:,}")

    # Simulate a few transitions
    for _ in range(100):
        state = np.random.rand(12).astype(np.float32)
        action = agent.act(state)
        reward = np.random.uniform(-1, 0)
        next_state = np.random.rand(12).astype(np.float32)
        agent.remember(state, action, reward, next_state, False)

    loss = agent.learn()
    print(f"First learning step loss: {loss}")
    agent.end_episode()
    print(f"After episode: epsilon={agent.epsilon:.4f}")
