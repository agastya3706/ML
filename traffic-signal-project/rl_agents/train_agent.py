"""
train_agent.py — DQN Training Loop
=====================================
Trains the DQN agent in the SUMO traffic environment.

Training flow per episode:
    1. Reset SUMO environment
    2. Fill history buffer for STGCN
    3. Loop: act → SUMO step → collect transition → learn
    4. Decay epsilon, sync target net
    5. Log + checkpoint best agent

Usage:
    python rl_agents/train_agent.py
    or:
    python main.py --mode train_rl
"""

import os
import sys
import logging
import time
from typing import Optional, List

import numpy as np

logger = logging.getLogger(__name__)


def _make_env(config: dict, use_gui: bool = False):
    """Auto-detect SUMO; fall back to SyntheticTrafficEnv if not installed."""
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        try:
            from simulation.sumo_env import SumoEnvironment
            logger.info("SUMO detected — using SumoEnvironment.")
            return SumoEnvironment(config, use_gui=use_gui)
        except Exception as e:
            logger.warning(f"SUMO failed ({e}). Using synthetic env.")
    from simulation.synthetic_env import SyntheticTrafficEnv
    logger.info("Using SyntheticTrafficEnv (SUMO not found).")
    return SyntheticTrafficEnv(config)


def run_rl_training(
    config: dict,
    stgcn_model=None,
    use_gui: bool = False,
    resume: bool = False,
) -> dict:
    """
    Full DQN training loop.

    Parameters
    ----------
    config : dict — project config
    stgcn_model : STGCN (optional) — if None, predicted traffic = zeros
    use_gui : bool — launch SUMO-GUI (slow) or headless (fast)
    resume : bool — load existing checkpoint and continue training

    Returns
    -------
    training_history : dict with episode rewards, losses, etc.
    """
    from rl_agents.environment import TrafficEnvironment
    from rl_agents.agent import DQNAgent
    import torch

    rl_cfg = config.get("rl", {})
    num_episodes = rl_cfg.get("episodes", 300)
    checkpoint_path = rl_cfg.get("checkpoint_path", "rl_agents/best_dqn.pth")
    log_interval = config.get("logging", {}).get("log_interval", 10)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs("evaluation/results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"RL Training on device: {device}")

    # --------------- Initialize Components ---------------
    base_env = _make_env(config, use_gui=use_gui)
    env = TrafficEnvironment(config, base_env, stgcn_model=stgcn_model, device=device)
    agent = DQNAgent(config, device=device)

    if resume and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        logger.info("Resumed from checkpoint.")

    # Training history
    history = {
        "episode_rewards": [],
        "episode_lengths": [],
        "avg_queue_lengths": [],
        "avg_wait_times": [],
        "losses": [],
        "epsilons": [],
    }
    best_avg_reward = float("-inf")

    logger.info("=" * 60)
    logger.info(f"DQN Training: {num_episodes} episodes")
    logger.info(f"Agent: {agent}")
    logger.info("=" * 60)

    for episode in range(1, num_episodes + 1):
        t0 = time.time()

        # --------------- Episode Reset ---------------
        state = env.reset()
        episode_reward = 0.0
        episode_losses: List[float] = []
        episode_queues: List[float] = []
        episode_waits: List[float] = []
        done = False

        # --------------- Episode Loop ---------------
        while not done:
            # Act
            action = agent.act(state, training=True)

            # Step environment
            next_state, reward, done, info = env.step(action)

            # Store transition
            agent.remember(state, action, reward, next_state, done)

            # Learn
            loss = agent.learn()
            if loss is not None:
                episode_losses.append(loss)

            # Track metrics
            episode_reward += reward
            episode_queues.append(float(info["queue_lengths"].sum()))
            episode_waits.append(float(info["waiting_times"].sum()))

            state = next_state

        # --------------- End of Episode ---------------
        agent.end_episode()
        elapsed = time.time() - t0

        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        avg_queue = np.mean(episode_queues) if episode_queues else 0.0
        avg_wait = np.mean(episode_waits) if episode_waits else 0.0

        history["episode_rewards"].append(episode_reward)
        history["episode_lengths"].append(env.current_step)
        history["avg_queue_lengths"].append(avg_queue)
        history["avg_wait_times"].append(avg_wait)
        history["losses"].append(avg_loss)
        history["epsilons"].append(agent.epsilon)

        # --------------- Save Best Checkpoint ---------------
        # Use rolling average of last 20 episodes
        recent_rewards = history["episode_rewards"][-20:]
        avg_recent = np.mean(recent_rewards)

        if avg_recent > best_avg_reward and episode >= 20:
            best_avg_reward = avg_recent
            agent.save(checkpoint_path, episode=episode, reward=avg_recent)
            logger.info(f"  ★ New best! Avg reward: {avg_recent:.4f}")

        # --------------- Logging ---------------
        if episode % log_interval == 0 or episode == 1:
            logger.info(
                f"Ep [{episode:4d}/{num_episodes}] | "
                f"Reward: {episode_reward:8.4f} | "
                f"Avg(20): {avg_recent:8.4f} | "
                f"Queue: {avg_queue:6.2f} | "
                f"Wait: {avg_wait:7.2f}s | "
                f"Loss: {avg_loss:.6f} | "
                f"eps: {agent.epsilon:.3f} | "
                f"Time: {elapsed:.1f}s"
            )

    # --------------- End of Training ---------------
    env.close()
    logger.info(f"\nTraining complete! Best model saved to: {checkpoint_path}")

    # Save training history
    try:
        np.savez(
            "evaluation/results/rl_training_history.npz",
            episode_rewards=np.array(history["episode_rewards"]),
            avg_queue_lengths=np.array(history["avg_queue_lengths"]),
            avg_wait_times=np.array(history["avg_wait_times"]),
            losses=np.array(history["losses"]),
            epsilons=np.array(history["epsilons"]),
        )
        logger.info("Training history saved.")
    except Exception as e:
        logger.warning(f"Could not save history: {e}")

    # Plot training curves
    _plot_training_history(history)

    return history


def _plot_training_history(history: dict) -> None:
    """Plot and save training metrics."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("DQN Training Progress", fontsize=14)

        episodes = range(1, len(history["episode_rewards"]) + 1)

        # Reward
        ax = axes[0, 0]
        ax.plot(episodes, history["episode_rewards"], alpha=0.3, color="#4C9BE8", label="Raw")
        # Smoothed
        rewards = np.array(history["episode_rewards"])
        window = min(20, len(rewards))
        smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
        ax.plot(range(window, len(rewards)+1), smoothed, color="#1E5FA8", label="Smoothed (20ep)")
        ax.set_title("Episode Reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Queue length
        ax = axes[0, 1]
        ax.plot(episodes, history["avg_queue_lengths"], color="#E74C3C")
        ax.set_title("Avg Queue Length per Step")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Vehicles")
        ax.grid(True, alpha=0.3)

        # Waiting time
        ax = axes[1, 0]
        ax.plot(episodes, history["avg_wait_times"], color="#E67E22")
        ax.set_title("Avg Waiting Time per Step")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Seconds")
        ax.grid(True, alpha=0.3)

        # Epsilon
        ax = axes[1, 1]
        ax.plot(episodes, history["epsilons"], color="#27AE60")
        ax.set_title("Epsilon (Exploration Rate)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = "evaluation/results/rl_training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Training plots saved to {plot_path}")

    except Exception as e:
        logger.warning(f"Could not save training plots: {e}")


if __name__ == "__main__":
    import yaml
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Try to load STGCN if available
    stgcn_model = None
    try:
        import sys, torch
        sys.path.insert(0, ".")
        from preprocessing.graph_builder import build_graph
        from models.stgcn.stgcn_model import build_stgcn
        from models.stgcn.train_stgcn import load_stgcn_checkpoint

        ckpt = cfg["stgcn"]["checkpoint_path"]
        if os.path.exists(ckpt):
            graph = build_graph(cfg)
            stgcn_model = build_stgcn(cfg, graph["cheb_polys"])
            stgcn_model = load_stgcn_checkpoint(stgcn_model, ckpt)
            logger.info("STGCN model loaded for RL training.")
        else:
            logger.warning("No STGCN checkpoint found. Running without predictions.")
    except Exception as e:
        logger.warning(f"STGCN loading failed: {e}. Continuing without predictions.")

    run_rl_training(cfg, stgcn_model=stgcn_model, use_gui=False)
