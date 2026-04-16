"""
comparison.py — Baseline Comparison Evaluator
===============================================
Evaluates and compares all methods:
    1. DQN Agent (trained)
    2. Fixed Timing (default SUMO plan)
    3. Random Agent (random action each step)
    4. Linear Regression predictor (greedy based on predicted queues)

Produces:
    - Metrics table (printed + saved as CSV)
    - Bar chart comparison (saved as PNG)

Usage:
    python evaluation/comparison.py
    or:
    python main.py --mode eval
"""

import os
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _make_env(config: dict, use_gui: bool = False):
    """Auto-detect SUMO; fall back to SyntheticTrafficEnv."""
    if os.environ.get("SUMO_HOME", ""):
        try:
            from simulation.sumo_env import SumoEnvironment
            return SumoEnvironment(config, use_gui=use_gui)
        except Exception:
            pass
    from simulation.synthetic_env import SyntheticTrafficEnv
    return SyntheticTrafficEnv(config)

# ---------------------------------------------------------------------------
# Policy Runners
# ---------------------------------------------------------------------------

def run_fixed_timing(config: dict, num_episodes: int = 5) -> List[Dict]:
    """
    Baseline: No RL control — let SUMO run with its default fixed signal plan.
    """
    from simulation.sumo_env import SumoEnvironment
    from evaluation.metrics import compute_all_metrics

    results = []
    for ep in range(num_episodes):
        env = _make_env(config, use_gui=False)
        env.start()

        queues, waits = [], []
        max_steps = config.get("sumo", {}).get("max_steps", 3600)
        for _ in range(max_steps):
            state = env.get_state()
            queues.append(state["queue_lengths"])
            waits.append(state["waiting_times"])
            if not env.step(1):
                break
        env.close()

        q_arr = np.array(queues)
        w_arr = np.array(waits)
        # Total reward approximation
        from rl_agents.reward import RewardCalculator
        rc = RewardCalculator(config)
        total_r = sum(rc.compute(q_arr[t], w_arr[t]) for t in range(len(q_arr)))
        results.append(compute_all_metrics(q_arr, w_arr, total_r, label="Fixed Timing"))

    logger.info(f"Fixed timing evaluated over {num_episodes} episodes.")
    return results


def run_random_agent(config: dict, num_episodes: int = 5) -> List[Dict]:
    """
    Baseline: Random signal control (uniform random action each decision step).
    """
    import random
    from simulation.sumo_env import SumoEnvironment
    from rl_agents.reward import RewardCalculator
    from evaluation.metrics import compute_all_metrics

    rc = RewardCalculator(config)
    signal_step = config.get("intersection", {}).get("min_green", 10)
    max_steps = config.get("sumo", {}).get("max_steps", 3600)
    action_dim = config.get("rl", {}).get("action_dim", 4)

    results = []
    for ep in range(num_episodes):
        env = _make_env(config, use_gui=False)
        env.start()

        queues, waits, total_r = [], [], 0.0
        step = 0
        while step < max_steps:
            action = random.randint(0, action_dim - 1)
            env.apply_action(action)
            still = env.step(signal_step)
            state = env.get_state()
            queues.append(state["queue_lengths"])
            waits.append(state["waiting_times"])
            total_r += rc.compute(state["queue_lengths"], state["waiting_times"])
            step += signal_step
            if not still:
                break
        env.close()

        q_arr = np.array(queues)
        w_arr = np.array(waits)
        results.append(compute_all_metrics(q_arr, w_arr, total_r, label="Random"))

    return results


def run_dqn_agent(
    config: dict,
    agent,
    stgcn_model=None,
    num_episodes: int = 5,
    use_gui: bool = False,
) -> List[Dict]:
    """
    Evaluate trained DQN agent.
    """
    import torch
    from simulation.sumo_env import SumoEnvironment
    from rl_agents.environment import TrafficEnvironment
    from evaluation.metrics import compute_all_metrics

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for ep in range(num_episodes):
        base_env = _make_env(config, use_gui=use_gui)
        env = TrafficEnvironment(config, base_env, stgcn_model=stgcn_model, device=device)
        state = env.reset()

        queues, waits = [], []
        done = False
        while not done:
            action = agent.act_greedy(state)
            state, _, done, info = env.step(action)
            queues.append(info["queue_lengths"])
            waits.append(info["waiting_times"])

        q_arr = np.array(queues)
        w_arr = np.array(waits)
        results.append(
            compute_all_metrics(q_arr, w_arr, env.total_reward, label="DQN Agent")
        )
        env.close()

    return results


# ---------------------------------------------------------------------------
# Full Comparison
# ---------------------------------------------------------------------------

def run_full_comparison(
    config: dict,
    agent=None,
    stgcn_model=None,
    num_episodes: int = 5,
) -> List[Dict]:
    """
    Run all baselines and DQN, return aggregated metrics for each.

    Parameters
    ----------
    config : dict
    agent : DQNAgent (None to skip DQN)
    stgcn_model : STGCN (None ok)
    num_episodes : int — episodes per method

    Returns
    -------
    all_metrics : list of aggregated metric dicts (one per method)
    """
    from evaluation.metrics import aggregate_metrics, print_metrics_table

    all_aggregated = []

    logger.info("\n--- Running: Fixed Timing Baseline ---")
    fixed_results = run_fixed_timing(config, num_episodes)
    all_aggregated.append(aggregate_metrics(fixed_results))

    logger.info("\n--- Running: Random Agent Baseline ---")
    random_results = run_random_agent(config, num_episodes)
    all_aggregated.append(aggregate_metrics(random_results))

    if agent is not None:
        logger.info("\n--- Running: DQN Agent ---")
        dqn_results = run_dqn_agent(config, agent, stgcn_model, num_episodes)
        all_aggregated.append(aggregate_metrics(dqn_results))

    print_metrics_table(all_aggregated)
    _save_results(all_aggregated)
    _plot_comparison(all_aggregated)

    return all_aggregated


def _save_results(metrics_list: List[Dict]) -> None:
    """Save metrics to CSV."""
    os.makedirs("evaluation/results", exist_ok=True)
    import csv
    path = "evaluation/results/comparison.csv"
    keys = ["label", "avg_queue_length", "avg_waiting_time",
            "throughput_ratio", "peak_queue", "total_reward", "episodes"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics_list)
    logger.info(f"Results saved to {path}")


def _plot_comparison(metrics_list: List[Dict]) -> None:
    """Generate bar chart comparing all methods."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        labels = [m["label"] for m in metrics_list]
        colors = ["#E74C3C", "#E67E22", "#27AE60", "#3498DB"][:len(labels)]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle("Traffic Signal Control — Method Comparison", fontsize=14, fontweight="bold")

        # Avg queue
        vals = [m["avg_queue_length"] for m in metrics_list]
        axes[0].bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
        axes[0].set_title("Avg Queue Length (↓ better)")
        axes[0].set_ylabel("Vehicles")
        axes[0].grid(True, alpha=0.3, axis="y")

        # Avg wait
        vals = [m["avg_waiting_time"] for m in metrics_list]
        axes[1].bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
        axes[1].set_title("Avg Waiting Time (↓ better)")
        axes[1].set_ylabel("Seconds")
        axes[1].grid(True, alpha=0.3, axis="y")

        # Total reward
        vals = [m["total_reward"] for m in metrics_list]
        axes[2].bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
        axes[2].set_title("Total Reward (↑ better)")
        axes[2].set_ylabel("Reward")
        axes[2].grid(True, alpha=0.3, axis="y")

        for ax in axes:
            ax.tick_params(axis="x", rotation=15)

        plt.tight_layout()
        path = "evaluation/results/comparison_chart.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Comparison chart saved to {path}")

    except Exception as e:
        logger.warning(f"Could not save comparison chart: {e}")


if __name__ == "__main__":
    import yaml, sys
    sys.path.insert(0, ".")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    run_full_comparison(cfg, agent=None, num_episodes=3)
