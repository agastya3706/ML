"""
metrics.py — Evaluation Metrics
==================================
Computes traffic signal performance metrics from a completed episode.

Metrics:
    - Average queue length (vehicles)
    - Average waiting time (seconds)
    - Throughput (vehicles completed)
    - Average travel time (seconds)
    - Total reward
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_avg_queue_length(queue_history: np.ndarray) -> float:
    """
    Average queue length across all roads and timesteps.

    Parameters
    ----------
    queue_history : np.ndarray, shape (T, N)
        Queue length per road per timestep.

    Returns
    -------
    float — mean halting vehicles
    """
    return float(np.mean(queue_history))


def compute_avg_waiting_time(wait_history: np.ndarray) -> float:
    """
    Average waiting time across all roads and timesteps.

    Parameters
    ----------
    wait_history : np.ndarray, shape (T, N)

    Returns
    -------
    float — mean waiting seconds
    """
    return float(np.mean(wait_history))


def compute_throughput(wait_history: np.ndarray, threshold: float = 0.0) -> float:
    """
    Approximate throughput as count of timesteps where waiting time is zero
    (vehicle passed through without waiting).

    Parameters
    ----------
    wait_history : (T, N)
    threshold : float — max wait considered "zero"

    Returns
    -------
    float — throughput ratio [0, 1]
    """
    zero_wait = (wait_history <= threshold).sum()
    return float(zero_wait) / wait_history.size


def compute_peak_queue(queue_history: np.ndarray) -> float:
    """Maximum queue length observed across any road and time."""
    return float(np.max(queue_history))


def compute_all_metrics(
    queue_history: np.ndarray,
    wait_history: np.ndarray,
    total_reward: float,
    label: str = "Agent",
) -> Dict[str, float]:
    """
    Compute and return all metrics as a dict.

    Parameters
    ----------
    queue_history : (T, N)
    wait_history  : (T, N)
    total_reward  : float
    label         : str — identifier for this run

    Returns
    -------
    dict with metric names and float values
    """
    metrics = {
        "label": label,
        "avg_queue_length": compute_avg_queue_length(queue_history),
        "avg_waiting_time": compute_avg_waiting_time(wait_history),
        "throughput_ratio": compute_throughput(wait_history),
        "peak_queue": compute_peak_queue(queue_history),
        "total_reward": total_reward,
        "episodes": 1,
    }

    logger.info(
        f"[{label}] "
        f"AvgQueue={metrics['avg_queue_length']:.2f}veh | "
        f"AvgWait={metrics['avg_waiting_time']:.2f}s | "
        f"Throughput={metrics['throughput_ratio']:.2%} | "
        f"PeakQ={metrics['peak_queue']:.1f}veh | "
        f"TotalReward={total_reward:.4f}"
    )
    return metrics


def aggregate_metrics(metrics_list: List[Dict]) -> Dict[str, float]:
    """
    Average metrics across multiple evaluation episodes.

    Parameters
    ----------
    metrics_list : list of dicts from compute_all_metrics

    Returns
    -------
    aggregated dict with means
    """
    keys = ["avg_queue_length", "avg_waiting_time", "throughput_ratio",
            "peak_queue", "total_reward"]
    agg = {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}
    agg["label"] = metrics_list[0]["label"] if metrics_list else "Unknown"
    agg["episodes"] = len(metrics_list)
    return agg


def print_metrics_table(metrics_list: List[Dict]) -> None:
    """Print a formatted comparison table."""
    header = f"\n{'Method':<20} {'AvgQueue':>10} {'AvgWait(s)':>12} {'Throughput':>12} {'PeakQueue':>11} {'TotalReward':>13}"
    print(header)
    print("-" * len(header))
    for m in metrics_list:
        print(
            f"{m['label']:<20} "
            f"{m['avg_queue_length']:>10.2f} "
            f"{m['avg_waiting_time']:>12.2f} "
            f"{m['throughput_ratio']:>11.2%} "
            f"{m['peak_queue']:>11.1f} "
            f"{m['total_reward']:>13.4f}"
        )
    print()
