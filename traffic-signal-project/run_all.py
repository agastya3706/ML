"""
run_all.py — One-Click Full Pipeline Runner
============================================
Deletes stale checkpoints, retrains STGCN + DQN with fixed settings,
then runs full analytics and generates the report.

Usage (from inside traffic-signal-project):
    python run_all.py
"""

import os
import sys
import time
import shutil
import logging

# ── Force UTF-8 output on Windows ──────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Make sure we're in the project root ───────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

BANNER = "=" * 62

def banner(title: str) -> None:
    logger.info(BANNER)
    logger.info(f"  {title}")
    logger.info(BANNER)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Delete stale files
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 0 | Cleaning stale checkpoints & cache")

stale_files = [
    "data/processed/stgcn_history.npz",
    "rl_agents/best_dqn.pth",
    "models/stgcn/best_model.pth",
    "evaluation/results/analytics_report.png",
    "evaluation/results/analytics_summary.json",
    "evaluation/results/rl_training_history.npz",
    "evaluation/results/rl_training_curves.png",
    "evaluation/results/stgcn_training_loss.png",
]

for f in stale_files:
    if os.path.exists(f):
        os.remove(f)
        logger.info(f"  Removed: {f}")
    else:
        logger.info(f"  Already clean: {f}")

# Make sure output dirs exist
os.makedirs("data/processed",      exist_ok=True)
os.makedirs("models/stgcn",        exist_ok=True)
os.makedirs("rl_agents",           exist_ok=True)
os.makedirs("evaluation/results",  exist_ok=True)
os.makedirs("logs",                exist_ok=True)

logger.info("Clean complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Load config
# ─────────────────────────────────────────────────────────────────────────────
import yaml

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

# Override: force 300 episodes & aligned params for reproducibility
N_EPISODES = 300
CONFIG["sumo"]["max_steps"]                  = 1500   # 1500 steps/episode (25 min of traffic)
CONFIG["rl"]["episodes"]                     = N_EPISODES
CONFIG["rl"]["state_dim"]                    = 14     # extended state: +phase +timer
CONFIG["preprocessing"]["max_queue_length"]  = 20     # realistic single-approach cap
CONFIG["preprocessing"]["max_waiting_time"]  = 60     # realistic signalled intersection wait


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Collect data & Train STGCN
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 1 | Collecting Traffic Data & Training STGCN Predictor")

from models.stgcn.train_stgcn import run_stgcn_training
from preprocessing.graph_builder import build_graph

graph = build_graph(CONFIG)
logger.info(f"Graph built: {graph['num_nodes']} nodes")

stgcn_hist = run_stgcn_training(CONFIG)
logger.info("STGCN training complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Load STGCN for RL training
# ─────────────────────────────────────────────────────────────────────────────
import torch
from models.stgcn.stgcn_model import build_stgcn
from models.stgcn.train_stgcn import load_stgcn_checkpoint

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cheb       = graph["cheb_polys"]
stgcn_model = build_stgcn(CONFIG, cheb)
ckpt_path  = CONFIG["stgcn"]["checkpoint_path"]
stgcn_model = load_stgcn_checkpoint(stgcn_model, ckpt_path, device)
logger.info(f"STGCN loaded from checkpoint: {ckpt_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Train DQN RL Agent
# ─────────────────────────────────────────────────────────────────────────────
banner(f"STEP 2 | Training DQN Agent ({N_EPISODES} episodes)")

from rl_agents.train_agent import run_rl_training

def _make_env(cfg):
    """Auto-detect SUMO, fall back to SyntheticTrafficEnv."""
    if os.environ.get("SUMO_HOME", ""):
        try:
            from simulation.sumo_env import SumoEnvironment
            return SumoEnvironment(cfg, use_gui=False)
        except Exception:
            pass
    from simulation.synthetic_env import SyntheticTrafficEnv
    return SyntheticTrafficEnv(cfg)

rl_hist = run_rl_training(
    CONFIG,
    stgcn_model=stgcn_model,
    device=device,
    num_episodes=N_EPISODES,
    use_gui=False,
)
logger.info("DQN training complete.\n")

# Save RL history for analytics
rl_ckpt_path = CONFIG["rl"]["checkpoint_path"]
hist_path = "evaluation/results/rl_training_history.npz"
if rl_hist:
    ep_rewards = rl_hist.get("episode_rewards", [])
    avg_q      = rl_hist.get("avg_queue_lengths", [])
    avg_w      = rl_hist.get("avg_wait_times", [])
    epsilons   = rl_hist.get("epsilons", [])
    import numpy as np
    np.savez(
        hist_path,
        episode_rewards   = np.array(ep_rewards,  dtype=np.float32),
        avg_queue_lengths = np.array(avg_q,        dtype=np.float32),
        avg_wait_times    = np.array(avg_w,        dtype=np.float32),
        epsilons          = np.array(epsilons,     dtype=np.float32),
    )
    logger.info(f"RL history saved to {hist_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Run Full Analytics
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 3 | Generating Analytics Report")

from evaluation.analyze import run_analysis

results = run_analysis(CONFIG, save=True)
summary = results["summary"]


# ─────────────────────────────────────────────────────────────────────────────
# Final Summary
# ─────────────────────────────────────────────────────────────────────────────
banner("ALL DONE — Results Summary")

if "stgcn" in summary:
    s = summary["stgcn"]
    logger.info("STGCN Predictor Accuracy:")
    logger.info(f"  R2 Score  : {s.get('r2',  0):.4f}  (1.0 = perfect)")
    logger.info(f"  MAE       : {s.get('mae', 0):.5f}")
    logger.info(f"  RMSE      : {s.get('rmse',0):.5f}")

if "rl" in summary:
    r = summary["rl"]
    logger.info("\nDQN Agent Training:")
    logger.info(f"  Episodes   : {r['episodes']}")
    logger.info(f"  Best Reward: {r['best_reward']:.4f}")
    logger.info(f"  Last20 Avg : {r['last20_avg']:.4f}")
    logger.info(f"  Improvement: {r['improvement_pct']:+.1f}%")

if "comparison" in summary:
    logger.info("\nPolicy Comparison (live evaluation):")
    for label, m in summary["comparison"].items():
        logger.info(
            f"  {label:<14}  Queue: {m['avg_queue']:5.2f}  "
            f"Wait: {m['avg_wait']:6.1f}s  "
            f"Reward: {m['avg_reward']:.4f}"
        )

logger.info("\nFiles saved:")
logger.info("  evaluation/results/analytics_report.png")
logger.info("  evaluation/results/analytics_summary.json")
logger.info("  evaluation/results/rl_training_history.npz")
logger.info("  models/stgcn/best_model.pth")
logger.info("  rl_agents/best_dqn.pth")
logger.info("\nTo view the live dashboard:")
logger.info("  python dashboard/app.py  ->  http://localhost:5000")
logger.info("  Analtics page            ->  http://localhost:5000/analytics")
logger.info(BANNER + "\n")
