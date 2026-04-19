"""
analyze.py — Comprehensive Model Analytics & Reward Analysis
=============================================================
Tracks reward system, analyses STGCN prediction accuracy, DQN learning
curves, and compares all metrics against baselines.

Generates:
    evaluation/results/analytics_report.png  — full visual report
    evaluation/results/analytics_summary.json — machine-readable summary

Usage:
    python evaluation/analyze.py
    OR visit http://localhost:5000/analytics (when dashboard is running)
"""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

logger = logging.getLogger(__name__)

# ─── Style Constants ──────────────────────────────────────────────────────────
BG      = "#0d1520"
BG2     = "#111b28"
GRID    = "#1e2d3e"
TEXT    = "#e2e8f0"
MUTED   = "#64748b"
GREEN   = "#22c55e"
BLUE    = "#3b82f6"
CYAN    = "#06b6d4"
ORANGE  = "#f97316"
RED     = "#ef4444"
PURPLE  = "#a855f7"
YELLOW  = "#eab308"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG2,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "grid.color":        GRID,
    "text.color":        TEXT,
    "legend.facecolor":  BG2,
    "legend.edgecolor":  GRID,
    "font.family":       "DejaVu Sans",
    "font.size":         9,
})


# ─── Data Loaders ─────────────────────────────────────────────────────────────

def load_rl_history(path: str = "evaluation/results/rl_training_history.npz") -> dict:
    """Load saved RL training history."""
    if not os.path.exists(path):
        return {}
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}


def load_stgcn_checkpoint(config: dict) -> dict:
    """Load STGCN checkpoint metadata."""
    import torch
    ckpt_path = config.get("stgcn", {}).get("checkpoint_path", "models/stgcn/best_model.pth")
    if not os.path.exists(ckpt_path):
        return {}
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {
        "epoch":    ckpt.get("epoch", "?"),
        "val_loss": ckpt.get("val_loss", float("nan")),
    }


def evaluate_stgcn_accuracy(config: dict) -> dict:
    """Run STGCN on validation data and compute accuracy metrics."""
    try:
        import torch
        from preprocessing.graph_builder import build_graph
        from models.stgcn.stgcn_model import build_stgcn
        from models.stgcn.train_stgcn import load_stgcn_checkpoint, TrafficDataset
        from preprocessing.data_cleaning import prepare_stgcn_data
        from torch.utils.data import DataLoader

        cache = "data/processed/stgcn_history.npz"
        if not os.path.exists(cache):
            return {}
        npz = np.load(cache)
        history = {k: npz[k] for k in npz.files}
        X, y, _ = prepare_stgcn_data(history, config)

        ckpt_path = config.get("stgcn", {}).get("checkpoint_path", "models/stgcn/best_model.pth")
        if not os.path.exists(ckpt_path):
            return {}

        graph  = build_graph(config)
        model  = build_stgcn(config, graph["cheb_polys"])
        device = torch.device("cpu")
        model  = load_stgcn_checkpoint(model, ckpt_path, device)

        dataset = TrafficDataset(X, y)
        loader  = DataLoader(dataset, batch_size=64, shuffle=False)

        all_pred, all_true = [], []
        with torch.no_grad():
            for xb, yb in loader:
                pred = model(xb.to(device)).cpu().numpy()
                all_pred.append(pred)
                all_true.append(yb.numpy())

        pred_arr = np.vstack(all_pred)
        true_arr = np.vstack(all_true)

        mae  = float(np.mean(np.abs(pred_arr - true_arr)))
        mse  = float(np.mean((pred_arr - true_arr) ** 2))
        rmse = float(np.sqrt(mse))
        # R² score
        ss_res = np.sum((true_arr - pred_arr) ** 2)
        ss_tot = np.sum((true_arr - true_arr.mean()) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-10))

        # Per-road accuracy
        per_road_mae = [float(np.mean(np.abs(pred_arr[:,i] - true_arr[:,i]))) for i in range(4)]

        return {
            "mae": mae, "mse": mse, "rmse": rmse, "r2": r2,
            "per_road_mae": per_road_mae,
            "pred_sample":  pred_arr[:60].tolist(),
            "true_sample":  true_arr[:60].tolist(),
            "n_samples":    len(true_arr),
        }
    except Exception as e:
        logger.warning(f"STGCN accuracy eval failed: {e}")
        return {}


def run_episode_comparison(config: dict, n_steps: int = 600) -> dict:
    """
    Run one episode each for DQN, Fixed, and Random — collect metrics.
    Actions are applied every signal_step steps (same as during training).
    """
    try:
        import random as rnd
        from simulation.synthetic_env import SyntheticTrafficEnv
        from rl_agents.reward import RewardCalculator

        rc = RewardCalculator(config)
        # Must match training: one action per signal_step = 20 (hardcoded in TrafficEnvironment)
        signal_step = 20
        max_c = config.get("preprocessing", {}).get("max_vehicle_count", 50)
        max_q = config.get("preprocessing", {}).get("max_queue_length", 30)
        results = {}

        def run_policy(policy_fn, label, color, seed=42):
            """Run a policy for n_steps, applying action every signal_step."""
            env = SyntheticTrafficEnv(config, seed=seed)
            env.start()
            rs, qs, ws = [], [], []
            step = 0
            while step < n_steps:
                # Observe state
                s = env.get_state()
                # Decide action
                action = policy_fn(s, step)
                # Apply action then advance signal_step steps
                env.apply_action(action, green_duration=signal_step)
                for _ in range(signal_step):
                    if step >= n_steps:
                        break
                    still = env.step(1)
                    s2 = env.get_state()
                    r  = rc.compute(s2["queue_lengths"], s2["waiting_times"])
                    rs.append(r)
                    qs.append(float(s2["queue_lengths"].mean()))
                    ws.append(float(s2["waiting_times"].mean()))
                    step += 1
                    if not still:
                        break
            env.close()
            return {"rewards": rs, "queues": qs, "waits": ws, "color": color}

        # ── DQN Agent ──
        try:
            import torch
            from rl_agents.agent import DQNAgent
            agent = DQNAgent(config)
            ckpt = config.get("rl", {}).get("checkpoint_path", "rl_agents/best_dqn.pth")
            if os.path.exists(ckpt):
                agent.load(ckpt)

                def dqn_policy(s, step):
                    max_wait_ = config.get("preprocessing", {}).get("max_waiting_time", 90)
                    cn = np.clip(s["vehicle_counts"] / (max_c + 1e-8), 0, 1)
                    qn = np.clip(s["queue_lengths"]   / (max_q + 1e-8), 0, 1)
                    # STGCN prediction unavailable here — use zeros
                    pred = np.zeros(4, dtype=np.float32)
                    # Phase feature: 1.0 = NS green
                    phase = s.get("current_phase", 0)
                    ns_green = np.array([1.0 if phase == 0 else 0.0], dtype=np.float32)
                    timer = np.array([0.5], dtype=np.float32)  # mid-phase assumption
                    state_dim = config.get("rl", {}).get("state_dim", 14)
                    raw = np.concatenate([cn, qn, pred, ns_green, timer]).astype(np.float32)
                    # Handle state_dim mismatch gracefully
                    if len(raw) != state_dim:
                        raw = np.resize(raw, (state_dim,))
                    return agent.act_greedy(raw)

                results["DQN Agent"] = run_policy(dqn_policy, "DQN Agent", GREEN)
        except Exception as e:
            logger.warning(f"DQN run failed: {e}")

        # ── Fixed Timing (natural auto-cycle, no apply_action) ──
        env_f = SyntheticTrafficEnv(config, seed=42)
        env_f.start()
        rs_f, qs_f, ws_f = [], [], []
        for step in range(n_steps):
            env_f.step(1)
            s2 = env_f.get_state()
            r = rc.compute(s2["queue_lengths"], s2["waiting_times"])
            rs_f.append(r); qs_f.append(float(s2["queue_lengths"].mean())); ws_f.append(float(s2["waiting_times"].mean()))
        env_f.close()
        results["Fixed Timing"] = {"rewards": rs_f, "queues": qs_f, "waits": ws_f, "color": RED}

        # ── Random Agent (same signal_step interval) ──
        def random_policy(s, step):
            return rnd.randint(0, 3)
        results["Random"] = run_policy(random_policy, "Random", ORANGE, seed=42)

        return results
    except Exception as e:
        logger.warning(f"Episode comparison failed: {e}")
        return {}



def smooth(data: list, window: int = 15) -> np.ndarray:
    arr = np.array(data, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


# ─── Main Analysis & Plot ──────────────────────────────────────────────────────

def run_analysis(config: dict, save: bool = True) -> dict:
    """
    Run complete analysis. Returns summary dict, saves PNG report.
    """
    logger.info("Running comprehensive analytics...")
    os.makedirs("evaluation/results", exist_ok=True)

    rl_hist   = load_rl_history()
    stgcn_acc = evaluate_stgcn_accuracy(config)
    comparison = run_episode_comparison(config)
    ckpt_meta = load_stgcn_checkpoint(config)

    # ─── Figure Layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    fig.suptitle(
        "Traffic Signal AI  —  Training Analytics & Model Accuracy",
        fontsize=16, fontweight="bold", color=TEXT, y=0.98
    )

    gs = gridspec.GridSpec(
        3, 4,
        figure=fig,
        hspace=0.52, wspace=0.38,
        left=0.06, right=0.97, top=0.93, bottom=0.07
    )

    axes = {
        "reward_curve":   fig.add_subplot(gs[0, :2]),
        "queue_curve":    fig.add_subplot(gs[0, 2:]),
        "episode_reward": fig.add_subplot(gs[1, :2]),
        "stgcn_pred":     fig.add_subplot(gs[1, 2:]),
        "metric_bars":    fig.add_subplot(gs[2, :2]),
        "per_road":       fig.add_subplot(gs[2, 2]),
        "kpi_text":       fig.add_subplot(gs[2, 3]),
    }

    def style_ax(ax, title, xlabel="", ylabel=""):
        ax.set_facecolor(BG2)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8, color=TEXT)
        ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
        ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.spines[:].set_color(GRID)
        ax.tick_params(labelsize=8)

    # ── Plot 1: Training reward curve ──────────────────────────────────────────
    ax = axes["reward_curve"]
    style_ax(ax, "DQN Training — Episode Reward", "Episode", "Reward")
    if "episode_rewards" in rl_hist:
        ep_r = rl_hist["episode_rewards"]
        eps  = np.arange(1, len(ep_r) + 1)
        ax.plot(eps, ep_r, alpha=0.2, color=BLUE, linewidth=0.8, label="Raw")
        if len(ep_r) > 5:
            sm = smooth(ep_r.tolist(), window=min(20, len(ep_r)//3 or 1))
            ax.plot(np.arange(len(sm)) + 1, sm, color=BLUE, linewidth=2.2, label="Smoothed (20ep)")
        ax.axhline(y=float(np.mean(ep_r[-20:])) if len(ep_r) >= 20 else float(np.mean(ep_r)),
                   color=GREEN, linestyle="--", linewidth=1.2, alpha=0.7, label="Recent avg")
        ax.legend(fontsize=8)
        ax.set_xlim(1, len(ep_r))
    else:
        ax.text(0.5, 0.5, "No training history found.\nRun: python main.py --mode train_all",
                ha="center", va="center", transform=ax.transAxes, color=MUTED, fontsize=9)

    # ── Plot 2: Training queue + wait ──────────────────────────────────────────
    ax = axes["queue_curve"]
    style_ax(ax, "DQN Training — Queue & Wait Trends", "Episode", "Value")
    if "avg_queue_lengths" in rl_hist and "avg_wait_times" in rl_hist:
        eps   = np.arange(1, len(rl_hist["avg_queue_lengths"]) + 1)
        q_sm  = smooth(rl_hist["avg_queue_lengths"].tolist())
        w_sm  = smooth(rl_hist["avg_wait_times"].tolist())
        ax2   = ax.twinx()
        ax2.spines[:].set_color(GRID)
        ax.plot(np.arange(len(q_sm))+1, q_sm, color=ORANGE, linewidth=2, label="Avg Queue (veh)")
        ax2.plot(np.arange(len(w_sm))+1, w_sm, color=CYAN, linewidth=2, linestyle="--", label="Avg Wait (s)")
        ax.set_ylabel("Avg Queue (vehicles)", color=ORANGE, fontsize=8)
        ax2.set_ylabel("Avg Wait (seconds)", color=CYAN, fontsize=8)
        ax.tick_params(axis="y", labelcolor=ORANGE)
        ax2.tick_params(axis="y", labelcolor=CYAN)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No training history found.", ha="center", va="center",
                transform=ax.transAxes, color=MUTED, fontsize=9)

    # ── Plot 3: Comparison — episode reward over time ──────────────────────────
    ax = axes["episode_reward"]
    style_ax(ax, "Policy Comparison — Cumulative Reward (500 Steps)", "Step", "Reward")
    if comparison:
        for label, data in comparison.items():
            rs_sm = smooth(data["rewards"], window=10)
            steps = np.arange(1, len(rs_sm) + 1)
            ax.plot(steps, rs_sm, color=data["color"], linewidth=2, label=label)
            ax.fill_between(steps, rs_sm, alpha=0.08, color=data["color"])
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No comparison data.", ha="center", va="center",
                transform=ax.transAxes, color=MUTED, fontsize=9)

    # ── Plot 4: STGCN Prediction vs Actual ────────────────────────────────────
    ax = axes["stgcn_pred"]
    style_ax(ax, "STGCN Traffic Prediction — Predicted vs Actual", "Sample", "Normalized Flow")
    if stgcn_acc and "pred_sample" in stgcn_acc:
        pred = np.array(stgcn_acc["pred_sample"])[:, 0]  # Road 0
        true = np.array(stgcn_acc["true_sample"])[:, 0]
        xs   = np.arange(len(pred))
        ax.plot(xs, true, color=BLUE,  linewidth=1.8, label="Actual (Road N)",  alpha=0.9)
        ax.plot(xs, pred, color=GREEN, linewidth=1.5, label="Predicted",  linestyle="--", alpha=0.9)
        ax.fill_between(xs, true, pred, alpha=0.08, color=PURPLE, label="Error")
        ax.legend(fontsize=8)
        r2_str = f"R²={stgcn_acc['r2']:.3f}  MAE={stgcn_acc['mae']:.4f}  RMSE={stgcn_acc['rmse']:.4f}"
        ax.set_title(f"STGCN Prediction Accuracy\n{r2_str}", fontsize=9, fontweight="bold", color=TEXT, pad=6)
    else:
        ax.text(0.5, 0.5, "No STGCN data.\nTrain with: python main.py --mode train_stgcn",
                ha="center", va="center", transform=ax.transAxes, color=MUTED, fontsize=9)

    # ── Plot 5: Bar comparison of metrics ────────────────────────────────────
    ax = axes["metric_bars"]
    style_ax(ax, "Method Comparison — Avg Queue & Wait Time", "", "")
    if comparison:
        methods = list(comparison.keys())
        colors  = [comparison[m]["color"] for m in methods]
        avg_q   = [float(np.mean(comparison[m]["queues"])) for m in methods]
        avg_w   = [float(np.mean(comparison[m]["waits"]))  for m in methods]
        x = np.arange(len(methods))
        w = 0.35
        b1 = ax.bar(x - w/2, avg_q, w, color=[c + "cc" for c in colors], edgecolor=GRID, linewidth=0.8, label="Avg Queue (veh)")
        b2 = ax.bar(x + w/2, avg_w, w, color=colors, edgecolor=GRID, linewidth=0.8, alpha=0.6, label="Avg Wait (s)")
        ax.bar_label(b1, fmt="%.1f", fontsize=8, color=TEXT, padding=3)
        ax.bar_label(b2, fmt="%.1f", fontsize=8, color=TEXT, padding=3)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=8)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No comparison data.", ha="center", va="center",
                transform=ax.transAxes, color=MUTED, fontsize=9)

    # ── Plot 6: Per-road MAE ───────────────────────────────────────────────────
    ax = axes["per_road"]
    style_ax(ax, "STGCN Per-Road MAE", "Road", "MAE")
    road_labels = ["North", "South", "East", "West"]
    if stgcn_acc and "per_road_mae" in stgcn_acc:
        vals   = stgcn_acc["per_road_mae"]
        colors = [GREEN, BLUE, CYAN, ORANGE]
        bars = ax.bar(road_labels, vals, color=colors, edgecolor=GRID, linewidth=0.8)
        ax.bar_label(bars, fmt="%.4f", fontsize=8, color=TEXT, padding=3)
        ax.set_ylim(0, max(vals) * 1.4 if vals else 1.0)
    else:
        ax.text(0.5, 0.5, "No STGCN accuracy data.", ha="center", va="center",
                transform=ax.transAxes, color=MUTED, fontsize=9)

    # ── Plot 7: KPI Text Panel ─────────────────────────────────────────────────
    ax = axes["kpi_text"]
    ax.set_facecolor(BG2)
    ax.axis("off")
    ax.spines[:].set_color(GRID)

    kpi_lines = ["ANALYTICS SUMMARY\n"]
    # STGCN
    if stgcn_acc:
        kpi_lines += [
            "STGCN Predictor",
            f"  R² Score:  {stgcn_acc['r2']:.4f}",
            f"  MAE:       {stgcn_acc['mae']:.5f}",
            f"  RMSE:      {stgcn_acc['rmse']:.5f}",
            f"  Val Loss:  {ckpt_meta.get('val_loss', float('nan')):.5f}",
            f"  Best Ep:   {ckpt_meta.get('epoch', '?')}",
            f"  Samples:   {stgcn_acc.get('n_samples', '?')}",
            "",
        ]
    # RL
    if "episode_rewards" in rl_hist:
        ep_r = rl_hist["episode_rewards"]
        kpi_lines += [
            "DQN Agent",
            f"  Episodes:  {len(ep_r)}",
            f"  Best Ep.R: {float(max(ep_r)):.4f}",
            f"  Last20 R:  {float(np.mean(ep_r[-20:])):.4f}",
            f"  Final eps: {float(rl_hist.get('epsilons', [0])[-1]) if 'epsilons' in rl_hist else '?'}",
            "",
        ]
    # Comparison
    if comparison and "DQN Agent" in comparison:
        dqn_q = np.mean(comparison["DQN Agent"]["queues"])
        dqn_w = np.mean(comparison["DQN Agent"]["waits"])
        kpi_lines += ["vs Baselines (500 steps)"]
        for label, data in comparison.items():
            q_imp = (np.mean(data["queues"]) - dqn_q) / (np.mean(data["queues"]) + 1e-8) * 100
            kpi_lines.append(f"  {label:<12} +{q_imp:5.1f}% queue")

    full_text = "\n".join(kpi_lines)
    ax.text(
        0.05, 0.97, full_text,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=8.5,
        color=TEXT,
        fontfamily="monospace",
        linespacing=1.7,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = "evaluation/results/analytics_report.png"
    if save:
        plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
        plt.close()
        logger.info(f"Analytics report saved to {out_path}")

    # ── JSON Summary ──────────────────────────────────────────────────────────
    summary = {}
    if stgcn_acc:
        summary["stgcn"] = {k: v for k, v in stgcn_acc.items() if k not in ("pred_sample", "true_sample")}
    if "episode_rewards" in rl_hist:
        ep_r = rl_hist["episode_rewards"]
        summary["rl"] = {
            "episodes":         int(len(ep_r)),
            "best_reward":      float(max(ep_r)),
            "last20_avg":       float(np.mean(ep_r[-20:])),
            "first_reward":     float(ep_r[0]),
            "improvement_pct":  float((ep_r[-1] - ep_r[0]) / (abs(ep_r[0]) + 1e-8) * 100),
        }
    if comparison:
        summary["comparison"] = {
            label: {
                "avg_queue": float(np.mean(data["queues"])),
                "avg_wait":  float(np.mean(data["waits"])),
                "avg_reward": float(np.mean(data["rewards"])),
            }
            for label, data in comparison.items()
        }
    json_path = "evaluation/results/analytics_summary.json"
    if save:
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Analytics summary saved to {json_path}")

    return {"summary": summary, "image_path": out_path}


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    results = run_analysis(cfg)
    summary = results["summary"]

    print("\n" + "=" * 60)
    print("  ANALYTICS COMPLETE")
    print("=" * 60)
    if "stgcn" in summary:
        s = summary["stgcn"]
        print(f"\n  STGCN Prediction Accuracy")
        print(f"    R² Score  : {s.get('r2', 0):.4f}  (1.0 = perfect)")
        print(f"    MAE       : {s.get('mae', 0):.5f}")
        print(f"    RMSE      : {s.get('rmse', 0):.5f}")
    if "rl" in summary:
        r = summary["rl"]
        print(f"\n  DQN Agent Training")
        print(f"    Episodes  : {r['episodes']}")
        print(f"    Best Ep.R : {r['best_reward']:.4f}")
        print(f"    Last20 Avg: {r['last20_avg']:.4f}")
        print(f"    Improvement: {r['improvement_pct']:+.1f}%")
    if "comparison" in summary:
        print(f"\n  Policy Comparison (500 steps)")
        for label, m in summary["comparison"].items():
            print(f"    {label:<14} Avg Queue: {m['avg_queue']:.2f}  Avg Wait: {m['avg_wait']:.1f}s  R: {m['avg_reward']:.4f}")
    print(f"\n  Report saved to: evaluation/results/analytics_report.png")
    print("=" * 60 + "\n")
