"""
dashboard/app.py — Flask Backend for Traffic Simulation Dashboard
==================================================================
Runs the trained DQN agent + SyntheticTrafficEnv in a background thread,
streams state updates via Server-Sent Events (SSE) to the frontend.

Usage:
    cd traffic-signal-project
    python dashboard/app.py
    # Open http://localhost:5000
"""

import sys
import os
import json
import time
import threading
import math

# Add parent dir to path so we can import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, Response, jsonify, request
import yaml
import numpy as np

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global simulation state (shared between background thread and Flask routes)
# ---------------------------------------------------------------------------

SIM = {
    "running": False,
    "paused": False,
    "step": 0,
    "episode": 1,
    "max_steps": 3600,
    "speed": 1,
    "intersections": [
        {
            "phase": 0, "phase_timer": 30, "queues": [0.0]*4, 
            "waiting_times": [0.0]*4, "vehicle_counts": [0]*4, "action_name": "—", "action": 0
        },
        {
            "phase": 0, "phase_timer": 30, "queues": [0.0]*4, 
            "waiting_times": [0.0]*4, "vehicle_counts": [0]*4, "action_name": "—", "action": 0
        }
    ],
    "transit_east": 0,  # Cars currently travelling env1 -> env2
    "transit_west": 0,  # Cars currently travelling env2 -> env1
    "avg_queue": 0.0,
    "avg_wait": 0.0,
    "reward_history": [],
    "queue_history": [],
}

ACTION_NAMES = {
    0: "NS Green (30s)", 1: "EW Green (30s)",
    2: "NS Extended (50s)", 3: "EW Extended (50s)",
}

PHASE_NAMES = {0: "NS Green", 1: "NS Yellow", 2: "EW Green", 3: "EW Yellow"}

# ---------------------------------------------------------------------------
# Load config and trained models
# ---------------------------------------------------------------------------

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yaml")
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# Override for faster visual simulation
CONFIG["sumo"]["max_steps"] = 1000


def _load_agent():
    """Load the trained DQN agent if checkpoint exists."""
    try:
        import torch
        from rl_agents.agent import DQNAgent
        agent = DQNAgent(CONFIG)
        ckpt = CONFIG.get("rl", {}).get("checkpoint_path", "rl_agents/best_dqn.pth")
        if os.path.exists(ckpt):
            agent.load(ckpt)
            return agent
        else:
            return None
    except Exception as e:
        print(f"[dashboard] Could not load DQN agent: {e}")
        return None


def _load_stgcn():
    """Load the trained STGCN model if checkpoint exists."""
    try:
        import torch
        from preprocessing.graph_builder import build_graph
        from models.stgcn.stgcn_model import build_stgcn
        from models.stgcn.train_stgcn import load_stgcn_checkpoint
        graph = build_graph(CONFIG)
        model = build_stgcn(CONFIG, graph["cheb_polys"])
        ckpt = CONFIG.get("stgcn", {}).get("checkpoint_path", "models/stgcn/best_model.pth")
        if os.path.exists(ckpt):
            device = __import__("torch").device("cpu")
            return load_stgcn_checkpoint(model, ckpt, device)
        return None
    except Exception as e:
        print(f"[dashboard] Could not load STGCN: {e}")
        return None


# Load models once at startup
print("[dashboard] Loading trained models...")
AGENT = _load_agent()
STGCN = _load_stgcn()
print(f"[dashboard] DQN Agent: {'Loaded' if AGENT else 'Not found (using fixed timing)'}")
print(f"[dashboard] STGCN: {'Loaded' if STGCN else 'Not found (predictions disabled)'}")

# ---------------------------------------------------------------------------
# Background simulation thread
# ---------------------------------------------------------------------------

_lock = threading.Lock()

def _get_agent_state(env, s_dict):
    """Helper to build 14-dim state vector for the RL agent."""
    max_c = max(CONFIG.get("preprocessing", {}).get("max_vehicle_count", 50), 1)
    max_q = max(CONFIG.get("preprocessing", {}).get("max_queue_length", 30), 1)
    max_gr = max(CONFIG.get("intersection", {}).get("max_green", 60), 1)

    c_n = np.clip(s_dict["vehicle_counts"] / max_c, 0, 1)
    q_n = np.clip(s_dict["queue_lengths"] / max_q, 0, 1)
    pred_n = np.zeros(4, dtype=np.float32)
    ns_g = np.array([1.0 if s_dict.get("current_phase", 0) == 0 else 0.0], dtype=np.float32)
    t_norm = np.array([float(np.clip(getattr(env, "_phase_timer", 0) / max_gr, 0, 1))], dtype=np.float32)
    return np.concatenate([c_n, q_n, pred_n, ns_g, t_norm]).astype(np.float32)

def _run_simulation():
    """Background thread running TWO interconnected environments."""
    from simulation.synthetic_env import SyntheticTrafficEnv
    import torch, numpy as np

    while True:
        if not SIM["running"] or SIM["paused"]:
            time.sleep(0.1)
            continue

        seed = int(time.time())
        env1 = SyntheticTrafficEnv(CONFIG, seed=seed)
        env2 = SyntheticTrafficEnv(CONFIG, seed=seed + 1)
        env1.start()
        env2.start()

        # Transit delays (corridor buffers) [count, timer]
        transit_east = []  # Cars moving env1 East -> env2 West
        transit_west = []  # Cars moving env2 West -> env1 East
        TRANSIT_DELAY = 10 # 10 seconds travel time between intersections

        step_count = 0
        queue_acc = 0.0
        wait_acc = 0.0
        episode_reward = 0.0

        action1, action2 = 0, 0

        while SIM["running"] and step_count < CONFIG["sumo"]["max_steps"]:
            if SIM["paused"]:
                time.sleep(0.05)
                continue

            # --- Agent Decisions (Every 20s) ---
            if step_count % 20 == 0:
                s1 = env1.get_state()
                s2 = env2.get_state()
                state1 = _get_agent_state(env1, s1)
                state2 = _get_agent_state(env2, s2)
                
                if AGENT:
                    action1 = AGENT.act_greedy(state1)
                    action2 = AGENT.act_greedy(state2)
                else:
                    action1 = (step_count // 40) % 2
                    action2 = (step_count // 40) % 2

                green1 = 50 if action1 > 1 else 30
                green2 = 50 if action2 > 1 else 30
                env1.apply_action(action1, green_duration=green1)
                env2.apply_action(action2, green_duration=green2)

            # --- Step Environments ---
            env1.step(1)
            env2.step(1)
            step_count += 1

            # --- Physical Corridor Transfer ---
            # 1. Get served cars
            served1 = env1.get_last_served()
            served2 = env2.get_last_served()
            
            # env1 Road 3 (West inbound) serves cars that drive East. So they go to env2 Road 3 (West inbound).
            if served1[3] > 0: transit_east.append([served1[3], TRANSIT_DELAY])
            # env2 Road 2 (East inbound) serves cars that drive West. So they go to env1 Road 2 (East inbound).
            if served2[2] > 0: transit_west.append([served2[2], TRANSIT_DELAY])

            # 2. Advance transit timers
            for t in transit_east: t[1] -= 1
            for t in transit_west: t[1] -= 1

            # 3. Arrive at destination
            arrived_east = sum(t[0] for t in transit_east if t[1] <= 0)
            arrived_west = sum(t[0] for t in transit_west if t[1] <= 0)
            transit_east = [t for t in transit_east if t[1] > 0]
            transit_west = [t for t in transit_west if t[1] > 0]

            if arrived_east > 0: env2.add_arriving_cars(3, arrived_east)  # Arrive at env2 West inbound
            if arrived_west > 0: env1.add_arriving_cars(2, arrived_west)  # Arrive at env1 East inbound

            # --- Stats & Rewards ---
            s1 = env1.get_state()
            s2 = env2.get_state()
            q1, q2 = s1["queue_lengths"], s2["queue_lengths"]
            w1, w2 = s1["waiting_times"], s2["waiting_times"]
            
            queue_acc += float(np.mean(q1) + np.mean(q2)) / 2
            wait_acc += float(np.mean(w1) + np.mean(w2)) / 2
            
            # Calculate negative average penalty for the chart
            max_q = max(CONFIG.get("preprocessing", {}).get("max_queue_length", 30), 1)
            penalty = -((np.mean(q1) + np.mean(q2))/(2*max_q))
            episode_reward += penalty

            # Update shared state
            with _lock:
                SIM["step"] = step_count
                
                # Intersections
                SIM["intersections"][0] = {
                    "phase": int(s1["current_phase"]),
                    "phase_timer": int(env1._phase_timer),
                    "queues": [round(float(x), 1) for x in q1],
                    "waiting_times": [round(float(x), 1) for x in w1],
                    "vehicle_counts": [int(x) for x in s1["vehicle_counts"]],
                    "action": action1, "action_name": ACTION_NAMES.get(action1, "—")
                }
                SIM["intersections"][1] = {
                    "phase": int(s2["current_phase"]),
                    "phase_timer": int(env2._phase_timer),
                    "queues": [round(float(x), 1) for x in q2],
                    "waiting_times": [round(float(x), 1) for x in w2],
                    "vehicle_counts": [int(x) for x in s2["vehicle_counts"]],
                    "action": action2, "action_name": ACTION_NAMES.get(action2, "—")
                }
                
                # Grid KPIs
                SIM["transit_east"] = round(float(sum(t[0] for t in transit_east)), 1)
                SIM["transit_west"] = round(float(sum(t[0] for t in transit_west)), 1)
                SIM["avg_queue"] = round(queue_acc / step_count, 2)
                SIM["avg_wait"] = round(wait_acc / step_count, 2)
                SIM["reward_history"] = SIM["reward_history"][-59:] + [round(episode_reward, 3)]
                SIM["queue_history"] = SIM["queue_history"][-59:] + [round((np.mean(q1)+np.mean(q2))/2, 2)]

            speed = max(1, SIM.get("speed", 1))
            time.sleep(max(0.02, 0.5 / speed))

        env1.close()
        env2.close()

        with _lock:
            SIM["episode"] += 1
            SIM["reward_history"] = []
            SIM["queue_history"] = []

        time.sleep(1.0)


# Start background thread
_sim_thread = threading.Thread(target=_run_simulation, daemon=True)
_sim_thread.start()

# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    SIM["running"] = True
    SIM["paused"] = False
    return jsonify({"status": "started"})


@app.route("/pause", methods=["POST"])
def pause():
    SIM["paused"] = not SIM["paused"]
    return jsonify({"paused": SIM["paused"]})


@app.route("/reset", methods=["POST"])
def reset():
    with _lock:
        SIM["running"] = False
        SIM["step"] = 0
        SIM["episode"] = 1
        SIM["intersections"][0]["queues"] = [0]*4
        SIM["intersections"][1]["queues"] = [0]*4
        SIM["transit_east"] = 0
        SIM["transit_west"] = 0
        SIM["reward_history"] = []
        SIM["queue_history"] = []
    time.sleep(0.2)
    SIM["running"] = True
    SIM["paused"] = False
    return jsonify({"status": "reset"})


@app.route("/speed", methods=["POST"])
def set_speed():
    data = request.get_json()
    SIM["speed"] = int(data.get("speed", 1))
    return jsonify({"speed": SIM["speed"]})


@app.route("/state")
def get_state():
    with _lock:
        return jsonify({
            "step": int(SIM["step"]),
            "episode": int(SIM["episode"]),
            "avg_queue": float(SIM["avg_queue"]),
            "avg_wait": float(SIM["avg_wait"]),
            "transit_east": float(SIM["transit_east"]),
            "transit_west": float(SIM["transit_west"]),
            "intersections": SIM["intersections"],
            "reward_history": SIM["reward_history"],
            "queue_history": SIM["queue_history"],
            "paused": SIM["paused"],
            "running": SIM["running"],
            "agent_mode": "DQN Agent" if AGENT else "Fixed Timing (no checkpoint)",
            "max_steps": CONFIG["sumo"]["max_steps"],
        })

@app.route("/stream")
def stream():
    """Server-Sent Events endpoint for real-time push updates."""
    def event_stream():
        while True:
            with _lock:
                data = json.dumps({
                    "step": int(SIM["step"]),
                    "avg_queue": float(SIM["avg_queue"]),
                    "avg_wait": float(SIM["avg_wait"]),
                    "transit_east": float(SIM["transit_east"]),
                    "transit_west": float(SIM["transit_west"]),
                    "intersections": SIM["intersections"],
                    "reward_history": SIM["reward_history"],
                    "queue_history": SIM["queue_history"],
                })
            yield f"data: {data}\n\n"
            time.sleep(0.3)
    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/analytics")
def analytics_page():
    return render_template("analytics.html")


@app.route("/analytics_data")
def analytics_data():
    """Run analysis and return JSON summary + path to image."""
    import threading
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from evaluation.analyze import run_analysis
        result = run_analysis(CONFIG, save=True)
        return jsonify({"status": "ok", "summary": result["summary"]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/analytics_image")
def analytics_image():
    """Serve the latest analytics report PNG."""
    img_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "evaluation", "results", "analytics_report.png"
    )
    if os.path.exists(img_path):
        from flask import send_file
        return send_file(img_path, mimetype="image/png")
    return "No analytics image found. Click 'Run Analysis' first.", 404


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Traffic Signal AI  —  Web Dashboard")
    print("  Open your browser at: http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(debug=False, threaded=True, port=5000)
