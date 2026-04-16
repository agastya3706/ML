# Traffic Signal Optimization — SUMO + STGCN + DQN

A professional end-to-end AI system for autonomous traffic signal control combining:
- **SUMO** — Realistic traffic simulation via TraCI
- **STGCN** — Spatio-Temporal Graph Convolutional Network for traffic prediction
- **DQN** — Deep Q-Network reinforcement learning for signal timing control

---

## System Architecture

```
SUMO Simulation
      ↓ (TraCI — real-time data)
  vehicle counts, queue lengths, speeds
      ↓
  Preprocessing (normalize, sliding windows)
      ↓
  STGCN Prediction Agent → predicted future traffic
      ↓
  DQN Signal Agent → optimal green/red timing
      ↓ (TraCI — signal control)
  SUMO updates signal phases
      ↑__________________________________|
```

---

## Project Structure

```
traffic-signal-project/
├── config/
│   └── config.yaml              ← All hyperparameters
├── data/
│   ├── raw/                     ← Raw SUMO outputs
│   ├── processed/               ← Cleaned + cached arrays
│   └── graph/                   ← Adjacency matrices
├── preprocessing/
│   ├── data_cleaning.py         ← Normalize, fill NaN, sliding windows
│   ├── feature_engineering.py  ← Time encoding, rolling stats
│   └── graph_builder.py         ← Adjacency + Chebyshev matrices
├── models/
│   ├── baseline/
│   │   ├── linear_regression.py ← Ridge regression baseline
│   │   └── lstm_model.py        ← LSTM temporal baseline
│   └── stgcn/
│       ├── layers.py            ← TemporalConv, ChebConv, STConvBlock
│       ├── stgcn_model.py       ← Full STGCN model
│       └── train_stgcn.py       ← STGCN training loop
├── rl_agents/
│   ├── reward.py                ← Congestion-based reward
│   ├── agent.py                 ← DQN: QNetwork, ReplayBuffer
│   ├── environment.py           ← Gym-style env (SUMO + STGCN + reward)
│   └── train_agent.py           ← DQN training loop
├── simulation/
│   ├── sumo_env.py              ← TraCI Python-SUMO bridge
│   ├── network.net.xml          ← 4-way intersection road network
│   ├── routes.rou.xml           ← Vehicle flows (N/S/E/W)
│   └── config.sumocfg           ← SUMO simulation config
├── integration/
│   ├── pipeline.py              ← Full pipeline orchestrator
│   └── inference.py             ← Real-time inference interface
├── evaluation/
│   ├── metrics.py               ← Queue, wait, throughput metrics
│   └── comparison.py            ← DQN vs baselines comparison
├── main.py                      ← CLI entry point
└── requirements.txt
```

---

## Setup

### 1. Install SUMO

Download from: https://sumo.dlr.de/docs/Downloads.php

Set environment variable:
```powershell
[System.Environment]::SetEnvironmentVariable("SUMO_HOME", "C:\Program Files (x86)\Eclipse\Sumo", "User")
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation (No SUMO Required)

```bash
cd traffic-signal-project
python main.py --mode graph_test
```

This tests the graph builder, STGCN forward pass, and DQN agent without launching SUMO.

---

## Running the Pipeline

### Step 1 — Train STGCN Predictor

Collects traffic data from SUMO (headless) and trains the prediction model:

```bash
python main.py --mode train_stgcn
```

- Runs SUMO headless for 500 steps to collect data
- Trains STGCN with MSELoss + Adam
- Saves checkpoint: `models/stgcn/best_model.pth`

### Step 2 — Train DQN Agent

Trains the RL signal controller in SUMO environment:

```bash
python main.py --mode train_rl
```

- Loads STGCN predictions into state
- DQN controls signal phases
- Saves best agent: `rl_agents/best_dqn.pth`
- Saves training plots to `evaluation/results/`

### Train Both (End-to-End)

```bash
python main.py --mode train_all
```

### Step 3 — Evaluate

Compare DQN vs fixed-timing vs random baselines:

```bash
python main.py --mode eval
```

Outputs:
- Metrics table (printed)
- `evaluation/results/comparison.csv`
- `evaluation/results/comparison_chart.png`

### Step 4 — Demo (SUMO-GUI)

Watch the trained agent control real traffic signals:

```bash
python main.py --mode demo --gui
```

---

## Configuration

All hyperparameters in `config/config.yaml`:

| Section | Key Parameters |
|---|---|
| `sumo` | `binary`, `max_steps`, `step_length` |
| `intersection` | `num_roads`, `lane_ids`, `tl_id`, `min_green`, `max_green` |
| `stgcn` | `Ks`, `seq_len`, `epochs`, `checkpoint_path` |
| `rl` | `episodes`, `gamma`, `epsilon_*`, `batch_size`, `memory_size` |

---

## Agent Design

### State (12-dim)
```
[vehicle_count_N, vehicle_count_S, vehicle_count_E, vehicle_count_W,  ← from TraCI
 queue_length_N, queue_length_S, queue_length_E, queue_length_W,      ← from TraCI
 predicted_N, predicted_S, predicted_E, predicted_W]                  ← from STGCN
```

### Actions (4 discrete)
| Action | Meaning |
|---|---|
| 0 | North-South green, normal (30s) |
| 1 | East-West green, normal (30s) |
| 2 | North-South green, extended (45s) |
| 3 | East-West green, extended (45s) |

### Reward
```
reward = -(w_queue * sum(queue_lengths) + w_wait * sum(waiting_times))
```
Normalized to `[-1, 0]`. Closer to 0 = better (less congestion).

---

## Results

After training, `evaluation/results/` contains:
- `stgcn_training_loss.png` — STGCN MSE loss curve
- `rl_training_curves.png` — DQN reward, queue, wait trends
- `comparison_chart.png` — DQN vs Fixed vs Random bar chart
- `comparison.csv` — Numeric results

---

## Requirements

- Python 3.9+
- SUMO 1.18+
- PyTorch 2.0+
- See `requirements.txt` for full list
