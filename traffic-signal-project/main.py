# -*- coding: utf-8 -*-
"""
main.py — Traffic Signal Optimization System Entry Point
==========================================================
Run the complete pipeline with a single command.

Usage:
    python main.py --mode train_stgcn    # Step 1: Train STGCN predictor
    python main.py --mode train_rl       # Step 2: Train DQN agent
    python main.py --mode train_all      # Steps 1 + 2 combined
    python main.py --mode eval           # Evaluate: DQN vs baselines
    python main.py --mode demo           # Watch agent in SUMO-GUI
    python main.py --mode graph_test     # Verify graph builder (no SUMO needed)

Options:
    --config   PATH     Path to config.yaml (default: config/config.yaml)
    --gui               Force SUMO-GUI mode (overrides config)
    --headless          Force headless SUMO mode
    --episodes  N       Override number of training episodes
    --resume            Resume RL training from checkpoint

System Architecture:
    SUMO → TraCI → Python
       ↓
    preprocessing (data_cleaning, graph_builder)
       ↓
    STGCN (predict future traffic)
       ↓
    DQN Agent (decide signal timing)
       ↓
    TraCI → SUMO (apply signal changes)
       ↑___________________________|
"""

import argparse
import logging
import os
import sys

import yaml


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logging(config: dict) -> None:
    """Configure logging from config."""
    import io
    # Fix Windows cp1252 encoding — force UTF-8 stdout
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"), logging.INFO)
    log_dir = log_cfg.get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, "traffic_signal.log")),
        ],
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mode Handlers
# ---------------------------------------------------------------------------

def mode_graph_test(config: dict) -> None:
    """Quick sanity check — no SUMO required."""
    from preprocessing.graph_builder import build_graph
    import numpy as np

    logger.info("=== Graph Building Test ===")
    graph = build_graph(config)
    logger.info(f"Nodes: {graph['num_nodes']}")
    logger.info(f"A_hat:\n{graph['A_hat']}")
    logger.info(f"Chebyshev polynomials: {len(graph['cheb_polys'])}")

    # Test STGCN forward pass (no SUMO)
    try:
        import torch
        from models.stgcn.stgcn_model import build_stgcn

        model = build_stgcn(config, graph["cheb_polys"])
        N = graph["num_nodes"]
        seq = config.get("stgcn", {}).get("seq_len", 12)
        x = torch.randn(2, seq, N, 1)
        out = model(x)
        assert out.shape == (2, N), f"Shape error: {out.shape}"
        params = sum(p.numel() for p in model.parameters())
        logger.info(f"STGCN forward pass OK: {x.shape} -> {out.shape}")
        logger.info(f"STGCN parameters: {params:,}")
    except ImportError as e:
        logger.warning(f"PyTorch not available: {e}")
        logger.warning("Run: pip install torch torchvision")

    # Test DQN (no SUMO)
    try:
        import torch
        from rl_agents.agent import DQNAgent

        agent = DQNAgent(config)
        import numpy as np
        state = np.random.rand(config["rl"]["state_dim"]).astype(np.float32)
        action = agent.act(state)
        assert 0 <= action < config["rl"]["action_dim"]
        logger.info(f"DQN agent OK: action={action}, params={agent.num_parameters:,}")
    except ImportError as e:
        logger.warning(f"PyTorch not available for DQN test: {e}")

    logger.info("=== All tests passed [OK] ===")


def mode_train_stgcn(config: dict) -> None:
    from integration.pipeline import TrafficPipeline

    pipeline = TrafficPipeline(config)
    pipeline.build_graph()
    pipeline.train_stgcn()


def mode_train_rl(config: dict, use_gui: bool = False, resume: bool = False) -> None:
    from integration.pipeline import TrafficPipeline

    pipeline = TrafficPipeline(config)
    pipeline.build_graph()
    pipeline.train_rl(use_gui=use_gui, resume=resume)


def mode_train_all(config: dict, use_gui: bool = False) -> None:
    from integration.pipeline import TrafficPipeline

    pipeline = TrafficPipeline(config)
    pipeline.build_graph()
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Training STGCN Prediction Model")
    logger.info("=" * 60)
    pipeline.train_stgcn()

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Training DQN RL Agent")
    logger.info("=" * 60)
    pipeline.train_rl(use_gui=use_gui)


def mode_eval(config: dict, num_episodes: int = 5) -> None:
    from integration.pipeline import TrafficPipeline

    pipeline = TrafficPipeline(config)
    pipeline.build_graph()
    pipeline.evaluate(num_episodes=num_episodes)


def mode_demo(config: dict) -> None:
    from integration.pipeline import TrafficPipeline

    pipeline = TrafficPipeline(config)
    pipeline.build_graph()
    pipeline.demo()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traffic Signal Optimization — SUMO + STGCN + DQN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode graph_test        # No SUMO needed — quick test
  python main.py --mode train_stgcn       # Train prediction model
  python main.py --mode train_rl          # Train signal control agent
  python main.py --mode train_all         # Train STGCN + RL end-to-end
  python main.py --mode eval              # Compare against baselines
  python main.py --mode demo --gui        # Watch agent in SUMO-GUI
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["train_stgcn", "train_rl", "train_all", "eval", "demo", "graph_test"],
        default="graph_test",
        help="Pipeline mode to run (default: graph_test)",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Force SUMO-GUI mode (overrides config binary setting)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force headless SUMO (no GUI)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override number of training or evaluation episodes",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume RL training from existing checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        print("Run from the traffic-signal-project/ directory.")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.gui:
        config["sumo"]["binary"] = "sumo-gui"
    elif args.headless:
        config["sumo"]["binary"] = "sumo"

    if args.episodes is not None:
        if args.mode in ("train_rl", "train_all"):
            config["rl"]["episodes"] = args.episodes
        elif args.mode == "eval":
            config["evaluation"]["num_eval_episodes"] = args.episodes

    setup_logging(config)

    logger.info("=" * 60)
    logger.info("Traffic Signal Optimization System")
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Config: {args.config}")
    logger.info(f"SUMO binary: {config.get('sumo', {}).get('binary', 'sumo')}")
    logger.info("=" * 60)

    use_gui = "gui" in config.get("sumo", {}).get("binary", "sumo").lower()
    num_ep = config.get("evaluation", {}).get("num_eval_episodes", 5)

    dispatch = {
        "graph_test": lambda: mode_graph_test(config),
        "train_stgcn": lambda: mode_train_stgcn(config),
        "train_rl": lambda: mode_train_rl(config, use_gui=use_gui, resume=args.resume),
        "train_all": lambda: mode_train_all(config, use_gui=use_gui),
        "eval": lambda: mode_eval(config, num_episodes=num_ep),
        "demo": lambda: mode_demo(config),
    }

    try:
        dispatch[args.mode]()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
    except Exception as e:
        logger.exception(f"Error in mode '{args.mode}': {e}")
        sys.exit(1)

    logger.info("Done.")


if __name__ == "__main__":
    main()
