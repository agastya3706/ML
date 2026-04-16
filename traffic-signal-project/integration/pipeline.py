"""
pipeline.py — Full System Pipeline Orchestrator
=================================================
Connects all components into a single runnable pipeline:
    SUMO → Preprocessing → STGCN → DQN → Signal Control → SUMO

Also provides:
    - Data collection run (SUMO headless, collect history)
    - STGCN training run
    - RL training run
    - Evaluation run

Usage (via main.py):
    pipeline = TrafficPipeline(config)
    pipeline.train_stgcn()
    pipeline.train_rl()
    pipeline.evaluate()
    pipeline.demo()
"""

import os
import sys
import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TrafficPipeline:
    """
    Orchestrates the full traffic signal optimization pipeline.

    Parameters
    ----------
    config : dict — loaded project config
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stgcn_model = None
        self.dqn_agent = None
        self.graph = None
        logger.info(f"TrafficPipeline initialized on device: {self.device}")

    # -----------------------------------------------------------------------
    # Step 1: Build Graph
    # -----------------------------------------------------------------------

    def build_graph(self) -> dict:
        """Build road graph for STGCN. Returns graph dict."""
        from preprocessing.graph_builder import build_graph
        self.graph = build_graph(self.config)
        logger.info(f"Graph built: {self.graph['num_nodes']} nodes")
        return self.graph

    # -----------------------------------------------------------------------
    # Step 2: Train STGCN
    # -----------------------------------------------------------------------

    def train_stgcn(self) -> None:
        """
        Collect SUMO traffic data and train STGCN predictor.
        Saves checkpoint to stgcn.checkpoint_path in config.
        """
        from models.stgcn.train_stgcn import run_stgcn_training
        logger.info("Pipeline Step: Training STGCN...")
        run_stgcn_training(self.config)
        self._load_stgcn()

    def _load_stgcn(self) -> None:
        """Load STGCN from checkpoint if available."""
        from models.stgcn.stgcn_model import build_stgcn
        from models.stgcn.train_stgcn import load_stgcn_checkpoint

        if self.graph is None:
            self.build_graph()

        ckpt = self.config.get("stgcn", {}).get("checkpoint_path", "")
        if ckpt and os.path.exists(ckpt):
            model = build_stgcn(self.config, self.graph["cheb_polys"])
            self.stgcn_model = load_stgcn_checkpoint(model, ckpt, self.device)
            logger.info("STGCN model loaded from checkpoint.")
        else:
            logger.warning(
                f"STGCN checkpoint not found at '{ckpt}'. "
                "RL will run without traffic predictions."
            )

    # -----------------------------------------------------------------------
    # Step 3: Train RL Agent
    # -----------------------------------------------------------------------

    def train_rl(self, use_gui: bool = False, resume: bool = False) -> dict:
        """
        Train DQN agent in SUMO environment using STGCN predictions.

        Parameters
        ----------
        use_gui : bool — show SUMO-GUI during training (slow)
        resume : bool — continue from existing checkpoint

        Returns
        -------
        training_history : dict
        """
        from rl_agents.train_agent import run_rl_training

        if self.stgcn_model is None:
            try:
                self._load_stgcn()
            except Exception as e:
                logger.warning(f"Could not load STGCN: {e}")

        logger.info("Pipeline Step: Training DQN Agent...")
        history = run_rl_training(
            self.config,
            stgcn_model=self.stgcn_model,
            use_gui=use_gui,
            resume=resume,
        )
        self._load_dqn()
        return history

    def _load_dqn(self) -> None:
        """Load DQN agent from checkpoint if available."""
        from rl_agents.agent import DQNAgent

        agent = DQNAgent(self.config, device=self.device)
        ckpt = self.config.get("rl", {}).get("checkpoint_path", "")
        if ckpt and os.path.exists(ckpt):
            agent.load(ckpt)
            self.dqn_agent = agent
            logger.info("DQN agent loaded from checkpoint.")
        else:
            logger.warning(f"DQN checkpoint not found at '{ckpt}'.")

    # -----------------------------------------------------------------------
    # Step 4: Evaluate
    # -----------------------------------------------------------------------

    def evaluate(self, num_episodes: int = 5) -> None:
        """
        Run full comparison: DQN vs fixed timing vs random.
        """
        from evaluation.comparison import run_full_comparison

        if self.dqn_agent is None:
            self._load_dqn()
        if self.stgcn_model is None:
            self._load_stgcn()

        logger.info("Pipeline Step: Evaluating all methods...")
        run_full_comparison(
            self.config,
            agent=self.dqn_agent,
            stgcn_model=self.stgcn_model,
            num_episodes=num_episodes,
        )

    # -----------------------------------------------------------------------
    # Step 5: Demo (SUMO-GUI)
    # -----------------------------------------------------------------------

    def demo(self) -> None:
        """
        Run one episode with SUMO-GUI so you can watch the trained agent
        control real traffic signals in the simulation.
        """
        from rl_agents.environment import TrafficEnvironment

        def _make_env(config, use_gui=False):
            if os.environ.get("SUMO_HOME", ""):
                try:
                    from simulation.sumo_env import SumoEnvironment
                    return SumoEnvironment(config, use_gui=use_gui)
                except Exception:
                    pass
            from simulation.synthetic_env import SyntheticTrafficEnv
            return SyntheticTrafficEnv(config)

        if self.dqn_agent is None:
            self._load_dqn()
        if self.stgcn_model is None:
            self._load_stgcn()

        if self.dqn_agent is None:
            raise RuntimeError(
                "No trained DQN agent found. Run 'python main.py --mode train_rl' first."
            )

        logger.info("Pipeline Step: Demo with SUMO-GUI...")
        sumo_env = SumoEnvironment(self.config, use_gui=True)
        env = TrafficEnvironment(
            self.config,
            sumo_env,
            stgcn_model=self.stgcn_model,
            device=self.device,
        )

        state = env.reset()
        done = False
        step = 0

        logger.info("Demo running — watch SUMO-GUI window. Close it to stop.")
        while not done:
            action = self.dqn_agent.act_greedy(state)
            state, reward, done, info = env.step(action)
            step += 1
            if step % 50 == 0:
                logger.info(
                    f"Step {step}: action={action}, "
                    f"queue={info['queue_lengths'].sum():.1f}, "
                    f"reward={reward:.4f}"
                )

        env.close()
        logger.info(f"Demo complete. Total reward: {env.total_reward:.4f}")
