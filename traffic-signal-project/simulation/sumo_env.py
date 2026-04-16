"""
sumo_env.py — SUMO Environment Bridge (TraCI Interface)
=======================================================
Connects Python to SUMO via TraCI.

Responsibilities:
  - Launch/close SUMO subprocess
  - Extract per-lane state: vehicle count, queue length, speed, waiting time
  - Apply RL agent actions: change traffic signal phase/duration
  - Advance simulation by N timesteps

Usage:
    env = SumoEnvironment(config)
    env.start()
    state = env.get_state()
    env.set_phase(action=0, green_duration=30)
    env.step(steps=30)
    reward_info = env.get_reward_info()
    env.close()
"""

import os
import sys
import subprocess
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SUMO_HOME check — graceful error if SUMO not installed
# ---------------------------------------------------------------------------
def _check_sumo_home() -> str:
    """Verify SUMO_HOME is set and return the path."""
    sumo_home = os.environ.get("SUMO_HOME", "")
    if not sumo_home:
        raise EnvironmentError(
            "SUMO_HOME environment variable is not set.\n"
            "Please install SUMO from https://sumo.dlr.de/docs/Downloads.php\n"
            "Then set: SUMO_HOME = C:\\Program Files (x86)\\Eclipse\\Sumo"
        )
    tools_path = os.path.join(sumo_home, "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)
    return sumo_home


class SumoEnvironment:
    """
    Python bridge to SUMO via TraCI.

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary (from config.yaml).
    port : int
        TraCI connection port (default 8813).
    use_gui : bool, optional
        If True, launch sumo-gui; if False, launch sumo (headless).
        Overrides config if explicitly passed.
    """

    # Inbound lane IDs (one per approach road)
    DEFAULT_LANE_IDS = ["north_in_0", "south_in_0", "east_in_0", "west_in_0"]
    # Inbound edge IDs
    DEFAULT_EDGE_IDS = ["north_in", "south_in", "east_in", "west_in"]
    # Traffic light node ID
    DEFAULT_TL_ID = "center"

    def __init__(
        self,
        config: dict,
        port: int = 8813,
        use_gui: Optional[bool] = None,
    ):
        self.config = config
        self.port = port

        sumo_cfg = config.get("sumo", {})
        self.max_steps = sumo_cfg.get("max_steps", 3600)
        self.step_length = sumo_cfg.get("step_length", 1.0)
        self.config_file = sumo_cfg.get("config_file", "simulation/config.sumocfg")

        # GUI toggle
        if use_gui is not None:
            self.use_gui = use_gui
        else:
            binary = sumo_cfg.get("binary", "sumo")
            self.use_gui = "gui" in binary.lower()

        int_cfg = config.get("intersection", {})
        self.lane_ids = int_cfg.get("lane_ids", self.DEFAULT_LANE_IDS)
        self.edge_ids = int_cfg.get("road_ids", self.DEFAULT_EDGE_IDS)
        self.tl_id = int_cfg.get("tl_id", self.DEFAULT_TL_ID)
        self.min_green = int_cfg.get("min_green", 10)
        self.max_green = int_cfg.get("max_green", 60)

        self.num_roads = len(self.lane_ids)
        self._traci = None          # traci module (imported after SUMO_HOME check)
        self._current_step = 0
        self._is_running = False

        # History buffers
        self._vehicle_counts: List[List[float]] = []
        self._queue_lengths: List[List[float]] = []
        self._waiting_times: List[List[float]] = []

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Launch SUMO and establish TraCI connection."""
        _check_sumo_home()
        import traci  # noqa: PLC0415 — imported after SUMO_HOME check
        self._traci = traci

        binary = "sumo-gui" if self.use_gui else "sumo"

        # Resolve config file path relative to project root
        cfg_path = os.path.abspath(self.config_file)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"SUMO config not found: {cfg_path}\n"
                f"Make sure simulation/config.sumocfg exists."
            )

        sumo_cmd = [
            binary,
            "-c", cfg_path,
            "--step-length", str(self.step_length),
            "--no-step-log", "true",
            "--waiting-time-memory", "100",
            "--no-warnings", "true",
        ]

        logger.info(f"Starting SUMO: {' '.join(sumo_cmd)}")
        traci.start(sumo_cmd, port=self.port)

        self._current_step = 0
        self._is_running = True
        self._vehicle_counts = []
        self._queue_lengths = []
        self._waiting_times = []
        logger.info("SUMO started successfully via TraCI.")

    def close(self) -> None:
        """Close TraCI connection and stop SUMO."""
        if self._is_running and self._traci is not None:
            try:
                self._traci.close()
                logger.info("SUMO closed.")
            except Exception as e:
                logger.warning(f"Error closing SUMO: {e}")
            finally:
                self._is_running = False

    def reset(self) -> None:
        """Close and restart SUMO for a new episode."""
        self.close()
        self.start()

    # -----------------------------------------------------------------------
    # Simulation Step
    # -----------------------------------------------------------------------

    def step(self, steps: int = 1) -> bool:
        """
        Advance simulation by `steps` timesteps.

        Returns
        -------
        bool
            True if simulation is still running, False if max_steps reached.
        """
        if not self._is_running:
            raise RuntimeError("SUMO is not running. Call start() first.")

        for _ in range(steps):
            if self._current_step >= self.max_steps:
                return False
            self._traci.simulationStep()
            self._current_step += 1

        return self._current_step < self.max_steps

    # -----------------------------------------------------------------------
    # State Extraction (TraCI Queries)
    # -----------------------------------------------------------------------

    def get_vehicle_counts(self) -> np.ndarray:
        """Return vehicle count on each approach lane."""
        counts = []
        for lane_id in self.lane_ids:
            count = self._traci.lane.getLastStepVehicleNumber(lane_id)
            counts.append(float(count))
        return np.array(counts, dtype=np.float32)

    def get_queue_lengths(self) -> np.ndarray:
        """Return halting vehicle count on each approach lane (queue proxy)."""
        queues = []
        for lane_id in self.lane_ids:
            halting = self._traci.lane.getLastStepHaltingNumber(lane_id)
            queues.append(float(halting))
        return np.array(queues, dtype=np.float32)

    def get_speeds(self) -> np.ndarray:
        """Return mean vehicle speed on each approach lane (m/s)."""
        speeds = []
        for lane_id in self.lane_ids:
            speed = self._traci.lane.getLastStepMeanSpeed(lane_id)
            speeds.append(float(speed))
        return np.array(speeds, dtype=np.float32)

    def get_waiting_times(self) -> np.ndarray:
        """Return total waiting time per approach lane (seconds)."""
        waits = []
        for lane_id in self.lane_ids:
            wait = self._traci.lane.getWaitingTime(lane_id)
            waits.append(float(wait))
        return np.array(waits, dtype=np.float32)

    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Get complete intersection state.

        Returns
        -------
        dict with keys:
            'vehicle_counts'  : (num_roads,)
            'queue_lengths'   : (num_roads,)
            'speeds'          : (num_roads,)
            'waiting_times'   : (num_roads,)
            'current_phase'   : int
            'step'            : int
        """
        state = {
            "vehicle_counts": self.get_vehicle_counts(),
            "queue_lengths": self.get_queue_lengths(),
            "speeds": self.get_speeds(),
            "waiting_times": self.get_waiting_times(),
            "current_phase": self._traci.trafficlight.getPhase(self.tl_id),
            "step": self._current_step,
        }

        # Append to history buffers for STGCN
        self._vehicle_counts.append(state["vehicle_counts"].tolist())
        self._queue_lengths.append(state["queue_lengths"].tolist())
        self._waiting_times.append(state["waiting_times"].tolist())

        return state

    def get_reward_info(self) -> Dict[str, np.ndarray]:
        """Return data needed for reward calculation."""
        return {
            "queue_lengths": self.get_queue_lengths(),
            "waiting_times": self.get_waiting_times(),
        }

    def get_history_buffer(self) -> Dict[str, np.ndarray]:
        """
        Return historical traffic data collected so far.
        Used to build STGCN training sequences.

        Returns
        -------
        dict with keys 'vehicle_counts', 'queue_lengths', 'waiting_times'
        Each value: (T, num_roads) array
        """
        return {
            "vehicle_counts": np.array(self._vehicle_counts, dtype=np.float32),
            "queue_lengths": np.array(self._queue_lengths, dtype=np.float32),
            "waiting_times": np.array(self._waiting_times, dtype=np.float32),
        }

    # -----------------------------------------------------------------------
    # Signal Control (TraCI Commands)
    # -----------------------------------------------------------------------

    def set_phase(self, phase_index: int, duration: Optional[int] = None) -> None:
        """
        Set traffic light to a specific phase.

        Parameters
        ----------
        phase_index : int
            Phase index in the tlLogic (0 = NS green, 2 = EW green).
        duration : int, optional
            Override duration for this phase (seconds).
        """
        self._traci.trafficlight.setPhase(self.tl_id, phase_index)
        if duration is not None:
            self._traci.trafficlight.setPhaseDuration(self.tl_id, duration)

    def apply_action(self, action: int, green_duration: int = 30) -> None:
        """
        Apply RL agent action to traffic signal.

        Action mapping:
            0 → extend North-South green (phase 0)
            1 → switch to East-West green (phase 2)
            2 → extend North-South green longer
            3 → extend East-West green longer

        Parameters
        ----------
        action : int
            Action from DQN agent (0-3).
        green_duration : int
            Green phase duration in seconds.
        """
        green_duration = max(self.min_green, min(self.max_green, green_duration))

        if action == 0:
            # NS green, normal duration
            self.set_phase(phase_index=0, duration=green_duration)
        elif action == 1:
            # EW green, normal duration
            self.set_phase(phase_index=2, duration=green_duration)
        elif action == 2:
            # NS green, extended
            self.set_phase(phase_index=0, duration=min(green_duration + 15, self.max_green))
        elif action == 3:
            # EW green, extended
            self.set_phase(phase_index=2, duration=min(green_duration + 15, self.max_green))

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def num_lanes(self) -> int:
        return self.num_roads

    def __repr__(self) -> str:
        return (
            f"SumoEnvironment(gui={self.use_gui}, "
            f"step={self._current_step}/{self.max_steps}, "
            f"running={self._is_running})"
        )
