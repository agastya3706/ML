"""
synthetic_env.py — Synthetic Traffic Simulator (SUMO Fallback)
================================================================
When SUMO is not installed, this module provides a realistic
Poisson-based traffic simulator that mimics the SumoEnvironment API.

Drop-in replacement for SumoEnvironment — same interface:
    .start() / .reset() / .step() / .get_state() / .apply_action() / .close()

Traffic model:
    - Vehicle arrivals: Poisson process per road (configurable rates)
    - Queue dynamics: FIFO, vehicles served during green phase
    - Signal phases: 4-phase cycle (NS-green, NS-yellow, EW-green, EW-yellow)
    - Waiting time: accumulates for halted vehicles
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SyntheticTrafficEnv:
    """
    Realistic synthetic traffic simulator — SUMO-compatible API.

    Simulates a 4-way intersection with:
        - Poisson vehicle arrivals per approach
        - Queue-based service during green phase
        - Dynamic waiting time tracking

    Parameters
    ----------
    config : dict — project config
    seed : int — random seed for reproducibility
    """

    # Arrival rates (vehicles/step) per road — asymmetric like SUMO routes
    DEFAULT_ARRIVAL_RATES = [0.25, 0.17, 0.33, 0.11]  # N, S, E, W (matches routes.rou.xml)

    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)

        int_cfg = config.get("intersection", {})
        self.num_roads = int_cfg.get("num_roads", 4)
        self.min_green = int_cfg.get("min_green", 10)
        self.max_green = int_cfg.get("max_green", 60)

        sumo_cfg = config.get("sumo", {})
        self.max_steps = sumo_cfg.get("max_steps", 3600)
        self.step_length = sumo_cfg.get("step_length", 1.0)

        pre_cfg = config.get("preprocessing", {})
        self.max_queue = pre_cfg.get("max_queue_length", 30)

        # State
        self._queues = np.zeros(self.num_roads, dtype=np.float32)
        self._waiting_times = np.zeros(self.num_roads, dtype=np.float32)
        self._speeds = np.full(self.num_roads, 13.89, dtype=np.float32)  # 50 km/h in m/s
        self._vehicle_counts = np.zeros(self.num_roads, dtype=np.float32)
        self._current_phase = 0        # 0=NS green, 1=NS yellow, 2=EW green, 3=EW yellow
        self._phase_timer = 0          # steps remaining in current phase
        self._current_step = 0

        # Phase durations (seconds): [NS-green, NS-yellow, EW-green, EW-yellow]
        self._phase_durations = [30, 5, 30, 5]

        # History buffers (same as SumoEnvironment)
        self._count_history: List[List[float]] = []
        self._queue_history: List[List[float]] = []
        self._wait_history: List[List[float]] = []

        self._arrival_rates = np.array(self.DEFAULT_ARRIVAL_RATES, dtype=np.float32)
        self._is_running = False

        logger.info("SyntheticTrafficEnv initialized (SUMO fallback mode)")

    # -----------------------------------------------------------------------
    # Lifecycle (mirrors SumoEnvironment)
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Initialize simulation state."""
        self._queues = np.zeros(self.num_roads, dtype=np.float32)
        self._waiting_times = np.zeros(self.num_roads, dtype=np.float32)
        self._vehicle_counts = np.zeros(self.num_roads, dtype=np.float32)
        self._speeds = np.full(self.num_roads, 13.89, dtype=np.float32)
        self._current_phase = 0
        self._phase_timer = self._phase_durations[0]
        self._current_step = 0
        self._count_history = []
        self._queue_history = []
        self._wait_history = []
        self._is_running = True
        logger.debug("Synthetic simulation started.")

    def close(self) -> None:
        self._is_running = False

    def reset(self) -> None:
        self.close()
        self.start()

    # -----------------------------------------------------------------------
    # Simulation Step
    # -----------------------------------------------------------------------

    def step(self, steps: int = 1) -> bool:
        """
        Advance simulation by `steps` timesteps.
        Returns True if simulation still running.
        """
        for _ in range(steps):
            if self._current_step >= self.max_steps:
                return False
            self._simulate_one_step()
            self._current_step += 1
        return self._current_step < self.max_steps

    def _simulate_one_step(self) -> None:
        """Core traffic physics simulation for one timestep."""
        # --- Vehicle arrivals (Poisson) ---
        # Add time-of-day variability: peak at step 900 and 2700 (15min, 45min)
        t = self._current_step / self.max_steps
        peak_factor = 1.0 + 0.5 * np.sin(2 * np.pi * t)  # oscillates between 0.5 and 1.5
        arrivals = self.rng.poisson(self._arrival_rates * peak_factor * self.step_length)
        self._queues = np.clip(self._queues + arrivals, 0, self.max_queue)

        # --- Service during green phase ---
        green_roads = self._get_green_roads()
        service_rate = 0.4  # vehicles served per second per road (saturation flow)
        for road in green_roads:
            served = min(self._queues[road], service_rate * self.step_length)
            self._queues[road] = max(0.0, self._queues[road] - served)

        # --- Waiting time update ---
        for road in range(self.num_roads):
            if road in green_roads and self._queues[road] < 1:
                # Vehicles clearing — reduce waiting time
                self._waiting_times[road] = max(0.0, self._waiting_times[road] - self.step_length)
            else:
                # Vehicles accumulating wait
                self._waiting_times[road] += self._queues[road] * self.step_length * 0.1

        self._waiting_times = np.clip(self._waiting_times, 0.0, 120.0)

        # --- Vehicle counts on approach (includes through traffic) ---
        base_count = self._queues + self.rng.poisson(self._arrival_rates * 0.5)
        self._vehicle_counts = np.clip(base_count, 0, 50).astype(np.float32)

        # --- Speed: inversely related to queue ---
        self._speeds = np.clip(13.89 - self._queues * 0.3, 0.5, 13.89).astype(np.float32)

        # --- Phase timer advance ---
        self._phase_timer -= 1
        if self._phase_timer <= 0:
            self._current_phase = (self._current_phase + 1) % 4
            self._phase_timer = self._phase_durations[self._current_phase]

    def _get_green_roads(self) -> List[int]:
        """Return road indices that currently have green light."""
        # Phase 0 (NS green): roads 0,1 (North, South)
        # Phase 2 (EW green): roads 2,3 (East, West)
        if self._current_phase == 0:
            return [0, 1]
        elif self._current_phase == 2:
            return [2, 3]
        return []  # Yellow phase — no one gets green

    # -----------------------------------------------------------------------
    # State Queries (mimics TraCI API)
    # -----------------------------------------------------------------------

    def get_vehicle_counts(self) -> np.ndarray:
        return self._vehicle_counts.copy()

    def get_queue_lengths(self) -> np.ndarray:
        return self._queues.copy()

    def get_speeds(self) -> np.ndarray:
        return self._speeds.copy()

    def get_waiting_times(self) -> np.ndarray:
        return self._waiting_times.copy()

    def get_state(self) -> Dict[str, np.ndarray]:
        """Full state dict — same keys as SumoEnvironment.get_state()."""
        state = {
            "vehicle_counts": self.get_vehicle_counts(),
            "queue_lengths": self.get_queue_lengths(),
            "speeds": self.get_speeds(),
            "waiting_times": self.get_waiting_times(),
            "current_phase": self._current_phase,
            "step": self._current_step,
        }
        # Record history
        self._count_history.append(state["vehicle_counts"].tolist())
        self._queue_history.append(state["queue_lengths"].tolist())
        self._wait_history.append(state["waiting_times"].tolist())
        return state

    def get_reward_info(self) -> Dict[str, np.ndarray]:
        return {
            "queue_lengths": self.get_queue_lengths(),
            "waiting_times": self.get_waiting_times(),
        }

    def get_history_buffer(self) -> Dict[str, np.ndarray]:
        return {
            "vehicle_counts": np.array(self._count_history, dtype=np.float32),
            "queue_lengths":  np.array(self._queue_history, dtype=np.float32),
            "waiting_times":  np.array(self._wait_history, dtype=np.float32),
        }

    # -----------------------------------------------------------------------
    # Signal Control (mimics TraCI API)
    # -----------------------------------------------------------------------

    def set_phase(self, phase_index: int, duration: Optional[int] = None) -> None:
        """Set current signal phase."""
        self._current_phase = phase_index % 4
        if duration:
            self._phase_timer = int(duration)

    def apply_action(self, action: int, green_duration: int = 30) -> None:
        """
        Apply RL agent action — same API as SumoEnvironment.apply_action().

        Actions:
            0 → NS green, normal
            1 → EW green, normal
            2 → NS green, extended
            3 → EW green, extended
        """
        green_duration = max(self.min_green, min(self.max_green, green_duration))

        if action in (0, 2):
            # NS green
            self._current_phase = 0
            self._phase_timer = green_duration + (15 if action == 2 else 0)
        elif action in (1, 3):
            # EW green
            self._current_phase = 2
            self._phase_timer = green_duration + (15 if action == 3 else 0)

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
            f"SyntheticTrafficEnv(step={self._current_step}/{self.max_steps}, "
            f"phase={self._current_phase}, "
            f"queues={self._queues.astype(int).tolist()})"
        )


if __name__ == "__main__":
    import yaml, logging
    logging.basicConfig(level=logging.INFO)

    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    env = SyntheticTrafficEnv(cfg)
    env.start()

    for i in range(10):
        state = env.get_state()
        env.step(1)
        if i % 3 == 0:
            print(f"Step {i}: queues={state['queue_lengths'].astype(int)}, "
                  f"wait={state['waiting_times'].astype(int)}")
    env.close()
    print("Synthetic env test passed.")
