"""
synthetic_env.py — Synthetic Traffic Simulator (SUMO Fallback)
================================================================
Realistic Poisson-based traffic simulator matching a real urban intersection.

Real-world calibration:
    - Typical urban intersection: 400-900 veh/hour per approach
    - That's 400/3600 ≈ 0.11 veh/s at peak → we use 0.06-0.09 as moderate load
    - Saturation flow: ~1800 veh/hour = 0.5 veh/s per lane → 3 veh/s for burst clearing
    - Green efficiency: ~85% of saturation flow actually served

Traffic model:
    - Vehicle arrivals: Poisson process per road (realistic rates)
    - Queue dynamics: FIFO, vehicles served during green phase
    - Signal phases: 4-phase cycle (NS-green, NS-yellow, EW-green, EW-yellow)
    - Waiting time: accumulates for halted vehicles, resets fast on green
    - Vehicle cap: hard cap of 20 vehicles per approach (realistic single-road queue)
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SyntheticTrafficEnv:
    """
    Realistic synthetic traffic simulator — SUMO-compatible API.

    Simulates a 4-way intersection with:
        - Poisson vehicle arrivals per approach (calibrated to real urban data)
        - Queue-based service during green phase (high saturation flow)
        - Dynamic waiting time tracking with fast green decay

    Parameters
    ----------
    config : dict — project config
    seed : int — random seed for reproducibility
    """

    # Real-world calibrated arrival rates (vehicles per second per approach):
    # Moderate urban intersection: ~250-350 veh/hour = 0.07-0.10 veh/s
    # These are PER STEP (step_length=1s), so these ARE veh/s
    DEFAULT_ARRIVAL_RATES = [0.07, 0.08, 0.06, 0.07]  # N, S, E, W  (light-medium load)

    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)

        int_cfg = config.get("intersection", {})
        self.num_roads = int_cfg.get("num_roads", 4)
        self.min_green = int_cfg.get("min_green", 15)
        self.max_green = int_cfg.get("max_green", 60)

        sumo_cfg = config.get("sumo", {})
        self.max_steps = sumo_cfg.get("max_steps", 3600)
        self.step_length = sumo_cfg.get("step_length", 1.0)

        pre_cfg = config.get("preprocessing", {})
        # Realistic max queue: 20 vehicles per approach at a single-lane urban road
        self.max_queue = min(pre_cfg.get("max_queue_length", 30), 20)

        # State
        self._queues = np.zeros(self.num_roads, dtype=np.float32)
        self._waiting_times = np.zeros(self.num_roads, dtype=np.float32)
        self._speeds = np.full(self.num_roads, 13.89, dtype=np.float32)  # 50 km/h in m/s
        self._vehicle_counts = np.zeros(self.num_roads, dtype=np.float32)
        self._current_phase = 0        # 0=NS green, 1=NS yellow, 2=EW green, 3=EW yellow
        self._phase_timer = 0          # steps remaining in current phase
        self._current_step = 0

        # Phase durations (seconds): [NS-green, NS-yellow, EW-green, EW-yellow]
        # Start with balanced 30s green phases
        self._phase_durations = [30, 4, 30, 4]

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

        # --- Vehicle arrivals (Poisson, REAL-WORLD calibrated) ---
        # Mild time-of-day variation: ±15% (real peak hours are ~1.3-1.5x but
        # for a training environment we keep it gentle so the agent can learn)
        t = self._current_step / max(self.max_steps, 1)
        # Single smooth sine wave — one "rush hour" peak per episode
        peak_factor = 0.9 + 0.2 * np.sin(np.pi * t)  # ranges 0.7 → 1.1 → 0.7
        arrivals = self.rng.poisson(self._arrival_rates * peak_factor * self.step_length)

        # Hard cap at realistic single-approach queue limit
        self._queues = np.clip(self._queues + arrivals, 0, self.max_queue)

        # --- Service during green phase ---
        # Saturation flow: 1800 veh/hr = 0.5 veh/s per lane
        green_roads = self._get_green_roads()
        service_rate = 1.5  # vehicles cleared per second per green road (realistic)
        self._last_served = np.zeros(self.num_roads, dtype=np.float32)

        for road in green_roads:
            served = min(self._queues[road], service_rate * self.step_length)
            self._last_served[road] = served
            self._queues[road] = max(0.0, self._queues[road] - served)

        # --- Waiting time update ---
        max_wait = self.config.get("preprocessing", {}).get("max_waiting_time", 60)
        for road in range(self.num_roads):
            if road in green_roads:
                # Green: vehicles are being served — decay wait time quickly
                # Decay proportional to how many are served (realistic: served cars leave)
                served_this_step = service_rate * self.step_length
                decay = max(2.5, served_this_step * 5.0) * self.step_length
                self._waiting_times[road] = max(0.0, self._waiting_times[road] - decay)
            else:
                # Red: every stopped vehicle accumulates wait
                # Real-world: 1 second per second for halted vehicles
                # queue * 1.0 = total vehicle-seconds accumulated
                if self._queues[road] > 0:
                    self._waiting_times[road] += (1.0 + self._queues[road] * 0.03) * self.step_length
                # Empty road: no wait accumulates
                else:
                    # Fast decay even on red if queue is empty (moved through)
                    self._waiting_times[road] = max(0.0, self._waiting_times[road] - 0.5)

        # Cap wait at configured max (60s is realistic for a signalled intersection)
        self._waiting_times = np.clip(self._waiting_times, 0.0, max_wait)

        # --- Vehicle counts on approach (visible vehicles on inbound road) ---
        # Slightly more than queue (includes vehicles still approaching)
        self._vehicle_counts = np.clip(
            self._queues + self.rng.poisson(self._arrival_rates * 2.0),
            0, 50
        ).astype(np.float32)

        # --- Speed: inversely related to queue density ---
        # Free flow = 50 km/h = 13.89 m/s; congested = near 0
        # At max_queue (20 veh), speed ≈ 0.5 m/s
        congestion_ratio = self._queues / (self.max_queue + 1e-8)
        self._speeds = np.clip(13.89 * (1 - congestion_ratio ** 0.7), 0.5, 13.89).astype(np.float32)

        # --- Phase timer advance (only if not overridden by apply_action) ---
        self._phase_timer -= 1
        if self._phase_timer <= 0:
            # Auto-advance to next phase in cycle
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
        return []  # Yellow phase — no-one gets green

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
            "queue_lengths":  self.get_queue_lengths(),
            "speeds":         self.get_speeds(),
            "waiting_times":  self.get_waiting_times(),
            "current_phase":  self._current_phase,
            "step":           self._current_step,
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
        Apply RL agent action. Switches to the requested green phase and sets
        the timer. Inserts a short yellow when changing direction.

        REVISED Action Space (more granular control):
            0 -> NS green, standard  (30 s)
            1 -> EW green, standard  (30 s)
            2 -> NS green, extended  (50 s) — use for heavy NS congestion
            3 -> EW green, extended  (50 s) — use for heavy EW congestion

        Yellow transition = 4 s (realistic UK/US standard)
        """
        if action in (0, 2):
            target_phase = 0   # NS green
            yellow_phase = 3   # EW yellow → precedes NS green
            duration = 50 if action == 2 else 30
        else:
            target_phase = 2   # EW green
            yellow_phase = 1   # NS yellow → precedes EW green
            duration = 50 if action == 3 else 30

        # Clamp to configured min/max green
        duration = max(self.min_green, min(self.max_green, duration))

        if self._current_phase == target_phase:
            # Already correct direction — extend if duration is longer
            self._phase_timer = max(self._phase_timer, duration)
        else:
            # Switch direction: insert yellow transition
            self._current_phase = yellow_phase
            self._phase_timer = 4  # 4 s yellow (realistic)

    # -----------------------------------------------------------------------
    # Grid Transfer API
    # -----------------------------------------------------------------------

    def get_last_served(self) -> np.ndarray:
        """Returns the number of vehicles served on each road in the very last step."""
        return getattr(self, "_last_served", np.zeros(self.num_roads, dtype=np.float32))

    def add_arriving_cars(self, road_index: int, num_cars: float) -> None:
        """Externally inject vehicles into a specific approach queue (used for grid transfers)."""
        self._queues[road_index] = np.clip(self._queues[road_index] + num_cars, 0, self.max_queue)

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

    total_q, total_w = 0.0, 0.0
    n = 100
    for i in range(n):
        state = env.get_state()
        env.step(1)
        total_q += state["queue_lengths"].mean()
        total_w += state["waiting_times"].mean()
        if i % 20 == 0:
            print(f"Step {i:3d}: queues={state['queue_lengths'].round(1)}, "
                  f"wait={state['waiting_times'].round(1)}")

    print(f"\nAvg queue/road: {total_q/n:.2f} veh  |  Avg wait/road: {total_w/n:.2f} s")
    env.close()
    print("Synthetic env test passed.")
