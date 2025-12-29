# hydrion/env.py
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml

from .config import HydrionConfig
from .physics.hydraulics import HydraulicsModel
from .physics.clogging import CloggingModel
from .physics.electrostatics import ElectrostaticsModel
from .physics.particles import ParticleModel
from .sensors.optical import OpticalSensorArray

# Commit 1 v1.5 imports
from .runtime.run_context import RunContext
from .runtime.seeding import set_global_seed

# Commit 2 v1.5 imports
from pathlib import Path
from .logging.writer import RunLogger

# Commit 3 v1.5 imports
from .state.init import init_truth_state, init_sensor_state
from .sensors.sensor_fusion import build_observation


class HydrionEnv(gym.Env):
    """
    HydrionEnv — Multi-Physics Digital Twin Environment

    Truth vs Sensor separation (Commit 3):
    - truth_state: physics truth (internal)
    - sensor_state: measured outputs (may diverge later)
    - observation: derived from (truth_state, sensor_state) via sensor_fusion
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        render_mode=None,
        run_context: RunContext | None = None,
        version: str = "v1.5",
        seed: int | None = None,
        noise_enabled: bool | None = None,
        auto_reset: bool = True,
    ):
        super().__init__()
        self.render_mode = render_mode

        # Load YAML config
        with open(config_path, "r") as f:
            raw_cfg = yaml.safe_load(f) or {}
        self.cfg = HydrionConfig(raw_cfg)

        # ------------------------------
        # Commit 1: Identity + Determinism
        # ------------------------------
        cfg_hash = self.cfg.config_hash()
        resolved_seed = self.cfg.get_seed(0) if seed is None else int(seed)
        resolved_noise = self.cfg.get_noise_enabled(False) if noise_enabled is None else bool(noise_enabled)

        self.run_context = run_context or RunContext.create(
            version=version,
            seed=resolved_seed,
            noise_enabled=resolved_noise,
            config_hash=cfg_hash,
            deterministic_id=True,
        )

        # ------------------------------
        # Commit 2: Logging Skeleton
        # ------------------------------
        self._log_base_dir = Path(self.cfg.raw.get("logging", {}).get("base_dir", "runs"))
        self.logger = RunLogger(base_dir=self._log_base_dir, enabled=True, strict=False)
        self._episode_return = 0.0

        # Track which seed was actually used for the most recent reset
        self._active_seed: int = self.run_context.seed

        # Simulation params
        self.dt = float(self.cfg.raw.get("sim", {}).get("dt", 0.1))
        self.max_steps = int(600.0 / self.dt)

        # Physics & sensors
        self.hydraulics = HydraulicsModel(self.cfg)
        self.clogging = CloggingModel(self.cfg)
        self.electrostatics = ElectrostaticsModel(self.cfg)
        self.particles = ParticleModel(self.cfg)
        self.sensors = OpticalSensorArray(self.cfg)

        # Commit 3: explicit truth/sensor state containers
        self.truth_state: dict = {}
        self.sensor_state: dict = {}

        # Action space: [valve, pump, backflush, node_voltage]
        self.action_space = spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space (12-D)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32)

        self.steps = 0
        if auto_reset:
            self.reset()

    # ---------------------------------------------------------
    # RESET
    # ---------------------------------------------------------
    def reset(self, seed=None, options=None):
        resolved_seed = self.run_context.seed if seed is None else int(seed)
        self._active_seed = resolved_seed

        super().reset(seed=resolved_seed)
        set_global_seed(resolved_seed)

        self.steps = 0
        self._episode_return = 0.0

        # Start logging run (Commit 2)
        run_header = {
            "run_id": self.run_context.run_id,
            "version": self.run_context.version,
            "seed": self._active_seed,
            "noise_enabled": self.run_context.noise_enabled,
            "config_hash": self.run_context.config_hash,
        }
        self.logger.start_run(
            run_id=self.run_context.run_id,
            run_header=run_header,
            config=self.cfg.raw,
        )
        self.logger.log_step({
            "event": "reset",
            "run_id": self.run_context.run_id,
            "timestep": 0,
            "step": 0,
            "seed": self._active_seed,
        })

        # Commit 3: initialize truth/sensor states
        self.truth_state = init_truth_state().data
        self.sensor_state = init_sensor_state().data

        # Reset subsystems (physics uses truth)
        self.clogging.reset(self.truth_state)
        self.hydraulics.reset()
        self.electrostatics.reset(self.truth_state)
        self.particles.reset(self.truth_state)

        # Sensor reset writes to sensor_state (and mirrors to truth for compatibility)
        self.sensors.reset(self.truth_state, sensor_state=self.sensor_state)

        # Neutral kick to initialize derived values
        neutral = np.array([0.5, 0.5, 0.0, 0.5], dtype=np.float32)

        self.hydraulics.update(self.truth_state, dt=self.dt, action=neutral, clogging_model=self.clogging)
        self.electrostatics.update(self.truth_state, dt=self.dt, node_cmd=self.truth_state["node_voltage_cmd"])
        self.particles.update(
            self.truth_state,
            dt=self.dt,
            clogging_model=self.clogging,
            electrostatics_model=self.electrostatics,
        )
        self.sensors.update(self.truth_state, dt=self.dt, sensor_state=self.sensor_state)

        self._update_normalized_state()
        return self._observe(), {}

    # ---------------------------------------------------------
    # STEP
    # ---------------------------------------------------------
    def step(self, action):
        self.steps += 1
        action = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)

        # Actuators (truth)
        self.truth_state["valve_cmd"] = float(action[0])
        self.truth_state["pump_cmd"] = float(action[1])
        self.truth_state["bf_cmd"] = float(action[2])
        self.truth_state["node_voltage_cmd"] = float(action[3])

        # Physics order:
        # Hydraulics → Clogging → Electrostatics → Particles → Optical
        self.hydraulics.update(
            state=self.truth_state,
            dt=self.dt,
            action=action,
            clogging_model=self.clogging,
        )
        self.clogging.update(self.truth_state, dt=self.dt)
        self.electrostatics.update(self.truth_state, dt=self.dt, node_cmd=self.truth_state["node_voltage_cmd"])
        self.particles.update(
            self.truth_state,
            dt=self.dt,
            clogging_model=self.clogging,
            electrostatics_model=self.electrostatics,
        )
        self.sensors.update(self.truth_state, dt=self.dt, sensor_state=self.sensor_state)

        self._update_normalized_state()

        flow = float(self.truth_state.get("flow", 0.0))
        pressure = float(self.truth_state.get("pressure", 0.0))
        clog = float(self.truth_state.get("clog", 0.0))

        reward = 2.0 * flow - 1.0 * pressure - 0.5 * clog

        terminated = False
        truncated = self.steps >= self.max_steps

        info = {
            "Q_out_Lmin": float(self.truth_state.get("Q_out_Lmin", 0.0)),
            "P_in": float(self.truth_state.get("P_in", 0.0)),
            "mesh_loading_avg": float(self.truth_state.get("mesh_loading_avg", 0.0)),
            "capture_eff": float(self.truth_state.get("capture_eff", 0.0)),
            "E_norm": float(self.truth_state.get("E_norm", 0.0)),

            # measured values come from sensor_state
            "sensor_turbidity": float(self.sensor_state.get("sensor_turbidity", 0.0)),
            "sensor_scatter": float(self.sensor_state.get("sensor_scatter", 0.0)),
        }

        # Part D (optional but helpful): expose run identity
        info["run_id"] = self.run_context.run_id
        info["version"] = self.run_context.version
        info["seed"] = self._active_seed
        info["noise_enabled"] = self.run_context.noise_enabled
        info["config_hash"] = self.run_context.config_hash

        # Accumulate episode return
        self._episode_return += float(reward)

        # Commit 2 timestep spine logging (now truth/sensor aware)
        self.logger.log_step({
            "event": "step",
            "run_id": self.run_context.run_id,
            "timestep": int(self.steps),
            "step": int(self.steps),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),

            "flow": float(self.truth_state.get("flow", 0.0)),
            "pressure": float(self.truth_state.get("pressure", 0.0)),
            "clog": float(self.truth_state.get("clog", 0.0)),

            "sensor_turbidity": float(self.sensor_state.get("sensor_turbidity", 0.0)),
            "sensor_scatter": float(self.sensor_state.get("sensor_scatter", 0.0)),
        })

        if terminated or truncated:
            self.logger.end_run(summary={
                "episode_return": float(self._episode_return),
                "steps": int(self.steps),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })

        return self._observe(), reward, terminated, truncated, info

    # ---------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------
    def _update_normalized_state(self):
        p = self.hydraulics.params

        Q = float(self.truth_state.get("Q_out_Lmin", 0.0))
        P = float(self.truth_state.get("P_in", 0.0))

        self.truth_state["flow"] = float(np.clip(Q / max(p.Q_max_Lmin, 1e-6), 0.0, 1.0))
        self.truth_state["pressure"] = float(np.clip(P / max(p.P_max_Pa, 1e-6), 0.0, 1.0))
        self.truth_state["clog"] = float(np.clip(self.truth_state.get("mesh_loading_avg", 0.0), 0.0, 1.0))

    def _observe(self):
        # Commit 3: stable observation contract
        return build_observation(self.truth_state, self.sensor_state)

    def render(self):
        print(
            f"Flow={self.truth_state.get('flow', 0.0):.3f}, "
            f"P={self.truth_state.get('pressure', 0.0):.3f}, "
            f"Clog={self.truth_state.get('clog', 0.0):.3f}, "
            f"E_norm={self.truth_state.get('E_norm', 0.0):.3f}, "
            f"Turb={self.sensor_state.get('sensor_turbidity', 0.0):.3f}, "
            f"Scatter={self.sensor_state.get('sensor_scatter', 0.0):.3f}"
        )
