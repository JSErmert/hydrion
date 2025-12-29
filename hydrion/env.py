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


class HydrionEnv(gym.Env):
    """
    HydrionEnv — Multi-Physics Digital Twin Environment

    Includes:
    - Hydraulics
    - Clogging
    - Electrostatics (E-field)
    - Particle transport (C_in, C_out, capture efficiency)
    - Optical sensor array (turbidity, scatter)
    - Full 12-dimensional observation vector for PPO

    obs = [
        flow,
        pressure,
        clog,

        E_norm,

        C_out,
        particle_capture_eff,

        valve_cmd,
        pump_cmd,
        bf_cmd,
        node_voltage_cmd,

        sensor_turbidity,
        sensor_scatter
    ]
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
        # Default: logs go to project-root / runs / <run_id>/
        # If you want configurable later, we can add cfg.raw["logging"]["base_dir"]
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

        # State dictionary
        self.state: dict = {}

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
        # Resolve seed precedence:
        # 1) explicit reset(seed=...)
        # 2) run_context seed
        resolved_seed = self.run_context.seed if seed is None else int(seed)
        self._active_seed = resolved_seed

        # Gymnasium internal seeding
        super().reset(seed=resolved_seed)

        # Commit 1: global seeding for deterministic subsystem noise (np.random in sensors)
        set_global_seed(resolved_seed)

        self.steps = 0
        
        # Reset episode return accumulator
        self._episode_return = 0.0

        # Start logging run (Commit 2)
        run_header = {
            "run_id": self.run_context.run_id,
            "version": self.run_context.version,
            "seed": self._active_seed,
            "noise_enabled": self.run_context.noise_enabled,
            "config_hash": self.run_context.config_hash,
        }
        # Full config snapshot as logged evidence
        self.logger.start_run(
            run_id=self.run_context.run_id,
            run_header=run_header,
            config=self.cfg.raw,
        )

        # Log initial reset spine row
        self.logger.log_step({
            "event": "reset",
            "run_id": self.run_context.run_id,
            "timestep": 0,
            "step": 0,
            "seed": self._active_seed,
        })


        # Base state scaffold
        self.state = {
            "valve_cmd": 0.5,
            "pump_cmd": 0.5,
            "bf_cmd": 0.0,
            "node_voltage_cmd": 0.5,

            "Q_out_Lmin": 0.0,
            "P_in": 0.0,
            "P_m1": 0.0,
            "P_m2": 0.0,
            "P_m3": 0.0,
            "P_out": 0.0,

            # normalized placeholders (will be overwritten by _update_normalized_state)
            "flow": 0.5,
            "pressure": 0.4,
            "clog": 0.0,
        }

        # Reset subsystems
        self.clogging.reset(self.state)
        self.hydraulics.reset()
        self.electrostatics.reset(self.state)
        self.particles.reset(self.state)
        self.sensors.reset(self.state)

        # Neutral kick to initialize derived values
        neutral = np.array([0.5, 0.5, 0.0, 0.5], dtype=np.float32)

        self.hydraulics.update(self.state, dt=self.dt, action=neutral, clogging_model=self.clogging)
        self.electrostatics.update(self.state, dt=self.dt, node_cmd=self.state["node_voltage_cmd"])
        self.particles.update(
            self.state,
            dt=self.dt,
            clogging_model=self.clogging,
            electrostatics_model=self.electrostatics,
        )
        self.sensors.update(self.state, dt=self.dt)

        self._update_normalized_state()
        return self._observe(), {}

    # ---------------------------------------------------------
    # STEP
    # ---------------------------------------------------------
    def step(self, action):
        self.steps += 1
        action = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)

        # Actuators
        self.state["valve_cmd"] = float(action[0])
        self.state["pump_cmd"] = float(action[1])
        self.state["bf_cmd"] = float(action[2])
        self.state["node_voltage_cmd"] = float(action[3])

        # Physics order:
        # Hydraulics → Clogging → Electrostatics → Particles → Optical
        self.hydraulics.update(
            state=self.state,
            dt=self.dt,
            action=action,
            clogging_model=self.clogging,
        )
        self.clogging.update(self.state, dt=self.dt)
        self.electrostatics.update(self.state, dt=self.dt, node_cmd=self.state["node_voltage_cmd"])
        self.particles.update(
            self.state,
            dt=self.dt,
            clogging_model=self.clogging,
            electrostatics_model=self.electrostatics,
        )
        self.sensors.update(self.state, dt=self.dt)

        self._update_normalized_state()

        flow = float(self.state["flow"])
        pressure = float(self.state["pressure"])
        clog = float(self.state["clog"])

        reward = 2.0 * flow - 1.0 * pressure - 0.5 * clog

        terminated = False
        truncated = self.steps >= self.max_steps

        info = {
            "Q_out_Lmin": float(self.state.get("Q_out_Lmin", 0.0)),
            "P_in": float(self.state.get("P_in", 0.0)),
            "mesh_loading_avg": float(self.state.get("mesh_loading_avg", 0.0)),
            "capture_eff": float(self.state.get("capture_eff", 0.0)),
            "E_norm": float(self.state.get("E_norm", 0.0)),
            "sensor_turbidity": float(self.state.get("sensor_turbidity", 0.0)),
            "sensor_scatter": float(self.state.get("sensor_scatter", 0.0)),
        }

        # ------------------------------
        # Part D (OPTIONAL but HIGHLY helpful):
        # expose run identity for debugging now, logging later
        # ------------------------------
        info["run_id"] = self.run_context.run_id
        info["version"] = self.run_context.version
        info["seed"] = self._active_seed
        info["noise_enabled"] = self.run_context.noise_enabled
        info["config_hash"] = self.run_context.config_hash

                # Accumulate episode return
        self._episode_return += float(reward)

        # Commit 2: timestep spine logging (minimal, extensible)
        # We keep it small now: time index, reward, termination, and a few key scalars.
        self.logger.log_step({
            "event": "step",
            "run_id": self.run_context.run_id,
            "timestep": int(self.steps),
            "step": int(self.steps),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "flow": float(self.state.get("flow", 0.0)),
            "pressure": float(self.state.get("pressure", 0.0)),
            "clog": float(self.state.get("clog", 0.0)),
            "sensor_turbidity": float(self.state.get("sensor_turbidity", 0.0)),
            "sensor_scatter": float(self.state.get("sensor_scatter", 0.0)),
        })

        # If episode ended, close out run with a summary row
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

        Q = float(self.state.get("Q_out_Lmin", 0.0))
        P = float(self.state.get("P_in", 0.0))

        self.state["flow"] = float(np.clip(Q / max(p.Q_max_Lmin, 1e-6), 0.0, 1.0))
        self.state["pressure"] = float(np.clip(P / max(p.P_max_Pa, 1e-6), 0.0, 1.0))
        self.state["clog"] = float(np.clip(self.state.get("mesh_loading_avg", 0.0), 0.0, 1.0))

    def _observe(self):
        s = self.state
        return np.array(
            [
                # hydraulics
                float(s.get("flow", 0.0)),
                float(s.get("pressure", 0.0)),
                float(s.get("clog", 0.0)),

                # electrostatics
                float(s.get("E_norm", 0.0)),

                # particle transport
                float(s.get("C_out", 0.0)),
                float(s.get("particle_capture_eff", 0.0)),

                # actuator commands
                float(s.get("valve_cmd", 0.0)),
                float(s.get("pump_cmd", 0.0)),
                float(s.get("bf_cmd", 0.0)),
                float(s.get("node_voltage_cmd", 0.0)),

                # optical sensors
                float(s.get("sensor_turbidity", 0.0)),
                float(s.get("sensor_scatter", 0.0)),
            ],
            dtype=np.float32,
        )

    def render(self):
        print(
            f"Flow={self.state.get('flow', 0.0):.3f}, "
            f"P={self.state.get('pressure', 0.0):.3f}, "
            f"Clog={self.state.get('clog', 0.0):.3f}, "
            f"E_norm={self.state.get('E_norm', 0.0):.3f}, "
            f"Turb={self.state.get('sensor_turbidity', 0.0):.3f}, "
            f"Scatter={self.state.get('sensor_scatter', 0.0):.3f}"
        )
