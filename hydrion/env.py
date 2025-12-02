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


class HydrionEnv(gym.Env):
    """
    HydrionEnv v3.0 — Multi-Physics Digital Twin Environment

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

    def __init__(self, config_path: str = "configs/default.yaml", render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Load YAML config
        with open(config_path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        self.cfg = HydrionConfig(raw_cfg)

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
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )

        self.steps = 0
        self.reset()

    # ---------------------------------------------------------
    # RESET
    # ---------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

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

        # Neutral kick
        neutral = np.array([0.5, 0.5, 0.0, 0.5], dtype=np.float32)

        self.hydraulics.update(self.state, dt=self.dt, action=neutral, clogging_model=self.clogging)
        self.electrostatics.update(self.state, dt=self.dt, node_cmd=self.state["node_voltage_cmd"])
        self.particles.update(self.state, dt=self.dt, clogging_model=self.clogging, electrostatics_model=self.electrostatics)
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
        self.particles.update(self.state, dt=self.dt, clogging_model=self.clogging, electrostatics_model=self.electrostatics)
        self.sensors.update(self.state, dt=self.dt)

        self._update_normalized_state()

        flow = float(self.state["flow"])
        pressure = float(self.state["pressure"])
        clog = float(self.state["clog"])

        reward = 2.0 * flow - 1.0 * pressure - 0.5 * clog

        terminated = False
        truncated = self.steps >= self.max_steps

        info = {
            "Q_out_Lmin": float(self.state["Q_out_Lmin"]),
            "P_in": float(self.state["P_in"]),
            "mesh_loading_avg": float(self.state.get("mesh_loading_avg", 0.0)),
            "capture_eff": float(self.state.get("capture_eff", 0.0)),
            "E_norm": float(self.state.get("E_norm", 0.0)),
            "sensor_turbidity": float(self.state.get("sensor_turbidity", 0.0)),
            "sensor_scatter": float(self.state.get("sensor_scatter", 0.0)),
        }

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
        self.state["clog"] = float(
            np.clip(self.state.get("mesh_loading_avg", 0.0), 0.0, 1.0)
        )

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

    # ---------------------------------------------------------
    def render(self):
        print(
            f"Flow={self.state['flow']:.3f}, "
            f"P={self.state['pressure']:.3f}, "
            f"Clog={self.state['clog']:.3f}, "
            f"E_norm={self.state.get('E_norm', 0.0):.3f}, "
            f"Turb={self.state.get('sensor_turbidity', 0.0):.3f}, "
            f"Scatter={self.state.get('sensor_scatter', 0.0):.3f}"
        )
