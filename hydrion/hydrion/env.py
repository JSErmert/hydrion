import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml

from .config import HydrionConfig
from .physics.hydraulics import HydraulicsModel
from .physics.clogging import CloggingModel


class HydrionEnv(gym.Env):
    """
    HydrionEnv v1.7 — full fidelity environment integrating:
    - Long-form HydraulicsModel (pressure ladder, nonlinear pump curve)
    - Long-form CloggingModel (tri-mesh nonlinear clog dynamics)
    - Blackboard-style shared state dict
    - Normalized observations for RL
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config_path="hydrion/configs/default.yaml", render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Load configuration
        with open(config_path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        self.cfg = HydrionConfig(raw_cfg)

        # Simulation settings
        self.dt = float(self.cfg.raw.get("sim", {}).get("dt", 0.1))
        self.max_steps = int(600.0 / self.dt)

        # Physics models
        self.hydraulics = HydraulicsModel(self.cfg)
        self.clogging = CloggingModel(self.cfg)

        # Blackboard (shared state for all physics)
        self.state = {}

        # Action space: [valve, pump, backflush, node_voltage]
        self.action_space = spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: 8D normalized vector
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

        self.steps = 0
        self.reset()

    # --------------------------------------------------------------------
    # RESET
    # --------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        # Base structure
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

        # Reset clogging
        self.clogging.reset(self.state)

        # Stabilize hydraulics with a neutral action
        neutral_action = np.array([0.5, 0.5, 0.0, 0.5], dtype=np.float32)
        self.hydraulics.update(self.state, dt=self.dt, action=neutral_action, clogging_model=self.clogging)

        self._update_normalized_state()

        return self._observe(), {}

    # --------------------------------------------------------------------
    # STEP
    # --------------------------------------------------------------------
    def step(self, action):
        self.steps += 1

        # Prevent invalid inputs
        action = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)

        # Update commands
        self.state["valve_cmd"] = float(action[0])
        self.state["pump_cmd"] = float(action[1])
        self.state["bf_cmd"] = float(action[2])
        self.state["node_voltage_cmd"] = float(action[3])

        # Physics ordering: hydraulics FIRST → clogging SECOND
        self.hydraulics.update(self.state, dt=self.dt, action=action, clogging_model=self.clogging)
        self.clogging.update(self.state, dt=self.dt)

        # Normalize obs
        self._update_normalized_state()

        # Reward
        flow = self.state["flow"]
        pressure = self.state["pressure"]
        clog = self.state["clog"]

        reward = 2.0 * flow - 1.0 * pressure - 0.5 * clog

        terminated = False
        truncated = self.steps >= self.max_steps

        info = {
            "Q_out_Lmin": float(self.state["Q_out_Lmin"]),
            "P_in": float(self.state["P_in"]),
            "mesh_loading_avg": float(self.state.get("mesh_loading_avg", 0.0)),
            "capture_eff": float(self.state.get("capture_eff", 0.0)),
        }

        return self._observe(), reward, terminated, truncated, info

    # --------------------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------------------
    def _update_normalized_state(self):
        p = self.hydraulics.params

        Q = float(self.state["Q_out_Lmin"])
        P = float(self.state["P_in"])

        self.state["flow"] = float(np.clip(Q / max(p.Q_max_Lmin, 1e-6), 0.0, 1.0))
        self.state["pressure"] = float(np.clip(P / max(p.P_max_Pa, 1e-6), 0.0, 1.0))
        self.state["clog"] = float(np.clip(self.state.get("mesh_loading_avg", 0.0), 0.0, 1.0))

    def _observe(self):
        s = self.state

        n1 = s.get("n1", 0.0)
        n2 = s.get("n2", 0.0)
        n3 = s.get("n3", 0.0)

        mesh_var = float(np.var([n1, n2, n3])) if (n1 + n2 + n3) > 0 else 0.0
        mesh_var = float(np.clip(mesh_var, 0.0, 1.0))

        return np.array([
            s["flow"],
            s["pressure"],
            s["clog"],
            s["valve_cmd"],
            s["pump_cmd"],
            s["bf_cmd"],
            s.get("capture_eff", 0.0),
            mesh_var
        ], dtype=np.float32)

    def render(self):
        print(
            f"Flow={self.state['flow']:.3f}, "
            f"P={self.state['pressure']:.3f}, "
            f"Clog={self.state['clog']:.3f}, "
