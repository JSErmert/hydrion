import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml

from .config import HydrionConfig
from .physics.hydraulics import HydraulicsModel
from .physics.clogging import CloggingModel


class HydrionEnv(gym.Env):
    """
    HydrionEnv v1.6 — RL wrapper around HydraulicsModel + CloggingModel.

    Features:
    - Multi-physics flow + clogging
    - Clean blackboard state shared across subsystems
    - 8D normalized observation vector
    - Fully stable for PPO / Safe RL extensions
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config_path: str = "configs/default.yaml", render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # --- Load YAML configuration
        with open(config_path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        self.cfg = HydrionConfig(raw_cfg)

        # Simulation parameters
        self.dt: float = float(self.cfg.raw.get("sim", {}).get("dt", 0.1))
        self.max_steps: int = int(600.0 / self.dt)

        # --- Physics models
        self.hydraulics = HydraulicsModel(self.cfg)
        self.clogging = CloggingModel(self.cfg)

        # Global state dict
        self.state: dict = {}

        # Action space: [valve, pump, backflush, node_voltage]
        self.action_space = spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space: 8D normalized vector
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

        self.steps = 0
        self.reset()

    # ---------------------------------------------------------
    # RESET
    # ---------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        # Base scaffold
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

        # Reset clogging (writes mesh state)
        self.clogging.reset(self.state)

        # Neutral hydraulics update
        neutral = np.array([0.5, 0.5, 0.0, 0.5], dtype=np.float32)
        self.hydraulics.update(self.state, dt=self.dt, action=neutral, clogging_model=self.clogging)

        # Normalize for observation
        self._update_normalized_state()

        return self._observe(), {}

    # ---------------------------------------------------------
    # STEP
    # ---------------------------------------------------------
    def step(self, action):
        self.steps += 1

        # Clip action to valid range
        action = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)

        # Update commands
        self.state["valve_cmd"] = float(action[0])
        self.state["pump_cmd"] = float(action[1])
        self.state["bf_cmd"] = float(action[2])
        self.state["node_voltage_cmd"] = float(action[3])

        # PHYSICS ORDER: Hydraulics FIRST → Clogging SECOND
        self.hydraulics.update(
            state=self.state,
            dt=self.dt,
            action=action,
            clogging_model=self.clogging,
        )
        self.clogging.update(self.state, dt=self.dt)

        # Normalize
        self._update_normalized_state()

        # Reward
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

        n1 = float(s.get("n1", 0.0))
        n2 = float(s.get("n2", 0.0))
        n3 = float(s.get("n3", 0.0))

        mesh_var = float(np.var([n1, n2, n3])) if (n1 + n2 + n3) > 0 else 0.0
        mesh_var = float(np.clip(mesh_var, 0.0, 1.0))

        return np.array(
            [
                s["flow"],
                s["pressure"],
                s["clog"],
                s["valve_cmd"],
                s["pump_cmd"],
                s["bf_cmd"],
                s.get("capture_eff", 0.0),
                mesh_var,
            ],
            dtype=np.float32,
        )

    def render(self):
        print(
            f"Flow={self.state['flow']:.3f}, "
            f"P={self.state['pressure']:.3f}, "
            f"Clog={self.state['clog']:.3f}, "
            f"CaptureEff={self.state.get('capture_eff', 0.0):.3f}"
        )
