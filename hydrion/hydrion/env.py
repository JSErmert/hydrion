import gymnasium as gym
from gymnasium import spaces
import numpy as np

class HydrionEnv(gym.Env):
    """
    Minimal working Hydrion environment.

    Purpose:
    - Establish action_space and observation_space
    - Implement reset(), step(), reward
    - Provide a stable loop for PPO to train against
    - Allow future expansion (physics, sensors, anomalies, etc.)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config_path="configs/default.yaml", render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Simulation parameters
        self.dt = 0.1
        self.max_steps = int(600 / self.dt)

        # Internal state (placeholder)
        self.state = {
            "flow": 0.5,
            "pressure": 0.4,
            "clog": 0.0,
            "valve": 0.5,
            "pump": 0.5,
            "bf": 0.0,
        }

        # ACTIONS: [valve, pump, backflush, node_voltage]
        self.action_space = spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )

        # OBSERVATIONS: simple 8D placeholder
        # (This will be replaced by SensorFusion later)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        self.state = {
            "flow": 0.6,
            "pressure": 0.3,
            "clog": 0.0,
            "valve": 0.5,
            "pump": 0.5,
            "bf": 0.0,
        }

        return self._observe(), {}

    def step(self, action):
        self.steps += 1
        action = np.clip(action, 0.0, 1.0)

        # Update actuator states
        self.state["valve"] = float(action[0])
        self.state["pump"]  = float(action[1])
        self.state["bf"]    = float(action[2])

        # Simple placeholder physics
        self.state["clog"] += 0.001 * self.dt
        self.state["pressure"] = (
            0.2 + 0.6 * self.state["pump"] + 0.2 * self.state["clog"]
        )
        self.state["flow"] = (
            0.8 * self.state["valve"] * (1 - 0.5 * self.state["bf"])
        )

        # Reward: encourage high flow, low pressure, low clog
        reward = (
            + 2.0 * self.state["flow"]
            - 1.0 * self.state["pressure"]
            - 0.5 * self.state["clog"]
        )

        terminated = False
        truncated = self.steps >= self.max_steps

        info = {
            "flow": self.state["flow"],
            "pressure": self.state["pressure"],
            "clog": self.state["clog"],
        }

        return self._observe(), reward, terminated, truncated, info

    def _observe(self):
        """Return an 8D placeholder observation."""
        s = self.state
        return np.array([
            s["flow"],
            s["pressure"],
            s["clog"],
            s["valve"],
            s["pump"],
            s["bf"],
            np.random.rand() * 0.05,   # placeholder camera noise 1
            np.random.rand() * 0.05,   # placeholder camera noise 2
        ], dtype=np.float32)

    def render(self):
        print(
            f"Flow={self.state['flow']:.3f}, "
            f"P={self.state['pressure']:.3f}, "
            f"Clog={self.state['clog']:.3f}"
        )
