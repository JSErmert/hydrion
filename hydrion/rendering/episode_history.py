"""
hydrion/rendering/episode_history.py

Episode history recorder for visualization.
Pure observer: records truth_state, sensor_state, actions, rewards, info without side effects.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np


class EpisodeHistory:
    """
    Records episode data for visualization and analysis.
    
    Side-effect free: only reads and stores data, never modifies simulation state.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Clear all recorded history."""
        self.steps: List[int] = []
        self.timesteps: List[float] = []  # Cumulative time
        self.truth_states: List[Dict[str, Any]] = []
        self.sensor_states: List[Dict[str, Any]] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.infos: List[Dict[str, Any]] = []
        self.observations: List[np.ndarray] = []
        self.terminated: bool = False
        self.truncated: bool = False
        self.dt: Optional[float] = None
    
    def record_step(
        self,
        step: int,
        truth_state: Dict[str, Any],
        sensor_state: Dict[str, Any],
        action: np.ndarray,
        reward: float,
        info: Dict[str, Any],
        observation: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
    ):
        """
        Record a single step of episode data.
        
        Args:
            step: Step number (0-indexed)
            truth_state: Physics truth state dict
            sensor_state: Sensor state dict
            action: Action array [valve, pump, backflush, node_voltage]
            reward: Reward value
            info: Info dict (may contain safety, validation outputs, etc.)
            observation: Optional 12D observation vector
            dt: Time step duration (used to compute timesteps)
        """
        # Store shallow copies to avoid mutation
        self.steps.append(step)
        self.truth_states.append(dict(truth_state))
        self.sensor_states.append(dict(sensor_state))
        self.actions.append(np.array(action, copy=True))
        self.rewards.append(float(reward))
        self.infos.append(dict(info))
        if observation is not None:
            self.observations.append(np.array(observation, copy=True))
        
        # Compute cumulative time
        if dt is not None:
            self.dt = dt
            if len(self.timesteps) == 0:
                self.timesteps.append(dt)
            else:
                self.timesteps.append(self.timesteps[-1] + dt)
        else:
            self.timesteps.append(float(step))
    
    def finalize(self, terminated: bool = False, truncated: bool = False):
        """Mark episode as complete."""
        self.terminated = terminated
        self.truncated = truncated
    
    def get_time_array(self) -> np.ndarray:
        """Get time array for plotting."""
        return np.array(self.timesteps)
    
    def get_step_array(self) -> np.ndarray:
        """Get step array for plotting."""
        return np.array(self.steps)
    
    def get_truth_variable(self, key: str, default: float = 0.0) -> np.ndarray:
        """Extract a truth state variable across all steps."""
        return np.array([float(state.get(key, default)) for state in self.truth_states])
    
    def get_sensor_variable(self, key: str, default: float = 0.0) -> np.ndarray:
        """Extract a sensor state variable across all steps."""
        return np.array([float(state.get(key, default)) for state in self.sensor_states])
    
    def get_actions_array(self) -> np.ndarray:
        """Get actions as (n_steps, 4) array."""
        if not self.actions:
            return np.zeros((0, 4))
        return np.array(self.actions)
    
    def get_rewards_array(self) -> np.ndarray:
        """Get rewards as array."""
        return np.array(self.rewards)
    
    def has_psd(self) -> bool:
        """Check if PSD variables are present in truth_state."""
        if not self.truth_states:
            return False
        return "C_in_bin_0" in self.truth_states[0]
    
    def has_shape(self) -> bool:
        """Check if shape variables are present in truth_state."""
        if not self.truth_states:
            return False
        return "fiber_fraction" in self.truth_states[0] or "C_fibers" in self.truth_states[0]
    
    def get_psd_bin_keys(self) -> List[str]:
        """Get list of PSD bin keys (C_in_bin_0, C_out_bin_0, etc.) if PSD enabled."""
        if not self.has_psd():
            return []
        keys = []
        i = 0
        while f"C_in_bin_{i}" in self.truth_states[0]:
            keys.append(f"C_in_bin_{i}")
            i += 1
        return keys
    
    def get_safety_info(self) -> List[Dict[str, Any]]:
        """Extract safety info from all steps."""
        return [info.get("safety", {}) for info in self.infos]
    
    def __len__(self) -> int:
        """Number of recorded steps."""
        return len(self.steps)
