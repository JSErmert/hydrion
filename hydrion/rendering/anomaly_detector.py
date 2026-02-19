"""
hydrion/rendering/anomaly_detector.py

Anomaly detection and visualization for research observability.
Detects NaNs, bounds violations, shield events, and termination causes.
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .episode_history import EpisodeHistory


class AnomalyDetector:
    """
    Detects anomalies in episode history:
    - NaNs in state variables
    - Bounds violations (values outside expected ranges)
    - Shield events (safety interventions)
    - Termination causes
    """
    
    def __init__(self, history: EpisodeHistory):
        self.history = history
        self.anomalies: List[Dict[str, Any]] = []
        self._detect_all()
    
    def _detect_all(self):
        """Run all anomaly detection checks."""
        self.anomalies = []
        self._detect_nans()
        self._detect_bounds_violations()
        self._detect_shield_events()
        self._detect_termination_causes()
    
    def _detect_nans(self):
        """Detect NaN values in truth_state and sensor_state."""
        for step_idx, (truth, sensor) in enumerate(zip(self.history.truth_states, self.history.sensor_states)):
            # Check truth_state
            for key, value in truth.items():
                if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                    self.anomalies.append({
                        "step": step_idx,
                        "type": "nan" if np.isnan(value) else "inf",
                        "variable": key,
                        "value": value,
                        "source": "truth_state",
                    })
            
            # Check sensor_state
            for key, value in sensor.items():
                if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                    self.anomalies.append({
                        "step": step_idx,
                        "type": "nan" if np.isnan(value) else "inf",
                        "variable": key,
                        "value": value,
                        "source": "sensor_state",
                    })
    
    def _detect_bounds_violations(self):
        """Detect values outside expected bounds."""
        # Expected bounds for normalized variables [0, 1]
        normalized_vars = ["flow", "pressure", "clog", "E_norm", "C_out", "particle_capture_eff",
                          "valve_cmd", "pump_cmd", "bf_cmd", "node_voltage_cmd",
                          "sensor_turbidity", "sensor_scatter", "fiber_fraction"]
        
        for step_idx, truth in enumerate(self.history.truth_states):
            for var in normalized_vars:
                if var in truth:
                    value = float(truth[var])
                    if value < -0.01 or value > 1.01:  # Small tolerance for floating point
                        self.anomalies.append({
                            "step": step_idx,
                            "type": "bounds_violation",
                            "variable": var,
                            "value": value,
                            "expected": "[0, 1]",
                            "source": "truth_state",
                        })
        
        # Check sensor bounds
        for step_idx, sensor in enumerate(self.history.sensor_states):
            for var in ["sensor_turbidity", "sensor_scatter"]:
                if var in sensor:
                    value = float(sensor[var])
                    if value < -0.01 or value > 1.01:
                        self.anomalies.append({
                            "step": step_idx,
                            "type": "bounds_violation",
                            "variable": var,
                            "value": value,
                            "expected": "[0, 1]",
                            "source": "sensor_state",
                        })
    
    def _detect_shield_events(self):
        """Detect safety shield interventions."""
        safety_infos = self.history.get_safety_info()
        
        for step_idx, safety_info in enumerate(safety_infos):
            if not safety_info:
                continue
            
            # Check for action projection
            if safety_info.get("projected", False):
                self.anomalies.append({
                    "step": step_idx,
                    "type": "shield_projection",
                    "reason": safety_info.get("reason", "unknown"),
                    "source": "safety_shield",
                })
            
            # Check for violations
            violation_types = [
                "soft_pressure_violation",
                "hard_pressure_violation",
                "soft_clog_violation",
                "hard_clog_violation",
                "blockage_violation",
            ]
            
            for violation_type in violation_types:
                if safety_info.get(violation_type, False):
                    self.anomalies.append({
                        "step": step_idx,
                        "type": violation_type,
                        "penalty": safety_info.get("penalty", 0.0),
                        "source": "safety_shield",
                    })
    
    def _detect_termination_causes(self):
        """Detect episode termination causes."""
        if self.history.terminated or self.history.truncated:
            # Check last info dict for termination details
            if self.history.infos:
                last_info = self.history.infos[-1]
                if "terminated" in last_info or "truncated" in last_info:
                    self.anomalies.append({
                        "step": len(self.history.steps) - 1,
                        "type": "termination",
                        "terminated": self.history.terminated,
                        "truncated": self.history.truncated,
                        "source": "episode",
                    })
    
    def get_anomalies_by_type(self, anomaly_type: str) -> List[Dict[str, Any]]:
        """Get all anomalies of a specific type."""
        return [a for a in self.anomalies if a.get("type") == anomaly_type]
    
    def get_anomalies_by_step(self, step: int) -> List[Dict[str, Any]]:
        """Get all anomalies at a specific step."""
        return [a for a in self.anomalies if a.get("step") == step]
    
    def has_anomalies(self) -> bool:
        """Check if any anomalies were detected."""
        return len(self.anomalies) > 0
    
    def summary(self) -> Dict[str, int]:
        """Get summary counts by anomaly type."""
        summary = {}
        for anomaly in self.anomalies:
            anomaly_type = anomaly.get("type", "unknown")
            summary[anomaly_type] = summary.get(anomaly_type, 0) + 1
        return summary


def plot_anomalies(
    history: EpisodeHistory,
    detector: Optional[AnomalyDetector] = None,
    time_axis: str = "time",
    figsize: Tuple[float, float] = (12, 8),
) -> Optional[Tuple[Figure, List[Axes]]]:
    """
    Visualize anomalies detected in episode history.
    
    Args:
        history: EpisodeHistory instance
        detector: Optional AnomalyDetector (will create one if not provided)
        time_axis: "time" or "step" for x-axis
        figsize: Figure size
    
    Returns:
        (figure, axes_list) if anomalies found, None otherwise
    """
    if detector is None:
        detector = AnomalyDetector(history)
    
    if not detector.has_anomalies():
        return None
    
    # Get time array
    if time_axis == "time":
        x = history.get_time_array()
        xlabel = "Time (s)"
    else:
        x = history.get_step_array()
        xlabel = "Step"
    
    # Group anomalies by type
    anomaly_types = set(a.get("type", "unknown") for a in detector.anomalies)
    n_types = len(anomaly_types)
    
    if n_types == 0:
        return None
    
    # Create subplots
    fig, axes = plt.subplots(n_types, 1, figsize=figsize, sharex=True)
    if n_types == 1:
        axes = [axes]
    
    # Plot each anomaly type
    for ax_idx, anomaly_type in enumerate(sorted(anomaly_types)):
        ax = axes[ax_idx]
        
        # Get anomalies of this type
        type_anomalies = detector.get_anomalies_by_type(anomaly_type)
        
        # Create binary signal: 1 where anomaly occurs, 0 otherwise
        anomaly_signal = np.zeros(len(history.steps))
        for anomaly in type_anomalies:
            step_idx = anomaly.get("step", 0)
            if 0 <= step_idx < len(anomaly_signal):
                anomaly_signal[step_idx] = 1.0
        
        # Plot
        ax.plot(x, anomaly_signal, drawstyle="steps-post", linewidth=2, color="red", alpha=0.8)
        ax.fill_between(x, 0, anomaly_signal, alpha=0.3, color="red")
        ax.set_ylabel("Anomaly")
        ax.set_title(f"{anomaly_type.replace('_', ' ').title()} ({len(type_anomalies)} occurrences)")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    
    return fig, axes


def plot_shield_events(
    history: EpisodeHistory,
    time_axis: str = "time",
    figsize: Tuple[float, float] = (12, 6),
) -> Optional[Tuple[Figure, Axes]]:
    """
    Visualize safety shield events (projections and violations).
    
    Args:
        history: EpisodeHistory instance
        time_axis: "time" or "step" for x-axis
        figsize: Figure size
    
    Returns:
        (figure, axes) if shield events found, None otherwise
    """
    safety_infos = history.get_safety_info()
    
    if not safety_infos or not any(safety_infos):
        return None
    
    # Get time array
    if time_axis == "time":
        x = history.get_time_array()
        xlabel = "Time (s)"
    else:
        x = history.get_step_array()
        xlabel = "Step"
    
    # Extract shield signals
    projected = np.array([info.get("projected", False) for info in safety_infos], dtype=float)
    penalty = np.array([info.get("penalty", 0.0) for info in safety_infos])
    
    # Check for violations
    violation_types = [
        ("soft_pressure_violation", "Soft Pressure"),
        ("hard_pressure_violation", "Hard Pressure"),
        ("soft_clog_violation", "Soft Clog"),
        ("hard_clog_violation", "Hard Clog"),
        ("blockage_violation", "Blockage"),
    ]
    
    has_violations = any(
        any(info.get(vt[0], False) for info in safety_infos)
        for vt in violation_types
    )
    
    if not np.any(projected) and not has_violations:
        return None
    
    # Create plot
    n_subplots = 1 + (1 if has_violations else 0)
    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)
    if n_subplots == 1:
        axes = [axes]
    
    # Plot projections
    ax = axes[0]
    ax.plot(x, projected, drawstyle="steps-post", linewidth=2, color="orange", label="Action Projected", alpha=0.8)
    ax.fill_between(x, 0, projected, alpha=0.3, color="orange")
    ax.set_ylabel("Shield Active")
    ax.set_title("Safety Shield: Action Projections")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    
    # Plot violations if present
    if has_violations:
        ax = axes[1]
        for violation_type, label in violation_types:
            violation_signal = np.array([info.get(violation_type, False) for info in safety_infos], dtype=float)
            if np.any(violation_signal):
                ax.plot(x, violation_signal, drawstyle="steps-post", linewidth=1.5, label=label, alpha=0.7)
        ax.set_ylabel("Violation")
        ax.set_xlabel(xlabel)
        ax.set_title("Safety Shield: Violations")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    else:
        axes[0].set_xlabel(xlabel)
    
    fig.tight_layout()
    
    return fig, axes
