"""
hydrion/rendering/time_series.py

Time-series plotting utilities for research observability.
Professional, publishable plots with labeled axes, units, and legends.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .episode_history import EpisodeHistory


# Variable metadata: (label, unit, ylim)
VARIABLE_METADATA: Dict[str, Tuple[str, str, Optional[Tuple[float, float]]]] = {
    # Hydraulics
    "flow": ("Flow", "normalized [0,1]", (0.0, 1.0)),
    "pressure": ("Pressure", "normalized [0,1]", (0.0, 1.0)),
    "clog": ("Clog", "normalized [0,1]", (0.0, 1.0)),
    "Q_out_Lmin": ("Flow Rate", "L/min", None),
    "P_in": ("Inlet Pressure", "Pa", None),
    "P_m1": ("Mesh 1 Pressure", "Pa", None),
    "P_m2": ("Mesh 2 Pressure", "Pa", None),
    "P_m3": ("Mesh 3 Pressure", "Pa", None),
    "P_out": ("Outlet Pressure", "Pa", None),
    
    # Electrostatics
    "E_norm": ("Electric Field", "normalized [0,1]", (0.0, 1.0)),
    
    # Particles
    "C_in": ("Inlet Concentration", "normalized", None),
    "C_out": ("Outlet Concentration", "normalized", None),
    "particle_capture_eff": ("Capture Efficiency", "normalized [0,1]", (0.0, 1.0)),
    "capture_eff": ("Capture Efficiency", "normalized [0,1]", (0.0, 1.0)),
    
    # Sensors
    "sensor_turbidity": ("Turbidity", "normalized [0,1]", (0.0, 1.0)),
    "sensor_scatter": ("Scatter", "normalized [0,1]", (0.0, 1.0)),
    
    # Shape (if enabled)
    "fiber_fraction": ("Fiber Fraction", "fraction", (0.0, 1.0)),
    "C_fibers": ("Fiber Concentration", "normalized", None),
    
    # PSD (if enabled)
    "C_L": ("Large Particle Conc.", "normalized", None),
    "C_M": ("Medium Particle Conc.", "normalized", None),
    "C_S": ("Small Particle Conc.", "normalized", None),
}


def plot_time_series(
    history: EpisodeHistory,
    variables: List[str],
    time_axis: str = "time",
    figsize: Tuple[float, float] = (12, 6),
    sharex: bool = True,
) -> Tuple[Figure, List[Axes]]:
    """
    Plot time series for multiple variables.
    
    Args:
        history: EpisodeHistory instance
        variables: List of variable keys to plot
        time_axis: "time" or "step" for x-axis
        figsize: Figure size (width, height)
        sharex: Whether to share x-axis across subplots
    
    Returns:
        (figure, axes_list)
    """
    if not history:
        raise ValueError("History is empty")
    
    n_vars = len(variables)
    if n_vars == 0:
        raise ValueError("No variables specified")
    
    # Get time array
    if time_axis == "time":
        x = history.get_time_array()
        xlabel = "Time (s)"
    else:
        x = history.get_step_array()
        xlabel = "Step"
    
    # Create subplots
    fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=sharex)
    if n_vars == 1:
        axes = [axes]
    
    # Plot each variable
    for i, var_key in enumerate(variables):
        ax = axes[i]
        
        # Get metadata
        label, unit, ylim = VARIABLE_METADATA.get(var_key, (var_key, "", None))
        
        # Extract data
        if var_key.startswith("sensor_"):
            y = history.get_sensor_variable(var_key)
        else:
            y = history.get_truth_variable(var_key)
        
        # Plot
        ax.plot(x, y, linewidth=1.5, label=label)
        ax.set_ylabel(f"{label} ({unit})" if unit else label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        
        if ylim:
            ax.set_ylim(ylim)
    
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    
    return fig, axes


def plot_actions(
    history: EpisodeHistory,
    time_axis: str = "time",
    figsize: Tuple[float, float] = (12, 6),
) -> Tuple[Figure, Axes]:
    """
    Plot action traces synchronized with state.
    
    Args:
        history: EpisodeHistory instance
        time_axis: "time" or "step" for x-axis
        figsize: Figure size
    
    Returns:
        (figure, axes)
    """
    if not history:
        raise ValueError("History is empty")
    
    # Get time array
    if time_axis == "time":
        x = history.get_time_array()
        xlabel = "Time (s)"
    else:
        x = history.get_step_array()
        xlabel = "Step"
    
    # Get actions
    actions = history.get_actions_array()
    if len(actions) == 0:
        raise ValueError("No actions recorded")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each action component
    action_labels = ["Valve", "Pump", "Backflush", "Node Voltage"]
    colors = ["blue", "green", "red", "orange"]
    
    for i in range(4):
        ax.plot(x, actions[:, i], label=action_labels[i], color=colors[i], linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Action [0,1]")
    ax.set_title("Action Traces")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    
    fig.tight_layout()
    
    return fig, ax


def plot_psd_time_series(
    history: EpisodeHistory,
    time_axis: str = "time",
    figsize: Tuple[float, float] = (12, 8),
) -> Optional[Tuple[Figure, List[Axes]]]:
    """
    Plot PSD (particle size distribution) time series if enabled.
    
    Args:
        history: EpisodeHistory instance
        time_axis: "time" or "step" for x-axis
        figsize: Figure size
    
    Returns:
        (figure, axes_list) if PSD enabled, None otherwise
    """
    if not history.has_psd():
        return None
    
    # Get time array
    if time_axis == "time":
        x = history.get_time_array()
        xlabel = "Time (s)"
    else:
        x = history.get_step_array()
        xlabel = "Step"
    
    # Get bin keys
    bin_keys = history.get_psd_bin_keys()
    if not bin_keys:
        return None
    
    n_bins = len(bin_keys)
    
    # Create subplots: C_in bins and C_out bins
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot C_in bins
    ax_in = axes[0]
    for i, key in enumerate(bin_keys):
        y = history.get_truth_variable(key)
        ax_in.plot(x, y, label=f"Bin {i}", linewidth=1.5, alpha=0.8)
    ax_in.set_ylabel("C_in (normalized)")
    ax_in.set_title("Inlet PSD")
    ax_in.grid(True, alpha=0.3)
    ax_in.legend(loc="upper right")
    
    # Plot C_out bins
    ax_out = axes[1]
    for i, key in enumerate(bin_keys):
        out_key = key.replace("C_in", "C_out")
        y = history.get_truth_variable(out_key)
        ax_out.plot(x, y, label=f"Bin {i}", linewidth=1.5, alpha=0.8)
    ax_out.set_ylabel("C_out (normalized)")
    ax_out.set_xlabel(xlabel)
    ax_out.set_title("Outlet PSD")
    ax_out.grid(True, alpha=0.3)
    ax_out.legend(loc="upper right")
    
    fig.tight_layout()
    
    return fig, axes


def plot_psd_summary(
    history: EpisodeHistory,
    time_axis: str = "time",
    figsize: Tuple[float, float] = (12, 6),
) -> Optional[Tuple[Figure, Axes]]:
    """
    Plot PSD summary (C_L, C_M, C_S) if available.
    
    Args:
        history: EpisodeHistory instance
        time_axis: "time" or "step" for x-axis
        figsize: Figure size
    
    Returns:
        (figure, axes) if C_L/C_M/C_S available, None otherwise
    """
    if not history.has_psd():
        return None
    
    # Check if C_L, C_M, C_S exist
    if not history.truth_states or "C_L" not in history.truth_states[0]:
        return None
    
    # Get time array
    if time_axis == "time":
        x = history.get_time_array()
        xlabel = "Time (s)"
    else:
        x = history.get_step_array()
        xlabel = "Step"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot C_L, C_M, C_S
    for key, label, color in [("C_L", "Large", "red"), ("C_M", "Medium", "orange"), ("C_S", "Small", "blue")]:
        y = history.get_truth_variable(key)
        ax.plot(x, y, label=label, color=color, linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Concentration (normalized)")
    ax.set_title("PSD Summary (Outlet)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    
    fig.tight_layout()
    
    return fig, ax


def plot_shape_variables(
    history: EpisodeHistory,
    time_axis: str = "time",
    figsize: Tuple[float, float] = (12, 6),
) -> Optional[Tuple[Figure, List[Axes]]]:
    """
    Plot shape variables (fiber_fraction, C_fibers) if enabled.
    
    Args:
        history: EpisodeHistory instance
        time_axis: "time" or "step" for x-axis
        figsize: Figure size
    
    Returns:
        (figure, axes_list) if shape enabled, None otherwise
    """
    if not history.has_shape():
        return None
    
    # Get time array
    if time_axis == "time":
        x = history.get_time_array()
        xlabel = "Time (s)"
    else:
        x = history.get_step_array()
        xlabel = "Step"
    
    # Determine which variables exist
    vars_to_plot = []
    if history.truth_states and "fiber_fraction" in history.truth_states[0]:
        vars_to_plot.append("fiber_fraction")
    if history.truth_states and "C_fibers" in history.truth_states[0]:
        vars_to_plot.append("C_fibers")
    
    if not vars_to_plot:
        return None
    
    n_vars = len(vars_to_plot)
    fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=True)
    if n_vars == 1:
        axes = [axes]
    
    for i, var_key in enumerate(vars_to_plot):
        ax = axes[i]
        label, unit, ylim = VARIABLE_METADATA.get(var_key, (var_key, "", None))
        y = history.get_truth_variable(var_key)
        
        ax.plot(x, y, linewidth=1.5, label=label)
        ax.set_ylabel(f"{label} ({unit})" if unit else label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        
        if ylim:
            ax.set_ylim(ylim)
    
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    
    return fig, axes


def plot_reward_trace(
    history: EpisodeHistory,
    time_axis: str = "time",
    cumulative: bool = True,
    figsize: Tuple[float, float] = (12, 6),
) -> Tuple[Figure, Axes]:
    """
    Plot reward trace (instantaneous and/or cumulative).
    
    Args:
        history: EpisodeHistory instance
        time_axis: "time" or "step" for x-axis
        cumulative: Whether to plot cumulative reward
        figsize: Figure size
    
    Returns:
        (figure, axes)
    """
    if not history:
        raise ValueError("History is empty")
    
    # Get time array
    if time_axis == "time":
        x = history.get_time_array()
        xlabel = "Time (s)"
    else:
        x = history.get_step_array()
        xlabel = "Step"
    
    rewards = history.get_rewards_array()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot instantaneous reward
    ax.plot(x, rewards, label="Instantaneous Reward", linewidth=1.5, alpha=0.7, color="blue")
    
    # Plot cumulative reward if requested
    if cumulative:
        cum_reward = np.cumsum(rewards)
        ax.plot(x, cum_reward, label="Cumulative Reward", linewidth=1.5, alpha=0.8, color="green")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Reward")
    ax.set_title("Reward Trace")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    
    fig.tight_layout()
    
    return fig, ax
