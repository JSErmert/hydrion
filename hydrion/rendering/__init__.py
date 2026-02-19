"""
hydrion/rendering/__init__.py

Research Observatory Visualization Module for HydrOS.

This module provides side-effect free visualization and observability tools
for analyzing HydrOS episodes. All visualization logic lives here.

Main entry point: Observatory class
"""

from .observatory import Observatory
from .episode_history import EpisodeHistory
from .time_series import (
    plot_time_series,
    plot_actions,
    plot_psd_time_series,
    plot_psd_summary,
    plot_shape_variables,
    plot_reward_trace,
)
from .anomaly_detector import AnomalyDetector, plot_anomalies, plot_shield_events

# Re-export for convenience
__all__ = [
    "Observatory",
    "EpisodeHistory",
    "AnomalyDetector",
    "plot_time_series",
    "plot_actions",
    "plot_psd_time_series",
    "plot_psd_summary",
    "plot_shape_variables",
    "plot_reward_trace",
    "plot_anomalies",
    "plot_shield_events",
]
