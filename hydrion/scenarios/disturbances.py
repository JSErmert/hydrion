# hydrion/scenarios/disturbances.py
"""
Disturbance activation and effect application for HydrOS scenario execution.

Only disturbances with truth-state effects are applied here.
Sensor-only disturbances (sensor_noise, foam_event) are passed through
as active events so the marker system can emit them, but no truth mutation
is performed — consistent with v1 spec.
"""

from __future__ import annotations

from typing import List

from .types import DisturbanceEvent

Q_MAX_LMIN: float = 20.0


def get_active_disturbances(
    disturbances: List[DisturbanceEvent],
    t: float,
) -> List[DisturbanceEvent]:
    """Return every disturbance whose window contains t."""
    return [d for d in disturbances if d.time <= t < d.time + d.duration]


def apply_disturbances_to_flow(
    flow: float,
    active: List[DisturbanceEvent],
) -> float:
    """
    Overlay flow_spike disturbances onto the computed base flow.

    intensity is treated as a fraction of Q_MAX: intensity=1.0 → +20 L/min.
    """
    for d in active:
        if d.type == "flow_spike":
            flow += d.intensity * Q_MAX_LMIN * 0.1
    return float(max(0.0, min(flow, Q_MAX_LMIN)))


def apply_disturbances_to_particles(
    density: float,
    active: List[DisturbanceEvent],
) -> float:
    """
    Overlay particle_spike disturbances onto the base particle density.

    intensity is an additive offset (intensity=1.0 → +0.20 normalized).
    """
    for d in active:
        if d.type == "particle_spike":
            density += d.intensity * 0.20
    return float(max(0.0, min(density, 1.0)))
