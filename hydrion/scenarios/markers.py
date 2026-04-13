# hydrion/scenarios/markers.py
"""
Runtime event marker detection for HydrOS scenario execution.

detect_runtime_markers() is called every step.  It compares the current
truth_state against the previous flag snapshot and emits markers whenever
a meaningful state transition is detected.

Marker types (per spec):
  Required:  scenario_start, scenario_end
  Conditional:
    threshold_crossing  — maintenance_required: 0 → 1
    backflush_start     — bf_active: 0 → 1
    backflush_end       — bf_active: 1 → 0
    bypass_start        — bypass_active: 0 → 1
    bypass_end          — bypass_active: 1 → 0
    disturbance_start   — any new disturbance becomes active
    disturbance_end     — any active disturbance deactivates
"""

from __future__ import annotations

from typing import Any, Dict, List

from .types import DisturbanceEvent, ScenarioEventMarker


def detect_runtime_markers(
    t: float,
    prev_flags: Dict[str, Any],
    truth_state: Dict[str, Any],
    prev_active_disturbances: List[DisturbanceEvent],
    current_active_disturbances: List[DisturbanceEvent],
) -> List[ScenarioEventMarker]:
    """
    Compare current truth_state against prev_flags and active disturbance lists.
    Return any newly detected event markers.
    """
    markers: List[ScenarioEventMarker] = []

    # --- maintenance_required threshold crossing (0 → 1 only) ---
    prev_maint = int(prev_flags.get("maintenance_required", 0))
    curr_maint = int(truth_state.get("maintenance_required", 0.0) > 0.5)
    if prev_maint == 0 and curr_maint == 1:
        ff_s1 = truth_state.get("fouling_frac_s1", 0.0)
        ff_s2 = truth_state.get("fouling_frac_s2", 0.0)
        ff_s3 = truth_state.get("fouling_frac_s3", 0.0)
        markers.append(ScenarioEventMarker(
            time=t,
            type="threshold_crossing",
            label="Maintenance threshold reached",
            meta={"fouling_frac_s1": ff_s1, "fouling_frac_s2": ff_s2, "fouling_frac_s3": ff_s3},
        ))

    # --- backflush ---
    prev_bf = int(prev_flags.get("bf_active", 0))
    curr_bf = int(truth_state.get("bf_active", 0.0) > 0.5)
    if prev_bf == 0 and curr_bf == 1:
        markers.append(ScenarioEventMarker(
            time=t, type="backflush_start", label="Backflush burst started",
        ))
    elif prev_bf == 1 and curr_bf == 0:
        markers.append(ScenarioEventMarker(
            time=t, type="backflush_end", label="Backflush burst ended",
        ))

    # --- bypass ---
    prev_bypass = int(prev_flags.get("bypass_active", 0))
    curr_bypass = int(truth_state.get("bypass_active", 0.0) > 0.5)
    if prev_bypass == 0 and curr_bypass == 1:
        markers.append(ScenarioEventMarker(
            time=t, type="bypass_start", label="Bypass valve activated",
            meta={"P_in": truth_state.get("P_in", 0.0)},
        ))
    elif prev_bypass == 1 and curr_bypass == 0:
        markers.append(ScenarioEventMarker(
            time=t, type="bypass_end", label="Bypass valve deactivated",
        ))

    # --- disturbances ---
    prev_ids = {(d.type, d.time) for d in prev_active_disturbances}
    curr_ids = {(d.type, d.time) for d in current_active_disturbances}
    for d in current_active_disturbances:
        if (d.type, d.time) not in prev_ids:
            markers.append(ScenarioEventMarker(
                time=t, type="disturbance_start",
                label=f"Disturbance started: {d.type}",
                meta={"disturbance_type": d.type, "intensity": d.intensity},
            ))
    for d in prev_active_disturbances:
        if (d.type, d.time) not in curr_ids:
            markers.append(ScenarioEventMarker(
                time=t, type="disturbance_end",
                label=f"Disturbance ended: {d.type}",
                meta={"disturbance_type": d.type},
            ))

    return markers


def update_prev_flags(
    prev_flags: Dict[str, Any],
    truth_state: Dict[str, Any],
) -> None:
    """Update the prev_flags snapshot in-place from current truth_state."""
    prev_flags["maintenance_required"] = int(truth_state.get("maintenance_required", 0.0) > 0.5)
    prev_flags["bf_active"]            = int(truth_state.get("bf_active",            0.0) > 0.5)
    prev_flags["bypass_active"]        = int(truth_state.get("bypass_active",        0.0) > 0.5)
