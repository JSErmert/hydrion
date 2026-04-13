# hydrion/sensors/sensor_fusion.py
from __future__ import annotations
import numpy as np


def build_observation(truth: dict, sensors: dict) -> np.ndarray:
    """
    obs12_v2 — stable observation contract.

    Builds the 12D observation vector strictly from:
    - normalized truth_state values
    - measured sensor_state values

    Schema version: obs12_v2 (M3 — 2026-04-10)
    Change from v1: index 3 is now E_field_norm (physical kV/m normalised to [0,1])
                    replacing E_norm (dimensionless arbitrary reference in [0,2]).

    Index mapping:
        0   flow
        1   pressure
        2   clog
        3   E_field_norm        <- obs12_v2 (was E_norm in obs12_v1)
        4   C_out
        5   particle_capture_eff
        6   valve_cmd
        7   pump_cmd
        8   bf_cmd
        9   node_voltage_cmd
        10  sensor_turbidity
        11  sensor_scatter

    This function is the *single source of truth* for the RL observation.
    DO NOT change index ordering without bumping the schema version label.
    """
    return np.array(
        [
            # hydraulics (normalized)
            float(truth.get("flow", 0.0)),
            float(truth.get("pressure", 0.0)),
            float(truth.get("clog", 0.0)),

            # electrostatics (obs12_v2: E_field_norm replaces E_norm)
            float(truth.get("E_field_norm", 0.0)),

            # particle transport
            float(truth.get("C_out", 0.0)),
            float(truth.get("particle_capture_eff", 0.0)),

            # actuator commands
            float(truth.get("valve_cmd", 0.0)),
            float(truth.get("pump_cmd", 0.0)),
            float(truth.get("bf_cmd", 0.0)),
            float(truth.get("node_voltage_cmd", 0.0)),

            # optical sensors (measured)
            float(sensors.get("sensor_turbidity", 0.0)),
            float(sensors.get("sensor_scatter", 0.0)),
        ],
        dtype=np.float32,
    )
