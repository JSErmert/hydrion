# hydrion/state/init.py
from __future__ import annotations
from typing import Dict, Any
from .types import TruthState, SensorState


def init_truth_state() -> TruthState:
    # Minimal scaffold (same as your current env scaffold)
    return TruthState(
        data={
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
    )


def init_sensor_state() -> SensorState:
    return SensorState(
        data={
            "sensor_turbidity": 0.0,
            "sensor_scatter": 0.0,
        }
    )
