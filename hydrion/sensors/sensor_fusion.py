# hydrion/sensors/sensor_fusion.py
from __future__ import annotations
import numpy as np


def build_observation(truth: dict, sensors: dict) -> np.ndarray:
    """
    obs14_v1 — sensor-extended observation contract.

    Builds the 14D observation vector from:
    - normalized truth_state values (indices 0–9, unchanged from obs12_v2)
    - measured sensor_state values (indices 10–13)

    Schema version: obs14_v1 (M6.2B — 2026-04-13)
    Change from obs12_v2: indices 0–11 UNCHANGED. Two new sensor-derived indices appended:
        index 12: flow_sensor_norm  = sensors["flow_sensor_lmin"] / 20.0  (clip [0,1])
        index 13: dp_sensor_norm    = sensors["dp_sensor_kPa"]   / 80.0   (clip [0,1])

    SEMANTIC WARNING — indices 12 and 13 are NOT substitutes for indices 0 and 1:
        Index 0  (flow)     = truth["flow"]                = Q_out_Lmin / 20.0 (post-backflush-diversion)
        Index 12 (sensor)   = sensors["flow_sensor_lmin"] / 20.0               (q_processed, noisy, pre-diversion)
        Index 1  (pressure) = truth["pressure"]            = P_in / 80000       (inlet absolute, encodes bypass signal)
        Index 13 (sensor)   = sensors["dp_sensor_kPa"]    / 80.0               (dp_total, noisy, pure filter differential)
    Shared normalization denominator does NOT imply shared physical quantity.

    Index mapping:
        0   flow                  truth_state     Q_out_Lmin / 20.0
        1   pressure              truth_state     P_in / 80000
        2   clog                  truth_state     mesh_loading_avg
        3   E_field_norm          truth_state     radial E-field (obs12_v2: replaces E_norm)
        4   C_out                 truth_state     outlet particle concentration
        5   particle_capture_eff  truth_state     capture efficiency
        6   valve_cmd             truth_state     valve actuator command
        7   pump_cmd              truth_state     pump actuator command
        8   bf_cmd                truth_state     backflush command
        9   node_voltage_cmd      truth_state     node voltage command
        10  sensor_turbidity      sensor_state    optical turbidity (noisy)
        11  sensor_scatter        sensor_state    optical scatter (noisy)
        12  flow_sensor_norm      sensor_state    flow_sensor_lmin / 20.0, clip [0,1]  <- obs14_v1
        13  dp_sensor_norm        sensor_state    dp_sensor_kPa / 80.0,    clip [0,1]  <- obs14_v1

    This function is the *single source of truth* for the HydrionEnv RL observation.
    DO NOT change index ordering without bumping the schema version label.
    CCE uses a separate 12D truth-derived schema — obs14_v1 does not apply to CCE.
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

            # hydraulic sensors — obs14_v1 extension (M6.2B — 2026-04-13)
            float(np.clip(sensors.get("flow_sensor_lmin", 0.0) / 20.0, 0.0, 1.0)),
            float(np.clip(sensors.get("dp_sensor_kPa",   0.0) / 80.0, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


def build_observation_obs8(truth: dict, sensors: dict) -> np.ndarray:
    """
    obs8_deployment_v1 — deployment-realistic 8D observation contract.

    Retains only channels available at hardware deployment:
        - actuator_feedback channels (obs14_v1 indices 6–9, re-indexed 0–3)
        - sensor_derived channels    (obs14_v1 indices 10–13, re-indexed 4–7)

    Removed: obs14_v1 indices 0–5 (physics_truth — simulation-only quantities).

    Index mapping (obs8_deployment_v1):
        0  valve_cmd         truth_state    actuator command (obs14_v1 index 6)
        1  pump_cmd          truth_state    actuator command (obs14_v1 index 7)
        2  bf_cmd            truth_state    actuator command (obs14_v1 index 8)
        3  node_voltage_cmd  truth_state    actuator command (obs14_v1 index 9)
        4  sensor_turbidity  sensor_state   optical turbidity (obs14_v1 index 10)
        5  sensor_scatter    sensor_state   optical scatter   (obs14_v1 index 11)
        6  flow_sensor_norm  sensor_state   flow_sensor_lmin / 20.0, clip [0,1] (obs14_v1 index 12)
        7  dp_sensor_norm    sensor_state   dp_sensor_kPa / 80.0, clip [0,1]   (obs14_v1 index 13)

    Schema version: obs8_deployment_v1 (M8 — 2026-04-14)
    This schema is deployment-realistic, NOT sensor-only.
    Indices 0–3 are actuator command feedback (controller-generated, always self-known).
    Indices 4–7 are sensor-derived (hardware-measurable).

    Source traceability: M8.1R.4_sources_map.md S1 (POMDP self-knowledge),
    S4 (action history), S5 (asymmetric actor-critic deployment obs).

    DO NOT change index ordering without bumping the schema version label.
    obs14_v1 is preserved in build_observation() — these are parallel functions.
    """
    return np.array(
        [
            # actuator_feedback channels (controller-issued, always deployment-available)
            float(truth.get("valve_cmd",         0.0)),
            float(truth.get("pump_cmd",          0.0)),
            float(truth.get("bf_cmd",            0.0)),
            float(truth.get("node_voltage_cmd",  0.0)),

            # sensor_derived channels (hardware-measurable)
            float(sensors.get("sensor_turbidity", 0.0)),
            float(sensors.get("sensor_scatter",   0.0)),

            # hydraulic sensors — same normalization as obs14_v1 indices 12 and 13
            float(np.clip(sensors.get("flow_sensor_lmin", 0.0) / 20.0, 0.0, 1.0)),
            float(np.clip(sensors.get("dp_sensor_kPa",   0.0) / 80.0, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )
