# hydrion/state/init.py
from __future__ import annotations
from typing import Dict, Any
from .types import TruthState, SensorState


def init_truth_state() -> TruthState:
    """
    Initialize the canonical truth_state dict.

    Fields are grouped by owning subsystem.  All new Milestone 1 fields are
    initialised here so every module can safely read them with .get() without
    needing a fallback path.

    Aggregate clogging fields (Mc*, n*, mesh_loading_avg) are preserved as
    derived compatibility fields; they are recomputed by CloggingModel each step.
    """
    return TruthState(
        data={
            # ----------------------------------------------------------------
            # Actuator commands (written by env.step before physics pipeline)
            # ----------------------------------------------------------------
            "valve_cmd":        0.5,
            "pump_cmd":         0.5,
            "bf_cmd":           0.0,
            "node_voltage_cmd": 0.5,

            # ----------------------------------------------------------------
            # Hydraulics — raw physical outputs (written by HydraulicsModel)
            # ----------------------------------------------------------------
            "Q_out_Lmin":       0.0,   # net forward filtration flow [L/min]
            "q_in_lmin":        0.0,   # pump-driven total input flow [L/min]
            "q_processed_lmin": 0.0,   # flow through filter stages (q_in − q_bypass) [L/min]
            "q_bypass_lmin":    0.0,   # flow diverted around filter stages [L/min]
            "P_in":             0.0,   # inlet pressure [Pa]
            "P_m1":             0.0,   # cumulative pressure level at Stage 1 inlet [Pa]
            "P_m2":             0.0,   # cumulative pressure level at Stage 2 inlet [Pa]
            "P_m3":             0.0,   # cumulative pressure level at Stage 3 inlet [Pa]
            "P_out":            0.0,   # outlet pressure [Pa]
            "dp_stage1_pa":     0.0,   # pressure drop across Stage 1 [Pa]
            "dp_stage2_pa":     0.0,   # pressure drop across Stage 2 [Pa]
            "dp_stage3_pa":     0.0,   # pressure drop across Stage 3 [Pa]
            "dp_total_pa":      0.0,   # total filter pressure drop (sum of stages) [Pa]
            "bypass_active":    0.0,   # 1.0 when passive bypass valve is open, else 0.0

            # ----------------------------------------------------------------
            # Clogging — decomposed fouling per stage (primary state)
            # Written by CloggingModel
            # ----------------------------------------------------------------
            # Stage 1 (coarse ~500 µm, area 120 cm²)
            "cake_s1":          0.0,   # surface cake fractional loading [0, 1]
            "bridge_s1":        0.0,   # fiber bridge fractional loading [0, 1]
            "pore_s1":          0.0,   # pore restriction fractional loading [0, 1]
            "irreversible_s1":  0.0,   # irreversible fouling fraction [0, 1]
            "recoverable_s1":   0.0,   # recoverable = fouling_frac − irreversible [0, 1]
            "fouling_frac_s1":  0.0,   # total = cake + bridge + pore [0, 1]

            # Stage 2 (medium ~100 µm, area 220 cm²)
            "cake_s2":          0.0,
            "bridge_s2":        0.0,
            "pore_s2":          0.0,
            "irreversible_s2":  0.0,
            "recoverable_s2":   0.0,
            "fouling_frac_s2":  0.0,

            # Stage 3 (fine pleated ~5 µm, area 900 cm²)
            "cake_s3":          0.0,
            "bridge_s3":        0.0,
            "pore_s3":          0.0,
            "irreversible_s3":  0.0,
            "recoverable_s3":   0.0,
            "fouling_frac_s3":  0.0,

            # ----------------------------------------------------------------
            # Clogging — aggregate compatibility fields (derived by CloggingModel)
            # These are recomputed each step from the decomposed fields above.
            # Preserved for telemetry continuity and downstream compatibility.
            # ----------------------------------------------------------------
            "Mc1":              0.0,
            "Mc2":              0.0,
            "Mc3":              0.0,
            "Mc1_max":          1.0,
            "Mc2_max":          1.0,
            "Mc3_max":          1.0,
            "n1":               0.0,
            "n2":               0.0,
            "n3":               0.0,
            "mesh_loading_avg": 0.0,
            "capture_eff":      0.0,

            # ----------------------------------------------------------------
            # Backflush event state (written by env.py state machine)
            # Must be updated BEFORE clogging.update() each step.
            # ----------------------------------------------------------------
            "bf_active":               0.0,   # 1.0 during an active pulse, else 0.0
            "bf_pulse_idx":            0.0,   # current pulse index within burst (0, 1, 2)
            "bf_burst_elapsed":        0.0,   # elapsed time within current burst [s]
            "bf_cooldown_remaining":   0.0,   # time until next burst is permitted [s]
            "bf_n_bursts_completed":   0.0,   # cumulative burst count (for diminishing returns)
            "bf_source_efficiency":    0.70,  # cleaning efficiency of active fluid source [0, 1]

            # ----------------------------------------------------------------
            # Electrostatics (written by ElectrostaticsModel — unchanged)
            # ----------------------------------------------------------------
            "V_node":       0.0,
            "E_field":      0.0,
            "E_field_norm": 0.0,   # obs12_v2 key (was E_norm in v1)

            # ----------------------------------------------------------------
            # Particles (written by ParticleModel — unchanged)
            # ----------------------------------------------------------------
            "C_in":                0.0,
            "C_out":               0.0,
            "particle_capture_eff": 0.0,
            "C_fibers":            1.0,

            # ----------------------------------------------------------------
            # Normalized observation channels (written by _update_normalized_state)
            # ----------------------------------------------------------------
            "flow":     0.5,
            "pressure": 0.4,
            "clog":     0.0,

            # ----------------------------------------------------------------
            # Derived telemetry (written by _update_normalized_state)
            # Status flag only — not used in reward shaping.
            # ----------------------------------------------------------------
            "maintenance_required": 0.0,   # 1.0 when max stage fouling ≥ threshold
        }
    )


def init_sensor_state() -> SensorState:
    return SensorState(
        data={
            "sensor_turbidity": 0.0,
            "sensor_scatter":   0.0,
        }
    )
