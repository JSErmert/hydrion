# hydrion/physics/electrostatics.py
"""
ElectrostaticsModel v2 — Radial field, two-subsystem architecture.

Replaces the axial scalar-gap model (v1) with a cylindrical radial field
model grounded in the concentric electrode geometry of the HydrOS device.

Architecture (06_LOCKED_SYSTEM_CONSTRAINTS.md §D):
    InletPolarizationRing  — 30% capture contribution, upstream charge conditioning
    OuterWallCollectorNode — 70% capture contribution, radial field at collection wall

Field geometry:
    E_r(r) = V / (r × ln(r_outer / r_inner))       [V/m]
    At collection wall (r = r_outer):
        E_r_wall = V / (r_outer × ln(r_outer / r_inner))
        E_field_kVm = E_r_wall / 1000.0             [kV/m]

Output:
    E_capture_gain in [0, 1]  — normalized electrostatic capture boost
                                consumed by particles.py as a gain signal
    E_field_kVm               — physical field at collection wall [kV/m]
    E_field_norm              — E_field_kVm / E_field_kVm_max in [0, 1]
                                replaces E_norm in obs12_v2 (index 3)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class ElectrostaticsParams:
    # Voltage bounds (locked system constraints — 06_LOCKED_SYSTEM_CONSTRAINTS.md §D)
    V_max_realism: float = 2500.0   # [V] upper operational bound
    V_hard_clamp:  float = 3000.0   # [V] absolute safety ceiling — never exceed

    # First-order voltage dynamics
    tau_charge: float = 0.5         # [s] time constant to reach command
    leak_rate:  float = 0.1         # [1/s] passive voltage decay

    # Radial geometry — concentric cylindrical capacitor
    r_inner_m: float = 0.005        # [m] counter-electrode (central rod) radius
    r_outer_m: float = 0.040        # [m] outer collection wall radius

    # Residence time model
    t_E_ref_s:      float = 2.0     # [s] residence time at which tanh -> 0.76 saturation
    stage_volume_L: float = 0.25    # [L] effective stage volume (placeholder — bench calibration)

    # Ring conditioning reference
    V_ring_ref: float = 500.0       # [V] voltage at which ring tanh -> 0.76 saturation

    # 30/70 functional allocation (locked — 06_LOCKED_SYSTEM_CONSTRAINTS.md §D)
    ring_weight: float = 0.30       # share of E capture from InletPolarizationRing
    node_weight: float = 0.70       # share of E capture from OuterWallCollectorNode

    eps: float = 1e-8


class ElectrostaticsModel:
    """
    ElectrostaticsModel v2

    Reads from state / actions:
        node_voltage_cmd  in [0, 1]   (action vector index 3)
        q_processed_lmin               (from hydraulics, for residence time)

    Writes to truth_state:
        V_node            [V]     actual node voltage after first-order dynamics
        E_field_kVm       [kV/m]  radial field at collection wall
        E_field_norm      []      E_field_kVm / E_field_kVm_max in [0, 1] (obs12_v2 index 3)
        charge_factor     []      InletPolarizationRing contribution in [0, ring_weight]
        node_capture_gain []      OuterWallCollectorNode contribution in [0, node_weight]
        E_capture_gain    []      total electrostatic capture boost in [0, 1]
    """

    def __init__(self, cfg: Any | None = None) -> None:
        e_raw: Dict[str, float] = {}
        if cfg is not None and hasattr(cfg, "raw"):
            e_raw = getattr(cfg, "raw", {}).get("electrostatics", {}) or {}

        self.params = ElectrostaticsParams(
            V_max_realism   = float(e_raw.get("V_max_realism",  2500.0)),
            V_hard_clamp    = float(e_raw.get("V_hard_clamp",   3000.0)),
            tau_charge      = float(e_raw.get("tau_charge",     0.5)),
            leak_rate       = float(e_raw.get("leak_rate",      0.1)),
            r_inner_m       = float(e_raw.get("r_inner_m",      0.005)),
            r_outer_m       = float(e_raw.get("r_outer_m",      0.040)),
            t_E_ref_s       = float(e_raw.get("t_E_ref_s",      2.0)),
            stage_volume_L  = float(e_raw.get("stage_volume_L", 0.25)),
            V_ring_ref      = float(e_raw.get("V_ring_ref",     500.0)),
            ring_weight     = float(e_raw.get("ring_weight",    0.30)),
            node_weight     = float(e_raw.get("node_weight",    0.70)),
            eps             = float(e_raw.get("eps",            1e-8)),
        )

        p = self.params
        # Precompute geometric constant: ln(r_outer / r_inner)
        self._ln_ratio = float(np.log(
            max(p.r_outer_m, p.eps) / max(p.r_inner_m, p.eps)
        ))
        # Precompute maximum field at V_max_realism for normalisation reference
        self._E_field_kVm_max = float(
            (p.V_max_realism / 1000.0) /
            max(p.r_outer_m * self._ln_ratio, p.eps)
        )

        self.state: Dict[str, float] = {}
        self._reset_state()

    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self.state = {
            "V_node":            0.0,
            "E_field_kVm":       0.0,
            "E_field_norm":      0.0,
            "charge_factor":     0.0,
            "node_capture_gain": 0.0,
            "E_capture_gain":    0.0,
        }

    def reset(self, state: Dict[str, float]) -> None:
        self._reset_state()
        state.update(self.state)

    def get_state(self) -> Dict[str, float]:
        return dict(self.state)

    # ------------------------------------------------------------------

    def update(
        self,
        state: Dict[str, float],
        dt: float,
        node_cmd: float,
    ) -> None:
        p = self.params
        node_cmd = float(np.clip(node_cmd, 0.0, 1.0))

        # ── Voltage dynamics (first-order approach with leakage) ──────────
        V_node = float(self.state.get("V_node", 0.0))
        V_target = float(np.clip(node_cmd * p.V_max_realism, 0.0, p.V_hard_clamp))
        dV = ((V_target - V_node) / max(p.tau_charge, p.eps) - p.leak_rate * V_node) * dt
        V_node = float(np.clip(V_node + dV, 0.0, p.V_hard_clamp))

        # ── Radial field at collection wall ───────────────────────────────
        # E_r(r_outer) = V / (r_outer × ln(r_outer / r_inner))
        E_r_wall_Vm = V_node / max(p.r_outer_m * self._ln_ratio, p.eps)
        E_field_kVm = E_r_wall_Vm / 1000.0
        E_field_norm = float(np.clip(
            E_field_kVm / max(self._E_field_kVm_max, p.eps), 0.0, 1.0
        ))

        # ── Residence time ────────────────────────────────────────────────
        Q_proc_Ls = max(float(state.get("q_processed_lmin", 0.0)) / 60.0, p.eps)
        t_residence_s = p.stage_volume_L / Q_proc_Ls
        t_sat = float(np.tanh(t_residence_s / max(p.t_E_ref_s, p.eps)))

        # ── SubSystem A: InletPolarizationRing (30%) ──────────────────────
        # Charge conditioning — scales with voltage and residence time.
        # Output in [0, ring_weight].
        V_norm = float(np.clip(V_node / max(p.V_ring_ref, p.eps), 0.0, 5.0))
        charge_factor = p.ring_weight * float(np.tanh(V_norm)) * t_sat

        # ── SubSystem B: OuterWallCollectorNode (70%) ─────────────────────
        # Primary capture via radial field.
        # Scales with normalised E-field and residence time.
        # Output in [0, node_weight].
        node_capture_gain = p.node_weight * E_field_norm * t_sat

        # ── Combined gain (normalised in [0, 1]) ──────────────────────────
        E_capture_gain = float(np.clip(charge_factor + node_capture_gain, 0.0, 1.0))

        # ── Write state ───────────────────────────────────────────────────
        self.state.update(
            V_node            = float(V_node),
            E_field_kVm       = float(E_field_kVm),
            E_field_norm      = float(E_field_norm),
            charge_factor     = float(charge_factor),
            node_capture_gain = float(node_capture_gain),
            E_capture_gain    = float(E_capture_gain),
        )
        state.update(self.state)
