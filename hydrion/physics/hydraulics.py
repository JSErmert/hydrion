# hydrion/physics/hydraulics.py
"""
HydraulicsModel v2 — Area-normalized resistance, bypass logic, explicit stage ΔP.

Changes from v1
---------------
1. Area-normalized clog sensitivity
   k_mi_clog_eff = k_mi_clog × (A_s3_ref / A_si)
   Physical basis: resistance increase per unit fouling is inversely proportional
   to stage area at fixed pore-resistance density.
   CALIBRATION NOTE: This is a first-pass area-scaling approximation.
   Base resistances (R_m*_base) capture intrinsic pore resistance and are
   unchanged.  All area factors are derived from stage_geometry: in YAML.

2. Bypass logic (passive pressure-relief)
   Activates when estimated P_in exceeds bypass_pressure_threshold_pa.
   Hysteresis band prevents step-level oscillation.
   Splits Q_in into q_processed_lmin + q_bypass_lmin.

3. Explicit stage pressure drops
   dp_stage1_pa, dp_stage2_pa, dp_stage3_pa, dp_total_pa written to state.

4. Normalization bug fix
   max_Q_Lmin default aligned to Q_max_Lmin (20.0).  Was 50.0, which allowed
   Q_out > Q_max_Lmin and therefore flow > 1.0 in the observation.

Public API (unchanged from v1):
    reset() -> None
    update(state, dt, action, clogging_model) -> None
    get_state() -> dict
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class HydraulicsParams:
    # --- Normalization / clipping ---
    Q_max_Lmin: float = 20.0      # normalization reference and peak-flow ceiling [L/min]
    P_max_Pa:   float = 80_000.0  # max pressure for normalization [Pa]

    # --- Pipe and valve resistances [Pa·s/m³] ---
    R_pipe:       float = 2.0e7
    R_valve_min:  float = 0.5e7
    R_valve_max:  float = 8.0e7

    # --- Base mesh resistances (intrinsic pore resistance) [Pa·s/m³] ---
    R_m1_base:   float = 1.0e7
    R_m2_base:   float = 2.0e7
    R_m3_base:   float = 4.0e7

    # --- Clog sensitivity coefficients (before area scaling) [Pa·s/m³ per unit n] ---
    k_m1_clog:   float = 3.0e7
    k_m2_clog:   float = 4.0e7
    k_m3_clog:   float = 5.0e7

    # --- Backflush flow-diversion penalty (fraction of Q_processed diverted) ---
    bf_flow_penalty: float = 0.8

    # --- Flow bounds ---
    min_Q_Lmin:  float = 0.0
    max_Q_Lmin:  float = 20.0   # MUST equal Q_max_Lmin to keep normalized flow ≤ 1.0

    # --- Stage geometry (cm²) — loaded from stage_geometry: section ---
    area_s1_cm2: float = 120.0
    area_s2_cm2: float = 220.0
    area_s3_cm2: float = 900.0   # reference area for area-factor computation

    # --- Bypass parameters ---
    bypass_pressure_threshold_pa: float = 65_000.0
    bypass_flow_fraction:         float = 0.30
    bypass_hysteresis_fraction:   float = 0.90


class HydraulicsModel:
    """
    Hydraulics model v2 with area-normalized resistance and bypass.

    Reads from YAML sections: hydraulics:, stage_geometry:, bypass:
    All parameters are loadable from config; no hardcoded physics constants.
    """

    def __init__(self, cfg: Any) -> None:
        # Safe extraction helper
        def _section(name: str) -> Dict[str, Any]:
            if hasattr(cfg, "raw") and isinstance(cfg.raw, dict):
                return cfg.raw.get(name, {}) or {}
            return {}

        h_raw  = _section("hydraulics")
        sg_raw = _section("stage_geometry")
        bp_raw = _section("bypass")

        def gf(key: str, default: float, src: Dict = h_raw) -> float:
            val = src.get(key, default)
            try:
                return float(val)
            except Exception:
                return default

        # Keep Q_max_Lmin and max_Q_Lmin in sync: if YAML omits max_Q_Lmin,
        # default it to Q_max_Lmin (not the old 50.0 hardcode).
        Q_max = gf("Q_max_Lmin", 20.0)

        self.params = HydraulicsParams(
            Q_max_Lmin = Q_max,
            P_max_Pa   = gf("P_max_Pa", 80_000.0),

            R_pipe      = gf("R_pipe",      2.0e7),
            R_valve_min = gf("R_valve_min", 0.5e7),
            R_valve_max = gf("R_valve_max", 8.0e7),

            R_m1_base   = gf("R_m1_base", 1.0e7),
            R_m2_base   = gf("R_m2_base", 2.0e7),
            R_m3_base   = gf("R_m3_base", 4.0e7),

            k_m1_clog   = gf("k_m1_clog", 3.0e7),
            k_m2_clog   = gf("k_m2_clog", 4.0e7),
            k_m3_clog   = gf("k_m3_clog", 5.0e7),

            bf_flow_penalty = gf("bf_flow_penalty", 0.8),
            min_Q_Lmin      = gf("min_Q_Lmin", 0.0),
            max_Q_Lmin      = gf("max_Q_Lmin", Q_max),  # defaults to Q_max_Lmin if absent

            # Stage geometry
            area_s1_cm2 = gf("area_s1_cm2", 120.0, sg_raw),
            area_s2_cm2 = gf("area_s2_cm2", 220.0, sg_raw),
            area_s3_cm2 = gf("area_s3_cm2", 900.0, sg_raw),

            # Bypass
            bypass_pressure_threshold_pa = gf("bypass_pressure_threshold_pa", 65_000.0, bp_raw),
            bypass_flow_fraction         = gf("bypass_flow_fraction",          0.30,      bp_raw),
            bypass_hysteresis_fraction   = gf("bypass_hysteresis_fraction",    0.90,      bp_raw),
        )

        self.state: Dict[str, float] = {}
        self.reset()

    def reset(self) -> None:
        self.state = {
            "Q_out_Lmin":     0.0,
            "q_in_lmin":      0.0,
            "q_processed_lmin": 0.0,
            "q_bypass_lmin":  0.0,
            "P_in":           0.0,
            "P_m1":           0.0,
            "P_m2":           0.0,
            "P_m3":           0.0,
            "P_out":          0.0,
            "dp_stage1_pa":   0.0,
            "dp_stage2_pa":   0.0,
            "dp_stage3_pa":   0.0,
            "dp_total_pa":    0.0,
            "bypass_active":  0.0,
        }

    def get_state(self) -> Dict[str, float]:
        return dict(self.state)

    def update(
        self,
        state: Dict[str, Any],
        dt: float,
        action: np.ndarray,
        clogging_model: Any = None,
    ) -> None:
        p = self.params

        # ---- Decode action ----
        valve_open = float(np.clip(action[0], 0.0, 1.0))
        pump_cmd   = float(np.clip(action[1], 0.0, 1.0))
        bf_cmd     = float(np.clip(action[2], 0.0, 1.0))

        # ---- Clogging state (normalized fouling fractions) ----
        # v3: n_i now equals fouling_frac_si (derived by CloggingModel v3)
        n1 = n2 = n3 = 0.0
        if clogging_model is not None:
            cs = clogging_model.get_state()
            n1 = float(np.clip(cs.get("n1", 0.0), 0.0, 1.0))
            n2 = float(np.clip(cs.get("n2", 0.0), 0.0, 1.0))
            n3 = float(np.clip(cs.get("n3", 0.0), 0.0, 1.0))

        # ---- Area-normalized clog sensitivity ----
        # Physical basis: for flow through a porous medium at fixed pore properties,
        # resistance increase per unit fouling ∝ 1/area.
        # A_s3 (900 cm²) is the reference → area_factor_s3 = 1.0.
        # CALIBRATION NOTE: first-pass approximation; base resistances are unchanged.
        A_ref   = max(p.area_s3_cm2, 1e-6)
        k_m1_eff = p.k_m1_clog * (A_ref / max(p.area_s1_cm2, 1e-6))  # ≈ 7.5 × k_m1_clog
        k_m2_eff = p.k_m2_clog * (A_ref / max(p.area_s2_cm2, 1e-6))  # ≈ 4.1 × k_m2_clog
        k_m3_eff = p.k_m3_clog * 1.0                                   # area_factor = 1.0

        # ---- Stage resistances ----
        R_m1 = p.R_m1_base + k_m1_eff * n1
        R_m2 = p.R_m2_base + k_m2_eff * n2
        R_m3 = p.R_m3_base + k_m3_eff * n3

        # ---- Valve resistance ----
        R_valve = p.R_valve_min + (1.0 - valve_open) * (p.R_valve_max - p.R_valve_min)

        # ---- Total series resistance ----
        R_forward = max(p.R_pipe + R_valve + R_m1 + R_m2 + R_m3, 1e-6)

        # ---- Pump curve → Q_in ----
        # Pump curve: P_pump = P_max_eff × (1 - (Q / Q_max)²)
        # System curve: P_sys  = R_forward × Q  (SI)
        # Operating point (intersection): P_pump = P_sys
        #   → P_max_eff × (1 - u²) = R_forward × Q_max_m3s × u  where u = Q / Q_max_m3s
        #   → u² + β·u - 1 = 0  with β = R_forward × Q_max_m3s / P_max_eff
        #   → u = (-β + √(β² + 4)) / 2    (positive root, u ∈ [0, 1] by construction)
        # This avoids the q_ratio-clip bug where Q_raw > Q_max zeroed P_avail.
        P_max_eff   = p.P_max_Pa * pump_cmd
        Q_max_m3s   = p.Q_max_Lmin / 60000.0
        if P_max_eff < 1e-12:
            Q_in = p.min_Q_Lmin
        else:
            beta  = R_forward * Q_max_m3s / P_max_eff
            u     = (-beta + float(np.sqrt(beta ** 2 + 4.0))) / 2.0
            Q_in  = float(np.clip(u * p.Q_max_Lmin, p.min_Q_Lmin, p.Q_max_Lmin))

        # ---- Bypass logic ----
        # Compute tentative P_in at full Q_in (no bypass)
        P_tentative = R_forward * (Q_in / 60000.0)

        bypass_active_prev = float(state.get("bypass_active", 0.0)) > 0.5

        if P_tentative > p.bypass_pressure_threshold_pa:
            bypass_active = True
        elif bypass_active_prev and P_tentative > p.bypass_pressure_threshold_pa * p.bypass_hysteresis_fraction:
            bypass_active = True   # hold within hysteresis band
        else:
            bypass_active = False

        if bypass_active:
            Q_bypass    = Q_in * p.bypass_flow_fraction
            Q_processed = Q_in * (1.0 - p.bypass_flow_fraction)
        else:
            Q_bypass    = 0.0
            Q_processed = Q_in

        # ---- Actual P_in and stage pressure drops (based on Q_processed) ----
        Q_proc_m3s = Q_processed / 60000.0   # L/min → m³/s

        dp_stage1 = R_m1 * Q_proc_m3s
        dp_stage2 = R_m2 * Q_proc_m3s
        dp_stage3 = R_m3 * Q_proc_m3s
        dp_total  = dp_stage1 + dp_stage2 + dp_stage3

        # Reconstruct cumulative pressure levels from outlet upward
        dP_pipe  = p.R_pipe  * Q_proc_m3s
        dP_valve = R_valve   * Q_proc_m3s
        P_out = 0.0
        P_m3  = P_out  + dp_stage3
        P_m2  = P_m3   + dp_stage2
        P_m1  = P_m2   + dp_stage1
        P_in  = P_m1   + dP_valve + dP_pipe

        # ---- Q_out: processed flow minus backflush diversion ----
        diversion = float(np.clip(p.bf_flow_penalty * bf_cmd, 0.0, 1.0))
        Q_out = float(np.clip(Q_processed * (1.0 - diversion), p.min_Q_Lmin, p.max_Q_Lmin))

        # ---- Write state ----
        self.state.update(
            Q_out_Lmin      = Q_out,
            q_in_lmin       = float(Q_in),
            q_processed_lmin = float(Q_processed),
            q_bypass_lmin   = float(Q_bypass),
            P_in            = float(P_in),
            P_m1            = float(P_m1),
            P_m2            = float(P_m2),
            P_m3            = float(P_m3),
            P_out           = float(P_out),
            dp_stage1_pa    = float(dp_stage1),
            dp_stage2_pa    = float(dp_stage2),
            dp_stage3_pa    = float(dp_stage3),
            dp_total_pa     = float(dp_total),
            bypass_active   = 1.0 if bypass_active else 0.0,
        )

        state.update(self.state)
