# hydrion/physics/clogging.py
"""
CloggingModel v3 — Decomposed fouling with backflush pulse recovery.

Primary state per stage: cake_si, bridge_si, pore_si, irreversible_si
Derived (written for compatibility): n_i, Mc_i, mesh_loading_avg, capture_eff

Public API (identical to v2 — no upstream callers break):
    reset(state: dict) -> None
    update(state: dict, dt: float) -> None
    get_state() -> dict

Calibration note
----------------
dep_rate_s* multipliers and component weight distributions are first-pass
estimates based on physical intuition (pore size, stage area, fiber behavior).
They are NOT validated against lab data at Milestone 1.  All values are
exposed in default.yaml so they can be re-tuned without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class CloggingParams:
    """
    All tunable parameters for the decomposed tri-stage fouling model.

    Units
    -----
    - Mc* are in arbitrary units (AU), internally consistent.
    - Flow in L/min (consistent with HydraulicsModel).
    - dep_base in AU / (L/min · s).
    - shear_coeff in 1/s.
    """

    # --- Aggregate capacity (kept for Mc* compatibility) ---
    Mc1_max: float = 1.0
    Mc2_max: float = 1.0
    Mc3_max: float = 1.0

    # --- Core deposition dynamics ---
    dep_base:      float = 1e-3
    # dep_exponent=2 creates BISTABLE kinetics: each stage has an unstable fixed
    # point ff_u = shear_coeff / (dep_rate * dep_base * Q_ref).  Below ff_u the
    # stage self-cleans to zero; above it fouling accelerates to saturation.
    # Fixed points at default params: Stage 1 ≈ 0.67, Stage 2 ≈ 0.42, Stage 3 ≈ 0.22.
    # Consequence: the filter will NOT foul from a perfectly clean reset without a
    # seed perturbation above ff_u.  For monotone growth from clean state,
    # set dep_exponent=1.0 (linear; ff_ss → ∞, saturates at 1.0 via clip).
    dep_exponent:  float = 2.0
    shear_coeff:   float = 5e-3
    shear_Q_ref:   float = 15.0
    eps:           float = 1e-8

    # --- Per-stage deposition rate multipliers ---
    dep_rate_s1: float = 0.5
    dep_rate_s2: float = 0.8
    dep_rate_s3: float = 1.5

    # --- Fouling component weights per stage (must sum to 1.0 per stage) ---
    cake_weight_s1:   float = 0.20
    bridge_weight_s1: float = 0.60
    pore_weight_s1:   float = 0.20

    cake_weight_s2:   float = 0.30
    bridge_weight_s2: float = 0.45
    pore_weight_s2:   float = 0.25

    cake_weight_s3:   float = 0.55
    bridge_weight_s3: float = 0.25
    pore_weight_s3:   float = 0.20

    # --- Maintenance / irreversible fouling ---
    maintenance_fouling_threshold: float = 0.70
    irreversible_stress_threshold: float = 0.70
    irreversible_rate:             float = 0.05

    # --- Backflush recovery coefficients ---
    # Fraction of recoverable loading removed per dt, normalized over pulse_duration_s.
    # Relative to clean-water source (bf_source_efficiency scales all three).
    bf_cake_recovery:    float = 0.35
    bf_bridge_recovery:  float = 0.20
    bf_pore_recovery:    float = 0.08
    bf_diminishing_factor: float = 0.80
    bf_pulse_duration_s: float = 0.4   # mirrors backflush:pulse_duration_s for time-normalization

    # --- Capture efficiency curve (YAML-parameterized) ---
    # Equation: baseline + gain × n × (1 − n)^exponent, clipped to [floor, ceiling]
    # Non-monotonic: peaks at moderate fouling, softens near saturation.
    capture_eff_baseline:  float = 0.80
    capture_eff_gain:      float = 0.12
    capture_eff_exponent:  float = 0.50
    capture_eff_floor:     float = 0.30
    capture_eff_ceiling:   float = 0.98


class CloggingModel:
    """
    CloggingModel v3 — see module docstring.
    """

    def __init__(self, cfg: Any = None) -> None:
        # ---- Load clogging: section ----
        c_raw: Dict[str, Any] = {}
        f_raw: Dict[str, Any] = {}
        bf_raw: Dict[str, Any] = {}
        if cfg is not None:
            if hasattr(cfg, "raw"):
                raw = getattr(cfg, "raw", {})
                c_raw  = raw.get("clogging", {}) or {}
                f_raw  = raw.get("fouling",  {}) or {}
                bf_raw = raw.get("backflush", {}) or {}
            elif isinstance(cfg, dict):
                c_raw  = cfg.get("clogging", {}) or {}
                f_raw  = cfg.get("fouling",  {}) or {}
                bf_raw = cfg.get("backflush", {}) or {}

        def gf(key: str, default: float, src: Dict = c_raw) -> float:
            val = src.get(key, default)
            try:
                return float(val)
            except Exception:
                return default

        # Recovery coefficients: fouling: section takes precedence over clogging: defaults
        bf_cake_rec  = gf("bf_cake_recovery",    0.35, f_raw) if "bf_cake_recovery"    in f_raw else gf("bf_cake_recovery",    0.35)
        bf_brid_rec  = gf("bf_bridge_recovery",  0.20, f_raw) if "bf_bridge_recovery"  in f_raw else gf("bf_bridge_recovery",  0.20)
        bf_pore_rec  = gf("bf_pore_recovery",    0.08, f_raw) if "bf_pore_recovery"    in f_raw else gf("bf_pore_recovery",    0.08)
        bf_dim_fac   = gf("bf_diminishing_factor", 0.80, f_raw) if "bf_diminishing_factor" in f_raw else gf("bf_diminishing_factor", 0.80)

        self.params = CloggingParams(
            Mc1_max        = gf("Mc1_max", 1.0),
            Mc2_max        = gf("Mc2_max", 1.0),
            Mc3_max        = gf("Mc3_max", 1.0),
            dep_base       = gf("dep_base", 1e-3),
            dep_exponent   = gf("dep_exponent", 2.0),
            shear_coeff    = gf("shear_coeff", 5e-3),
            shear_Q_ref    = gf("shear_Q_ref", 15.0),
            eps            = gf("eps", 1e-8),
            dep_rate_s1    = gf("dep_rate_s1", 0.5),
            dep_rate_s2    = gf("dep_rate_s2", 0.8),
            dep_rate_s3    = gf("dep_rate_s3", 1.5),
            cake_weight_s1   = gf("cake_weight_s1",   0.20),
            bridge_weight_s1 = gf("bridge_weight_s1", 0.60),
            pore_weight_s1   = gf("pore_weight_s1",   0.20),
            cake_weight_s2   = gf("cake_weight_s2",   0.30),
            bridge_weight_s2 = gf("bridge_weight_s2", 0.45),
            pore_weight_s2   = gf("pore_weight_s2",   0.25),
            cake_weight_s3   = gf("cake_weight_s3",   0.55),
            bridge_weight_s3 = gf("bridge_weight_s3", 0.25),
            pore_weight_s3   = gf("pore_weight_s3",   0.20),
            maintenance_fouling_threshold = gf("maintenance_fouling_threshold", 0.70),
            irreversible_stress_threshold = gf("irreversible_stress_threshold", 0.70),
            irreversible_rate             = gf("irreversible_rate",             0.05),
            bf_cake_recovery    = bf_cake_rec,
            bf_bridge_recovery  = bf_brid_rec,
            bf_pore_recovery    = bf_pore_rec,
            bf_diminishing_factor = bf_dim_fac,
            bf_pulse_duration_s = gf("pulse_duration_s", 0.4, bf_raw),
            capture_eff_baseline = gf("capture_eff_baseline",  0.80),
            capture_eff_gain     = gf("capture_eff_gain",      0.12),
            capture_eff_exponent = gf("capture_eff_exponent",  0.50),
            capture_eff_floor    = gf("capture_eff_floor",     0.30),
            capture_eff_ceiling  = gf("capture_eff_ceiling",   0.98),
        )

        self._state: Dict[str, float] = {}
        self._build_zero_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, state: Dict[str, float]) -> None:
        """Reset all fouling to zero and write initial values into the shared state dict."""
        self._build_zero_state()
        state.update(self._state)

    def update(self, state: Dict[str, float], dt: float) -> None:
        """
        Advance clogging dynamics one timestep.

        Reads from state:
            Q_out_Lmin (or "flow" fallback)  — forward flow driving deposition/shear
            C_fibers                          — fiber concentration from ParticleModel
            bf_active                         — 1.0 during backflush pulse (from env.py)
            bf_n_bursts_completed             — cumulative burst count (diminishing returns)
            bf_source_efficiency              — cleaning efficiency of fluid source [0, 1]

        Writes to state:
            All decomposed fouling fields (cake_*, bridge_*, pore_*, irreversible_*,
            fouling_frac_*, recoverable_*) and all aggregate compatibility fields.
        """
        p = self.params

        # ---- Driving conditions ----
        Q_Lmin  = max(float(state.get("Q_out_Lmin", state.get("flow", 0.0))), 0.0)
        C_fibers = float(state.get("C_fibers", 1.0))

        # ---- Backflush state (written by env.py before this call) ----
        bf_active      = float(state.get("bf_active", 0.0)) > 0.5
        n_bursts       = float(state.get("bf_n_bursts_completed", 0.0))
        bf_source_eff  = float(state.get("bf_source_efficiency", 0.70))

        # Recovery scale: source efficiency × diminishing-returns factor
        # bf_diminishing_factor^max(n_bursts-1, 0):
        #   n_bursts=0 or 1 → factor^0 = 1.0 (full recovery)
        #   n_bursts=2 → factor^1, etc.
        if bf_active:
            diminishing    = p.bf_diminishing_factor ** max(n_bursts - 1.0, 0.0)
            recovery_scale = bf_source_eff * diminishing
        else:
            recovery_scale = 0.0

        # ---- Passive shear scale (always active, proportional to flow) ----
        shear_scale = max(Q_Lmin / max(p.shear_Q_ref, p.eps), 0.0)

        # ---- Update each stage ----
        (cake_s1, bridge_s1, pore_s1, irrev_s1) = self._update_stage(
            cake       = self._state["cake_s1"],
            bridge     = self._state["bridge_s1"],
            pore       = self._state["pore_s1"],
            irreversible = self._state["irreversible_s1"],
            dep_rate   = p.dep_rate_s1,
            cake_w     = p.cake_weight_s1,
            bridge_w   = p.bridge_weight_s1,
            pore_w     = p.pore_weight_s1,
            Q_Lmin     = Q_Lmin,
            C_fibers   = C_fibers,
            shear_scale = shear_scale,
            recovery_scale = recovery_scale,
            dt         = dt,
            p          = p,
        )
        (cake_s2, bridge_s2, pore_s2, irrev_s2) = self._update_stage(
            cake       = self._state["cake_s2"],
            bridge     = self._state["bridge_s2"],
            pore       = self._state["pore_s2"],
            irreversible = self._state["irreversible_s2"],
            dep_rate   = p.dep_rate_s2,
            cake_w     = p.cake_weight_s2,
            bridge_w   = p.bridge_weight_s2,
            pore_w     = p.pore_weight_s2,
            Q_Lmin     = Q_Lmin,
            C_fibers   = C_fibers,
            shear_scale = shear_scale,
            recovery_scale = recovery_scale,
            dt         = dt,
            p          = p,
        )
        (cake_s3, bridge_s3, pore_s3, irrev_s3) = self._update_stage(
            cake       = self._state["cake_s3"],
            bridge     = self._state["bridge_s3"],
            pore       = self._state["pore_s3"],
            irreversible = self._state["irreversible_s3"],
            dep_rate   = p.dep_rate_s3,
            cake_w     = p.cake_weight_s3,
            bridge_w   = p.bridge_weight_s3,
            pore_w     = p.pore_weight_s3,
            Q_Lmin     = Q_Lmin,
            C_fibers   = C_fibers,
            shear_scale = shear_scale,
            recovery_scale = recovery_scale,
            dt         = dt,
            p          = p,
        )

        # ---- Derived per-stage aggregates ----
        ff_s1  = float(np.clip(cake_s1 + bridge_s1 + pore_s1, 0.0, 1.0))
        ff_s2  = float(np.clip(cake_s2 + bridge_s2 + pore_s2, 0.0, 1.0))
        ff_s3  = float(np.clip(cake_s3 + bridge_s3 + pore_s3, 0.0, 1.0))
        rec_s1 = float(max(ff_s1 - irrev_s1, 0.0))
        rec_s2 = float(max(ff_s2 - irrev_s2, 0.0))
        rec_s3 = float(max(ff_s3 - irrev_s3, 0.0))

        # ---- Aggregate compatibility fields (derived, not primary) ----
        n1 = ff_s1
        n2 = ff_s2
        n3 = ff_s3
        mesh_loading_avg = (n1 + n2 + n3) / 3.0
        Mc1 = n1 * p.Mc1_max
        Mc2 = n2 * p.Mc2_max
        Mc3 = n3 * p.Mc3_max

        # ---- Capture efficiency (non-monotonic, YAML-parameterized) ----
        # Peaks at moderate fouling; softens near saturation where flow collapse dominates.
        # baseline + gain × n_avg × (1 − n_avg)^exponent
        capture_eff = (
            p.capture_eff_baseline
            + p.capture_eff_gain
            * mesh_loading_avg
            * (1.0 - mesh_loading_avg) ** p.capture_eff_exponent
        )
        capture_eff = float(np.clip(capture_eff, p.capture_eff_floor, p.capture_eff_ceiling))

        # ---- Write internal state ----
        self._state.update(
            # decomposed (primary)
            cake_s1=cake_s1,     bridge_s1=bridge_s1,     pore_s1=pore_s1,     irreversible_s1=irrev_s1,
            cake_s2=cake_s2,     bridge_s2=bridge_s2,     pore_s2=pore_s2,     irreversible_s2=irrev_s2,
            cake_s3=cake_s3,     bridge_s3=bridge_s3,     pore_s3=pore_s3,     irreversible_s3=irrev_s3,
            # derived per-stage
            fouling_frac_s1=ff_s1,  recoverable_s1=rec_s1,
            fouling_frac_s2=ff_s2,  recoverable_s2=rec_s2,
            fouling_frac_s3=ff_s3,  recoverable_s3=rec_s3,
            # aggregate compatibility
            n1=n1,    n2=n2,    n3=n3,
            Mc1=Mc1,  Mc2=Mc2,  Mc3=Mc3,
            Mc1_max=p.Mc1_max,  Mc2_max=p.Mc2_max,  Mc3_max=p.Mc3_max,
            mesh_loading_avg=float(mesh_loading_avg),
            capture_eff=capture_eff,
        )

        # Push all fields into the shared env state dict
        state.update(self._state)

    def get_state(self) -> Dict[str, float]:
        """Return a shallow copy of the current clogging state."""
        return dict(self._state)

    # ------------------------------------------------------------------
    # Testing utility
    # ------------------------------------------------------------------

    def _force_fouling_for_testing(
        self,
        state: Dict[str, float],
        fouling_frac: float,
    ) -> None:
        """
        FOR VALIDATION USE ONLY.

        Set all three stages to a uniform fouling fraction, distributed
        across components according to their configured stage weights.
        Irreversible is set to zero (clean pre-stress condition).

        This bypasses the normal update path and exists solely to allow
        validation tests to start from a known fouled state without
        running hundreds of warmup steps.
        """
        p = self.params
        ff = float(np.clip(fouling_frac, 0.0, 1.0))

        stage_cfg = [
            (1, p.cake_weight_s1, p.bridge_weight_s1, p.pore_weight_s1, p.Mc1_max),
            (2, p.cake_weight_s2, p.bridge_weight_s2, p.pore_weight_s2, p.Mc2_max),
            (3, p.cake_weight_s3, p.bridge_weight_s3, p.pore_weight_s3, p.Mc3_max),
        ]
        for i, cake_w, bridge_w, pore_w, mc_max in stage_cfg:
            self._state[f"cake_s{i}"]          = ff * cake_w
            self._state[f"bridge_s{i}"]        = ff * bridge_w
            self._state[f"pore_s{i}"]          = ff * pore_w
            self._state[f"irreversible_s{i}"]  = 0.0
            self._state[f"fouling_frac_s{i}"]  = ff
            self._state[f"recoverable_s{i}"]   = ff
            self._state[f"n{i}"]               = ff
            self._state[f"Mc{i}"]              = ff * mc_max

        self._state["mesh_loading_avg"] = ff
        # Recompute capture_eff for the forced state
        capture_eff = (
            p.capture_eff_baseline
            + p.capture_eff_gain * ff * (1.0 - ff) ** p.capture_eff_exponent
        )
        self._state["capture_eff"] = float(np.clip(capture_eff, p.capture_eff_floor, p.capture_eff_ceiling))

        state.update(self._state)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_zero_state(self) -> None:
        p = self.params
        self._state = {
            # decomposed fouling
            "cake_s1":   0.0,  "bridge_s1":   0.0,  "pore_s1":   0.0,  "irreversible_s1": 0.0,
            "cake_s2":   0.0,  "bridge_s2":   0.0,  "pore_s2":   0.0,  "irreversible_s2": 0.0,
            "cake_s3":   0.0,  "bridge_s3":   0.0,  "pore_s3":   0.0,  "irreversible_s3": 0.0,
            # derived per-stage
            "fouling_frac_s1": 0.0,  "recoverable_s1": 0.0,
            "fouling_frac_s2": 0.0,  "recoverable_s2": 0.0,
            "fouling_frac_s3": 0.0,  "recoverable_s3": 0.0,
            # aggregate compatibility
            "Mc1": 0.0,  "Mc1_max": p.Mc1_max,
            "Mc2": 0.0,  "Mc2_max": p.Mc2_max,
            "Mc3": 0.0,  "Mc3_max": p.Mc3_max,
            "n1":  0.0,  "n2":  0.0,  "n3":  0.0,
            "mesh_loading_avg": 0.0,
            "capture_eff":      p.capture_eff_baseline,  # clean state baseline
        }

    def _update_stage(
        self,
        cake: float,
        bridge: float,
        pore: float,
        irreversible: float,
        dep_rate: float,
        cake_w: float,
        bridge_w: float,
        pore_w: float,
        Q_Lmin: float,
        C_fibers: float,
        shear_scale: float,
        recovery_scale: float,
        dt: float,
        p: CloggingParams,
    ) -> Tuple[float, float, float, float]:
        """
        Advance one stage by dt.

        Returns (new_cake, new_bridge, new_pore, new_irreversible).
        All outputs are in [0, 1].
        """
        fouling_frac = float(np.clip(cake + bridge + pore, 0.0, 1.0))
        recoverable  = max(fouling_frac - irreversible, 0.0)

        # ---- Deposition ----
        # Total rate; nonlinear via (fouling_frac + eps)^dep_exponent.
        # With dep_exponent=2 this term is ≈0 at clean state (bistable — see
        # CloggingParams.dep_exponent docstring).  With dep_exponent=1 it produces
        # a constant background rate even at ff=0.
        dep_total = (
            dep_rate * p.dep_base
            * Q_Lmin * C_fibers
            * (fouling_frac + p.eps) ** p.dep_exponent
            * dt
        )
        d_cake   = cake_w   * dep_total
        d_bridge = bridge_w * dep_total
        d_pore   = pore_w   * dep_total

        # ---- Irreversible accumulation ----
        # Onset above stress threshold; grows proportional to deposition rate × stress excess.
        if fouling_frac > p.irreversible_stress_threshold:
            stress_excess = fouling_frac - p.irreversible_stress_threshold
            d_irrev = p.irreversible_rate * stress_excess * dep_total
        else:
            d_irrev = 0.0

        # ---- Passive shear removal (acts on recoverable fraction only) ----
        if fouling_frac > p.eps:
            shear_total  = p.shear_coeff * shear_scale * recoverable * dt
            # Distribute proportionally across components
            d_shear_cake   = shear_total * (cake   / fouling_frac)
            d_shear_bridge = shear_total * (bridge / fouling_frac)
            d_shear_pore   = shear_total * (pore   / fouling_frac)
        else:
            d_shear_cake = d_shear_bridge = d_shear_pore = 0.0

        # ---- Backflush pulse recovery ----
        # Applied per dt, normalized by pulse_duration_s so that the intended
        # per-pulse fraction is distributed across the timesteps within the pulse.
        d_bf_cake = d_bf_bridge = d_bf_pore = 0.0
        if recovery_scale > 0.0 and fouling_frac > p.eps:
            rec_ratio  = recoverable / max(fouling_frac, p.eps)
            time_scale = dt / max(p.bf_pulse_duration_s, p.eps)
            d_bf_cake   = p.bf_cake_recovery   * cake   * rec_ratio * recovery_scale * time_scale
            d_bf_bridge = p.bf_bridge_recovery * bridge * rec_ratio * recovery_scale * time_scale
            d_bf_pore   = p.bf_pore_recovery   * pore   * rec_ratio * recovery_scale * time_scale

        # ---- Integrate, clip to [0, 1] ----
        new_cake   = float(np.clip(cake   + d_cake   - d_shear_cake   - d_bf_cake,   0.0, 1.0))
        new_bridge = float(np.clip(bridge + d_bridge - d_shear_bridge - d_bf_bridge, 0.0, 1.0))
        new_pore   = float(np.clip(pore   + d_pore   - d_shear_pore   - d_bf_pore,   0.0, 1.0))

        # Irreversible bounded by new total fouling (cannot exceed what's there)
        new_ff    = new_cake + new_bridge + new_pore
        new_irrev = float(np.clip(irreversible + d_irrev, 0.0, new_ff))

        return new_cake, new_bridge, new_pore, new_irrev
