# hydrion/physics/particles.py
# ParticleModel v3 — M4: density classification, Stokes settling,
# per-stage size-dependent capture, formal eta_nominal definition.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

try:
    from scipy.stats import lognorm
except ImportError:
    lognorm = None


@dataclass
class ParticleParams:
    """
    Particle transport model parameters.

    Operates in dimensionless units:
        C_in, C_out in [0, 1] (relative concentration)
    """
    C_in_base: float = 0.7
    alpha_clog: float = 0.3        # legacy — not used in M4 main path
    alpha_E: float = 0.4           # electrostatic capture boost gain
    capture_floor: float = 0.3
    capture_ceiling: float = 0.99
    eps: float = 1e-8

    # M4: density classification (must sum to 1.0)
    dense_fraction: float = 0.70   # PET, PA, PVC, biofilm — primary capture target
    neutral_fraction: float = 0.15  # weathered, transitional fragments
    buoyant_fraction: float = 0.15  # PP, PE — pass-through, not captured (scope §E)

    # M4: Stokes settling geometry and fluid properties
    stage_height_m: float = 0.05    # representative stage height [m] — placeholder
    rho_dense_kgm3: float = 1380.0  # PET reference density [kg/m³]
    rho_water_kgm3: float = 1000.0  # water at ~20°C [kg/m³]
    mu_water_Pas: float = 1e-3      # dynamic viscosity [Pa·s]

    # M4: per-stage fouling coupling to capture
    fouling_gain_s1: float = 0.15   # capture gain per unit fouling (Stage 1)
    fouling_gain_s2: float = 0.20   # capture gain per unit fouling (Stage 2)
    fouling_gain_s3: float = 0.10   # capture gain per unit fouling (Stage 3)

    # M4: Stage 3 flow-rate penalty
    s3_flow_penalty_coeff: float = 0.04   # exp(-coeff × max(Q - onset, 0))
    s3_flow_onset_lmin: float = 10.0      # [L/min] onset threshold

    # M4: eta_nominal reference conditions (locked §G)
    eta_ref_d_um: float = 10.0     # [µm] reference particle diameter
    eta_ref_Q_lmin: float = 13.5   # [L/min] reference flow rate (mid-nominal envelope)


# ---------------------------------------------------------------------------
# M4 physics functions — module-level, independently testable
# ---------------------------------------------------------------------------

def stokes_velocity_ms(
    rho_p_kgm3: float,
    d_p_m: float,
    rho_w: float = 1000.0,
    mu: float = 1e-3,
) -> float:
    """
    Stokes settling velocity [m/s].
    Positive → sinking (dense, ρ > 1.0 g/cm³ — assists downward collection).
    Negative → rising  (buoyant, ρ < 1.0 g/cm³ — physically cannot be captured).
    """
    return (rho_p_kgm3 - rho_w) * 9.81 * d_p_m**2 / (18.0 * mu)


def _capture_eff_s1(d_p_um: float, fouling: float, fouling_gain: float) -> float:
    """Stage 1 — 500 µm coarse mesh. Size-power curve, fouling slightly improves capture."""
    base = float(np.clip((d_p_um / 500.0) ** 1.5, 0.0, 0.99))
    return float(np.clip(base * (1.0 + fouling_gain * fouling), 0.0, 0.99))


def _capture_eff_s2(d_p_um: float, fouling: float, fouling_gain: float) -> float:
    """Stage 2 — 100 µm medium mesh. Steeper size curve, moderate fouling coupling."""
    base = float(np.clip((d_p_um / 100.0) ** 1.2, 0.0, 0.98))
    return float(np.clip(base * (1.0 + fouling_gain * fouling), 0.0, 0.98))


def _capture_eff_s3(
    d_p_um: float,
    fouling: float,
    fouling_gain: float,
    Q_lmin: float,
    flow_penalty_coeff: float,
    flow_onset_lmin: float,
) -> float:
    """
    Stage 3 — 5 µm fine pleated cartridge.
    Size-power curve × flow-rate penalty × fouling factor.
    Flow penalty activates above flow_onset_lmin (default 10 L/min).
    At Q = 20 L/min: penalty ≈ exp(-0.4) ≈ 0.67.
    """
    base = float(np.clip((d_p_um / 5.0) ** 0.8, 0.0, 0.97))
    flow_penalty = float(np.exp(-flow_penalty_coeff * max(Q_lmin - flow_onset_lmin, 0.0)))
    return float(np.clip(base * flow_penalty * (1.0 + fouling_gain * fouling), 0.0, 0.97))


def _eta_system(eta_s1: float, eta_s2: float, eta_s3: float) -> float:
    """Compound system efficiency: η = 1 − (1−η₁)(1−η₂)(1−η₃)."""
    return 1.0 - (1.0 - eta_s1) * (1.0 - eta_s2) * (1.0 - eta_s3)


# ---------------------------------------------------------------------------
# PSD / shape helpers (unchanged from v2)
# ---------------------------------------------------------------------------

def _parse_psd_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    psd = raw.get("psd") or {}
    return {
        "enabled": bool(psd.get("enabled", False)),
        "mode": str(psd.get("mode", "hybrid")),
        "parametric": psd.get("parametric") or {},
        "bins": psd.get("bins") or [],
        "bin_edges_um": psd.get("bin_edges_um") or [],
    }


def _parse_shape_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    shape = raw.get("shape") or {}
    return {
        "enabled": bool(shape.get("enabled", False)),
        "fiber_fraction": float(shape.get("fiber_fraction", 1.0)),
    }


def _compute_bin_weights(
    mode: str,
    parametric: Dict[str, Any],
    bins: List[Dict[str, Any]],
    bin_edges_um: List[float],
) -> List[float]:
    if (mode == "bins" or (mode == "hybrid" and bins and all("w_in" in b for b in bins))) and bins:
        weights = [float(b.get("w_in", 0.0)) for b in bins]
        total = sum(weights)
        if total <= 0:
            return [1.0]
        return [w / total for w in weights]

    if mode in ("parametric", "hybrid") and parametric:
        dist = str(parametric.get("distribution", "lognormal")).lower()
        mean_um = float(parametric.get("mean_um", 5.0))
        std_um = float(parametric.get("std_um", 2.0))
        edges = bin_edges_um if bin_edges_um else [0.1, 1.0, 10.0, 100.0]
        if len(edges) < 2:
            return [1.0]

        if dist == "lognormal" and lognorm is not None:
            sigma_ln = np.sqrt(np.log(1.0 + (std_um / max(mean_um, 1e-9)) ** 2))
            mu_ln = np.log(mean_um) - 0.5 * sigma_ln ** 2
            cdf_vals = lognorm.cdf(edges, s=sigma_ln, scale=np.exp(mu_ln))
            weights = np.diff(cdf_vals)
            weights = np.clip(weights, 0.0, 1.0)
            total = float(np.sum(weights))
            if total <= 0:
                weights = np.ones(len(weights)) / len(weights)
            else:
                weights = weights / total
            return [float(w) for w in weights]

        n = len(edges) - 1
        return [1.0 / n] * n

    return [1.0]


# ---------------------------------------------------------------------------
# ParticleModel v3
# ---------------------------------------------------------------------------

class ParticleModel:
    """
    ParticleModel v3 — M4

    Reads from truth_state:
        q_processed_lmin     from HydraulicsModel
        fouling_frac_s1/2/3  from CloggingModel
        E_capture_gain        from ElectrostaticsModel (obs12_v2)

    Writes to truth_state:
        C_in, C_out, particle_capture_eff, C_fibers       (obs12_v2 unchanged)
        C_in_dense, C_in_neutral, C_in_buoyant            (M4 density classification)
        buoyant_fraction                                    (config trace)
        capture_eff_s1, capture_eff_s2, capture_eff_s3    (M4 per-stage)
        capture_boost_settling                             (M4 Stokes contribution)
        eta_system, eta_nominal                            (M4 efficiency)
    """

    def __init__(self, cfg: Any | None = None) -> None:
        p_raw: Dict[str, Any] = {}
        if cfg is not None and hasattr(cfg, "raw"):
            p_raw = getattr(cfg, "raw", {}).get("particles", {}) or {}

        self.params = ParticleParams(
            C_in_base=float(p_raw.get("C_in_base", 0.7)),
            alpha_clog=float(p_raw.get("alpha_clog", 0.3)),
            alpha_E=float(p_raw.get("alpha_E", 0.4)),
            capture_floor=float(p_raw.get("capture_floor", 0.3)),
            capture_ceiling=float(p_raw.get("capture_ceiling", 0.99)),
            eps=float(p_raw.get("eps", 1e-8)),
            # M4
            dense_fraction=float(p_raw.get("dense_fraction", 0.70)),
            neutral_fraction=float(p_raw.get("neutral_fraction", 0.15)),
            buoyant_fraction=float(p_raw.get("buoyant_fraction", 0.15)),
            stage_height_m=float(p_raw.get("stage_height_m", 0.05)),
            rho_dense_kgm3=float(p_raw.get("rho_dense_kgm3", 1380.0)),
            rho_water_kgm3=float(p_raw.get("rho_water_kgm3", 1000.0)),
            mu_water_Pas=float(p_raw.get("mu_water_Pas", 1e-3)),
            fouling_gain_s1=float(p_raw.get("fouling_gain_s1", 0.15)),
            fouling_gain_s2=float(p_raw.get("fouling_gain_s2", 0.20)),
            fouling_gain_s3=float(p_raw.get("fouling_gain_s3", 0.10)),
            s3_flow_penalty_coeff=float(p_raw.get("s3_flow_penalty_coeff", 0.04)),
            s3_flow_onset_lmin=float(p_raw.get("s3_flow_onset_lmin", 10.0)),
            eta_ref_d_um=float(p_raw.get("eta_ref_d_um", 10.0)),
            eta_ref_Q_lmin=float(p_raw.get("eta_ref_Q_lmin", 13.5)),
        )

        self._psd_cfg = _parse_psd_config(p_raw)
        self._shape_cfg = _parse_shape_config(p_raw)
        self._bin_weights: List[float] = []
        if self._psd_cfg["enabled"]:
            self._bin_weights = _compute_bin_weights(
                mode=self._psd_cfg["mode"],
                parametric=self._psd_cfg["parametric"],
                bins=self._psd_cfg["bins"],
                bin_edges_um=self._psd_cfg["bin_edges_um"],
            )

    def reset(self, state: Dict[str, float]) -> None:
        p = self.params
        state["C_in"] = p.C_in_base
        state["C_out"] = p.C_in_base
        state["particle_capture_eff"] = 0.0
        state["C_fibers"] = 1.0

        # M4: density classification
        state["C_in_dense"] = p.C_in_base * p.dense_fraction
        state["C_in_neutral"] = p.C_in_base * p.neutral_fraction
        state["C_in_buoyant"] = p.C_in_base * p.buoyant_fraction
        state["buoyant_fraction"] = p.buoyant_fraction

        # M4: per-stage capture and efficiency
        state["capture_eff_s1"] = 0.0
        state["capture_eff_s2"] = 0.0
        state["capture_eff_s3"] = 0.0
        state["capture_boost_settling"] = 0.0
        state["eta_system"] = 0.0
        state["eta_nominal"] = 0.0

        if self._psd_cfg["enabled"] and self._bin_weights:
            for i in range(len(self._bin_weights)):
                state[f"C_in_bin_{i}"] = 0.0
                state[f"C_out_bin_{i}"] = 0.0
            if len(self._bin_weights) == 3:
                state["C_L"], state["C_M"], state["C_S"] = 0.0, 0.0, 0.0

        if self._shape_cfg["enabled"]:
            state["fiber_fraction"] = self._shape_cfg["fiber_fraction"]

    def update(
        self,
        state: Dict[str, float],
        dt: float,
        clogging_model: Any | None = None,
        electrostatics_model: Any | None = None,
    ) -> None:
        p = self.params

        C_in = float(state.get("C_in", p.C_in_base))
        Q_lmin = float(state.get("q_processed_lmin", p.eta_ref_Q_lmin))

        # Per-stage fouling fractions written by CloggingModel
        ff_s1 = float(state.get("fouling_frac_s1", 0.0))
        ff_s2 = float(state.get("fouling_frac_s2", 0.0))
        ff_s3 = float(state.get("fouling_frac_s3", 0.0))

        # Per-stage capture efficiency at reference particle size (d = eta_ref_d_um)
        d_um = p.eta_ref_d_um
        eta_s1 = _capture_eff_s1(d_um, ff_s1, p.fouling_gain_s1)
        eta_s2 = _capture_eff_s2(d_um, ff_s2, p.fouling_gain_s2)
        eta_s3 = _capture_eff_s3(
            d_um, ff_s3, p.fouling_gain_s3,
            Q_lmin, p.s3_flow_penalty_coeff, p.s3_flow_onset_lmin,
        )
        eta_sys = _eta_system(eta_s1, eta_s2, eta_s3)

        # Electrostatic boost (E_capture_gain ∈ [0, 1] from ElectrostaticsModel v2)
        E_capture_gain = 0.0
        if electrostatics_model is not None:
            E_capture_gain = float(electrostatics_model.get_state().get("E_capture_gain", 0.0))
        else:
            E_capture_gain = float(state.get("E_capture_gain", 0.0))

        # Stokes settling boost — dense particles (ρ > 1.0 g/cm³) settle toward collection wall
        d_m = d_um * 1e-6
        v_s = stokes_velocity_ms(p.rho_dense_kgm3, d_m, p.rho_water_kgm3, p.mu_water_Pas)
        Q_proc_Ls = max(Q_lmin / 60.0, p.eps)
        stage_vol_L = float(state.get("stage_volume_L", 0.25))
        t_res_s = stage_vol_L / Q_proc_Ls
        capture_boost_settling = float(np.clip(
            v_s * t_res_s / max(p.stage_height_m, p.eps), 0.0, 0.05
        ))

        # Dense-phase: stage physics + electrostatics + settling
        capture_eff_dense = float(np.clip(
            eta_sys + p.alpha_E * E_capture_gain + capture_boost_settling,
            p.capture_floor, p.capture_ceiling,
        ))

        # Neutral-phase: stage physics + electrostatics (no settling contribution)
        capture_eff_neutral = float(np.clip(
            eta_sys + p.alpha_E * E_capture_gain,
            p.capture_floor, p.capture_ceiling,
        ))

        # Density-split concentrations and outputs
        C_in_dense = C_in * p.dense_fraction
        C_in_neutral = C_in * p.neutral_fraction
        C_in_buoyant = C_in * p.buoyant_fraction  # scope §E: pass-through

        C_out = (
            C_in_dense   * (1.0 - capture_eff_dense)
            + C_in_neutral * (1.0 - capture_eff_neutral)
            + C_in_buoyant                              # fully passes through
        )

        # η_nominal: deterministic reference (clean filter, d=10µm, Q=13.5 L/min, dense)
        eta_s1_ref = _capture_eff_s1(p.eta_ref_d_um, 0.0, p.fouling_gain_s1)
        eta_s2_ref = _capture_eff_s2(p.eta_ref_d_um, 0.0, p.fouling_gain_s2)
        eta_s3_ref = _capture_eff_s3(
            p.eta_ref_d_um, 0.0, p.fouling_gain_s3,
            p.eta_ref_Q_lmin, p.s3_flow_penalty_coeff, p.s3_flow_onset_lmin,
        )
        eta_nominal = _eta_system(eta_s1_ref, eta_s2_ref, eta_s3_ref)

        # C_fibers for clogging (fiber fraction of C_in)
        if self._psd_cfg["enabled"] and self._shape_cfg["enabled"]:
            C_fibers = self._shape_cfg["fiber_fraction"] * C_in
        else:
            C_fibers = 1.0

        # --- Write truth_state ---
        state["C_in"] = C_in
        state["C_out"] = C_out
        state["particle_capture_eff"] = capture_eff_dense  # obs12_v2 index 5: dense-phase
        state["C_fibers"] = float(np.clip(C_fibers, 0.0, 1.0))

        # M4 density classification
        state["C_in_dense"] = float(C_in_dense)
        state["C_in_neutral"] = float(C_in_neutral)
        state["C_in_buoyant"] = float(C_in_buoyant)
        state["buoyant_fraction"] = float(p.buoyant_fraction)

        # M4 per-stage capture
        state["capture_eff_s1"] = float(eta_s1)
        state["capture_eff_s2"] = float(eta_s2)
        state["capture_eff_s3"] = float(eta_s3)
        state["capture_boost_settling"] = float(capture_boost_settling)

        # M4 system efficiency
        state["eta_system"] = float(eta_sys)
        state["eta_nominal"] = float(eta_nominal)

        # Per-bin concentrations (PSD path, logging/validation only)
        if self._psd_cfg["enabled"] and self._bin_weights:
            for i, w in enumerate(self._bin_weights):
                C_in_i = w * C_in
                C_out_i = C_in_i * (1.0 - capture_eff_dense)
                state[f"C_in_bin_{i}"] = float(C_in_i)
                state[f"C_out_bin_{i}"] = float(C_out_i)
            if len(self._bin_weights) == 3:
                state["C_L"] = state["C_out_bin_2"]
                state["C_M"] = state["C_out_bin_1"]
                state["C_S"] = state["C_out_bin_0"]
