# M4 Particle Realism — Execution Prompt
**Version:** 1.0  
**Date:** 2026-04-10  
**Depends on:** M3 complete (ElectrostaticsModel v2, obs12_v2, E_capture_gain ✓)  
**Scope doc:** `docs/reports/M4_PARTICLE_REALISM_REPORT.md`  
**Constraints:** `docs/context/06_LOCKED_SYSTEM_CONSTRAINTS.md` §E (dense-phase scope)

---

## Execution Contract

This prompt is a complete, self-contained execution contract for Milestone 4.
Every task has exact code. Every verification has exact commands and expected output.
Do not deviate from the implementation unless a task fails at the verification step.
Raise the failure explicitly. Do not work around it silently.

---

## Pre-Flight Verification

Run before any task begins. All must pass.

```bash
cd /c/Users/JSEer/hydrOS
python -m pytest tests/ -q
```
**Expected:** 25 passed (minimum). Any failure — STOP. Fix M3 regression first.

```bash
python -c "
from hydrion.physics.electrostatics import ElectrostaticsModel
import numpy as np
e = ElectrostaticsModel(cfg=None)
s = {'q_processed_lmin': 13.5}
e.reset(s)
e.update(s, dt=0.1, node_cmd=0.8)
st = e.get_state()
assert 'E_field_kVm' in st, 'FAIL: E_field_kVm missing'
assert 'E_capture_gain' in st, 'FAIL: E_capture_gain missing'
assert 'E_norm' not in st, 'FAIL: E_norm still present (obs12_v1 key)'
print('PRE-FLIGHT OK: ElectrostaticsModel v2 confirmed')
"
```
**Expected:** `PRE-FLIGHT OK: ElectrostaticsModel v2 confirmed`

```bash
python -c "
from hydrion.sensors.sensor_fusion import build_obs
import numpy as np
truth = {k: 0.0 for k in ['flow','pressure','clog','E_field_norm','C_out',
         'particle_capture_eff','valve_cmd','pump_cmd','bf_cmd','node_voltage_cmd',
         'sensor_turbidity','sensor_scatter']}
obs = build_obs(truth, {})
assert obs.shape == (12,), f'FAIL: shape {obs.shape}'
print('PRE-FLIGHT OK: obs12_v2 shape confirmed')
"
```
**Expected:** `PRE-FLIGHT OK: obs12_v2 shape confirmed`

---

## Task 1 — Replace `hydrion/physics/particles.py` with M4 implementation

**Read the file first:** `hydrion/physics/particles.py`

Replace the entire file with the following:

```python
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
    eta_ref_Q_lmin: float = 13.5   # [L/min] reference flow rate


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
        q_processed_lmin   from HydraulicsModel
        fouling_frac_s1/2/3  from CloggingModel
        E_capture_gain       from ElectrostaticsModel (obs12_v2)

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
```

**Verify Task 1:**

```bash
cd /c/Users/JSEer/hydrOS
python -c "
from hydrion.physics.particles import (
    ParticleModel, stokes_velocity_ms,
    _capture_eff_s1, _capture_eff_s2, _capture_eff_s3, _eta_system,
)
print('TASK 1 IMPORT: OK')

# Physics gate 1: Stokes — dense sinks, buoyant rises
v_dense = stokes_velocity_ms(1380.0, 10e-6)
v_buoy  = stokes_velocity_ms(910.0,  10e-6)
assert v_dense > 0, f'FAIL: PET v_s={v_dense}'
assert v_buoy  < 0, f'FAIL: PP v_s={v_buoy}'
print(f'GATE 1 PASS: v_dense={v_dense:.2e} m/s (>0), v_buoy={v_buoy:.2e} m/s (<0)')

# Physics gate 2: S1 coarse mesh passes 10 µm particles
eta_s1_clean = _capture_eff_s1(10.0, 0.0, 0.15)
assert eta_s1_clean < 0.05, f'FAIL: eta_s1(10µm)={eta_s1_clean}'
print(f'GATE 2 PASS: eta_s1(10µm, clean)={eta_s1_clean:.4f} (<0.05)')

# Physics gate 3: S3 degrades with flow
eta_s3_low  = _capture_eff_s3(10.0, 0.0, 0.10, 5.0,  0.04, 10.0)
eta_s3_high = _capture_eff_s3(10.0, 0.0, 0.10, 20.0, 0.04, 10.0)
assert eta_s3_low > eta_s3_high, f'FAIL: eta_s3 not flow-dependent'
print(f'GATE 3 PASS: eta_s3(Q=5)={eta_s3_low:.3f} > eta_s3(Q=20)={eta_s3_high:.3f}')

# Physics gate 4: eta_nominal in plausible range
eta_s1_ref = _capture_eff_s1(10.0, 0.0, 0.15)
eta_s2_ref = _capture_eff_s2(10.0, 0.0, 0.20)
eta_s3_ref = _capture_eff_s3(10.0, 0.0, 0.10, 13.5, 0.04, 10.0)
eta_nom = _eta_system(eta_s1_ref, eta_s2_ref, eta_s3_ref)
assert 0.4 <= eta_nom <= 0.98, f'FAIL: eta_nominal={eta_nom}'
print(f'GATE 4 PASS: eta_nominal(ref)={eta_nom:.3f}')
print(f'  eta_s1={eta_s1_ref:.4f}, eta_s2={eta_s2_ref:.4f}, eta_s3={eta_s3_ref:.3f}')
"
```

**Expected output (exact values may vary slightly):**
```
TASK 1 IMPORT: OK
GATE 1 PASS: v_dense=2.07e-05 m/s (>0), v_buoy=-4.42e-06 m/s (<0)
GATE 2 PASS: eta_s1(10µm, clean)=0.0003 (<0.05)
GATE 3 PASS: eta_s3(Q=5)=0.745 > eta_s3(Q=20)=0.499
GATE 4 PASS: eta_nominal(ref)=0.752
  eta_s1=0.0003, eta_s2=0.0275, eta_s3=0.745
```

---

## Task 2 — Update `configs/default.yaml` (particles section)

**Read the file first:** `configs/default.yaml`

Find the existing `particles:` block:
```yaml
particles:
  C_in_base: 0.7
  psd:
    enabled: false
    mode: hybrid
  shape:
    enabled: false
    fiber_fraction: 1.0
```

Replace it with:
```yaml
# ---------------------------------------------------------------------------
# Particle model — M4: density classification, Stokes settling, per-stage capture
# ---------------------------------------------------------------------------
particles:
  C_in_base: 0.7

  # Density classification (locked §E — dense-phase scope, Option A)
  dense_fraction:   0.70    # PET, PA, PVC, biofilm-coated — primary capture target
  neutral_fraction: 0.15    # weathered, transitional fragments
  buoyant_fraction: 0.15    # PP, PE — pass-through, tracked not captured

  # Stokes settling — dense particle geometry and fluid properties
  stage_height_m:   0.05    # [m] representative stage height — placeholder
  rho_dense_kgm3:   1380.0  # PET reference density [kg/m³]
  rho_water_kgm3:   1000.0  # water at ~20°C [kg/m³]
  mu_water_Pas:     1.0e-3  # dynamic viscosity [Pa·s]

  # Per-stage fouling coupling to capture efficiency
  fouling_gain_s1:  0.15    # Stage 1 (500 µm coarse mesh)
  fouling_gain_s2:  0.20    # Stage 2 (100 µm medium mesh)
  fouling_gain_s3:  0.10    # Stage 3 (5 µm fine pleated)

  # Stage 3 flow-rate penalty (exp(-coeff × max(Q - onset, 0)))
  s3_flow_penalty_coeff: 0.04   # penalty rate coefficient
  s3_flow_onset_lmin:    10.0   # [L/min] penalty onset threshold

  # eta_nominal reference conditions (locked §G — do not change without versioning)
  eta_ref_d_um:     10.0    # [µm] reference particle diameter
  eta_ref_Q_lmin:   13.5    # [L/min] reference flow rate (mid-nominal envelope)

  psd:
    enabled: false
    mode: hybrid
  shape:
    enabled: false
    fiber_fraction: 1.0
```

**Verify Task 2:**

```bash
cd /c/Users/JSEer/hydrOS
python -c "
import yaml
with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
p = cfg['particles']
assert 'dense_fraction' in p,   'FAIL: dense_fraction missing'
assert 'eta_ref_d_um'   in p,   'FAIL: eta_ref_d_um missing'
assert 'fouling_gain_s3' in p,  'FAIL: fouling_gain_s3 missing'
assert abs(p['dense_fraction'] + p['neutral_fraction'] + p['buoyant_fraction'] - 1.0) < 1e-9, \
    f'FAIL: fractions do not sum to 1.0: {p[\"dense_fraction\"]+p[\"neutral_fraction\"]+p[\"buoyant_fraction\"]}'
print(f'TASK 2 OK: fractions sum={p[\"dense_fraction\"]+p[\"neutral_fraction\"]+p[\"buoyant_fraction\"]:.3f}')
print(f'  eta_ref_d_um={p[\"eta_ref_d_um\"]}, eta_ref_Q_lmin={p[\"eta_ref_Q_lmin\"]}')
"
```
**Expected:** `TASK 2 OK: fractions sum=1.000`

---

## Task 3 — Lock Section G in `docs/context/06_LOCKED_SYSTEM_CONSTRAINTS.md`

**Read the file first:** `docs/context/06_LOCKED_SYSTEM_CONSTRAINTS.md`

Append the following after the `# F. Realism Prioritization` section, before `# Implementation Doctrine`:

```markdown
---

# G. System Efficiency Definition

**Locked 2026-04-10**

```
η_nominal = η_system(d = 10 µm, Q = 13.5 L/min, clean filter, dense-phase particles)
```

This is the canonical reference efficiency for:
- Console display: always show as `"η = XX%  @ 10µm / 13.5 L/min"`
- Engineering claims: any efficiency percentage must reference this definition
- RL reward shaping (M6): `eta_nominal` in truth_state is the reward signal source
- Bench validation targets: physical measurement at these conditions = calibration target

## Display Requirement

**Never display a bare efficiency percentage.** The format is mandatory:
```
η = 75%  @ 10µm / 13.5 L/min
```

"99%" without qualification is not a valid HydrOS output.

## Implementation Reference

- Code: `eta_nominal` key in `truth_state`, computed in `hydrion/physics/particles.py`
- Config: `particles.eta_ref_d_um = 10.0`, `particles.eta_ref_Q_lmin = 13.5`
- Conditions: clean filter (`fouling_frac_s* = 0.0`), dense-phase, no electrostatics

---

```

**Verify Task 3:**

```bash
cd /c/Users/JSEer/hydrOS
python -c "
with open('docs/context/06_LOCKED_SYSTEM_CONSTRAINTS.md') as f:
    content = f.read()
assert '# G. System Efficiency Definition' in content, 'FAIL: Section G missing'
assert 'eta_nominal' in content, 'FAIL: eta_nominal not in §G'
assert '10 µm' in content, 'FAIL: reference size missing'
assert '13.5 L/min' in content, 'FAIL: reference flow missing'
print('TASK 3 OK: Section G locked in constraints doc')
"
```
**Expected:** `TASK 3 OK: Section G locked in constraints doc`

---

## Task 4 — Rewrite `tests/test_particles.py` with M4 tests

**Read the file first:** `tests/test_particles.py`

Replace the entire file with:

```python
# tests/test_particles.py
# M4 — density classification, Stokes settling, per-stage capture, eta_nominal
import numpy as np

from hydrion.physics.particles import (
    ParticleModel,
    _compute_bin_weights,
    stokes_velocity_ms,
    _capture_eff_s1,
    _capture_eff_s2,
    _capture_eff_s3,
    _eta_system,
)


# ---------------------------------------------------------------------------
# Regression: existing API contract preserved
# ---------------------------------------------------------------------------

def test_psd_disabled_regression():
    """PSD disabled: C_in, C_out, particle_capture_eff, C_fibers all present."""
    part = ParticleModel(cfg=None)
    state = {"mesh_loading_avg": 0.2, "capture_eff": 0.8}
    part.reset(state)
    part.update(state, dt=0.1, clogging_model=None, electrostatics_model=None)

    assert "C_in" in state
    assert "C_out" in state
    assert "particle_capture_eff" in state
    assert "C_fibers" in state
    assert "C_in_bin_0" not in state
    assert "C_L" not in state

    assert state["C_out"] <= state["C_in"] + 1e-9
    assert 0.0 <= state["particle_capture_eff"] <= 1.0 + 1e-9


def test_psd_enabled_bins_sum_to_one():
    """PSD enabled: bin weights sum to 1."""
    weights = _compute_bin_weights(
        mode="bins",
        parametric={},
        bins=[
            {"d_min_um": 0.1,  "d_max_um": 1.0,   "w_in": 0.2},
            {"d_min_um": 1.0,  "d_max_um": 10.0,  "w_in": 0.5},
            {"d_min_um": 10.0, "d_max_um": 100.0, "w_in": 0.3},
        ],
        bin_edges_um=[],
    )
    assert len(weights) == 3
    assert abs(sum(weights) - 1.0) < 1e-9

    weights2 = _compute_bin_weights(
        mode="parametric",
        parametric={"distribution": "lognormal", "mean_um": 5.0, "std_um": 2.0},
        bins=[],
        bin_edges_um=[0.1, 1.0, 10.0, 100.0],
    )
    assert len(weights2) == 3
    assert abs(sum(weights2) - 1.0) < 1e-9


def test_psd_enabled_mass_balance():
    """PSD enabled: per-bin C_out <= C_in."""
    cfg_raw = {
        "particles": {
            "C_in_base": 0.7,
            "psd": {"enabled": True, "mode": "bins", "bins": [
                {"d_min_um": 0.1,  "d_max_um": 1.0,   "w_in": 0.33},
                {"d_min_um": 1.0,  "d_max_um": 10.0,  "w_in": 0.34},
                {"d_min_um": 10.0, "d_max_um": 100.0, "w_in": 0.33},
            ]},
            "shape": {"enabled": True, "fiber_fraction": 0.5},
        }
    }
    from hydrion.config import HydrionConfig
    cfg = HydrionConfig(raw=cfg_raw)
    part = ParticleModel(cfg=cfg)
    state = {"mesh_loading_avg": 0.3, "capture_eff": 0.8}
    part.reset(state)
    part.update(state, dt=0.1, clogging_model=None, electrostatics_model=None)

    assert state["C_out"] <= state["C_in"] + 1e-9
    assert 0.0 <= state["particle_capture_eff"] <= 1.0 + 1e-9
    assert "C_fibers" in state
    for i in range(3):
        assert state[f"C_out_bin_{i}"] <= state[f"C_in_bin_{i}"] + 1e-9


# ---------------------------------------------------------------------------
# M4 Gate 1: Stokes settling physics
# ---------------------------------------------------------------------------

def test_stokes_dense_sinks():
    """Dense particles (PET, ρ=1380) have positive settling velocity."""
    v = stokes_velocity_ms(1380.0, 10e-6)
    assert v > 0, f"PET must sink: v_s={v}"


def test_stokes_buoyant_rises():
    """Buoyant particles (PP, ρ=910) have negative settling velocity."""
    v = stokes_velocity_ms(910.0, 10e-6)
    assert v < 0, f"PP must rise: v_s={v}"


def test_stokes_neutral_near_zero():
    """Neutral particles (ρ=1000) have near-zero settling velocity."""
    v = stokes_velocity_ms(1000.0, 10e-6)
    assert abs(v) < 1e-12, f"Neutral must have ~0 settling: v_s={v}"


# ---------------------------------------------------------------------------
# M4 Gate 2: Per-stage capture physics
# ---------------------------------------------------------------------------

def test_s1_passes_fine_particles():
    """Stage 1 (500 µm mesh) captures < 5% of 10 µm particles at clean state."""
    eta = _capture_eff_s1(10.0, 0.0, 0.15)
    assert eta < 0.05, f"S1 should pass 10µm particles: eta_s1={eta}"


def test_s2_moderate_capture():
    """Stage 2 (100 µm mesh) captures < 20% of 10 µm particles at clean state."""
    eta = _capture_eff_s2(10.0, 0.0, 0.20)
    assert eta < 0.20, f"S2 at 10µm: {eta}"


def test_s3_primary_capture():
    """Stage 3 (5 µm mesh) captures > 50% of 10 µm at low flow, clean state."""
    eta = _capture_eff_s3(10.0, 0.0, 0.10, 5.0, 0.04, 10.0)
    assert eta > 0.50, f"S3 at Q=5, clean, 10µm: {eta}"


def test_s3_flow_degradation():
    """Stage 3 capture efficiency decreases at Q=20 vs Q=5."""
    eta_low  = _capture_eff_s3(10.0, 0.0, 0.10, 5.0,  0.04, 10.0)
    eta_high = _capture_eff_s3(10.0, 0.0, 0.10, 20.0, 0.04, 10.0)
    assert eta_low > eta_high, f"Flow degradation absent: Q5={eta_low}, Q20={eta_high}"


def test_s3_fouling_improves_capture():
    """Fouling slightly improves S3 capture (pore restriction effect)."""
    eta_clean = _capture_eff_s3(10.0, 0.0,  0.10, 13.5, 0.04, 10.0)
    eta_fouled = _capture_eff_s3(10.0, 0.5, 0.10, 13.5, 0.04, 10.0)
    assert eta_fouled >= eta_clean, f"Fouling should improve S3 capture: clean={eta_clean}, fouled={eta_fouled}"


def test_eta_system_compound():
    """Compound efficiency exceeds single-stage maximum."""
    eta_s1 = _capture_eff_s1(10.0, 0.0, 0.15)
    eta_s2 = _capture_eff_s2(10.0, 0.0, 0.20)
    eta_s3 = _capture_eff_s3(10.0, 0.0, 0.10, 13.5, 0.04, 10.0)
    eta_sys = _eta_system(eta_s1, eta_s2, eta_s3)
    assert eta_sys > max(eta_s1, eta_s2, eta_s3), \
        f"Compound must exceed single-stage: sys={eta_sys}, max={max(eta_s1, eta_s2, eta_s3)}"


# ---------------------------------------------------------------------------
# M4 Gate 3: Density classification in full update()
# ---------------------------------------------------------------------------

def test_density_fractions_sum_to_one():
    """C_in_dense + C_in_neutral + C_in_buoyant = C_in."""
    part = ParticleModel(cfg=None)
    state: dict = {}
    part.reset(state)
    part.update(state, dt=0.1)
    total = state["C_in_dense"] + state["C_in_neutral"] + state["C_in_buoyant"]
    assert abs(total - state["C_in"]) < 1e-9, f"Density fractions do not sum to C_in: {total} vs {state['C_in']}"


def test_buoyant_pass_through():
    """C_in_buoyant exits system uncaptured."""
    part = ParticleModel(cfg=None)
    state: dict = {"C_in": 0.7, "q_processed_lmin": 13.5}
    part.reset(state)
    part.update(state, dt=0.1)
    # C_out must include at least C_in_buoyant unmodified
    assert state["C_out"] >= state["C_in_buoyant"] - 1e-9, \
        f"Buoyant pass-through missing from C_out: C_out={state['C_out']}, C_in_buoyant={state['C_in_buoyant']}"


def test_m4_truth_state_keys_present():
    """All M4 truth_state keys written after update."""
    part = ParticleModel(cfg=None)
    state: dict = {}
    part.reset(state)
    part.update(state, dt=0.1)

    required = [
        "C_in_dense", "C_in_neutral", "C_in_buoyant", "buoyant_fraction",
        "capture_eff_s1", "capture_eff_s2", "capture_eff_s3",
        "capture_boost_settling", "eta_system", "eta_nominal",
    ]
    for key in required:
        assert key in state, f"Missing M4 truth_state key: {key}"


# ---------------------------------------------------------------------------
# M4 Gate 4: eta_nominal determinism and physical range
# ---------------------------------------------------------------------------

def test_eta_nominal_deterministic():
    """eta_nominal is identical across two separate update() calls."""
    part = ParticleModel(cfg=None)
    state1: dict = {}
    part.reset(state1)
    part.update(state1, dt=0.1)

    state2: dict = {}
    part.reset(state2)
    part.update(state2, dt=0.1)

    assert state1["eta_nominal"] == state2["eta_nominal"], \
        f"eta_nominal not deterministic: {state1['eta_nominal']} vs {state2['eta_nominal']}"


def test_eta_nominal_range():
    """eta_nominal is in [0.4, 0.98] — physically plausible for 10µm at 13.5 L/min."""
    part = ParticleModel(cfg=None)
    state: dict = {}
    part.reset(state)
    part.update(state, dt=0.1)
    assert 0.4 <= state["eta_nominal"] <= 0.98, \
        f"eta_nominal out of expected range: {state['eta_nominal']}"


def test_eta_nominal_independent_of_fouling():
    """eta_nominal uses clean-filter reference conditions — fouling does not affect it."""
    part = ParticleModel(cfg=None)
    state_clean: dict = {"fouling_frac_s1": 0.0, "fouling_frac_s2": 0.0, "fouling_frac_s3": 0.0}
    part.reset(state_clean)
    part.update(state_clean, dt=0.1)

    state_fouled: dict = {"fouling_frac_s1": 0.5, "fouling_frac_s2": 0.5, "fouling_frac_s3": 0.5}
    part.reset(state_fouled)
    part.update(state_fouled, dt=0.1)

    assert abs(state_clean["eta_nominal"] - state_fouled["eta_nominal"]) < 1e-9, \
        f"eta_nominal should not change with fouling: clean={state_clean['eta_nominal']}, fouled={state_fouled['eta_nominal']}"


# ---------------------------------------------------------------------------
# M4 Gate 5: mass balance
# ---------------------------------------------------------------------------

def test_mass_balance_total():
    """C_out <= C_in at all operating conditions."""
    part = ParticleModel(cfg=None)
    for Q in [5.0, 13.5, 20.0]:
        for ff in [0.0, 0.3, 0.7]:
            state: dict = {
                "q_processed_lmin": Q,
                "fouling_frac_s1": ff,
                "fouling_frac_s2": ff,
                "fouling_frac_s3": ff,
            }
            part.reset(state)
            part.update(state, dt=0.1)
            assert state["C_out"] <= state["C_in"] + 1e-9, \
                f"Mass balance violated at Q={Q}, ff={ff}: C_out={state['C_out']:.4f} > C_in={state['C_in']:.4f}"


if __name__ == "__main__":
    test_m4_truth_state_keys_present()
    test_eta_nominal_range()
    print("M4 particle tests: OK")
```

**Verify Task 4:**

```bash
cd /c/Users/JSEer/hydrOS
python -m pytest tests/test_particles.py -v
```

**Expected:** All tests pass. Minimum 18 test functions in this file. Zero failures.

---

## Task 5 — Full test suite, documentation updates, commit

### Step 5.1 — Full regression suite

```bash
cd /c/Users/JSEer/hydrOS
python -m pytest tests/ -v
```

**Expected:** All tests pass (≥ 32 total — 25 pre-M4 + 7+ new M4 tests). Zero failures. If any pre-M4 test fails, identify root cause before proceeding.

### Step 5.2 — Inline physics verification

```bash
cd /c/Users/JSEer/hydrOS
python -c "
from hydrion.physics.particles import ParticleModel

part = ParticleModel(cfg=None)
state = {
    'q_processed_lmin': 13.5,
    'fouling_frac_s1': 0.0,
    'fouling_frac_s2': 0.0,
    'fouling_frac_s3': 0.0,
}
part.reset(state)
part.update(state, dt=0.1)

print('--- M4 ParticleModel v3 Inline Check ---')
print(f'  eta_nominal  = {state[\"eta_nominal\"]:.4f}  (ref: d=10µm, Q=13.5, clean, dense)')
print(f'  eta_system   = {state[\"eta_system\"]:.4f}  (live: current fouling + flow)')
print(f'  eta_s1       = {state[\"capture_eff_s1\"]:.4f}')
print(f'  eta_s2       = {state[\"capture_eff_s2\"]:.4f}')
print(f'  eta_s3       = {state[\"capture_eff_s3\"]:.4f}')
print(f'  C_in         = {state[\"C_in\"]:.3f}')
print(f'  C_out        = {state[\"C_out\"]:.3f}')
print(f'  C_in_dense   = {state[\"C_in_dense\"]:.3f}  ({state[\"C_in_dense\"]/state[\"C_in\"]*100:.0f}% of C_in)')
print(f'  C_in_buoyant = {state[\"C_in_buoyant\"]:.3f}  (pass-through)')
print(f'  settling boost = {state[\"capture_boost_settling\"]:.5f}')
"
```

**Expected (approximate — exact values will vary):**
```
--- M4 ParticleModel v3 Inline Check ---
  eta_nominal  = 0.752x  (ref: d=10µm, Q=13.5, clean, dense)
  eta_system   = 0.752x  (live: current fouling + flow)
  eta_s1       = 0.000x
  eta_s2       = 0.027x
  eta_s3       = 0.74xx
  C_in         = 0.700
  C_out        = 0.xxx
  C_in_dense   = 0.490  (70% of C_in)
  C_in_buoyant = 0.105  (pass-through)
  settling boost = 0.0000x
```

### Step 5.3 — obs12_v2 schema unchanged

```bash
cd /c/Users/JSEer/hydrOS
python -c "
from hydrion.sensors.sensor_fusion import build_obs
import numpy as np
truth = {
    'flow': 0.8, 'pressure': 0.4, 'clog': 0.2, 'E_field_norm': 0.5,
    'C_out': 0.1, 'particle_capture_eff': 0.75,
    'valve_cmd': 0.7, 'pump_cmd': 0.9, 'bf_cmd': 0.0, 'node_voltage_cmd': 0.3,
}
sensor = {'sensor_turbidity': 0.15, 'sensor_scatter': 0.05}
obs = build_obs(truth, sensor)
assert obs.shape == (12,), f'FAIL: shape {obs.shape}'
assert 0.0 <= obs[3] <= 1.0, f'FAIL: obs[3]={obs[3]} out of range (E_field_norm)'
assert 0.0 <= obs[5] <= 1.0, f'FAIL: obs[5]={obs[5]} out of range (particle_capture_eff)'
print(f'SCHEMA OK: obs shape={obs.shape}, obs[3]=E_field_norm={obs[3]:.3f}, obs[5]=particle_capture_eff={obs[5]:.3f}')
"
```
**Expected:** `SCHEMA OK: obs shape=(12,), obs[3]=E_field_norm=0.500, obs[5]=particle_capture_eff=0.750`

### Step 5.4 — Update `docs/context/04_CURRENT_ENGINE_STATUS.md`

**Read the file first.** Locate `# 8. Particle Module` and replace the **entire section** with:

```markdown
# 8. Particle Module

**[UPDATED: Milestone 4 — 2026-04-10]**

## Implementation


hydrion/physics/particles.py  (ParticleModel v3)


### Current behavior (M4)

- Density classification: `C_in` split into `C_in_dense` (70%), `C_in_neutral` (15%), `C_in_buoyant` (15%)
- Buoyant fraction (PP, PE) exits as full pass-through — scope constraint §E
- Stokes settling: `v_s = (ρ_p − ρ_w) × g × d² / (18μ)` — positive for dense (assists collection), capped `capture_boost_settling ≤ 0.05`
- Per-stage size-dependent capture:
  - `η_s1(d) = clip((d/500)^1.5, 0, 0.99)` — coarse mesh, passes fine particles
  - `η_s2(d) = clip((d/100)^1.2, 0, 0.98)` — medium mesh
  - `η_s3(d, Q) = clip((d/5)^0.8 × exp(−0.04 × max(Q−10, 0)), 0, 0.97)` — fine pleated, flow-rate dependent
- Compound system efficiency: `η_system = 1 − (1−η_s1)(1−η_s2)(1−η_s3)`
- `η_nominal` locked at reference conditions: d=10µm, Q=13.5 L/min, clean filter, dense-phase
- Electrostatic boost: `E_capture_gain` (from ElectrostaticsModel v2) added to dense and neutral paths
- `particle_capture_eff` at obs index 5: now represents dense-phase compound capture

### Outputs

- `C_in`, `C_out`, `particle_capture_eff`, `C_fibers` (obs12_v2 unchanged)
- `C_in_dense`, `C_in_neutral`, `C_in_buoyant`, `buoyant_fraction`
- `capture_eff_s1`, `capture_eff_s2`, `capture_eff_s3`, `capture_boost_settling`
- `eta_system`, `eta_nominal`

### Strengths

- Density classification physically grounded (§E locked constraint)
- Stage-specific capture tied to pore size — Stage 3 dominance is now physically derived
- `η_nominal` creates a single, unambiguous efficiency anchor for all future claims
- Flow-rate degradation at S3 is the first dynamic efficiency coupling in HydrOS

### Limitations

- Bulk d=10µm used for live efficiency — full PSD integration deferred to M4.5
- `stage_height_m`, `rho_dense_kgm3`, `rho_water_kgm3` are placeholders — bench geometry required
- Dense vs neutral capture distinction is settling only — shape/charge differences deferred
- Neutral fraction at same mesh efficiency as dense (conservative — could differ at fine scale)
```

### Step 5.5 — Commit

```bash
cd /c/Users/JSEer/hydrOS
git add hydrion/physics/particles.py
git add configs/default.yaml
git add docs/context/06_LOCKED_SYSTEM_CONSTRAINTS.md
git add docs/context/04_CURRENT_ENGINE_STATUS.md
git add tests/test_particles.py
git commit -m "$(cat <<'EOF'
feat(m4): particle realism — density classification, Stokes settling, per-stage capture, eta_nominal

- ParticleModel v3: splits C_in into dense/neutral/buoyant fractions (70/15/15%)
- Buoyant fraction (PP, PE) exits as pass-through per locked §E constraint
- Stokes settling boost for dense particles; capped at 0.05 contribution
- Per-stage size-dependent capture: S1(500µm), S2(100µm), S3(5µm flow-dependent)
- S3 flow penalty: exp(-0.04 × max(Q-10, 0)) — physically correct residence-time effect
- Compound eta_system = 1-(1-s1)(1-s2)(1-s3); eta_nominal locked at d=10µm/Q=13.5/clean
- Section §G locked in 06_LOCKED_SYSTEM_CONSTRAINTS.md
- All M4 parameters YAML-exposed; obs12_v2 schema unchanged

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Exit Checklist

All items must be confirmed before M4 is declared complete.

- [ ] `C_in_dense`, `C_in_neutral`, `C_in_buoyant` tracked separately in truth_state
- [ ] `C_in_buoyant` exits as full pass-through — no capture applied
- [ ] `stokes_velocity_ms(1380, 10e-6) > 0` (dense sinks)
- [ ] `stokes_velocity_ms(910, 10e-6) < 0` (buoyant rises)
- [ ] `capture_eff_s1(10µm, clean) < 0.05` (coarse mesh passes fine particles)
- [ ] `capture_eff_s3(Q=5) > capture_eff_s3(Q=20)` (flow degradation confirmed)
- [ ] `eta_system = 1 − (1−η_s1)(1−η_s2)(1−η_s3)` implemented
- [ ] `eta_nominal` computed at clean/d=10µm/Q=13.5, stored in truth_state
- [ ] `eta_nominal` deterministic across episodes
- [ ] Section §G locked in `06_LOCKED_SYSTEM_CONSTRAINTS.md`
- [ ] `configs/default.yaml` updated with all M4 particle parameters
- [ ] `04_CURRENT_ENGINE_STATUS.md` §8 updated
- [ ] obs12_v2 schema unchanged: shape (12,), index 5 = `particle_capture_eff`
- [ ] Full test suite passing — minimum 32 tests, zero failures
- [ ] Commit on `HydrOS-x-Claude-Code` branch

---

## Physics Summary

At reference conditions (d=10µm, Q=13.5 L/min, clean filter):

| Stage | Pore size | η (approx) | Role |
|-------|-----------|------------|------|
| S1    | 500 µm    | ~0.03%     | passes fine particles |
| S2    | 100 µm    | ~2.8%      | minor contribution |
| S3    | 5 µm      | ~74.5%     | primary capture stage |
| **System** | — | **~75%** | **η_nominal** |

S3 at Q=20 L/min: ~50% (flow penalty active). S3 at Q=5 L/min: ~74.5% (no penalty).
Electrostatics (+E_capture_gain × 0.4) adds up to ~+40% boost when voltage is at max.
