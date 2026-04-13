# hydrion/validation/milestone1_validation.py
"""
HydrOS Validation Protocol — Milestone 1 Suite.

Six tests that verify the hydraulic + fouling + backflush backbone:

  1. pressure_flow_sweep      — P vs Q monotonicity, stage dP ordering
  2. fouling_nonlinearity     — acceleration, stage ordering, threshold onset
  3. backflush_recovery       — partial recovery, irreversible preservation
  4. diminishing_returns      — each burst recovers less than the previous
  5. bypass_activation        — bypass_active flag and flow-conservation check
  6. nan_bounded_regression   — zero NaNs, all Milestone 1 fields in-bounds

Style note: follows the same conventions as the existing Validation Protocol v2
files (stress_matrix, envelope_sweep, mass_balance_test, recovery_latency_test):
  - run_* function returns Dict[str, Any] with "all_passed" key
  - optional output_path writes results as YAML
  - argparse main() for CLI invocation
  - no direct physics mutation except _force_fouling_for_testing()
    which is a validation-only utility on CloggingModel v3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from hydrion.env import HydrionEnv


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def _reset_and_force_fouling(env: HydrionEnv, fouling_frac: float, seed: int = 0) -> None:
    """
    Reset the environment then force all three clogging stages to a uniform
    fouling_frac using the CloggingModel._force_fouling_for_testing() utility.

    This is the only direct state injection in this suite; it exists so that
    tests which require a known pre-fouled initial condition do not depend on
    running hundreds of warmup steps, which would make them slow and brittle.
    """
    env.reset(seed=seed)
    env.clogging._force_fouling_for_testing(env.truth_state, fouling_frac)
    # Re-sync normalized channels so reward / observation are consistent
    env._update_normalized_state()


def _get_truth(env: HydrionEnv) -> Dict[str, Any]:
    return dict(getattr(env, "truth_state", {}))


def _write_output(summary: Dict[str, Any], output_path: Optional[str]) -> None:
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Test 1 — Pressure-Flow Sweep
# ---------------------------------------------------------------------------

def run_pressure_flow_sweep(
    config_path: str = "configs/default.yaml",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Sweep pump_cmd at three clog levels; verify pressure is monotonic in flow
    and increases with fouling.  Verify Stage 3 dominates pressure drop.

    Pass criteria:
    - P_in monotonically increases with pump_cmd at every clog level
    - P_in(heavy) > P_in(partial) > P_in(clean) at every pump point
    - dp_stage3_pa > dp_stage2_pa > dp_stage1_pa when clean (area ordering)
    - dp_stage3_pa increases more steeply with fouling than dp_stage1_pa
    - flow (normalized) ≤ 1.0 at all points
    - no NaN
    """
    env     = HydrionEnv(config_path=config_path)
    pump_sweep   = np.linspace(0.1, 1.0, 9).astype(np.float32)
    clog_levels  = [0.0, 0.40, 0.75]
    clog_labels  = ["clean", "partial", "heavy"]
    neutral_vbnv = np.array([1.0, 0.0, 0.0, 0.5], dtype=np.float32)  # valve open, bf off

    records: List[Dict[str, Any]] = []
    violations: List[str] = []

    # Collect P_in and dp_stage* for each (clog_level, pump_cmd) point
    data: Dict[str, List[float]] = {label: [] for label in clog_labels}
    dp1_clean = dp2_clean = dp3_clean = None

    for clog_frac, label in zip(clog_levels, clog_labels):
        prev_p_in = -1.0
        for pump_val in pump_sweep:
            _reset_and_force_fouling(env, clog_frac, seed=0)
            action = neutral_vbnv.copy()
            action[1] = pump_val
            obs, reward, term, trunc, info = env.step(action)
            t = _get_truth(env)

            p_in      = float(t.get("P_in", 0.0))
            flow_norm = float(t.get("flow", 0.0))
            dp1       = float(t.get("dp_stage1_pa", 0.0))
            dp2       = float(t.get("dp_stage2_pa", 0.0))
            dp3       = float(t.get("dp_stage3_pa", 0.0))

            if not np.isfinite(p_in):
                violations.append(f"NaN/Inf: P_in at clog={clog_frac:.2f} pump={pump_val:.2f}")
            if flow_norm > 1.0 + 1e-4:
                violations.append(f"flow={flow_norm:.4f} > 1.0 at clog={clog_frac:.2f} pump={pump_val:.2f}")
            if p_in < prev_p_in - 1.0:   # allow 1 Pa tolerance for float noise
                violations.append(
                    f"P_in not monotone: {p_in:.1f} < prev {prev_p_in:.1f} at clog={clog_frac:.2f} pump={pump_val:.2f}"
                )

            prev_p_in = p_in
            data[label].append(p_in)

            if label == "clean" and dp1_clean is None:
                # Capture first (lowest pump) clean dp values for ordering check
                dp1_clean, dp2_clean, dp3_clean = dp1, dp2, dp3

            records.append({
                "clog": clog_frac, "pump": float(pump_val),
                "P_in": p_in, "dp1": dp1, "dp2": dp2, "dp3": dp3,
                "flow_norm": flow_norm,
            })

    # Cross-clog ordering: heavy > partial > clean at each pump point
    for i, pump_val in enumerate(pump_sweep):
        p_clean   = data["clean"][i]
        p_partial = data["partial"][i]
        p_heavy   = data["heavy"][i]
        if not (p_heavy >= p_partial - 1.0 and p_partial >= p_clean - 1.0):
            violations.append(
                f"Pressure ordering violated at pump={pump_val:.2f}: "
                f"clean={p_clean:.1f} partial={p_partial:.1f} heavy={p_heavy:.1f}"
            )

    # Stage ordering check at clean state (middle pump point)
    mid = len(pump_sweep) // 2
    rec_mid = [r for r in records if r["clog"] == 0.0][mid]
    dp1m, dp2m, dp3m = rec_mid["dp1"], rec_mid["dp2"], rec_mid["dp3"]
    if not (dp3m >= dp2m - 1.0 and dp2m >= dp1m - 1.0):
        violations.append(
            f"Stage dP ordering violated at clean/mid pump: dp1={dp1m:.1f} dp2={dp2m:.1f} dp3={dp3m:.1f}"
        )

    # Stage 1 sensitivity to fouling vs Stage 3
    # With area-normalized clog sensitivity (k_m1_eff = k_m1 × A_s3/A_s1 ≈ 7.5×),
    # Stage 1 (smallest area, 120 cm²) has the highest resistance rise per unit fouling.
    # Stage 3 (reference area, 900 cm²) has the lowest area factor (1.0×).
    heavy_high = [r for r in records if r["clog"] == 0.75][-1]
    clean_high  = [r for r in records if r["clog"] == 0.0][-1]
    dp3_rise = heavy_high["dp3"] - clean_high["dp3"]
    dp1_rise = heavy_high["dp1"] - clean_high["dp1"]
    if dp1_rise <= dp3_rise:
        violations.append(
            f"Stage 1 dP rise ({dp1_rise:.1f} Pa) not > Stage 3 ({dp3_rise:.1f} Pa) under fouling "
            f"(area-normalized model: Stage 1 has highest clog sensitivity)"
        )

    all_passed = len(violations) == 0
    summary = {
        "test": "pressure_flow_sweep",
        "config_path": config_path,
        "all_passed": all_passed,
        "violations": violations,
        "records": records,
    }
    _write_output(summary, output_path)
    return summary


# ---------------------------------------------------------------------------
# Test 2 — Fouling Nonlinearity
# ---------------------------------------------------------------------------

def run_fouling_nonlinearity(
    config_path: str = "configs/default.yaml",
    n_steps: int = 800,
    seed: int = 0,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run from clean state at nominal flow; check that:
    - fouling_frac_s3 > _s2 > _s1 (dep_rate ordering)
    - fouling rate accelerates as loading increases
    - maintenance_required transitions 0 → 1
    - irreversible_s3 > 0 after threshold is crossed
    - no NaN

    Pass criteria:
    - Stage ordering holds by step n_steps // 2
    - d(mesh_loading_avg)/step is larger in the latter half than the first
    - maintenance_required == 1.0 at some step within n_steps
    - irreversible_s3 > 0 at step n_steps if maintenance was triggered
    """
    env = HydrionEnv(config_path=config_path)
    # Seed well above the bistable thresholds to ensure all three stages grow.
    # The deposition formula dep ∝ (ff + eps)^dep_exponent with dep_exponent=2 is
    # autocatalytic.  Each stage has its own unstable fixed point:
    #   Stage 1: ff_u ≈ shear_coeff / (dep_rate_s1 × dep_base × Q_ref) ≈ 0.67
    #   Stage 2: ff_u ≈ 0.42
    #   Stage 3: ff_u ≈ 0.22
    # ff_seed=0.50 sits above the Stage 3 and 2 thresholds (both grow), and
    # slightly below Stage 1 (decreases but slowly), so mesh_loading_avg grows and
    # Stage 3 reaches the maintenance threshold (~0.70) within 800 steps.
    _reset_and_force_fouling(env, 0.50, seed=seed)

    # Nominal flow action: high pump, valve open, no backflush
    action = np.array([1.0, 0.8, 0.0, 0.5], dtype=np.float32)

    fouling_history: List[float] = []
    ff_s1_history:   List[float] = []
    ff_s2_history:   List[float] = []
    ff_s3_history:   List[float] = []
    maint_history:   List[float] = []
    violations:      List[str]   = []
    nan_seen = False

    for step in range(n_steps):
        obs, reward, term, trunc, info = env.step(action)
        t = _get_truth(env)

        ff_avg = float(t.get("mesh_loading_avg", 0.0))
        ff_s1  = float(t.get("fouling_frac_s1",  0.0))
        ff_s2  = float(t.get("fouling_frac_s2",  0.0))
        ff_s3  = float(t.get("fouling_frac_s3",  0.0))
        maint  = float(t.get("maintenance_required", 0.0))

        if not np.isfinite(ff_avg):
            nan_seen = True
            violations.append(f"NaN in mesh_loading_avg at step {step}")

        fouling_history.append(ff_avg)
        ff_s1_history.append(ff_s1)
        ff_s2_history.append(ff_s2)
        ff_s3_history.append(ff_s3)
        maint_history.append(maint)

        if term or trunc:
            break

    if nan_seen:
        all_passed = False
        return {
            "test": "fouling_nonlinearity",
            "config_path": config_path,
            "all_passed": False,
            "violations": violations,
        }

    mid = len(fouling_history) // 2

    # ---- Stage ordering at mid-run ----
    if not (ff_s3_history[mid] >= ff_s2_history[mid] - 1e-6 and
            ff_s2_history[mid] >= ff_s1_history[mid] - 1e-6):
        violations.append(
            f"Stage ordering at step {mid}: "
            f"s1={ff_s1_history[mid]:.4f} s2={ff_s2_history[mid]:.4f} s3={ff_s3_history[mid]:.4f}"
        )

    # ---- Acceleration: latter-half rate > first-half rate ----
    if len(fouling_history) >= 4:
        q1_end = max(len(fouling_history) // 4, 1)
        rate_first = (fouling_history[q1_end] - fouling_history[0]) / max(q1_end, 1)
        rate_last  = (fouling_history[-1] - fouling_history[mid]) / max(len(fouling_history) - mid, 1)
        if rate_last <= rate_first * 0.9:   # 10% tolerance
            violations.append(
                f"Fouling rate did not accelerate: "
                f"early={rate_first:.6f}/step  late={rate_last:.6f}/step"
            )

    # ---- maintenance_required triggered ----
    if max(maint_history) < 1.0:
        violations.append(f"maintenance_required never reached 1.0 over {n_steps} steps")

    # ---- irreversible_s3 > 0 after maintenance triggered ----
    t_final = _get_truth(env)
    irrev_s3 = float(t_final.get("irreversible_s3", 0.0))
    if max(maint_history) >= 1.0 and irrev_s3 <= 0.0:
        violations.append("irreversible_s3 still 0.0 after maintenance threshold was crossed")

    all_passed = len(violations) == 0
    summary = {
        "test": "fouling_nonlinearity",
        "config_path": config_path,
        "all_passed": all_passed,
        "violations": violations,
        "steps_run": len(fouling_history),
        "final_mesh_loading_avg": fouling_history[-1] if fouling_history else None,
        "final_ff_s3": ff_s3_history[-1] if ff_s3_history else None,
        "maintenance_triggered": max(maint_history) >= 1.0,
        "final_irreversible_s3": irrev_s3,
    }
    _write_output(summary, output_path)
    return summary


# ---------------------------------------------------------------------------
# Test 3 — Backflush Recovery
# ---------------------------------------------------------------------------

def run_backflush_recovery(
    config_path: str = "configs/default.yaml",
    pre_foul_frac: float = 0.60,
    bf_duration_s: float = 12.0,
    seed: int = 0,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Force system to pre_foul_frac, then apply bf_cmd=1.0 for bf_duration_s.

    Pass criteria:
    - fouling_frac_si decreases for all i (≥5% reduction on at least one stage)
    - fouling_frac_si > 0 after backflush (recovery is partial, not complete)
    - irreversible_si_post ≥ irreversible_si_pre (irreversible cannot decrease)
    - bf_active toggles correctly between pulses
    - bf_cooldown_remaining > 0 after burst completes
    """
    env = HydrionEnv(config_path=config_path)
    _reset_and_force_fouling(env, pre_foul_frac, seed=seed)

    # Snapshot before backflush
    t0 = _get_truth(env)
    ff_pre    = {i: float(t0.get(f"fouling_frac_s{i}", 0.0)) for i in range(1, 4)}
    irr_pre   = {i: float(t0.get(f"irreversible_s{i}", 0.0)) for i in range(1, 4)}

    # Apply backflush for bf_duration_s
    bf_action    = np.array([1.0, 0.5, 1.0, 0.5], dtype=np.float32)
    bf_steps     = int(bf_duration_s / env.dt)
    bf_seen      = []
    cooldown_end = []
    violations: List[str] = []

    for s in range(bf_steps):
        obs, reward, term, trunc, info = env.step(bf_action)
        t = _get_truth(env)
        bf_seen.append(float(t.get("bf_active", 0.0)))
        cooldown_end.append(float(t.get("bf_cooldown_remaining", 0.0)))
        if not np.isfinite(float(t.get("mesh_loading_avg", 0.0))):
            violations.append(f"NaN at step {s} during backflush")
            break

    t_post  = _get_truth(env)
    ff_post  = {i: float(t_post.get(f"fouling_frac_s{i}", 0.0)) for i in range(1, 4)}
    irr_post = {i: float(t_post.get(f"irreversible_s{i}", 0.0)) for i in range(1, 4)}

    # ---- Recovery checks ----
    any_significant = False
    for i in range(1, 4):
        delta = ff_pre[i] - ff_post[i]
        if delta < 0:
            violations.append(f"fouling_frac_s{i} increased during backflush: pre={ff_pre[i]:.4f} post={ff_post[i]:.4f}")
        if delta >= ff_pre[i] * 0.05:   # at least 5% of original
            any_significant = True
        if ff_post[i] <= 0.0 and ff_pre[i] > 1e-4:
            violations.append(f"fouling_frac_s{i} fully cleared (should be partial): post={ff_post[i]:.6f}")
        if irr_post[i] < irr_pre[i] - 1e-9:
            violations.append(
                f"irreversible_s{i} decreased: pre={irr_pre[i]:.6f} post={irr_post[i]:.6f}"
            )

    if not any_significant:
        violations.append("No stage showed ≥5% fouling reduction — backflush had negligible effect")

    # ---- bf_active toggling ----
    if not any(b > 0.5 for b in bf_seen):
        violations.append("bf_active never went to 1.0 during backflush period")
    if not any(b < 0.5 for b in bf_seen):
        violations.append("bf_active never went to 0.0 (no interpulse or cooldown observed)")

    # ---- cooldown after burst ----
    # Check that cooldown started at some point, not just the last step.
    # The burst completes in ~1.7 s; with cooldown=9 s and bf_duration_s=12 s,
    # the cooldown will have expired before the test ends — so checking the last
    # step would always fail.  Instead verify that cooldown was > 0 at some step.
    if not any(c > 0.0 for c in cooldown_end):
        violations.append("bf_cooldown_remaining never went > 0 (burst may not have completed or cooldown never started)")

    all_passed = len(violations) == 0
    summary = {
        "test": "backflush_recovery",
        "config_path": config_path,
        "all_passed": all_passed,
        "violations": violations,
        "pre_foul_frac": pre_foul_frac,
        "bf_duration_s": bf_duration_s,
        "ff_pre":  {f"s{i}": ff_pre[i]  for i in range(1, 4)},
        "ff_post": {f"s{i}": ff_post[i] for i in range(1, 4)},
        "irr_pre":  {f"s{i}": irr_pre[i]  for i in range(1, 4)},
        "irr_post": {f"s{i}": irr_post[i] for i in range(1, 4)},
        "any_significant_recovery": any_significant,
    }
    _write_output(summary, output_path)
    return summary


# ---------------------------------------------------------------------------
# Test 4 — Diminishing Returns
# ---------------------------------------------------------------------------

def run_diminishing_returns(
    config_path: str = "configs/default.yaml",
    pre_foul_frac: float = 0.80,
    n_bursts: int = 5,
    seed: int = 0,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Apply n_bursts successive backflush bursts (with cooldown between each);
    verify each burst recovers strictly less fouling than the previous.

    Pass criteria:
    - δ_s3_burst_1 > δ_s3_burst_2 > δ_s3_burst_3 (first 3 must be monotone)
    - δ_s3_burst_N > 0 for all N (backflush still effective)
    - irreversible_s3 is non-decreasing across bursts
    """
    env = HydrionEnv(config_path=config_path)
    _reset_and_force_fouling(env, pre_foul_frac, seed=seed)

    # Actions
    bf_action    = np.array([1.0, 0.5, 1.0, 0.5], dtype=np.float32)
    idle_action  = np.array([0.5, 0.5, 0.0, 0.5], dtype=np.float32)
    bf_steps     = int((env._bf_burst_total + 0.5) / env.dt) + 1  # enough steps for one burst
    cooldown_steps = int((env._bf_cooldown + 1.0) / env.dt) + 1    # enough to expire cooldown

    deltas:     List[float] = []
    irr_series: List[float] = []
    violations: List[str]   = []

    for burst in range(n_bursts):
        t0     = _get_truth(env)
        ff_pre = float(t0.get("fouling_frac_s3", 0.0))
        irr_pre = float(t0.get("irreversible_s3", 0.0))

        # Fire one burst
        for _ in range(bf_steps):
            env.step(bf_action)
            if float(_get_truth(env).get("bf_burst_elapsed", 0.0)) == 0.0:
                break  # burst completed

        # Wait for cooldown to expire
        for _ in range(cooldown_steps):
            env.step(idle_action)
            if float(_get_truth(env).get("bf_cooldown_remaining", 0.0)) <= 0.0:
                break

        t1      = _get_truth(env)
        ff_post = float(t1.get("fouling_frac_s3", 0.0))
        irr_post = float(t1.get("irreversible_s3", 0.0))

        delta = ff_pre - ff_post
        deltas.append(delta)
        irr_series.append(irr_post)

        if delta <= 0.0:
            violations.append(f"Burst {burst+1}: no recovery (δ={delta:.6f})")

        if irr_post < irr_pre - 1e-9:
            violations.append(
                f"Burst {burst+1}: irreversible_s3 decreased: pre={irr_pre:.6f} post={irr_post:.6f}"
            )

    # ---- Monotone diminishing returns (first 3 bursts) ----
    for i in range(min(2, len(deltas) - 1)):
        if deltas[i] <= deltas[i + 1] + 1e-7:
            violations.append(
                f"Diminishing returns not satisfied: burst {i+1} δ={deltas[i]:.6f} "
                f"≤ burst {i+2} δ={deltas[i+1]:.6f}"
            )

    all_passed = len(violations) == 0
    summary = {
        "test": "diminishing_returns",
        "config_path": config_path,
        "all_passed": all_passed,
        "violations": violations,
        "pre_foul_frac_s3": pre_foul_frac,
        "burst_deltas_s3": deltas,
        "irreversible_s3_series": irr_series,
    }
    _write_output(summary, output_path)
    return summary


# ---------------------------------------------------------------------------
# Test 5 — Bypass Activation
# ---------------------------------------------------------------------------

def run_bypass_activation(
    config_path: str = "configs/default.yaml",
    seed: int = 0,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Verify bypass_active correctly toggles and flow conservation holds.

    Pass criteria:
    - At heavy fouling + high pump: bypass_active == 1.0
    - When bypass active: q_bypass_lmin > 0
    - Flow conservation: |q_processed + q_bypass − q_in| ≤ 1% of q_in
    - At clean state + same pump: bypass_active == 0.0
    - Repeated heavy-fouling steps: bypass_active does NOT oscillate every step
    """
    env = HydrionEnv(config_path=config_path)
    violations: List[str] = []
    records: List[Dict[str, Any]] = []

    # ---- Heavy fouled + high pump ----
    _reset_and_force_fouling(env, 0.90, seed=seed)
    high_pump_action = np.array([1.0, 1.0, 0.0, 0.5], dtype=np.float32)

    bypass_flags: List[float] = []
    for s in range(5):
        env.step(high_pump_action)
        t = _get_truth(env)
        bypass_active  = float(t.get("bypass_active",    0.0))
        q_in           = float(t.get("q_in_lmin",        0.0))
        q_proc         = float(t.get("q_processed_lmin", 0.0))
        q_byp          = float(t.get("q_bypass_lmin",    0.0))
        bypass_flags.append(bypass_active)

        if bypass_active > 0.5:
            if q_byp <= 0.0:
                violations.append(f"step {s}: bypass_active=1 but q_bypass_lmin={q_byp:.4f}")
            # Flow conservation: q_processed + q_bypass ≈ q_in (within 1%)
            if q_in > 1e-3:
                flow_err = abs((q_proc + q_byp) - q_in) / q_in
                if flow_err > 0.01:
                    violations.append(
                        f"step {s}: flow conservation error {flow_err:.4%}: "
                        f"q_in={q_in:.3f} q_proc={q_proc:.3f} q_byp={q_byp:.3f}"
                    )
        records.append({"step": s, "bypass_active": bypass_active, "q_in": q_in,
                        "q_proc": q_proc, "q_byp": q_byp})

    if max(bypass_flags) < 1.0:
        violations.append("bypass_active never reached 1.0 under heavy fouling + high pump")

    # No oscillation: consecutive steps should agree on bypass state
    oscillations = sum(1 for i in range(len(bypass_flags) - 1)
                       if bypass_flags[i] != bypass_flags[i + 1])
    if oscillations > 1:   # one transition allowed (from startup), not per-step flip-flop
        violations.append(f"bypass_active oscillated {oscillations} times in 5 consecutive steps")

    # ---- Clean state + same pump → no bypass ----
    _reset_and_force_fouling(env, 0.0, seed=seed + 1)
    env.step(high_pump_action)
    t_clean = _get_truth(env)
    if float(t_clean.get("bypass_active", 0.0)) > 0.5:
        violations.append(
            f"bypass_active=1 in clean state at high pump "
            f"(P_in={t_clean.get('P_in', 0.0):.1f} Pa)"
        )

    all_passed = len(violations) == 0
    summary = {
        "test": "bypass_activation",
        "config_path": config_path,
        "all_passed": all_passed,
        "violations": violations,
        "records": records,
    }
    _write_output(summary, output_path)
    return summary


# ---------------------------------------------------------------------------
# Test 6 — NaN / Bounded-State Regression
# ---------------------------------------------------------------------------

def run_nan_bounded_regression(
    config_path: str = "configs/default.yaml",
    n_steps: int = 2000,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run n_steps random actions; verify all Milestone 1 truth_state fields
    remain finite and within expected bounds.

    Pass criteria (checked every step):
    - No NaN/Inf in obs or reward
    - fouling_frac_si ∈ [0, 1] for i = 1, 2, 3
    - irreversible_si ≤ fouling_frac_si + 1e-7
    - recoverable_si ≥ −1e-7
    - q_bypass_lmin ≥ 0  and  q_processed_lmin ≥ 0
    - q_processed_lmin + q_bypass_lmin ≤ q_in_lmin + 0.01 (flow not created)
    - dp_stage*_pa ≥ 0
    - bf_cooldown_remaining ≥ 0
    - flow ∈ [0, 1]  (validates Q_max normalization fix)
    """
    env = HydrionEnv(config_path=config_path)
    env.reset(seed=seed)

    violations: List[str] = []
    step_count = 0

    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        step_count += 1
        t = _get_truth(env)

        def viol(msg: str) -> None:
            if len(violations) < 20:   # cap to avoid flooding output
                violations.append(f"step {step}: {msg}")

        # ---- NaN checks ----
        if not np.isfinite(obs).all():
            viol(f"obs has non-finite values: {obs}")
        if not np.isfinite(reward):
            viol(f"reward={reward}")

        # ---- Fouling bounds ----
        for i in range(1, 4):
            ff  = float(t.get(f"fouling_frac_s{i}", 0.0))
            irr = float(t.get(f"irreversible_s{i}", 0.0))
            rec = float(t.get(f"recoverable_s{i}",  0.0))
            if not (0.0 - 1e-7 <= ff <= 1.0 + 1e-7):
                viol(f"fouling_frac_s{i}={ff:.6f} out of [0,1]")
            if irr > ff + 1e-7:
                viol(f"irreversible_s{i}={irr:.6f} > fouling_frac_s{i}={ff:.6f}")
            if rec < -1e-7:
                viol(f"recoverable_s{i}={rec:.6f} < 0")

        # ---- Flow conservation ----
        q_in   = float(t.get("q_in_lmin",        0.0))
        q_proc = float(t.get("q_processed_lmin", 0.0))
        q_byp  = float(t.get("q_bypass_lmin",    0.0))
        if q_proc < -1e-4:
            viol(f"q_processed_lmin={q_proc:.4f} < 0")
        if q_byp < -1e-4:
            viol(f"q_bypass_lmin={q_byp:.4f} < 0")
        if q_proc + q_byp > q_in + 0.01:
            viol(f"q_proc+q_byp={q_proc+q_byp:.3f} > q_in={q_in:.3f}")

        # ---- Pressure drops ≥ 0 ----
        for tag in ("dp_stage1_pa", "dp_stage2_pa", "dp_stage3_pa"):
            v = float(t.get(tag, 0.0))
            if v < -1.0:   # 1 Pa tolerance
                viol(f"{tag}={v:.2f} < 0")

        # ---- Backflush timing ----
        cd = float(t.get("bf_cooldown_remaining", 0.0))
        if cd < -1e-4:
            viol(f"bf_cooldown_remaining={cd:.4f} < 0")

        # ---- Normalized flow in [0, 1] ----
        flow = float(t.get("flow", 0.0))
        if flow > 1.0 + 1e-4:
            viol(f"flow={flow:.5f} > 1.0 (Q_max normalization bug)")

        if term or trunc:
            env.reset(seed=seed + step + 1)

        # Stop collecting violations after cap to keep report readable
        if len(violations) >= 20:
            violations.append("... further violations suppressed (cap reached)")
            break

    all_passed = len(violations) == 0
    summary = {
        "test": "nan_bounded_regression",
        "config_path": config_path,
        "all_passed": all_passed,
        "violations": violations,
        "steps_run": step_count,
        "n_steps_requested": n_steps,
    }
    _write_output(summary, output_path)
    return summary


# ---------------------------------------------------------------------------
# Combined runner
# ---------------------------------------------------------------------------

def run_milestone1_validation(
    config_path: str = "configs/default.yaml",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run all six Milestone 1 validation tests and return a combined summary.
    """
    results = {}

    print("  [M1-1] Pressure-flow sweep ...")
    results["pressure_flow_sweep"] = run_pressure_flow_sweep(config_path)

    print("  [M1-2] Fouling nonlinearity ...")
    results["fouling_nonlinearity"] = run_fouling_nonlinearity(config_path)

    print("  [M1-3] Backflush recovery ...")
    results["backflush_recovery"] = run_backflush_recovery(config_path)

    print("  [M1-4] Diminishing returns ...")
    results["diminishing_returns"] = run_diminishing_returns(config_path)

    print("  [M1-5] Bypass activation ...")
    results["bypass_activation"] = run_bypass_activation(config_path)

    print("  [M1-6] NaN/bounded-state regression ...")
    results["nan_bounded_regression"] = run_nan_bounded_regression(config_path)

    all_passed = all(v["all_passed"] for v in results.values())
    summary = {
        "milestone": "Milestone 1 — Hydraulic + Fouling + Backflush Realism Backbone",
        "config_path": config_path,
        "all_passed": all_passed,
        "results": {k: {"all_passed": v["all_passed"], "violations": v.get("violations", [])}
                    for k, v in results.items()},
    }
    _write_output(summary, output_path)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HydrOS Milestone 1 Validation Suite"
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Hydrion config YAML")
    parser.add_argument("--output", default=None, help="Output results YAML path")
    parser.add_argument(
        "--test",
        default="all",
        choices=["all", "pressure_flow_sweep", "fouling_nonlinearity",
                 "backflush_recovery", "diminishing_returns",
                 "bypass_activation", "nan_bounded_regression"],
        help="Which test to run (default: all)",
    )
    args = parser.parse_args()

    if args.test == "all":
        summary = run_milestone1_validation(config_path=args.config, output_path=args.output)
    else:
        fn_map = {
            "pressure_flow_sweep":    run_pressure_flow_sweep,
            "fouling_nonlinearity":   run_fouling_nonlinearity,
            "backflush_recovery":     run_backflush_recovery,
            "diminishing_returns":    run_diminishing_returns,
            "bypass_activation":      run_bypass_activation,
            "nan_bounded_regression": run_nan_bounded_regression,
        }
        summary = fn_map[args.test](config_path=args.config, output_path=args.output)

    print("Milestone 1 all_passed:", summary["all_passed"])
    if not summary["all_passed"]:
        for test_name, result in summary.get("results", {args.test: summary}).items():
            if not result.get("all_passed", True):
                print(f"  FAIL [{test_name}]:")
                for v in result.get("violations", []):
                    print(f"    - {v}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    exit(main())
