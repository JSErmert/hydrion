# Phase 2 — M6 Observation Handoff
## Execution Document

**Document type:** Phase execution specification
**Phase:** M6 Phase 2 — Observation Source Migration
**Status:** Ready for planning activation
**Date:** 2026-04-13
**Blocked by:** Phase 1 (M6 core sensor realism) — **SATISFIED** (committed 2026-04-13)
**Blocking:** Phase 3 (M7 baseline RL rebuild under M6)

---

## 1. Context

### Phase 1 Completion Status

M6 Phase 1 is complete and committed as of 2026-04-13:

- `hydrion/sensors/pressure.py` — DifferentialPressureSensor: AWGN + random walk drift +
  fouling offset + 1-step latency buffer. Reads `truth_state["dp_total_pa"]` and
  `truth_state["mesh_loading_avg"]`. Writes `sensor_state["dp_sensor_kPa"]` only.
- `hydrion/sensors/flow.py` — FlowRateSensor: multiplicative AWGN + per-episode calibration
  bias. Reads `truth_state["q_processed_lmin"]`. Writes `sensor_state["flow_sensor_lmin"]` only.
- `hydrion/state/init.py` — `dp_sensor_kPa` and `flow_sensor_lmin` initialized to 0.0 in
  `init_sensor_state()`. Keys always present; never undefined.
- `configs/default.yaml` — `sensors.pressure` and `sensors.flow` blocks with all
  source-grounded defaults.
- `truth_state` / `sensor_state` separation: intact. Verified by tests 5 and 6 in
  `tests/test_sensors_m6.py`.
- Test suite: 97/97 passing.

### What Phase 1 Did Not Do

Phase 1 populated `sensor_state` but made no changes to how RL observations are constructed.
The RL policy (both HydrionEnv and ConicalCascadeEnv) continues to read from `truth_state`
at every observation dimension that existed before Phase 1. The two new `sensor_state` keys
are logged but not yet fed into any policy observation.

Phase 2 is the handoff: move selected observation dimensions from truth-derived to
sensor-derived without breaking schema discipline.

---

## 2. Scope Boundary

### In Scope — Phase 2

- Updating `hydrion/sensors/sensor_fusion.py` observation construction
- Updating `hydrion/environments/conical_cascade_env.py` `_obs()` method if and only if
  a clean sensor injection path can be defined without adding a sensor layer to CCE
- Field-by-field audit of both observation paths
- Schema version analysis and minimum justifiable version bump decision
- Validation plan for both paths

### Out of Scope — Phase 2

- No new sensor model implementation (Phase 1 closed)
- No reward function changes
- No modification to truth_state physics (hydraulics, clogging, electrostatics, particles)
- No M7 RL training or curriculum changes
- No changes to `HydraulicsModel`, `CloggingModel`, `ElectrostaticsModel`, or `ParticleModel`
- No changes to `ShieldedEnv` or `ScenarioRunner`

---

## 3. Architectural Constraints (Non-Negotiable)

These constraints are inherited from the HydrOS Co-Orchestrator Execution Contract and must
not be violated at any point during Phase 2 implementation:

**State separation:**
- `truth_state` remains authoritative for all physics. No Phase 2 change modifies truth_state.
- `sensor_state` remains observational. Phase 2 reads from it; never writes physics back to it.
- No cross-contamination in either direction.

**Schema discipline:**
- `obs12_v2` is the current stable schema. Its index ordering is immutable as long as the
  version label is unchanged. No silent index drift is permitted.
- If any index changes its data source from truth to sensor (changing the observation semantics
  for a trained policy), this constitutes a schema version event. The version must be bumped
  explicitly and documented with justification.
- Do not assume obs12_v3 unless implementation analysis confirms at least one index changes
  source. If both candidates remain truth-derived after analysis, obs12_v2 is preserved as-is.

**Two-path rule:**
- HydrionEnv and ConicalCascadeEnv have independent observation construction paths.
  They must be treated separately. Changes to sensor_fusion.py do not cascade to CCE.
  CCE has no sensor_state layer as of Phase 1 close.

**No backward-compat shims:**
- Do not add version aliases, deprecated key bridges, or dual-path fallbacks. Pick a version,
  implement cleanly, document it.

---

## 4. Observation Path Audit (Pre-Phase 2 Baseline)

### Path A — HydrionEnv / sensor_fusion.py

**File:** `hydrion/sensors/sensor_fusion.py`
**Function:** `build_observation(truth: dict, sensors: dict) -> np.ndarray`
**Version:** obs12_v2

| Index | Key | Source dict | Physical quantity | Unit / normalization | Phase 2 candidate? |
|---|---|---|---|---|---|
| 0 | flow | truth_state | Q_out_Lmin (net filtration flow) | / Q_max_Lmin, [0,1] | YES — see note A |
| 1 | pressure | truth_state | P_in (inlet pressure) | / P_max_Pa, [0,1] | CONDITIONAL — see note B |
| 2 | clog | truth_state | mesh_loading_avg (avg fouling) | [0,1] direct | No — no sensor for clogging |
| 3 | E_field_norm | truth_state | Normalized radial E-field | [0,1] | No — no E-field sensor |
| 4 | C_out | truth_state | Outlet particle concentration | [0,1] | No — no particle sensor |
| 5 | particle_capture_eff | truth_state | Capture efficiency | [0,1] | No — no particle sensor |
| 6 | valve_cmd | truth_state | Valve actuator command | [0,1] | No — actuator command, always truth |
| 7 | pump_cmd | truth_state | Pump actuator command | [0,1] | No — actuator command, always truth |
| 8 | bf_cmd | truth_state | Backflush command | [0,1] | No — actuator command, always truth |
| 9 | node_voltage_cmd | truth_state | Node voltage command | [0,1] | No — actuator command, always truth |
| 10 | sensor_turbidity | sensor_state | Optical turbidity (ALREADY sensor) | [0,1] | Already done |
| 11 | sensor_scatter | sensor_state | Optical scatter (ALREADY sensor) | [0,1] | Already done |

**Note A — Index 0 (flow):**
`truth_state["flow"]` is computed in `_update_normalized_state()` as
`Q_out_Lmin / Q_max_Lmin`. The Phase 1 flow sensor (`flow_sensor_lmin`) measures
`q_processed_lmin` (filter-stage flow, excluding bypass). These are related but not identical:
- `Q_out_Lmin`: net filtration flow written by HydraulicsModel
- `q_processed_lmin`: explicitly `q_in - q_bypass`

Implementation analysis must confirm whether `Q_out_Lmin ≈ q_processed_lmin` in all
operating modes (bypass active, backflush active, steady-state). If equivalent, a
sensor-normalized flow index is clean: `flow_sensor_lmin / Q_max_Lmin`. If not, the
substitution changes physical meaning and must be documented.

**Note B — Index 1 (pressure) — CRITICAL SEMANTIC MISMATCH:**
`truth_state["pressure"]` is computed as `P_in / P_max_Pa` (inlet absolute pressure
normalized over the system maximum). The Phase 1 pressure sensor (`dp_sensor_kPa`)
measures `dp_total_pa` — the *differential* pressure across all filter stages. These
are different physical quantities:

- `P_in` (inlet pressure): depends on pump command + total system resistance
- `dp_total_pa` (filter differential): depends on filter fouling state

They correlate but have different normalization references and different dynamic ranges.
A direct substitution of `dp_sensor_kPa / (P_max_Pa / 1000)` into index 1 would change
the physical meaning of that observation dimension. This is a schema-semantic change.

Implementation analysis must choose one of three options:
- **Option A:** Keep index 1 truth-derived (`P_in / P_max_Pa`). Introduce `dp_sensor_kPa`
  as a new index 12+ (schema extension, requires version bump and dimensionality change).
- **Option B:** Replace index 1 with `dp_sensor_kPa` normalized appropriately.
  Document the semantic change. Requires version bump. Justified only if the filter
  differential is more informative than inlet pressure for the RL policy.
- **Option C:** Keep index 1 truth-derived. Introduce sensor-derived pressure as an
  additional context signal outside the core 12D vector (e.g., separate info dict only).
  Preserves obs12_v2 without dimensionality change.

**Default conservative position:** Option C. Only change what the analysis confirms
improves policy informativeness without ambiguity.

---

### Path B — CCE / conical_cascade_env.py

**File:** `hydrion/environments/conical_cascade_env.py`
**Method:** `ConicalCascadeEnv._obs() -> np.ndarray`
**Version:** obs12 compatible (same dimensionality as obs12_v2; semantically DIFFERENT)

| Index | Comment | Source | Physical quantity | Phase 2 candidate? |
|---|---|---|---|---|
| 0 | q_in | _state (truth) | q_processed_lmin / OBS_Q_MAX (25 L/min) | YES — see note C |
| 1 | delta_p | _state (truth) | dp_total_pa / 1000 / OBS_DP_MAX (150 kPa) | YES — see note D |
| 2 | fouling_mean | _state (truth) | avg(fouling_frac_s1/s2/s3) | No — no clogging sensor |
| 3 | eta_cascade | _state (truth) | Cascade capture efficiency (M5 physics) | No |
| 4 | C_in | _state (truth) | Inlet particle concentration | No |
| 5 | C_out | _state (truth) | Outlet particle concentration | No |
| 6 | E_field_norm | _state (truth) | voltage_norm (normalized V) | No |
| 7 | v_crit_norm | _state (truth) | v_crit_s3 / OBS_VCRIT_MAX | No |
| 8 | step_norm | episode counter | _step / _max_steps | No — not a physical sensor |
| 9 | bf_active | _state (truth) | bf_active binary | No — actuator state |
| 10 | eta_PP | _state (truth) | Buoyant species capture efficiency | No |
| 11 | eta_PET | _state (truth) | Dense species capture efficiency | No |

**Critical observation — CCE vs HydrionEnv schema divergence:**
CCE index 3 is `eta_cascade` (M5 physics capture efficiency). HydrionEnv index 3 is
`E_field_norm` (radial electric field). These are completely different. The two environments
are NOT obs12_v2 compatible with each other despite the same dimensionality. This divergence
was already present before Phase 2 and must not be conflated.

**Note C — CCE index 0 (flow):**
CCE reads `q_processed_lmin` directly — the same physical quantity that `flow_sensor_lmin`
estimates. Normalization: `/ OBS_Q_MAX` (25 L/min, vs HydrionEnv `Q_max_Lmin` = 20 L/min).
If sensor injection is implemented, the normalization constant must match CCE's not
HydrionEnv's.

**Note D — CCE index 1 (delta_p):**
CCE reads `dp_total_pa / 1000 / OBS_DP_MAX` (in kPa, then normalized over 150 kPa). The
Phase 1 sensor `dp_sensor_kPa` is a direct estimate of `dp_total_pa / 1000`. This is a
**clean semantic match** — the first time a direct substitution is unambiguous. If any
index is migrated to sensor in CCE, index 1 is the natural candidate.

**CCE sensor state architecture problem:**
CCE has no `sensor_state`. Its `_obs()` reads from `self._state` (truth). Injecting sensor
readings requires one of:
- **Option I:** Add a `sensor_state` dict to CCE (parallel to HydrionEnv architecture).
  DifferentialPressureSensor and FlowRateSensor would need to be instantiated in CCE.
- **Option II:** Pass sensor values from outside via a reset/step argument.
- **Option III:** Defer CCE observation migration to a separate phase (Phase 2b).
  CCE truth-reading is internally consistent; the ppo_cce_v2 benchmark was trained on
  truth values and all comparisons reference it — changing CCE obs before retraining
  invalidates the benchmark. Deferring is architecturally safe.

**Default conservative position:** Option III. Migrate HydrionEnv first, validate, then
address CCE as a separate scoped effort with explicit benchmark versioning.

---

## 5. Schema Version Analysis

### Current Version: obs12_v2

Established in M3 (2026-04-10). Change from v1: index 3 changed from `E_norm` (arbitrary)
to `E_field_norm` (physical kV/m normalized). This was a semantic change → version bump.
Precedent: any semantic change to an index source warrants a version bump.

### Phase 2 Version Decision

Determined by which indices actually change source:

| Scenario | Version Decision | Rationale |
|---|---|---|
| No index changes source (all analysis leads to Option C / Option III) | obs12_v2 preserved | No semantic change |
| Index 0 (flow) changes to sensor-derived in sensor_fusion.py | obs12_v3 | Source change: truth-derived → sensor-derived |
| Index 1 (pressure) changes to dp_sensor in sensor_fusion.py | obs12_v3 | Source change + physical quantity change |
| CCE deferred (HydrionEnv only changes) | obs12_v3 scoped to HydrionEnv path only | CCE remains v2 until CCE-specific bump |
| New dimension added (Option A) | obs13_vX | Dimensionality change is a major version event |

**Rule:** Do not pre-assign obs12_v3. The implementation plan must derive the version bump
from the field-by-field source audit, not assume it.

**If obs12_v3 is justified:**
- Update the version label comment in `sensor_fusion.py`
- Document the delta from v2 (which index, old source, new source, why)
- ppo_cce_v2 (trained on obs12_v2 truth) becomes incompatible with obs12_v3 observations —
  this must be explicitly noted in the Phase 3 (RL rebuild) brief

---

## 6. Implementation-Planning Outputs Required

The implementation plan for Phase 2 must include the following, in this order:

### 6.1 Field-by-Field Source Audit (Both Paths)

For every observation index in both paths, document:
- Current source dict and key
- Current physical quantity and normalization
- Available sensor_state key (if any)
- Whether sensor key measures the same physical quantity (or a correlated proxy)
- Decision: migrate to sensor / keep truth / extend schema / defer

This is a pre-code analysis step. Do not write code before completing the audit.

### 6.2 Affected Files

**Minimum (HydrionEnv path only):**
- `hydrion/sensors/sensor_fusion.py` — observation source changes
- `hydrion/state/init.py` — if any new normalized sensor key written here
- `hydrion/env.py` — if normalization logic moves from `_update_normalized_state` to
  sensor path

**Additional (if CCE path included):**
- `hydrion/environments/conical_cascade_env.py` — `_obs()` method
- `hydrion/sensors/pressure.py` / `flow.py` — if CCE instantiates them

**Documentation (always):**
- `docs/orchestration/M6_Phase2/obs_schema_audit.md` — field-by-field audit record
- Version bump justification inline in `sensor_fusion.py` docstring

### 6.3 Backward Compatibility Assessment

- ppo_cce_v2 was trained with obs12_v2 truth-derived observations
- Any schema change in HydrionEnv affects ppo_cce_v2 comparability for that path
- If the version bumps to obs12_v3, ppo_cce_v2 is explicitly invalidated for Phase 3
  comparisons on the HydrionEnv path — Phase 3 must retrain the baseline
- CCE ppo_cce_v2 benchmark remains valid as long as CCE observation is unchanged

### 6.4 Validation Plan

For each index migrated to sensor-derived:

1. **Divergence test:** confirm sensor obs[i] diverges measurably from truth obs[i] over N
   steps (mean absolute difference > 0). This is the evidence that the sensor layer is active.
2. **Range test:** confirm sensor obs[i] stays within [0, 1] after normalization across the
   full operating envelope (Q_low to Q_peak, fouling 0–100%).
3. **No contamination test:** confirm truth_state[original_key] is unchanged after observation
   construction.
4. **End-to-end rollout test:** run 1000 steps; confirm no NaN, no out-of-bounds obs values.
5. **Benchmark rollout delta:** run ppo_cce_v2 (trained on v2 truth) on the new obs schema.
   Record mean return degradation. This quantifies the policy sensitivity to the obs change.

### 6.5 Acceptance Criteria

Phase 2 is complete when all of the following are satisfied:

1. Field-by-field audit doc written and committed
2. Version decision documented and justified (or explicitly preserved as v2 with rationale)
3. All migrated indices confirmed sensor-derived (divergence tests passing)
4. truth_state unmodified by any Phase 2 change (contamination tests passing)
5. Observation range [0, 1] maintained for all migrated indices
6. All existing 97 tests continue to pass
7. New tests cover: sensor obs diverges, truth unchanged, range valid, end-to-end populated
8. If obs12_v3 declared: version label updated in sensor_fusion.py docstring with delta table
9. Benchmark impact documented (ppo_cce_v2 degradation quantified if schema changes)

---

## 7. Blocked-By Statement

```
Blocked by: M6 Phase 1 (core sensor realism) — SATISFIED
  - dp_sensor_kPa and flow_sensor_lmin now exist in sensor_state
  - Truth/sensor separation intact and test-verified
  - YAML parameters exposed in configs/default.yaml
  - Committed 2026-04-13

Blocking: M6 Phase 3 (M7 RL baseline rebuild under M6)
  - Phase 3 cannot retrain until the observation schema is stable
  - If obs12_v3 is declared in Phase 2, Phase 3 must use v3 for all new baselines
  - ppo_cce_v2 compatibility status must be resolved here, not in Phase 3
```

---

## 8. Open Questions for Implementation Planning

The following questions must be answered during implementation planning (before writing code):

**Q1:** Is `Q_out_Lmin` (used in `truth["flow"]`) equivalent to `q_processed_lmin` (measured
by flow sensor) in all operating modes? Verify by inspecting `HydraulicsModel` output logic.

**Q2:** For HydrionEnv index 1 (pressure): is `P_in` normalized the more policy-informative
signal, or is `dp_total_pa` normalized more informative? This requires analysis of which
signal the policy actually needs for optimal backflush / bypass decisions.

**Q3:** For CCE: is the bench-test-validated ppo_cce_v2 benchmark at risk if CCE observation
changes? If yes, defer CCE migration to avoid invalidating the only validated benchmark before
a replacement exists.

**Q4:** Can a sensor-normalized pressure key be added to `sensor_state` (not truth_state)
to bridge the normalization gap? E.g., `dp_sensor_norm = dp_sensor_kPa / dp_max_kPa` written
by DifferentialPressureSensor. This decouples normalization from observation construction.

---

## 9. Notes for Planning Activation

This document defines the scope, constraints, field audit requirements, and acceptance criteria
for Phase 2. It does not prescribe the exact code changes — those emerge from the implementation
plan after the Q1–Q4 questions are answered.

The implementation plan must start with the field-by-field audit (Section 6.1) and derive all
code decisions from that analysis. Do not write sensor_fusion.py changes before the audit is
complete.

The minimum viable Phase 2 may be: audit confirms HydrionEnv index 0 (flow) migrates cleanly,
index 1 (pressure) defers to Option C, CCE defers to Option III. Result: obs12_v3 with a
single-index change. This is the conservative path and is architecturally sound.

---

*Document written 2026-04-13. Status: ready for planning activation.*
*Prerequisite: M6 Phase 1 complete (verified). Next step: invoke writing-plans for Phase 2.*
