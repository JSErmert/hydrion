# Phase 2 — M6 Observation Handoff
## Implementation Plan

**Document type:** Implementation plan (pre-code analysis)
**Phase:** M6 Phase 2 — Observation Source Migration
**Date:** 2026-04-13
**Governing spec:** docs/orchestration/M6_Phase2/Phase2_M6_observation_handoff.md
**Status:** ANALYSIS COMPLETE — see Section 7 for minimum safe scope recommendation

---

## 1. Field-by-Field Source Audit

### 1.1 HydrionEnv / sensor_fusion.py Path

**Version: obs12_v2**
**Function:** `build_observation(truth: dict, sensors: dict)` in `hydrion/sensors/sensor_fusion.py`
**Normalization context:**
- `truth["flow"]` written by `_update_normalized_state()`: `Q_out_Lmin / Q_max_Lmin` (= 20.0)
- `truth["pressure"]` written by `_update_normalized_state()`: `P_in / P_max_Pa` (= 80,000)

---

| Index | Label | Source dict | Source key | Physical quantity | Normalization | Sensor state equivalent | Semantic match? | Decision |
|---|---|---|---|---|---|---|---|---|
| 0 | flow | truth | truth["flow"] | Q_out_Lmin (net forward filtration output, post-backflush-diversion) | / Q_max_Lmin [0,1] | flow_sensor_lmin (q_processed_lmin estimate) | **NO — see §1.1a** | **DEFER** |
| 1 | pressure | truth | truth["pressure"] | P_in (absolute inlet pressure) | / P_max_Pa [0,1] | dp_sensor_kPa (dp_total_pa estimate) | **NO — see §1.1b** | **DEFER** |
| 2 | clog | truth | truth["clog"] | mesh_loading_avg (avg fouling fraction) | [0,1] direct | None | N/A — no clogging sensor | Keep truth |
| 3 | E_field_norm | truth | truth["E_field_norm"] | Normalized radial E-field (physical kV/m) | [0,1] | None | N/A — no E-field sensor | Keep truth |
| 4 | C_out | truth | truth["C_out"] | Outlet particle concentration | [0,1] | None | N/A — no inline particle sensor | Keep truth |
| 5 | particle_capture_eff | truth | truth["particle_capture_eff"] | Capture efficiency (computed) | [0,1] | None | N/A | Keep truth |
| 6 | valve_cmd | truth | truth["valve_cmd"] | Valve actuator command (written by env.step) | [0,1] | N/A | Actuator commands are always authoritative from truth | Keep truth |
| 7 | pump_cmd | truth | truth["pump_cmd"] | Pump actuator command | [0,1] | N/A | Same | Keep truth |
| 8 | bf_cmd | truth | truth["bf_cmd"] | Backflush command | [0,1] | N/A | Same | Keep truth |
| 9 | node_voltage_cmd | truth | truth["node_voltage_cmd"] | Node voltage command | [0,1] | N/A | Same | Keep truth |
| 10 | sensor_turbidity | sensors | sensors["sensor_turbidity"] | Optical turbidity (noisy) | [0,1] | Already sensor_state | Already migrated | No change |
| 11 | sensor_scatter | sensors | sensors["sensor_scatter"] | Optical scatter (noisy) | [0,1] | Already sensor_state | Already migrated | No change |

**Result: 0 migrations. 2 indices already sensor-derived. 10 indices remain truth-derived (8 with no sensor equivalent; 2 deferred).**

---

#### §1.1a — Index 0 (flow): Semantic Mismatch Detail

From `hydrion/physics/hydraulics.py` lines 261–263:

```python
diversion = float(np.clip(p.bf_flow_penalty * bf_cmd, 0.0, 1.0))
Q_out = float(np.clip(Q_processed * (1.0 - diversion), p.min_Q_Lmin, p.max_Q_Lmin))
```

And the state writes (lines 266–269):
```python
Q_out_Lmin      = Q_out,
q_processed_lmin = float(Q_processed),
```

**Q_out_Lmin and q_processed_lmin are NOT equivalent:**

- `q_processed_lmin` = `Q_processed` = `Q_in × (1 - bypass_fraction)` — flow entering the
  filter stages (excludes bypass; unaffected by backflush)
- `Q_out_Lmin` = `Q_processed × (1 - bf_flow_penalty × bf_cmd)` — net forward output after
  backflush flow diversion

During any backflush command (bf_cmd > 0): `Q_out_Lmin < q_processed_lmin`. The diversion
factor `bf_flow_penalty × bf_cmd` can be as large as 0.8 at full backflush command. The
difference is not negligible.

`FlowRateSensor` measures `truth_state["q_processed_lmin"]`. The obs index 0 reads
`truth_state["flow"]` = `Q_out_Lmin / Q_max_Lmin`. These diverge during backflush.

Substituting `flow_sensor_lmin / Q_max_Lmin` into index 0 would give the policy a *different*
signal during backflush events — one that does not reflect the backflush diversion. The policy
would see a higher-than-true forward output when backflush is active.

**Verdict: Semantic mismatch is real and operationally significant. DEFER.**

---

#### §1.1b — Index 1 (pressure): Semantic Mismatch Detail

From `hydrion/physics/hydraulics.py` lines 253–259:

```python
dP_pipe  = p.R_pipe  * Q_proc_m3s
dP_valve = R_valve   * Q_proc_m3s
P_out = 0.0
P_m3  = P_out  + dp_stage3
P_m2  = P_m3   + dp_stage2
P_m1  = P_m2   + dp_stage1
P_in  = P_m1   + dP_valve + dP_pipe
```

And the dp_total:
```python
dp_total = dp_stage1 + dp_stage2 + dp_stage3
```

**P_in and dp_total_pa are fundamentally different physical quantities:**

- `dp_total_pa` = `(R_m1 + R_m2 + R_m3) × Q_proc_m3s` — pure filter resistance signal;
  scales with fouling state only (for fixed flow)
- `P_in` = `dp_total + dP_valve + dP_pipe` = filter resistance + valve resistance + pipe
  resistance — the total system inlet pressure

`P_in` encodes the full system state: pump output, valve position, and filter fouling combined.
The bypass valve activates when `P_in > bypass_pressure_threshold_pa` (65,000 Pa). This
threshold is defined in absolute `P_in` terms, not `dp_total` terms.

If index 1 were replaced with `dp_sensor_kPa / dp_max_kPa`:
- The bypass activation signal would disappear from the observation
- The policy would not know when P_in is near the bypass threshold
- Bypass events would become unobservable from the normalized pressure channel

`dp_sensor_kPa` is a useful signal (it's the direct fouling indicator), but it does not
substitute for P_in in terms of policy-relevant content. The fouling state is already partially
encoded at index 2 (`clog` = `mesh_loading_avg`). Adding dp_sensor as a replacement for P_in
would degrade bypass observability without adding new information not already covered.

**Verdict: Semantic mismatch is real and policy-relevant. Replacement would remove bypass
signal from observation. DEFER.**

---

### 1.2 CCE / conical_cascade_env.py Path

**Method:** `ConicalCascadeEnv._obs()` in `hydrion/environments/conical_cascade_env.py`
**Note:** CCE uses `self._state` (internal physics dict, functionally equivalent to truth_state).
CCE has **no sensor_state layer**. DifferentialPressureSensor and FlowRateSensor are not
instantiated in CCE.

| Index | Label | Source | Physical quantity | Normalization constant | Sensor equivalent | Semantic match | Decision |
|---|---|---|---|---|---|---|---|
| 0 | q_in | _state | q_processed_lmin | / OBS_Q_MAX = 25.0 L/min | flow_sensor_lmin | Partial — same qty; different norm (25 vs 20 L/min) | **DEFER** |
| 1 | delta_p | _state | dp_total_pa / 1000 (kPa) | / OBS_DP_MAX = 150 kPa | dp_sensor_kPa | **Clean match — same physical quantity** | **DEFER — see §1.2a** |
| 2 | fouling_mean | _state | avg(fouling_frac_s1/s2/s3) | [0,1] direct | None | N/A | Keep truth |
| 3 | eta_cascade | _state | M5 cascade capture efficiency | [0,1] | None | N/A | Keep truth |
| 4 | C_in | _state | Inlet concentration | [0,1] | None | N/A | Keep truth |
| 5 | C_out | _state | Outlet concentration | [0,1] | None | N/A | Keep truth |
| 6 | E_field_norm | _state | voltage_norm (normalized V) | [0,1] | None | N/A | Keep truth |
| 7 | v_crit_norm | _state | v_crit_s3 / OBS_VCRIT_MAX | / 1.0 m/s | None | N/A | Keep truth |
| 8 | step_norm | episode counter | _step / _max_steps | [0,1] | N/A | Not a physical sensor | Keep as-is |
| 9 | bf_active | _state | bf_active binary | {0,1} | N/A | Actuator state | Keep truth |
| 10 | eta_PP | _state | Buoyant species capture | [0,1] | None | N/A | Keep truth |
| 11 | eta_PET | _state | Dense species capture | [0,1] | None | N/A | Keep truth |

**Critical observation documented:** CCE index 3 = `eta_cascade`; HydrionEnv index 3 =
`E_field_norm`. The two schemas are NOT obs12_v2 compatible with each other. This is
pre-existing divergence, not Phase 2 scope.

#### §1.2a — CCE Index 1: Clean Match, Still Deferred

CCE index 1 reads `dp_total_pa / 1000 / OBS_DP_MAX`. The Phase 1 dp sensor measures
`dp_total_pa`. This is the only index in either path where a sensor measurement physically
matches the observation quantity exactly.

However, CCE is deferred for infrastructure reasons independent of semantic match:

1. **No sensor_state layer in CCE.** Injecting sensor readings requires either adding a
   sensor_state dict to CCE (significant architecture change) or passing values in from
   outside the env (non-standard interface).
2. **ppo_cce_v2 benchmark integrity.** The canonical ppo_cce_v2 benchmark was trained on
   truth-derived CCE observations. Changing CCE obs — even at a clean-match index — would
   invalidate ppo_cce_v2 as a baseline before a replacement benchmark exists.
3. **No current Phase 3 substitute.** Phase 3 (M7 RL rebuild) has not started. Invalidating
   ppo_cce_v2 before Phase 3 produces a replacement is architecturally unsafe.

**Verdict: CCE observation migration is FULLY DEFERRED. Revisit after Phase 3 baseline is
established under the new schema.**

---

## 2. Resolution of Open Questions Q1–Q4

### Q1: Is Q_out_Lmin ≈ q_processed_lmin in all operating modes?

**Answer: NO.**

Evidence from `hydraulics.py` lines 261–263:
```python
diversion = float(np.clip(p.bf_flow_penalty * bf_cmd, 0.0, 1.0))
Q_out = float(np.clip(Q_processed * (1.0 - diversion), ...))
```

The two quantities differ by a factor of `(1 - bf_diversion)` which equals
`(1 - bf_flow_penalty × bf_cmd)`. With `bf_flow_penalty = 0.8` and `bf_cmd = 1.0`:
`Q_out = 0.2 × Q_processed` — a 5× difference at full backflush command.

During bypass (bypass_active = True): `Q_processed = Q_in × 0.70` (30% diverted).
`Q_out` then applies the backflush diversion on top of that. The two diverge in both
active bypass and active backflush modes.

Equivalence holds only during steady-state operation (bf_cmd = 0, bypass_active = False).
That equivalence is never guaranteed during an episode.

**Remaining uncertainty:** None. The hydraulics model is deterministic; the divergence
conditions are fully characterized.

---

### Q2: Is P_in or dp_total more informative for the RL policy?

**Answer: P_in (truth["pressure"]) is more informative for policy decisions; dp_total
(dp_sensor_kPa) is more informative for fouling state.**

Evidence:
- Bypass activation condition: `P_tentative > bypass_pressure_threshold_pa` where
  `P_tentative = R_forward × Q_in / 60000`. This is an inlet pressure threshold.
  The policy needs to know when P_in is near 65,000 Pa to anticipate bypass. `dp_total`
  does not encode pipe and valve contributions; a policy reading only dp_total cannot
  infer proximity to the bypass threshold from observation alone.
- Index 2 (`clog` = `mesh_loading_avg`) already encodes the fouling state that drives
  the filter resistance component of both P_in and dp_total. Replacing index 1 with
  dp_sensor would create partial redundancy with index 2 while removing the bypass-relevant
  absolute pressure information.
- If dp_total were added as an additional observation dimension (schema extension), both
  signals would be present simultaneously. But replacing P_in with dp_total is a net
  information loss for bypass decisions.

**Remaining uncertainty:** Whether an RL policy trained from scratch would find dp_total
more or less useful than P_in for backflush triggering in the M7 regime. This is an empirical
question for Phase 3. Phase 2 does not run RL training and cannot answer it.

---

### Q3: Is ppo_cce_v2 at risk if CCE observation changes?

**Answer: Yes, unambiguously.**

ppo_cce_v2 is the canonical benchmark artifact (500k steps, seed=42, obs12_v2 truth-derived
CCE observations). Any change to CCE's `_obs()` method that alters the values at any index
produces observations that the ppo_cce_v2 policy was not trained on. The policy behavior
becomes undefined; any performance metric is not comparable to the original benchmark.

There is currently no replacement benchmark for CCE. Phase 3 will produce one, but Phase 3
has not started and its observation schema is not yet fixed.

**Conclusion:** CCE obs changes are prohibited until Phase 3 produces a new CCE baseline
under the updated schema. This is an architectural integrity requirement, not a preference.

---

### Q4: Can a normalized sensor key be added to sensor_state to bridge the normalization gap?

**Answer: Yes technically, but it does not resolve the semantic mismatch.**

A `dp_sensor_norm` key could be written by `DifferentialPressureSensor.update()` as
`dp_sensor_kPa / dp_max_kPa`. This is a clean operation and does not violate any constraint.

However, normalization is not the obstacle. The obstacle is semantic:
- Index 1 represents inlet pressure (P_in); the sensor measures differential pressure
  (dp_total). These are different physical quantities regardless of normalization scale.
- Adding a normalized sensor key gives `sensor_fusion.py` a clean normalized float to
  read, but the physical meaning of the substitution remains incorrect.

**Conclusion:** A normalized sensor key in sensor_state is useful for logging and future
schema extension work, but does not justify migrating index 1. Can be implemented
independently as a logging enhancement without a schema version event.

---

## 3. Default Conservative Implementation Recommendation

### Minimum Safe Phase 2 Scope

**Recommendation: Zero observation index migrations. obs12_v2 preserved. Phase 2 closes
as a completed analysis phase, not a code-change phase.**

Rationale:

Both candidate indices in HydrionEnv have confirmed semantic mismatches that are
operationally significant (backflush divergence at index 0; bypass signal loss at index 1).
No clean migration exists without either accepting a semantic change or extending the schema
to 14D. The CCE path has no sensor layer and an active benchmark that must be protected.

The sensor readings (`dp_sensor_kPa`, `flow_sensor_lmin`) measure real and useful physical
quantities, but they do not correspond to the physical quantities currently at obs indices
0 and 1. They are available in `sensor_state`, logged in the step info dict, and accessible
to any future analysis. They are not wasted.

The correct next step for exposing these signals in RL is a deliberate schema extension
(Phase 2b), not a substitution into existing slots.

### Recommended Phase 2 Deliverables (Revised)

1. **This analysis document** — field audit, Q1-Q4 answers, version decision — written and
   committed. This is the implementation plan.
2. **No changes to sensor_fusion.py**, conical_cascade_env.py, or env.py observation path.
3. **Optional logging enhancement:** Add `dp_sensor_norm` to sensor_state as a normalized
   float (dp_sensor_kPa / dp_max_kPa) for logging convenience. This is zero-impact on
   schema and does not require a version event. Separate PR; not blocking.
4. **Phase 2b scoping note** recorded in this document (Section 8).

---

## 4. Schema / Version Assessment

### Can obs12_v2 Remain Valid?

**Yes. obs12_v2 is preserved.**

No index in either observation path has a migration decision of MIGRATE. All deferred
decisions preserve existing source keys and physical quantities. obs12_v2's index ordering
and semantic contract are unchanged.

### Version Bump Decision Table

| Scenario | Status |
|---|---|
| Index 0 (HydrionEnv flow) migrated to sensor | **NOT happening — semantic mismatch** |
| Index 1 (HydrionEnv pressure) migrated to sensor | **NOT happening — semantic mismatch** |
| CCE indices migrated | **NOT happening — no sensor layer + benchmark risk** |
| Schema extension to 14D (new indices for dp_sensor, flow_sensor) | **Not Phase 2 scope — Phase 2b** |

**Version outcome: obs12_v2 remains the stable observation contract for both HydrionEnv
and CCE. No version bump is warranted or permitted in Phase 2.**

### Minimum Acceptable Schema Strategy

Preserve obs12_v2 exactly as defined. The sensor readings available in sensor_state are
logged and available for analysis but do not enter the RL observation vector until a
deliberate schema extension phase defines the new dimensional contract, normalization
references, and validation criteria.

### Path-Specific Version Status

- **HydrionEnv / sensor_fusion.py:** obs12_v2 — unchanged
- **CCE / conical_cascade_env.py:** CCE-specific obs (not obs12_v2 in full semantic sense,
  already documented as divergent at index 3) — unchanged

---

## 5. File-by-File Implementation Plan

Given the zero-migration recommendation, the file changes are minimal.

### Files to Modify

**None required for the minimum safe scope.**

Optional (logging enhancement, not blocking):
- `hydrion/sensors/pressure.py` — add `dp_sensor_norm` write to sensor_state in `update()`
  if decided: `sensor_state["dp_sensor_norm"] = dp_sensor_kPa / dp_max_kPa`
- `hydrion/state/init.py` — add `dp_sensor_norm: 0.0` to init_sensor_state() if above implemented
- `configs/default.yaml` — add `dp_max_kPa` parameter under `sensors.pressure` if above implemented

### Files to Create

- `docs/orchestration/M6_Phase2/Phase2_M6_implementation_plan.md` — **this document**
- `docs/orchestration/M6_Phase2/obs_schema_audit.md` — optional standalone audit record
  (the field audit in Section 1 of this document satisfies this requirement; separate file
  is unnecessary)

### Files That Must Remain Untouched

The following files must not be modified as part of Phase 2:

| File | Reason |
|---|---|
| `hydrion/sensors/sensor_fusion.py` | obs12_v2 schema is stable; no migration occurs |
| `hydrion/environments/conical_cascade_env.py` | CCE obs deferred; ppo_cce_v2 protected |
| `hydrion/env.py` | Observation path unchanged; sensor logging already in place |
| `hydrion/physics/hydraulics.py` | Physics is unchanged |
| `configs/default.yaml` | No new observation parameters required |
| All existing test files | No behavior change to test against |

---

## 6. Validation Plan

Because no index migrations occur in Phase 2, the validation plan covers the zero-migration
confirmation tests and the optional logging enhancement.

### 6.1 Confirmation Tests (Minimum — No Code Change)

These tests already pass. They confirm the Phase 2 baseline is intact:

1. **Truth/sensor separation (existing):** tests 5 and 6 in `test_sensors_m6.py` confirm
   dp_sensor writes do not contaminate truth_state["dp_total_pa"] and flow_sensor writes
   do not contaminate truth_state["q_processed_lmin"]. Pass.

2. **obs12_v2 range (implicit in existing tests):** test 9 in `test_sensors_m6.py` confirms
   sensor_state keys are populated after env.step(). The observation returned by env.step()
   continues to come from build_observation(truth, sensors), which is unchanged.

3. **Divergence confirmation (existing):** tests 3 and 4 in `test_sensors_m6.py` confirm
   sensor readings diverge from truth values. The divergence is in sensor_state, not in obs.

### 6.2 If Optional Logging Enhancement Is Implemented (dp_sensor_norm)

4. **Normalization range test:** assert `0.0 <= sensor_state["dp_sensor_norm"] <= 1.0`
   across the operating envelope (Q_low to Q_peak, fouling 0–100%). Run envelope sweep.

5. **Key presence test:** after env.reset(), assert `"dp_sensor_norm" in env.sensor_state`.

### 6.3 Diagnostic ppo_cce_v2 Rollout (Informational Only)

Run ppo_cce_v2 policy on unchanged HydrionEnv (obs12_v2 truth). Record:
- Mean episode return
- Mean dp_sensor_kPa vs mean dp_total_pa/1000 (divergence from truth)
- Mean flow_sensor_lmin vs mean q_processed_lmin (divergence from truth)

This is informational only — it characterizes how large the sensor/truth gap would be
if future phases migrate these indices. It does NOT constitute a new benchmark.

---

## 7. Benchmark Integrity Note

**ppo_cce_v2 is the canonical HydrOS benchmark artifact.** It was trained under the
following conditions:
- Environment: ConicalCascadeEnv
- Observation schema: obs12 (CCE-specific; truth-derived all 12 dimensions)
- 500,000 steps, seed=42, d_p=1.0 µm submicron regime
- Committed and logged; reproducible from artifact

**Any policy rollout on a modified observation schema is diagnostic only.** It does not
represent a comparison against ppo_cce_v2 because the policy was trained on different inputs.
The return degradation is a measure of observation-change sensitivity, not policy quality.

Specifically:
- If Phase 2b (schema extension) introduces new obs dimensions, any rollout with the
  new obs must be treated as a new benchmark run, not a ppo_cce_v2 comparison.
- Phase 3 (M7 RL rebuild) must produce a new baseline trained on the Phase 2b schema
  before benchmark comparisons resume.
- ppo_cce_v2 retains its status as the M5 physics benchmark for truth-state observation.
  It cannot be invalidated by Phase 2 because Phase 2 makes no changes to CCE obs.

---

## 8. Risks and Failure Conditions

### Risk 1 — Schema Dishonesty

**Description:** Substituting `flow_sensor_lmin / Q_max_Lmin` into obs index 0 and labeling
the schema obs12_v2 (unchanged). The index now carries a different physical quantity
(q_processed vs Q_out) but the version label suggests nothing changed.

**Why it's dangerous:** Downstream RL policies trained on the "unchanged" obs12_v2 after
this substitution would behave differently (especially during backflush) compared to policies
trained before it, but the version label provides no warning. Reproducing ppo_cce_v2
comparisons would be silently invalid.

**Mitigation:** Zero migrations in Phase 2. Any future source change at an existing index
requires an explicit version bump with documented rationale.

---

### Risk 2 — Semantic Substitution Mistake

**Description:** Replacing `truth["pressure"]` (P_in) with `dp_sensor_kPa / scale` under
the reasoning that "both are pressure signals and the normalization can be adjusted."

**Why it's dangerous:** P_in and dp_total_pa are correlated but encode different information.
The policy's bypass decision depends on P_in approaching the threshold. After substitution,
the bypass-relevant pressure information is absent from the observation. The policy may fail
to anticipate bypass events even if its clog observation is accurate.

**Mitigation:** Q2 is resolved: P_in is more informative for bypass. Index 1 is deferred.

---

### Risk 3 — Accidental CCE Benchmark Invalidation

**Description:** Making an "innocuous" change to `conical_cascade_env.py` (e.g., adding a
sensor_state attribute or changing a normalization constant) that alters the obs values
returned by `_obs()`.

**Why it's dangerous:** ppo_cce_v2 rollouts would no longer reproduce the benchmark return,
with no obvious explanation. The artifact's reference status would be silently compromised.

**Mitigation:** CCE is on the "must not modify" list. Any future CCE obs change is a new
scoped phase with an explicit benchmark replacement plan.

---

### Risk 4 — Hidden truth/sensor Contamination via Normalization

**Description:** Adding a `dp_sensor_norm` to sensor_state computed inside
`DifferentialPressureSensor.update()` that reads from and writes to a truth_state key by
mistake (e.g., writing `dp_sensor_norm` into `truth_state` instead of `sensor_state`).

**Why it's dangerous:** Would violate the truth_state / sensor_state separation silently.
Tests may not catch it if the contaminated key is not the exact key tested.

**Mitigation:** Any optional logging enhancement must include an explicit test asserting
the new key is present in `sensor_state` and NOT present in `truth_state`.

---

### Risk 5 — Misleading Backward-Compat Assumptions

**Description:** Assuming that because indices 0 and 1 are "close numerically" to
flow_sensor_lmin / Q_max and dp_sensor_kPa / P_scale, policies trained on obs12_v2 truth
would "approximately" transfer to a sensor-derived obs. Using this reasoning to skip the
version bump.

**Why it's dangerous:** Numerical closeness does not equal semantic equivalence. The
backflush divergence at index 0 can be 5× during active backflush. The bypass-signal absence
at index 1 can cause catastrophic bypass mismanagement. "Close enough" reasoning is the
path to undetected behavioral regression.

**Mitigation:** Documented in this plan. If any future phase proposes numeric proximity as
a substitute for semantic equivalence analysis, reference this section.

---

## 9. Acceptance Criteria

Phase 2 implementation is complete when all of the following are satisfied:

1. Field audit completed and committed (Section 1 of this document — DONE)
2. Q1–Q4 resolved with code evidence (Section 2 — DONE)
3. Zero index migrations confirmed as the correct decision for obs12_v2 preservation
   (Section 3 — DONE)
4. obs12_v2 explicitly declared preserved in this document (Section 4 — DONE)
5. All existing 97 tests continue to pass (no code changed; passes by definition)
6. HydrionEnv `_observe()` continues to call `build_observation(truth, sensor)` unchanged
7. CCE `_obs()` continues to read from `self._state` unchanged
8. No new sensor index in obs12_v2 (confirmed by no changes to sensor_fusion.py)
9. Phase 2b scoping note recorded (see below)
10. This implementation plan committed to docs/orchestration/M6_Phase2/

---

## 10. Phase 2b Scoping Note (Forward Reference)

The correct mechanism for introducing `dp_sensor_kPa` and `flow_sensor_lmin` into RL
observation is a schema extension — adding new dimensions, not substituting existing ones.

A Phase 2b spec should address:
- Whether to extend to 14D (add 2 new sensor indices) or restructure to a new schema entirely
- Normalization references for the new dimensions (dp_max_kPa, Q_max_Lmin)
- Whether the extension applies to HydrionEnv only, CCE only, or both
- Benchmark strategy: Phase 3 RL training under the new 14D schema as the new baseline
- Version label: obs14_v1 or equivalent

Phase 2b is NOT a prerequisite for Phase 3. Phase 3 can proceed with obs12_v2 (all
truth-derived) as the starting schema, and the sensor extension can be layered in during
or after Phase 3 if empirical evidence shows the additional dimensions improve policy quality.

---

## Summary

**Phase 2 analysis result:** Zero migrations. obs12_v2 preserved. No code changes required.

Both HydrionEnv candidates (index 0: flow, index 1: pressure) have confirmed semantic
mismatches between the available sensor readings and the current observation quantities.
The CCE path has no sensor infrastructure and carries an active benchmark that must be
protected.

The sensor readings from Phase 1 are real, valid, and logged. They are not used in RL
observation at this time. The correct path to using them is a deliberate schema extension
(Phase 2b), not a substitution that would silently change the physical meaning of existing
observation indices.

**Phase 3 can proceed on obs12_v2 (truth-derived) as the starting schema.**

**Schema/version recommendation:** obs12_v2 preserved. No bump.
**Minimum safe implementation scope:** Analysis only. No code changes to observation path.
**Plan status:** Ready for execution (execution = commit this document and close Phase 2).

---

*Implementation plan written 2026-04-13. Analysis is complete.*
*Phase 2 execution: commit this document. Phase 3 is now unblocked.*
