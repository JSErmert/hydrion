Operate under the HydrOS Co-Orchestrator Execution Contract.

Do not ask for confirmation or permissions.
Proceed autonomously unless you encounter a true architectural conflict, missing file, or destructive ambiguity that would risk schema integrity or benchmark comparability.

Objective:
Transition the RL observation vector from truth_state direct reads to sensor_state reads for the sensor-available dimensions. Version the observation schema explicitly. This is the M6 observation handoff — the step that converts RL training from "perfect information" to "realistic information."

Context:
- Phase 1 (M6 core sensor realism) is complete.
- sensor_state["dp_sensor_kPa"] and sensor_state["flow_sensor_lmin"] now exist with realistic noise/drift.
- Current observation schema is obs12_v2 (12-dimensional, all from truth_state or truth-derived values).
- Trained artifact ppo_cce_v2 used obs12_v2 / truth_state — this is the truth-state benchmark baseline.
- Any model trained after this phase uses obs_v3 and cannot be directly compared to ppo_cce_v2 without explicit annotation.
- Schema versioning is mandatory — any schema change must produce a new version identifier; no silent updates.

Constraints:
- Create obs12_v3 (or obs_N_v3 if dimensions change) — do NOT overwrite obs12_v2.
- Document which observation dimensions come from sensor_state vs truth_state.
- Do not retrain a full 500k-step model in this phase — that is Phase 3.
- Do not change reward logic — that is Phase 4.
- Preserve obs12_v2 for backward compatibility with ppo_cce_v2 evaluation.
- Update meta.json schema field for any training runs under the new schema.

Tasks:
1. Define obs12_v3 (or new schema). CRITICAL SCOPING NOTE: HydrionEnv and CCE build observations through
   different code paths. Do NOT conflate them.
   - HydrionEnv path: hydrion/sensors/sensor_fusion.py (sensor_fusion builds the observation)
   - CCE path: hydrion/environments/conical_cascade_env.py (CCE constructs its own observation internally,
     independently of sensor_fusion.py)
   - CCE obs index 3 = eta_cascade (not E_field_norm as in HydrionEnv). This semantic difference is
     architectural, not a bug. Do not overwrite CCE index 3 with sensor_fusion logic.
   - obs12_v3 must be applied to BOTH paths, but the update is performed in the respective files for each env.
     Audit conical_cascade_env.py explicitly to identify which CCE observation dimensions currently read from
     truth_state and can be redirected to sensor_state equivalents.
   - Minimum for both paths: flow dimension → sensor_state["flow_sensor_lmin"] normalized
   - Minimum for both paths: pressure dimension → sensor_state["dp_sensor_kPa"] normalized
   - Remaining dimensions: document whether they remain truth_state (as explicit placeholders).
   - Version string: "obs12_v3" or equivalent.

2. Update sensor_fusion.py (HydrionEnv path) to build the v3 observation, clearly marking each dimension's source.
   Update conical_cascade_env.py (CCE path) separately for the same sensor-state dimensions.

3. Add a schema registry or version constant so training infrastructure can validate schema at load time.

4. Update ConicalCascadeEnv to accept obs schema version parameter (default to v3 going forward; v2 for legacy runs).

5. Run a training smoke test (1000 steps) under obs12_v3 to confirm:
   - Observation is numerically valid (no NaNs, all in [0,1]).
   - Schema version is correctly recorded in meta.json.
   - Agent receives noisy/drifted observations as expected.

6. Confirm ppo_cce_v2 can still be loaded and evaluated under obs12_v2 path (backward compatibility test).

7. Document obs12_v3 semantics explicitly:
   - Dimension table: index, name, source (sensor_state or truth_state), unit, normalization.
   - Mark any truth_state dimensions as [PLACEHOLDER — sensor not yet modeled].

Acceptance criteria:
- obs12_v3 (or equivalent) defined, documented, and implemented.
- At least pressure and flow dimensions draw from sensor_state (noisy/drifted).
- Training smoke test passes under new schema.
- ppo_cce_v2 backward compatibility preserved — v2 evaluation still runnable.
- meta.json for any new run encodes correct schema version.
- No silent schema changes — version bump is visible in all artifact metadata.

Blocked by: Phase 1 (M6 core sensor realism complete — dp_sensor and flow_sensor in sensor_state).

Return a concise decision report with:
- obs12_v3 dimension table (index, name, source, normalization)
- which dimensions are now sensor-mediated vs truth_state placeholders
- schema version constant location
- smoke test result
- backward compatibility confirmation for ppo_cce_v2
- blockers before Phase 3 (baseline RL rebuild under M6)
