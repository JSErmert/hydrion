# M8 — Achievements: Deployment Gap Quantification

**Milestone:** M8 — Deployment Gap Quantification / Privileged-Channel Ablation
**Closed:** 2026-04-14
**Governed by:** HydrOS x AlignFlow Orchestration Contract v3.1

---

## 1. One-Line Summary

M8 produced the first deployment-realistic HydrionEnv policy under an 8D observation
schema stripped of all physics truth channels, measured a positive deployment gap of
+21.2% vs the full 14D baseline, and formally corrected the actuator-channel
misclassification that had inflated the perceived deployment information deficit since M7.

---

## 2. Core Technical Achievements

| Artifact | What it is |
|----------|-----------|
| `obs8_deployment_v1` | First deployment-realistic HydrionEnv observation schema — 8D, all channels hardware-available (4 actuator feedback + 4 sensor-derived). Formally specified in M8.3, implemented in M8.6. |
| `obs_schema` selector in `HydrionEnv` | `obs_schema: str = "obs14_v1"` parameter with `_VALID_SCHEMAS` validation and `_observe()` dispatch. Default unchanged — all existing callers unaffected. Extensible by adding one function and one dict entry per new schema. |
| `build_observation_obs8()` | Additive parallel function in `hydrion/sensors/sensor_fusion.py`. Existing `build_observation()` untouched. Assembles 8D obs from `truth_state` (actuator indices) and `sensor_state` (sensor indices) with matched normalizations. |
| `ppo_hydrienv_v2` | Second canonical HydrionEnv policy. obs8_deployment_v1, seed=42, 360k steps, fresh VecNormalize. Named-constant VecNorm path separation prevents schema swap. Full metadata with channel taxonomy and confound declaration. |
| `eval_ppo_hydrienv_v2.py` | Three-agent evaluation script (v2 vs v1 vs random). Schema lock assertions on both loaded models. AC9/AC10/AC11 checks. Four-scenario dispatch per M8.3 Section 9. Prohibited-claims printed in header. MVC-3 confound block printed in output. |
| `tests/test_obs8_schema.py` | 17 tests: obs8 shape, actuator channel traces (T2a–d), sensor channel traces (T3a–d), 200-step rollout validity, obs14_v1 backward-compatibility gate, CCE 12D guard, unknown schema rejection, obs8 ≠ obs14 shape regression. |

---

## 3. Major Conceptual Achievement

**The actuator-channel misclassification was formally corrected.**

M7 metadata (`ppo_hydrienv_v1_meta.json`) recorded `"truth_channels": "0-9"`, labeling
all of obs14_v1 indices 0–9 as "privileged / not available at hardware deployment."

M8 inspection of `sensor_fusion.py` established:

| Indices | True class | Deployment availability |
|---------|-----------|------------------------|
| 0–5 | physics_truth | NOT available — requires running physics simulation |
| 6–9 | actuator_feedback | AVAILABLE — controller-issued commands, always self-known |
| 10–13 | sensor_derived | AVAILABLE — hardware-measurable |

The deployment gap is the removal of **6 channels** (0–5), not 10. A "remove all truth
channels" ablation that also removed actuator feedback (6–9) would have tested a scenario
more restrictive than any realistic deployment configuration, conflating information loss
with a deliberate design choice. M8 corrected the ablation target before training.

The three-class taxonomy (physics_truth / actuator_feedback / sensor_derived) is now the
canonical channel classification standard for all HydrOS observation design going forward.

---

## 4. Benchmark and Validation Achievements

| Check | Result |
|-------|--------|
| Test suite | **128/128 passing** — 111 existing (zero regressions) + 17 new obs8 schema tests |
| AC9: v2 return > random | **PASS** — 841.839 vs 577.689 |
| AC10: actuator non-constant; sensor active | **PASS** — act_std=0.8751; sensor_std=0.7358 |
| AC11: v1 return within 5% of canonical 694.639 | **PASS** — exact canonical match (694.639) |
| v1 lineage | `ppo_hydrienv_v1.zip`, `_vecnorm.pkl`, `_meta.json` — untouched throughout M8 |
| CCE lineage | `ppo_cce_v2`, `conical_cascade_env.py` — untouched throughout M8 |
| obs14_v1 schema | `build_observation()` — untouched; additive pattern preserved backward compatibility exactly |

---

## 5. Runtime Outcome Achievements

**Training:** 360,448 steps, seed=42, runtime ~10:22

```
Metric                        v2 (obs8)   v1 (obs14)    Random
--------------------------------------------------------------
Mean return                    841.839      694.639     577.689
Std return                       3.294       27.718      11.857
Deployment gap (v2 - v1) %      +21.2%
v2 vs random %                  +45.7%
```

**Scenario classification (M8.3 Section 9 binding):**
v2 approximately equals or exceeds v1 (v2 ≥ v1)

Must NOT conclude: "deployment-ready" or "deployment gap solved."

**Schema lock confirmed at runtime:**
```
[SCHEMA LOCK OK] observation_space.shape = (8,)  (obs8_deployment_v1)
[SCHEMA LOCK OK] v2 obs space: (8,)  |  [SCHEMA LOCK OK] v1 obs space: (14,)
```

---

## 6. Governance and Workflow Achievements

**Research layer worked as designed.**
M8.1R.2 (research brief) and M8.1R.4 (sources map) grounded the ablation methodology
in peer-reviewed literature (Kaelbling et al. 1998 for POMDP actuator retention;
Andrychowicz et al. 2021 for VecNorm refit requirement) before any code was written.
Architecture decisions were labeled as such — no fabricated citations.

**Execution document prevented scope drift.**
M8.3 defined the four-scenario interpretation table, methodology validity conditions
(MVC-1–4), prohibited claims, and AC11 halt rule before implementation began. The eval
script enforced these as runtime output, not post-hoc editorial decisions.

**Prohibited-claims discipline held under a positive result.**
The eval output printed "Do NOT conclude: deployment-ready or deployment gap solved" when
v2 exceeded v1. The governance structure was designed before results were known — it held
symmetrically regardless of gap direction.

**The additive implementation plan eliminated regression risk.**
The two-checkpoint test protocol (Step 5 after env changes, Step 12 after all scripts)
was specified in M8.5 before execution. Both checkpoints passed clean. The parallel
`build_observation_obs8()` pattern made obs14_v1 behavior provably unchanged by test
rather than by inspection.

**Manual-runtime constraint was handled cleanly.**
The 600-second autonomous execution ceiling was identified as a platform constraint in
M8.1 (carry-forward risks) and M8.6 (insight). Training was executed at the maximum
feasible autonomous step count (360k), with the deviation explicitly documented in M8.7
and assessed as non-blocking for closeout.

---

## 7. What M8 Unlocked

**Sensor calibration (M9 — highest priority).**
obs8 in-simulation viability is now established. The primary confound blocking
hardware-relevant interpretation is placeholder sensor noise. Calibrating
`dp_drift_rate`, `dp_fouling_gain`, and `flow_calibration_bias_std` would produce the
first deployment gap estimate with hardware-realistic noise. All RL infrastructure exists.

**Domain randomization.**
With obs8 schema fixed and v2 trained, randomizing sensor noise parameters over
calibrated ranges is the natural next step for sim-to-real robustness. Follows
calibration; does not require new observation infrastructure.

**Reward alignment.**
The obs8 policy works, removing the open question that previously blocked reward redesign:
"is it worth reshaping the reward if the observation space might collapse policy
performance anyway?" That question is now resolved. Reward work can proceed without
observation-space uncertainty as a confound.

**obs4_sensor_only ablation.**
An obs4 schema (indices 10–13 only, 4D) would isolate the policy contribution of the
4 actuator command channels specifically. Now that obs8 is established as a reference,
obs4 has a clear comparison point.

**Frontend telemetry alignment.**
The obs8 result — a policy that operates without physics truth channels — makes the
dual rendering of truth vs sensed behavior more meaningful. The console can now
visualize what a deployed policy actually perceives vs what the simulation knows.

---

## 8. Boundaries Preserved

M8 does **not** establish:

- **Deployment readiness.** No hardware was used. No real sensors. No device calibration.
- **Hardware validity.** Sensor channels in obs8 are governed by placeholder noise, not
  measured physical behavior. The gap result is internally valid, not hardware-predictive.
- **Sensor-only sufficiency.** obs8 contains 4 actuator command channels. The result
  does not test what a pure-sensor (4D) observation would achieve.
- **That channels 0–5 are unnecessary.** The policy functions without them under current
  conditions; this does not establish they carry no policy-useful information.
- **That obs8 is globally superior to obs14.** The positive gap may reflect dimensional
  reduction benefit or noise regime, not information content.
- **That the deployment gap is solved.** Calibration pending. Single seed. 360k ≠ 500k.
  No domain randomization. No cross-seed variance estimate.

---

## 9. Resume / Portfolio Summary

- **Designed and implemented a deployment-realistic RL observation schema** (`obs8_deployment_v1`) for a microplastic extraction digital twin, formally classifying all 14 observation channels by hardware deployment availability and correcting a misclassification that had inflated the perceived deployment information deficit since the prior milestone.

- **Trained a second canonical reinforcement learning policy** (`ppo_hydrienv_v2`) under a reduced, deployment-realistic 8D observation space using PPO (Stable-Baselines3), achieving a mean episodic return of 841.8 — a +21.2% positive deployment gap vs the 14D privileged-channel baseline — with zero safety violations across all evaluation episodes.

- **Built a three-way comparative evaluation pipeline** with schema lock assertions, named VecNormalize path separation, and automated four-scenario interpretation dispatch, enforcing pre-specified prohibited-claim constraints and methodology validity checks at runtime regardless of result direction.

- **Expanded the test suite from 111 to 128 tests** with zero regressions, including channel-trace tests, rollout validity tests, backward-compatibility guards, and schema isolation assertions across three environment types.

- **Executed a full AlignFlow milestone chain** (refinement → research brief with source traceability → execution document → implementation plan → execution → closeout → insight capture) on a governed engineering timeline, producing a formally closed milestone record with explicit confound accounting and next-milestone inheritance state.
