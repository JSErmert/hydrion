# M7 Achievements

**Milestone:** M7 — Phase 3 RL Rebuild (HydrionEnv Baseline)  
**Closed:** 2026-04-13

---

## 1. Milestone Objective

M7's objective was to produce the first canonical HydrionEnv RL policy baseline trained
under the `obs14_v1` observation schema — a clean, reproducible, metadata-documented
reference artifact. M7 was not a reward redesign, robustness expansion, or deployment
exercise. Its purpose was to establish a fixed starting point for all future HydrionEnv
policy work, with correct schema version-lock, seeding, and a complete governance record.

---

## 2. Major Technical Achievements

**First canonical HydrionEnv baseline under obs14_v1.**  
`ppo_hydrienv_v1` is the first PPO policy trained on HydrionEnv with the 14D obs schema
active. It establishes the benchmark reference for HydrionEnv — the counterpart to
`ppo_cce_v2` for the CCE path.

**New dedicated train/eval script path.**  
`train_ppo_hydrienv_v1.py` and `eval_ppo_hydrienv_v1.py` are clean, purpose-built scripts
with schema locks, correct VecNormalize handling, safety wrapping, and metadata output.
They replace the pre-M7 stale scripts (`train_ppo.py`, `train_ppo_v15.py`, `eval_ppo.py`)
as the authoritative HydrionEnv RL entry points.

**Schema version-lock discipline.**  
Both scripts assert `env.observation_space.shape == (14,)` at startup before any training
infrastructure is initialized. Schema drift silently breaking a training run is no longer
possible.

**Sensor-channel activation confirmed in RL evaluation.**  
Sensor-derived indices 12 (`flow_sensor_norm`) and 13 (`dp_sensor_norm`) were confirmed
non-zero and varying during evaluation. Dual-channel distinction values (2.56 and 3.73
normalized units) confirm the sensor readings differ meaningfully from their truth
counterparts — the sensor pipeline established in M6.1 is reachable in RL context.

**Protected CCE benchmark lineage.**  
`ppo_cce_v2`, `train_ppo_cce.py`, and `eval_ppo_cce.py` were verified untouched throughout
M7 implementation. Two independent, non-comparable benchmark paths now exist with clean
lineage.

---

## 3. Validation / Proof Points

| Proof point | Value |
|-------------|-------|
| PPO mean return | 694.639 (vs random baseline 586.028) |
| AC1–AC14 | All 14 PASS |
| Test suite | 111/111 pass |
| obs[12] sensor channel | mean=-0.3578, std=1.2785 (non-zero, varying) |
| obs[13] sensor channel | mean=-0.2648, std=1.2427 (non-zero, varying) |
| max\|obs[12]−obs[0]\| | 2.5559 (dual-channel distinction confirmed) |
| max\|obs[13]−obs[1]\| | 3.7331 (dual-channel distinction confirmed) |
| Schema lock | `[SCHEMA LOCK OK] shape = (14,)` confirmed before training |
| CCE diff | `git diff models/ppo_cce_v2*` empty |
| Training steps | 501,760 from seed=42 |

---

## 4. Governance / Process Achievements

M7 completed a 9-document governance chain:

| Document | Purpose |
|----------|---------|
| `M7.0_refinement_prompt.md` | Scoping refinement |
| `M7.1R.1_research_brief.md` | Research grounding (POMDP, sim-to-real, privileged info) |
| `M7.1R.2_sources_map.md` | Citation registry: source-derived vs internal decisions |
| `M7.1_execution_document_prompt.md` | Activation prompt for execution document |
| `M7.2_baseline_RL_execution_document.md` | Binding execution document with 14 ACs |
| `M7.3_planning_refinement_prompt.md` | Refinement prompt with stale-script disposition |
| `M7.4_implementation_plan.md` | Full implementation plan |
| `M7.5_implementation_execution_prompt.md` | Execution activation prompt |
| `M7.6_execution_report.md` | Full execution record with AC pass/fail |
| `M7.7_closeout_prompt.md` + `M7.8_closeout_record.md` | Formal closeout |

Every design decision is either sourced (with author/year/DOI) or explicitly labeled as an
internal architecture decision. No fabricated citations.

---

## 5. Accepted Limitations

| Limitation | Status |
|------------|--------|
| Not deployment-ready | Truth channels 0–9 are simulation-only; `deployment_ready=false` in metadata |
| Truth channels still privileged | Policy depends on inputs unavailable at hardware |
| Calibration-pending sensor noise | `dp_drift_rate`, `dp_fouling_gain` are placeholder values |
| No domain randomization | Deferred to post-calibration phase |
| Not comparable to ppo_cce_v2 | Different env, physics, reward, schema — not interchangeable |

These are documented, accepted states — not defects.

---

## 6. What M7 Unlocked for M8

M8 can now:

- **Measure improvement.** Any future HydrionEnv policy has a fixed, seeded baseline
  (694.639 mean return, 0 safety violations) to compare against.

- **Ablate truth channels.** With a working obs14_v1 pipeline, M8 can train a sensor-only
  variant (remove indices 0–9) and measure the performance gap between privileged and
  sensor-only regimes.

- **Reshape reward with a reference.** M8 reward redesign work can measure progress as
  improvement over `ppo_hydrienv_v1`, rather than against an undefined baseline.

- **Expand calibration-aware realism.** When sensor calibration data is available, M8 can
  replace placeholder noise parameters and retrain, using `ppo_hydrienv_v1` as the
  pre-calibration comparison point.

- **Maintain CCE independence.** The CCE/`ppo_cce_v2` path remains clean and can evolve
  independently through M8 if a CCE-specific objective is scoped.
