# M5 → M6 Transition Package

**Document type:** Authoritative transition record
**Status:** FINAL
**Date:** 2026-04-13
**Branch:** main | Head: 9c767b9
**Prepared by:** HydrOS Co-Orchestrator (Claude Code session)

---

## 1. M5 Final Status

M5 is **closed**. All ten tasks (A–J) in the M5 Task Execution Report are complete.

| Area | Status |
|---|---|
| S3 bed-depth physics fix | COMPLETE — L = d_c enforced; RT saturation eliminated |
| Wall-slide trajectory semantics | COMPLETE |
| CCE telemetry fix (E_field_norm key) | COMPLETE — commit 634dcb4 |
| HydrionEnv observation key fix (E_norm → E_field_norm) | COMPLETE — commit 634dcb4 |
| Policy selection fix (app.py branching) | COMPLETE — commit 634dcb4 |
| ppo_cce_v2 training (500k steps, seed=42, d_p=1.0µm) | COMPLETE |
| Benchmark evaluation (PPO vs Heuristic vs Random) | COMPLETE — both criteria PASS |
| Per-stage eta audit | COMPLETE — saturation residual ruled out for submicron regime |
| Test suite | COMPLETE — 88 tests passing |
| Species-morphology rendering (console) | COMPLETE |

**One item remains open at M5 closeout:**
- `hydrion/service/app.py` line 25: `_PPO_CCE_MODEL_PATH` still points to `ppo_cce_v1.zip`.
  Addressed as Task 1 of Phase 0 before M6 execution begins.

---

## 2. Canonical Documents Now in Force

The following documents are authoritative for all work from this transition point forward.

| Document | Path | Role |
|---|---|---|
| Locked blueprint | `docs/reports/blueprint/official_locked_blueprint_postM5.md` | System ground truth — architecture, gaps, maturity, constraints |
| M5 task execution report | `docs/reports/blueprint/M5_TASK_EXECUTION_REPORT.md` | M5 closeout authority |
| Phase chain (Phase 0–5) | `docs/reports/blueprint/next+phase_execution_postM5/` | Execution sequence for M6 and M7 |
| Phase chain integrity audit | `docs/orchestration/M5_Closeout/phase_chain_integrity_audit.md` | Wrapper readiness verification |

No other document supersedes the locked blueprint. The blueprint was reconciled on 2026-04-13
against the post-M5-merge main branch state.

---

## 3. Benchmark Truth Now Established

**Canonical benchmark artifact:** `models/ppo_cce_v2.zip`

| Parameter | Value |
|---|---|
| Training steps | 500,000 |
| Seed | 42 |
| Particle diameter | d_p = 1.0 µm (submicron benchmark regime) |
| Observation schema | obs12_v2 (truth_state, perfect information) |
| Action schema | act4_v1 |
| Reward version | phase1_v1 |
| DEP threshold | Q_crit = 7.7 L/min / pump_cmd = 0.22 |
| Artifacts | ppo_cce_v2.zip, ppo_cce_v2_meta.json, ppo_cce_v2_vecnorm.pkl |
| Created | 2026-04-13 09:11 |

**Benchmark results (simulation):**

| Comparison | Ratio | Criterion | Result |
|---|---|---|---|
| PPO / Random | 1.87× | > 1.5× | PASS |
| PPO / Heuristic | 1.96× | > 1.5× | PASS |

**ppo_cce_v1 status:** archived, benchmark-invalid. Trained under RT-saturated conditions
(d_p = 10.0 µm, pre-S3 bed-depth fix) — η_cascade was artificially capped at 1.000 regardless
of action. No valid benchmark claims may reference ppo_cce_v1.

All future Phase 3 comparisons must use ppo_cce_v2 as the prior-state anchor.

---

## 4. What M5 Proved

1. **Physics pipeline is capture-sensitive.** Post-S3 bed-depth fix, η_cascade varies meaningfully
   with flow rate and voltage. RT formula (Rajagopalan-Tien 1976) is correctly applied to the
   liquid-phase filtration context. The saturation regime that made ppo_cce_v1 invalid has been
   eliminated.

2. **PPO learns a non-trivial control policy.** Under submicron benchmark conditions, PPO
   outperforms an independently coded heuristic by 1.96× and random baseline by 1.87×, both
   exceeding the 1.5× PASS threshold. The policy is not a degenerate constant.

3. **Negative DEP is confirmed active for PP, PE, PET.** Re[K(ω)] ≈ −0.48 for all three target
   polymers in water (RSC 2025, literature-calibrated). nDEP capture is physically grounded and
   the model correctly reflects the DEP threshold at V = 500V, tip_radius = 3µm.

4. **Observation schema obs12_v2 is functional and versioned.** Benchmark training and evaluation
   both used obs12_v2 consistently. The schema version is recorded in all artifact metadata.

5. **The CCE environment is instrumentable and benchmark-stable.** Telemetry, policy evaluation,
   and per-stage capture efficiency readout all work cleanly post-M5.

---

## 5. What M5 Did Not Prove

These are not gaps to close before M6 begins — they are known, documented simulation-reality
boundaries that must not be overclaimed.

**1. Deployment-valid observation.** obs12_v2 reads from truth_state (perfect information). No
sensor noise, drift, or latency is present. Real hardware has none of this. PPO has never been
trained or evaluated under realistic sensing.

**2. Hardware flow compatibility.** PPO's learned operating point stays at or below Q ≤ 7.7 L/min
(pump_cmd ≤ 0.22). Residential laundry drains operate at 12–15 L/min. PPO's optimal control regime
is incompatible with real-world inlet flow without hardware architecture changes (e.g., bypass
splitting, post-drain buffer tank). This is an unresolved hardware architecture question — not
a software gap.

**3. Deployment-representative particle regime.** d_p = 1.0 µm is a physics-validation benchmark
chosen to activate DEP sensitivity. It is not the dominant laundry microplastic distribution
(fibers 100–5000 µm, fragments 10–500 µm). All M5 results are valid within their stated regime
and must be labelled accordingly.

**4. Full drain cycle coverage.** Episodes are 400 steps × 0.1s = 40s simulated. A real laundry
drain cycle runs 180–480s. Current training covers 8–20% of a real cycle. Behaviour across
full-cycle dynamics is untested.

**5. Reward mission-alignment.** Reward version phase1_v1 is a working training signal, not a
deployment-aligned objective function. Capture mass, energy cost, maintenance timing, and
backflush incentives have not been formally weighted against real operational targets.

**6. Dense-phase scope limitation.** The system targets dense-phase particles (ρ > 1.0 g/cm³:
PET, PA, PVC, biofilm-coated fibers). Buoyant polymers (PP, PE, ρ < 1.0) are not captured and
are outside system scope. This is a design boundary, not a defect, but must not be ignored in
any deployment claim.

---

## 6. Post-M5 Phase Chain Status

| Phase | Title | Status | Gate |
|---|---|---|---|
| Phase 0 | M5 Closeout | READY TO EXECUTE | No predecessor |
| Phase 1 | M6 Core Sensor Realism | READY AFTER PHASE 0 | Blocked by Phase 0 |
| Phase 2 | M6 Observation Handoff | READY AFTER PHASE 1 | Blocked by Phase 1 |
| Phase 3 | M7 Baseline RL Rebuild | READY AFTER PHASE 2 | Blocked by Phase 2 |
| Phase 4 | M7 Reward Shaping | READY AFTER PHASE 3 | Blocked by Phase 3 |
| Phase 5 | M7 Robustness + Generalization | READY AFTER PHASE 4 | Blocked by Phase 4 |

All six phases have been audited and corrected. Explicit `Blocked by:` gates are in place.
The chain is sequentially enforced — no phase may begin until its predecessor's acceptance
criteria are met.

Phase files are located at: `docs/reports/blueprint/next+phase_execution_postM5/`

---

## 7. Wrapper Readiness Status

**READY.**

The phase chain integrity audit (`docs/orchestration/M5_Closeout/phase_chain_integrity_audit.md`,
2026-04-13) verified:

- All six required fixes applied and confirmed in the phase files
- All four recommended additions applied and confirmed
- Explicit dependency gating across the full Phase 0–5 sequence
- ppo_cce_v2 named as comparison baseline in Phase 3
- DEP flow-rate compatibility check mandated in Phase 3 return report
- Energy parameter guard condition explicit in Phase 4
- High-flow stress case quantified (pump_cmd ≥ 0.85, Q ≈ 12–15 L/min) in Phase 5
- Simulation-only curriculum qualification required in Phase 5 return report

A master wrapper prompt covering Phase 0 through Phase 5 may be constructed and executed against
current main branch state.

---

## 8. Remaining Open Blockers Before and During M6

### Before Phase 0 execution

- `hydrion/service/app.py:25` — `_PPO_CCE_MODEL_PATH` = ppo_cce_v1.zip (must be updated to v2)
- `_PPO_CCE_VECNORM_PATH` = ppo_cce_v1_vecnorm.pkl (same, must be updated)

### During M6 (Phase 1 and Phase 2)

- `hydrion/sensors/` — no ΔP or flow sensor exists; must be built from scratch in Phase 1
- `hydrion/state/init.py` — sensor_state["dp_sensor_kPa"] and sensor_state["flow_sensor_lmin"]
  must be initialized before Phase 2 can read them
- `conical_cascade_env.py` — observation construction is independent from sensor_fusion.py;
  obs12_v3 must be applied to both CCE and HydrionEnv paths separately. CCE obs index 3 =
  eta_cascade (not E_field_norm) — this semantic difference must not be overwritten
- YAML `configs/default.yaml` — sensor parameters (dp_noise_kPa, dp_drift_rate_kPa_per_step,
  dp_drift_max_kPa, dp_fouling_gain, dp_latency_steps, flow_noise_frac, flow_bias_std_lmin)
  do not yet exist

### Persistent architectural constraints (remain throughout M6 and M7)

- Physics modules must NOT read from sensor_state
- Sensor modules must NOT write to truth_state
- obs12_v2 must remain loadable for backward compatibility with ppo_cce_v2 evaluation
- All benchmark claims must carry the submicron regime label
- PPO flow-rate hardware incompatibility must be noted in any deployment-direction statement

---

## 9. Formal Orchestration Logging

**Formal orchestration logging begins after this transition document.**

This package represents the last pre-M6 state capture. From this point:

- All phase executions are to be logged against the canonical phase files in
  `docs/reports/blueprint/next+phase_execution_postM5/`
- Any deviation from a phase's acceptance criteria must be recorded before proceeding to the
  next phase
- Artifact metadata (meta.json files) is the authoritative record of schema version, training
  configuration, and benchmark regime for every trained model
- No benchmark claim may be made against ppo_cce_v1 or any artifact produced under a
  non-current obs schema
- This document is the M5→M6 handoff boundary. Any question of "what was true at M5 closeout"
  is answered here first, then in the locked blueprint if deeper detail is required

---

*M5 is closed. M6 begins after Phase 0 execution and acceptance.*
