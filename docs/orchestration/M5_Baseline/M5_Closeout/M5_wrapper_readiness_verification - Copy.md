  HYDROS FOUNDER/VENTURE BLUEPRINT INPUT PACKAGE

  Extraction date: 2026-04-12 | Reconciled: 2026-04-13 (post-M5-merge) | Branch surveyed: main (explore/conical-cascade-arch merged at commit 2bb5cd6) | Evidence base: repository code + context
  docs

  ---
  1. SYSTEM IDENTITY SNAPSHOT

  Mission: Build a software-first digital twin of a handheld microplastic extraction device for laundry outflow, validated
  against peer-reviewed physics, progressing toward hardware-ready autonomous control logic.

  Simulated device: A three-stage conical cascade filtration system. Each stage applies graduated mechanical filtration
  (coarse → medium → fine mesh) combined with dielectrophoretic (nDEP) electrostatic capture and autonomous backflush
  maintenance. Target input: residential laundry wastewater. Target output: filtered effluent + captured dense-phase
  microplastics (PET, PA/nylon, PVC).

  Intended real-world target: A consumer or near-consumer device for laundry appliance integration. Scale is handheld.
  Operating envelope: 5–20 L/min continuous flow, 3-stage mesh filtration, electrostatic field-assisted capture of particles
   ≥5 µm.

  Present development phase: Phase 1.5 — "Research Console + Realism Backbone." Physics pipeline through M4 (hydraulics,
  clogging, backflush, electrostatics, particle realism) is implemented. M5 sensor realism is absent. RL training
  infrastructure exists with one trained artifact (500k steps, CCE). No hardware device exists. No lab-bench calibration
  data has been ingested.

  What it is NOT:
  - Not a hardware prototype or physical device
  - Not a calibrated model (all device geometry is [DESIGN_DEFAULT] — not measured from a physical build)
  - Not a production RL controller (observations come from truth_state, not sensor_state; M5 sensor realism required first)
  - Not a general-purpose simulation framework
  - Not validated against external lab data

  ---
  2. LOCKED CONSTRAINTS

  Constraint: Truth/sensor state separation — physics modules write ONLY to truth_state; sensor modules write ONLY to
    sensor_state; no cross-contamination
  Evidence: docs/context/02_ARCHITECTURE_CONSTRAINTS.md §1; enforced structurally in hydrion/env.py and
  hydrion/state/init.py
  Why it matters: Violating this collapses the measurement realism model and makes future sensor noise injection impossible
    without rearchitecting
  ────────────────────────────────────────
  Constraint: Immutable pipeline order: Hydraulics → Clogging → Electrostatics → Particles → Sensors → Observation → Safety
  Evidence: docs/context/02_ARCHITECTURE_CONSTRAINTS.md §2; implemented in hydrion/env.py step() method
  Why it matters: Semantic ordering is load-bearing — changing it alters what the reward signal observes
  ────────────────────────────────────────
  Constraint: Observation schema version lock — current schema is obs12_v2 (12-dim, all [0,1]); no silent
  additions/removals;
    version bump required for any change
  Evidence: hydrion/sensors/sensor_fusion.py (sole observation authority); models/ppo_cce_v2_meta.json encodes schema as
    "obs_schema": "obs12_v2"
  Why it matters: Any trained checkpoint is inseparable from the schema it was trained on; schema drift silently corrupts
    inference
  ────────────────────────────────────────
  Constraint: Action space: 4D continuous [0,1] — [valve_cmd, pump_cmd, bf_cmd, node_voltage_cmd]
  Evidence: hydrion/env.py:156–163; hydrion/environments/conical_cascade_env.py:213
  Why it matters: Hardware actuator mapping is 1:1 to these four dimensions; changes require hardware redesign
  ────────────────────────────────────────
  Constraint: Realism sequence ordering — must implement in order: 1 Hydraulics, 2 Clogging, 3 Backflush, 4 Electrostatics,
  5
     Particles, 6 Sensors, 7 RL+reward
  Evidence: docs/context/09_REALISM_ROADMAP.md; docs/context/06_LOCKED_SYSTEM_CONSTRAINTS.md §F
  Why it matters: Each stage depends on the previous for physically meaningful outputs
  ────────────────────────────────────────
  Constraint: Dense-phase only (§E) — buoyant fraction (PP, PE) passes through uncaptured; system efficiency is defined over

    dense-phase only
  Evidence: docs/context/06_LOCKED_SYSTEM_CONSTRAINTS.md §E (locked 2026-04-10); hydrion/physics/particles.py
  Why it matters: Product scope constraint — defines what the device claims to capture and what it does not
  ────────────────────────────────────────
  Constraint: η_nominal reference definition — η_system(d=10µm, Q=13.5 L/min, clean filter, dense-phase)
  Evidence: docs/context/06_LOCKED_SYSTEM_CONSTRAINTS.md §G (locked 2026-04-10); hydrion/physics/particles.py
  Why it matters: All external efficiency claims must cite this definition; "99%" without qualification is not a valid
  output
  ────────────────────────────────────────
  Constraint: Safety as external wrapper only — safety logic must NOT be embedded in physics modules
  Evidence: docs/context/02_ARCHITECTURE_CONSTRAINTS.md §5; hydrion/wrappers/shielded_env.py, hydrion/safety/shield.py
  Why it matters: Physics remains pure and testable in isolation
  ────────────────────────────────────────
  Constraint: YAML-driven configuration — no hardcoded physics constants
  Evidence: docs/context/02_ARCHITECTURE_CONSTRAINTS.md §4; all physics params in configs/default.yaml
  Why it matters: Reproducibility and parameter sweep capability depend on this
  ────────────────────────────────────────
  Constraint: Deterministic simulation — same seed → same result
  Evidence: hydrion/runtime/seeding.py; tests/test_reproducibility_smoke.py
  Why it matters: Validation and training comparisons are meaningless without this
  ────────────────────────────────────────
  Constraint: RL exploratory until M6 sensor realism — current RL observation uses truth_state directly; trained policies
  are
     physics-optimal under perfect observation, not deployment-realistic
  Evidence: docs/superpowers/plans/2026-04-12-ppo-cce-training-pipeline.md §Critical Constraints §1
  Why it matters: A trained policy is not deployment-ready until the observation is rebuilt on sensor_state (M6)

  ---
  3. CURRENT IMPLEMENTATION INVENTORY

  Hydraulics

  - Status: Implemented (M1 complete)
  - Files: hydrion/physics/hydraulics.py (HydraulicsModel v2)
  - Does: Pump-system curve intersection (quadratic solve); area-normalized clog resistance; passive bypass with hysteresis
  (65 kPa threshold); per-stage ΔP outputs; splits Q_in into processed + bypass flow
  - Does NOT: Model transient surges; temperature/viscosity variation; flow sensor noise
  - Confidence: HIGH — 10/10 M1 validation tests pass; no calibration data ingested (all params are YAML design defaults)

  Clogging

  - Status: Implemented (M1 + M1.5 fixes applied)
  - Files: hydrion/physics/clogging.py (CloggingModel v3)
  - Does: Per-stage decomposed fouling (cake/bridge/pore/irreversible); non-monotonic capture curve; passive shear removal;
  component sum normalization; dep_exponent=1.0 (bistable fix confirmed in configs/default.yaml:63)
  - Does NOT: Provide hardware-validated kinetics parameters; model fiber vs fragment clogging behavior differently
  - Confidence: HIGH structurally; MEDIUM numerically (parameters are first-pass estimates)

  Backflush

  - Status: Implemented (M2, merged into M1 sprint)
  - Files: hydrion/env.py (state machine), hydrion/physics/clogging.py (recovery)
  - Does: 3-pulse square-wave state machine (0.4s pulse, 0.25s spacing, 9s cooldown); per-component recovery (cake 35%,
  bridge 20%, pore 8%); diminishing returns (80% factor per burst); recirculated effluent source
  - Does NOT: Model clean water service mode with calibrated data; provide real-time backflush efficiency measurement
  - Confidence: HIGH structurally; MEDIUM quantitatively (recovery percentages not lab-validated)

  Electrostatics

  - Status: Implemented (M3)
  - Files: hydrion/physics/electrostatics.py (M3 model); hydrion/physics/m5/dep_ndep.py, hydrion/physics/m5/field_models.py
  (M5 CCE research physics)
  - Does (HydrionEnv/M3): Cylindrical radial field model; InletPolarizationRing (30%) + OuterWallCollectorNode (70%);
  residence time coupling; E_capture_gain passed to particles; V_max_realism=2500V, hard clamp 3000V
  - Does (CCE/M5): nDEP force balance from first principles using Clausius-Mossotti factors sourced from peer-reviewed
  literature (Pethig 1992, Gascoyne & Vykoukal 2002, Brandrup & Immergut 1999); analytical conical field; per-material CM
  factors for PP, PE, PET
  - Does NOT: Incorporate conductivity dependence; per-stage separate electrostatic parameters; particle-size dependence in
  E_capture_gain for M3 path
  - Confidence: M3 model: HIGH structurally, MEDIUM quantitatively. M5/CCE: HIGH for physics formulas (peer-reviewed), LOW
  for geometry inputs (all [DESIGN_DEFAULT])

  Particles

  - Status: Implemented (M4 for HydrionEnv; M5/CCE for research track)
  - Files: hydrion/physics/particles.py (ParticleModel v3 — M4); hydrion/physics/m5/particle_dynamics.py,
  hydrion/physics/m5/capture_rt.py (CCE M5)
  - Does (M4/HydrionEnv): Density classification (dense 70%, neutral 15%, buoyant 15%); Stokes settling; per-stage
  size-dependent capture (S1: d/500^1.5, S2: d/100^1.2, S3: d/5^0.8); buoyant pass-through; η_nominal locked at ~0.854 at
  reference conditions; electrostatic boost from E_capture_gain
  - Does (M5/CCE): RT 1976 single-collector efficiency formula; per-particle trajectory integration; nDEP force balance;
  gravity-driven radial velocity; apex capture detection
  - Does NOT: Full PSD integration (bulk d=10µm used); fiber vs fragment shape differentiation; material-dependent charge
  behavior
  - Confidence: M4: HIGH structurally. M5/CCE: HIGH for physics formulas; LOW for geometry (all [DESIGN_DEFAULT])

  Sensors

  - Status: Partial — optical sensors present; critical sensors absent
  - Files: hydrion/sensors/optical.py, hydrion/sensors/sensor_fusion.py
  - Does: Turbidity proxy; scatter proxy; noise injection; correct separation from truth_state; builds obs12_v2 observation
  vector
  - Does NOT: Model differential pressure sensor; flow sensor; fouling-induced sensor drift; latency; calibration bias;
  dropout. Camera proxy exists but has no AI interpretation layer
  - Confidence: What exists is solid. The gap is substantial — the two most critical sensors for hardware deployment (ΔP and
   flow) are absent

  RL / Training

  - Status: Infrastructure complete; ppo_cce_v2 trained and benchmark-evaluated; convergence criteria PASS (simulation)
  - Files: hydrion/train_ppo_cce.py, hydrion/eval_ppo_cce.py, hydrion/train_ppo.py, models/ppo_cce_v2.zip,
  models/ppo_cce_v2_meta.json, models/ppo_cce_v2_vecnorm.pkl (canonical benchmark artifact, Apr-13); ppo_cce_v1 artifacts
  also present but trained in RT-saturated regime (d_p=10µm, S3 bed-depth bug present) — not valid for benchmark comparison
  - Does: PPO + VecNormalize + ShieldedEnv training on CCE; deterministic seed (42); 500k-step training on submicron
  benchmark regime (d_p=1.0µm, below S3 collector diameter); ppo_cce_v2 evaluated against three baselines — both
  convergence criteria pass: PPO/Random η ratio 1.87× (threshold >1.5×), PPO/Heuristic η ratio 1.96× (threshold >1.2×);
  three-baseline evaluation script (PPO vs Heuristic vs Random); checkpoint cadence (10k steps); meta.json schema versioning
  - Does NOT: Use sensor_state in observation (uses truth_state directly — perfect information); have Phase 2 reward
  (capture mass, energy, species-weighted efficiency); have curriculum for disturbance generalization; have a confirmed
  per-stage η breakdown to distinguish PPO's η_cascade=1.000 from residual RT saturation (per-stage audit pending)
  - Confidence: Infrastructure HIGH; ppo_cce_v2 benchmark PASS (simulation-only); deployment validity requires M6 sensor
  realism; ppo_cce_v1 is archived and must not be cited as a valid benchmark artifact

  Reward Logic

  - Status: Two parallel implementations (HydrionEnv vs CCE); both Phase 1; neither is final

  ┌───────────────────┬───────────────────────────────────────────────────────────────────────────────────┬─────────────┐
  │        Env        │                                  Reward formula                                   │    Phase    │
  ├───────────────────┼───────────────────────────────────────────────────────────────────────────────────┼─────────────┤
  │ HydrionEnv        │ w_processed_flow × q_proc/Q_nom - w_pressure × max(0, P-0.5)² - w_fouling ×       │ Phase 1     │
  │                   │ max(0, ff-0.4) - w_bypass × q_byp/Q_nom - w_backflush × bf_active                 │ interim     │
  ├───────────────────┼───────────────────────────────────────────────────────────────────────────────────┼─────────────┤
  │ ConicalCascadeEnv │ eta_cascade + flow_bonus - dp_penalty - volt_penalty                              │ Phase 1 v1  │
  └───────────────────┴───────────────────────────────────────────────────────────────────────────────────┴─────────────┘

  - Does NOT: Directly reward capture efficiency in HydrionEnv (major gap); reflect energy usage; use sensor_state
  - Confidence: Structurally correct; neither is production-aligned

  Scenario Runner

  - Status: Implemented
  - Files: hydrion/scenarios/runner.py, hydrion/scenarios/examples/ (2 scenarios: baseline_nominal, backflush_recovery_demo)
  - Does: Loads YAML scenario definitions; applies initial state; computes per-step flow/particle profiles; emits event
  markers (threshold_crossing, backflush_start/end, bypass_start/end, disturbance_start/end); records full step history
  - Does NOT: Run against CCE directly (scenario runner uses HydrionEnv; CCE used directly in /api/scenarios/run); have
  high-fouling or high-flow stress scenarios
  - Confidence: HIGH

  Console / Telemetry

  - Status: Partial but significantly more complete than docs claim

  DOC CLAIM (docs/context/03_REPO_MAP.md): "static scaffold — no live data binding — no telemetry integration"
  CODE VERIFIED: Console has live API binding, scenario playback, ConicalCascadeView animated visualization, PlaybackBar
  with event marker navigation, RunLibrarySidebar, TopTelemetryBand, CoreMetricsPanel, ValidationPanel

  - Files: apps/hydros-console/src/ (30 source files); FastAPI backend at hydrion/service/app.py
  - Does: Scenario selection + execution via /api/scenarios/run; step-by-step playback; event marker navigation; animated
  particle stream rendering; conical cascade physics visualization; run library
  - Does NOT: Show live RL training progress; surface η_nominal with proper reference qualifiers; display differential
  pressure sensor data (no ΔP sensor); show PPO vs random policy comparison live

  Validation / Tests

  - Status: Substantial (88 tests across 17 test files)
  - Files: tests/ (17 files); hydrion/validation/ (4 validation protocols)
  - Does: Hydraulics unit tests; clogging unit tests; particle unit tests (including Stokes, density fractions, mass
  balance, η_nominal); electrostatics tests; sensor tests; CCE environment tests; shield environment smoke tests; training
  infrastructure tests (8 tests); reproducibility smoke test; full validation protocol v2 (stress matrix, envelope sweep,
  mass balance, recovery latency)
  - Does NOT: Have automated benchmark reporting against calibration targets; test sensor_state vs truth_state divergence
  under noise; have tests for scenario playback data integrity; test PPO convergence in CI

  Data Logging / Observability

  - Status: Implemented (run logging); partial (no live streaming)
  - Files: hydrion/logging/ (writer.py, artifacts.py, paths.py); hydrion/rendering/ (observatory.py, time_series.py,
  anomaly_detector.py)
  - Does: Per-run artifact logging; episode history recording; TensorBoard integration in training; anomaly detection;
  time-series visualization; video generation
  - Does NOT: Stream training metrics to console in real-time; produce standardized benchmark reports for cross-run
  comparison

  ---
  4. INTERFACE TRUTH TABLE

  Interface: Action space
  Doc claim: 4D continuous [0,1]: valve, pump, bf, voltage
  Code reality: CODE VERIFIED — spaces.Box(low=0, high=1, shape=(4,)) in both HydrionEnv and CCE
  Verified file(s): hydrion/env.py:156; hydrion/environments/conical_cascade_env.py:213
  Risk if misunderstood: None — consistent
  ────────────────────────────────────────
  Interface: Observation space (HydrionEnv)
  Doc claim: 12D, schema labeled obs12_v1 in env.py comments but obs12_v2 in sensor_fusion.py
  Code reality: FIXED (Task E, commit 634dcb4) — env.py:322 now writes E_field_norm (v2 key) to truth_state; matches
    sensor_fusion.py:42. Obs index 3 now correctly reflects electrostatic field strength. Regression test added
    (test_e_field_norm_obs3_nonzero_after_voltage in test_env_api.py).
  Verified file(s): hydrion/env.py:322 (confirmed E_field_norm); hydrion/sensors/sensor_fusion.py:14,42
  Risk if misunderstood: None — resolved. Previously HIGH risk; now closed.
  ────────────────────────────────────────
  Interface: Observation space (CCE)
  Doc claim: "obs12_v2 compatible with HydrionEnv" (docstring)
  Code reality: MISMATCH — CCE obs index 3 = eta_cascade; HydrionEnv obs index 3 = E_field_norm. Shape matches (12,) but
    semantics differ
  Verified file(s): hydrion/environments/conical_cascade_env.py:150,158,641
  Risk if misunderstood: HIGH — a model trained on CCE cannot be directly applied to HydrionEnv despite shape compatibility
  ────────────────────────────────────────
  Interface: Policy selection (/api/run)
  Doc claim: policy_type field accepted
  Code reality: CODE VERIFIED — ppo_cce path uses CCE + ShieldedEnv + loaded PPO; all other values use HydrionEnv + random
    actions
  Verified file(s): hydrion/service/app.py:109,129–157
  Risk if misunderstood: FIXED (Task F, commit 634dcb4) — /api/run now branches telemetry reads on _use_ppo_cce, reading
    CCE state directly (cce._state) when in ppo_cce mode. Previously all logged step data was stale HydrionEnv initial
    state; now reflects actual simulation. Remaining open item: _PPO_CCE_MODEL_PATH in app.py:25 still points to
    ppo_cce_v1.zip; must be updated to ppo_cce_v2.zip before ppo_cce service path uses the correct model.
  ────────────────────────────────────────
  Interface: Trained model checkpoint
  Doc claim: None claimed in docs
  Code reality: TWO ARTIFACTS — ppo_cce_v1.zip (Apr-12, trained at d_p=10µm in RT-saturated regime; benchmark-invalid) and
    ppo_cce_v2.zip (Apr-13, submicron regime, d_p=1.0µm, seed=42; canonical benchmark artifact). v2 evaluated: PPO/Random
    1.87×, PPO/Heuristic 1.96×, both criteria PASS.
  Verified file(s): models/ppo_cce_v2_meta.json (benchmark_regime: submicron); models/ppo_cce_v1_meta.json (archived)
  Risk if misunderstood: ppo_cce_v1 is NOT a valid benchmark artifact — trained before S3 bed-depth fix, RT-saturated
    regime. Any benchmark claim must cite ppo_cce_v2. app.py:25 _PPO_CCE_MODEL_PATH still references v1 (open item).
  ────────────────────────────────────────
  Interface: Scenario runner env
  Doc claim: Not specified
  Code reality: CCE used directly in /api/scenarios/run; HydrionEnv used in /api/run
  Verified file(s): hydrion/service/app.py:279–283
  Risk if misunderstood: Two different envs behind two different endpoints — comparison is not apples-to-apples
  ────────────────────────────────────────
  Interface: Available scenarios
  Doc claim: "Partially built (2 scenarios exist)"
  Code reality: CODE VERIFIED — baseline_nominal.yaml, backflush_recovery_demo.yaml
  Verified file(s): hydrion/scenarios/examples/
  Risk if misunderstood: Limited scenario coverage; no high-fouling or high-flow stress scenarios
  ────────────────────────────────────────
  Interface: Sensor fusion
  Doc claim: sensor_fusion.py is sole observation authority
  Code reality: CODE VERIFIED for CCE. HydrionEnv uses a different internal _observe() method
  Verified file(s): hydrion/sensors/sensor_fusion.py; hydrion/env.py:459
  Risk if misunderstood: HydrionEnv and CCE build their observations differently — they are not interchangeable
  ────────────────────────────────────────
  Interface: dep_exponent fix (M1.5)
  Doc claim: Documented as "FIXED"
  Code reality: CODE VERIFIED — dep_exponent: 1.0 in configs/default.yaml:63
  Verified file(s): configs/default.yaml
  Risk if misunderstood: Fixed — clean-start fouling growth now works
  ────────────────────────────────────────
  Interface: Safety wrapper
  Doc claim: Two implementations documented as a known risk
  Code reality: CONFIRMED — hydrion/safety/shield.py (SafetyConfig dataclass) and hydrion/wrappers/shielded_env.py
    (ShieldedEnv gym wrapper)
  Verified file(s): docs/context/08_VALIDATION_AND_SAFETY.md §8
  Risk if misunderstood: Duplication risk noted in docs; functional but the canonical safety logic is not singular

  ---
  5. REALISM ROADMAP STATE

  Stage 1 — Hydraulics

  - Maturity: COMPLETE (M1, 2026-04-09)
  - Prerequisites: N/A (first stage)
  - Blockers: No functional blockers. Quantitative calibration against hardware ΔP data outstanding
  - Validation dependency: 10/10 M1 tests pass. No hardware calibration curve ingested
  - Downstream wait: No — clogging and backflush built on top and are operational

  Stage 2 — Clogging

  - Maturity: COMPLETE (M1 + M1.5 fixes)
  - Prerequisites: Hydraulics ✓
  - Blockers: None functional. Component kinetics parameters (dep_rate_s1/s2/s3) are first-pass estimates
  - Validation dependency: Tests pass. No hardware fouling progression data ingested
  - Downstream wait: No

  Stage 3 — Backflush

  - Maturity: COMPLETE (M2, merged with M1)
  - Prerequisites: Hydraulics ✓, Clogging ✓
  - Blockers: None functional. Recovery percentages (cake 35%, bridge 20%, pore 8%) are unvalidated estimates
  - Validation dependency: State machine behavior tested; quantitative recovery not bench-validated
  - Downstream wait: No

  Stage 4 — Electrostatics

  - Maturity: COMPLETE at abstraction level (M3 in HydrionEnv; M5 first-principles in CCE)
  - Prerequisites: Hydraulics ✓, Clogging ✓, Backflush ✓
  - Blockers: Geometry parameters (r_inner_m, r_outer_m, stage_volume_L, electrode gaps) are all [DESIGN_DEFAULT]. No bench
  measurement
  - Validation dependency: Physics formulas are peer-reviewed and verified. Device-specific constants require physical
  hardware measurement before numbers are meaningful
  - Downstream wait: Conditional — physics structure is correct; quantitative claims require hardware

  Stage 5 — Particle Realism (Selectivity)

  - Maturity: PARTIAL — M4 (HydrionEnv) and M5 (CCE) both implement particle selectivity at different fidelity levels
  - Prerequisites: All prior stages ✓
  - Blockers: Full PSD integration deferred; fiber vs fragment shape effects absent; dense particle geometry
  (stage_height_m, rho_dense_kgm3) are placeholders
  - Validation dependency: η_nominal locked at ~0.854 (d=10µm, Q=13.5 L/min). No bench measurement to validate against
  - Downstream wait: Sensors can begin (M6) — particle module is functionally sufficient for this purpose

  Stage 6 — Sensors

  - Maturity: ABSENT for critical sensors (ΔP, flow); PARTIAL for optical (turbidity, scatter)
  - Prerequisites: All prior stages ✓
  - Blockers: No differential pressure sensor; no flow sensor; no drift/latency/fouling modeling. These are the sensors that
   would actually be on the device
  - Validation dependency: This stage is the prerequisite for production-realistic RL — without it, observation =
  truth_state and trained policy has no deployment validity
  - Downstream wait: YES — RL (Stage 7) should not be treated as deployment-ready until M6 is complete

  Stage 7 — RL + Reward

  - Maturity: PARTIAL — infrastructure complete, Phase 1 reward only; ppo_cce_v2 trained and benchmark-evaluated
  (simulation); deployment validity blocked on M6 sensor realism
  - Prerequisites: Stages 1–5 satisfied; Stage 6 (sensors) NOT satisfied
  - Blockers: (1) No Phase 2 reward (capture mass as primary signal); (2) observation uses truth_state not sensor_state;
  (3) no curriculum for disturbance generalization
  - Validation dependency: Three-baseline evaluation complete on ppo_cce_v2 (PPO/Random 1.87×, PPO/Heuristic 1.96×,
  both PASS). M6 must be completed before RL results have deployment validity. Per-stage η audit at PPO operating point
  pending (confirm η_cascade=1.000 is not residual saturation).
  - Downstream wait: RL training can continue for research benchmarking; should NOT be positioned as a deployable controller

  ---
  6. VALIDATION AND SAFETY STATUS

  Existing tests (88 total, 17 test files)

  ┌──────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────┐
  │            Test file             │                                     Coverage                                      │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ test_hydraulics.py               │ HydraulicsModel v2 behavior                                                       │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ test_clogging.py                 │ CloggingModel v3, dep_exponent fix                                                │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ test_electrostatics.py           │ ElectrostaticsModel M3                                                            │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ test_particles.py                │ ParticleModel M4: Stokes, density fractions, η_nominal, mass balance, buoyant     │
  │                                  │ pass-through (16 tests)                                                           │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ test_sensors.py                  │ OpticalSensorArray basic                                                          │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ test_env_api.py                  │ HydrionEnv Gymnasium API contract                                                 │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ test_conical_cascade_env.py      │ CCE environment                                                                   │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ test_field_models.py             │ M5 analytical field models                                                        │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ test_particle_dynamics.py        │ M5 ParticleDynamicsEngine                                                         │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ test_particle_engine_env.py      │ CCE + particle engine integration                                                 │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ test_particle_scenario_runner.py │ CCE scenario runner integration                                                   │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ test_artifact_schema.py          │ Run artifact schema                                                               │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ test_cce_training_infra.py       │ PPO training pipeline (8 tests: shield aliases, reward, randomized reset,         │
  │                                  │ training smoke, API fallback + happy path)                                        │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ test_shield_env_smoke.py         │ ShieldedEnv smoke                                                                 │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ test_shield_integrity.py         │ Shield integrity                                                                  │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ test_reproducibility_smoke.py    │ Deterministic replay                                                              │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ test_logging_smoke.py            │ Logging infrastructure                                                            │
  ├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ test_validation_protocol_v2.py   │ Full validation suite (stress matrix, envelope sweep, mass balance, recovery      │
  │                                  │ latency) — 8 tests                                                                │
  └──────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────┘

  Implicit validation mechanisms

  - η_nominal locked and tested to be deterministic and independent of fouling (test_eta_nominal_*)
  - Mass balance verified: C_out ≤ C_in, capture efficiency ∈ [0,1]
  - Reproducibility: same seed → same rollout (tested)
  - Physical plausibility: bounded state variables confirmed by envelope sweep

  Missing validation coverage

  - Sensor/truth separation under noise: No test verifies that sensor_state diverges from truth_state by a realistic
  magnitude under noise
  - Fouling progression curve: No test validates that fouling reaches 70% capacity within a specified number of steps at
  nominal flow (M1.5 acceptance criterion — documented but not automated)
  - Backflush recovery quantification: No test measures actual post-backflush fouling reduction percentage against the spec
  (cake 35%, bridge 20%, pore 8%)
  - Electrostatics on/off capture delta: No test verifies measurable capture improvement when electrostatics enabled vs
  disabled
  - PPO convergence: Training smoke test runs 1000 steps only; no CI test confirms 500k-step policy beats baselines
  - Cross-scenario generalization: No automated test for behavior under unseen disturbance conditions

  Failure conditions modeled

  - Pressure hard limit → episode termination (ShieldedEnv)
  - Clog hard limit → episode termination (ShieldedEnv)
  - Bypass activation (hydraulics, passive)
  - Irreversible fouling accumulation above 70% threshold

  Failure conditions still missing

  - Catastrophic mesh rupture (no model)
  - Pump cavitation / flow collapse (no model)
  - Electrostatic arc / voltage limit trip (voltage clamped to 3kV but no failure mode)
  - Sensor failure / dropout (no model)

  ---
  7. TECHNICAL GAP MAP

  Simulation Realism Gaps

  Gap: Sensor realism absent (M6) — no ΔP sensor, no flow sensor, no drift/latency
  Why it matters: Without sensor realism, the RL observation is perfect information; trained policies have zero deployment
    validity
  Dependency chain: Blocks Stage 7 (production RL); requires hardware build for calibration
  Evidence: docs/context/04_CURRENT_ENGINE_STATUS.md §9
  Severity: CRITICAL
  ────────────────────────────────────────
  Gap: All device geometry is [DESIGN_DEFAULT] — cone dimensions, electrode gaps, mesh specs, polymer fractions are
    placeholder
  Why it matters: Physics formulas are correct; the numbers driving them are not measured from hardware
  Dependency chain: Requires physical device build and characterization
  Evidence: hydrion/environments/conical_cascade_env.py:57–90 — 12+ [DESIGN_DEFAULT] tags
  Severity: CRITICAL for credibility
  ────────────────────────────────────────
  Gap: Buoyant fraction (PP/PE) uncaptured — product scope constraint
  Why it matters: PP and PE are ~45% of microplastic mass in laundry outflow; device does not capture them
  Dependency chain: Separate upstream treatment required; out of current scope
  Evidence: docs/context/06_LOCKED_SYSTEM_CONSTRAINTS.md §E
  Severity: MAJOR — affects market size claims
  ────────────────────────────────────────
  Gap: [RESOLVED — Task E, commit 634dcb4] HydrionEnv obs index 3 E_norm/E_field_norm key mismatch
  Resolution: env.py:322 now writes E_field_norm (v2 key); sensor_fusion.py:42 reads E_field_norm. Obs index 3 is no longer
    silently zero. Regression test added to test_env_api.py. No remaining action required.
  Severity: CLOSED
  ────────────────────────────────────────
  Gap: PSD integration deferred — bulk d=10µm for all efficiency calculations
  Why it matters: Real laundry outflow has a full particle size distribution; system behavior at other sizes is not
  simulated
    in the reward loop
  Dependency chain: M4.5 / M5 scope item
  Evidence: docs/context/04_CURRENT_ENGINE_STATUS.md §8
  Severity: MODERATE

  Control / RL Gaps

  Gap: CCE obs schema differs from HydrionEnv despite "obs12_v2 compatible" claim
  Why it matters: Models trained on CCE cannot be applied to HydrionEnv; the two environments are not interchangeable
  Dependency chain: Any cross-environment comparison is invalid
  Evidence: conical_cascade_env.py:158 vs sensor_fusion.py:42
  Severity: MAJOR
  ────────────────────────────────────────
  Gap: Phase 1 reward does not optimize capture efficiency (HydrionEnv)
  Why it matters: The reward maximizes processed flow throughput; a policy could score well while capturing poorly
  Dependency chain: Must be corrected before policy quality can be claimed
  Evidence: docs/context/04_CURRENT_ENGINE_STATUS.md §14
  Severity: MAJOR
  ────────────────────────────────────────
  Gap: [RESOLVED — Task D, 2026-04-13] ppo_cce_v2 evaluated against three baselines; both criteria pass
  Resolution: PPO/Random η ratio 1.87× (threshold >1.5×, PASS); PPO/Heuristic η ratio 1.96× (threshold >1.2×, PASS).
    Mean η_cascade: PPO=1.000, Heuristic=0.511, Random=0.535. PPO discovered DEP threshold: pump_cmd ≤ 0.22 keeps
    Q ≤ 7.7 L/min, activating nDEP capture. Heuristic locked pump=0.7 exceeds threshold regardless of voltage.
  Remaining caveat: PPO η_cascade=1.000 requires per-stage η audit to confirm it reflects genuine nDEP optimum rather
    than residual RT saturation. This is a simulation-internal verification item, not a gap in the benchmark comparison.
  Evidence: M5 Task Execution Report §Task D; models/ppo_cce_v2_meta.json
  Severity: CLOSED (benchmark); MINOR (per-stage audit pending)
  ────────────────────────────────────────
  Gap: No curriculum / disturbance generalization
  Why it matters: Agent trained on randomized fouling may not generalize to high-flow stress or rapid clogging scenarios
  Dependency chain: M6+ scope
  Evidence: Plan doc §4.4
  Severity: MODERATE
  ────────────────────────────────────────
  Gap: PPO optimal operating point is hardware-incompatible with residential laundry drain flow
  Why it matters: PPO learned to hold pump_cmd ≤ 0.22 (Q ≤ 7.7 L/min) to activate nDEP capture. Residential washing
    machines drain at 12–15 L/min. The device cannot throttle its own inlet flow — it would require a buffer tank or
    bypass manifold upstream to operate at PPO's learned flow rate. This makes the "optimal policy" physically
    unimplementable without hardware modifications not currently designed or costed.
  Dependency chain: Hardware architecture question — requires device design decision before PPO result can be deployed
  Evidence: CCE hydraulics model: Q_nom = 13.5 L/min; PPO operating point: Q ≤ 7.7 L/min; laundry drain: 12–15 L/min
  Severity: MAJOR — PPO benchmark result is simulation-valid but hardware-deployment requires architectural resolution
  ────────────────────────────────────────
  Gap: Benchmark episode length (40 s) captures only 8–20% of a real drain cycle (180–480 s)
  Why it matters: The submicron benchmark runs 400 steps × 0.1 s/step = 40 seconds simulated. A real laundry drain
    cycle is 3–8 minutes. Fouling dynamics, backflush timing, and concentration variation all evolve over 4–12× the
    simulated window. Capture efficiency ratios from benchmark cannot be extrapolated to "particles captured per wash
    cycle" without a full-cycle integration that does not exist.
  Dependency chain: Requires episode length extension and time-varying concentration model
  Evidence: M5 Task Execution Report §Task D — simulation parameters table
  Severity: MAJOR — benchmark results are steady-state operating point analysis, not full wash-cycle simulation
  ────────────────────────────────────────
  Gap: Submicron benchmark regime (d_p=1.0µm) is a physics-validation regime, not a deployment-representative regime
  Why it matters: d_p=1.0µm was chosen because it is below the S3 collector diameter (d_c=1.5µm), making the RT formula
    sensitive and DEP physics active. Dominant laundry microplastics are much larger: fibers 100–5000µm, fragments
    10–500µm. The RT formula is highly size-dependent — efficiency at 1.0µm tells you about submicron DEP physics, not
    about the full particle distribution a real device encounters. The CCE default particle size is d_p=10µm.
  Dependency chain: Full PSD integration across realistic size bins required for deployment-representative benchmarks
  Evidence: conical_cascade_env.py:213 (default d_p_um=10.0); M5 Task Execution Report §Task B (benchmark rationale)
  Severity: MODERATE for RL validity; MAJOR for any public efficiency claim that cites the submicron benchmark

  Observability Gaps

  Gap: [RESOLVED — Task F, commit 634dcb4] CCE telemetry stale-read in /api/run ppo_cce path
  Resolution: /api/run now branches on _use_ppo_cce; reads cce._state directly when in ppo_cce mode. All telemetry fields
    (flow_norm, pressure_norm, clog_norm, E_field_norm, C_out, particle_capture_eff) now reflect actual CCE simulation
    state. Analytics from ppo_cce API runs are now reliable.
  Remaining open item: _PPO_CCE_MODEL_PATH at app.py:25 still references ppo_cce_v1.zip; must be updated to
    ppo_cce_v2.zip to serve the correct benchmark model via API.
  Evidence: hydrion/service/app.py:153–178 (confirmed branched reads)
  Severity: CLOSED (telemetry bug); MINOR (v1→v2 model path update pending)
  ────────────────────────────────────────
  Gap: No benchmark reporting pipeline
  Why it matters: Validation tests pass/fail but do not emit structured calibration metrics
  Dependency chain: Required before external validation claims
  Evidence: docs/context/04_CURRENT_ENGINE_STATUS.md §11
  Severity: MODERATE

  Validation Gaps

  Gap: No hardware calibration data ingested
  Why it matters: All physics constants and device geometry are design defaults; model cannot be said to represent a
  specific
    physical device
  Dependency chain: Requires hardware build
  Evidence: Documented throughout docs/context/04_CURRENT_ENGINE_STATUS.md
  Severity: CRITICAL for external claims
  ────────────────────────────────────────
  Gap: Fouling progression not automatically validated against time target
  Why it matters: M1.5 acceptance criterion (70% loading in ≤500 steps) documented but not enforced in CI
  Dependency chain: tests/test_clogging.py exists but does not test this specific criterion
  Evidence: docs/context/09_REALISM_ROADMAP.md §M1.5
  Severity: MODERATE

  Commercialization-Relevant Gaps

  Gap: No physical device exists
  Why it matters: Digital twin without hardware cannot be validated, demonstrated, or sold
  Dependency chain: Prerequisite for all hardware claims
  Evidence: Entire codebase is simulation-only
  Severity: CRITICAL
  ────────────────────────────────────────
  Gap: Buoyant fraction (PP/PE ~45% of laundry microplastics) is excluded
  Why it matters: Market addressability claim is constrained to dense-phase particles only; this must be communicated
    accurately
  Dependency chain: Locked constraint §E
  Evidence: docs/context/06_LOCKED_SYSTEM_CONSTRAINTS.md §E
  Severity: MAJOR
  ────────────────────────────────────────
  Gap: Regulatory pathway undefined
  Why it matters: No documentation on EPA/EU compliance, NSF certification, or appliance integration standards
  Dependency chain: Not in scope yet
  Evidence: Absent from all docs
  Severity: MAJOR for commercialization
  ────────────────────────────────────────
  Gap: No customer or field-test partner documented
  Why it matters: No evidence of design partner engagement or real-world wastewater characterization
  Dependency chain: Not in scope yet
  Evidence: Absent from all docs
  Severity: MAJOR

  ---
  8. BLUEPRINT INPUT PACKAGE

  A. What HydrOS can credibly claim today

  1. A functioning multi-physics digital twin with a complete simulation pipeline (hydraulics, clogging, backflush,
  electrostatics, particle capture) running in a Gymnasium-compatible environment. Physics formulas are sourced from
  peer-reviewed literature for filtration (Rajagopalan-Tien 1976), nDEP (Pethig 1992, Gascoyne & Vykoukal 2002), and
  Clausius-Mossotti factors (Brandrup & Immergut 1999; Neagu 2002). Code-verified.
  2. A stable, hardware-aligned control interface: 4-dimensional continuous action space (valve, pump, backflush trigger,
  voltage) that maps directly to physical actuators.
  3. A reference efficiency definition: η_nominal = ~85.4% at d=10µm / Q=13.5 L/min / clean filter / dense-phase. This is a
  reproducible, code-verified anchor point for all future claims.
  4. A trained PPO artifact (ppo_cce_v2.zip, 500k steps, seed=42, submicron regime d_p=1.0µm) with a confirmed
  three-baseline benchmark result: PPO achieves 1.87× random and 1.96× heuristic on mean η_cascade in simulation. PPO
  discovered the non-trivial DEP flow-rate threshold without it being pre-programmed. All results are simulation outputs
  on design-default geometry — not hardware measurements.
  5. A working research console with physics playback, animated visualization, event marker navigation, and scenario
  execution.
  6. 88 passing tests covering the physics pipeline, environment API, training infrastructure, and validation protocols.
  7. A locked product scope: dense-phase microplastics (PET, PA, PVC, biofilm-coated fragments, ρ > 1.0 g/cm³). Buoyant
  plastics (PP, PE) are explicitly excluded and documented.

  B. What HydrOS cannot credibly claim yet

  1. Calibrated performance claims — no physical device exists; all geometry parameters are design defaults not measured
  from hardware. Any efficiency number is a model output at assumed geometry, not a measurement.
  2. A deployable RL controller — the RL observation uses truth_state directly (perfect information). A production
  controller requires sensor realism (M6) before it can be trusted in deployment.
  3. Coverage of buoyant microplastics (PP/PE ~45% of laundry outflow by count) — by design constraint, these pass through
  uncaptured.
  4. Hardware-validated performance claims — PPO benchmark superiority is demonstrated in simulation (1.87× random,
  1.96× heuristic on η_cascade) but all geometry parameters are design defaults, not measurements from a physical device.
  Simulation efficiency numbers cannot be directly compared to real-world capture rates without hardware calibration.
  5. Regulatory or standards compliance — no EPA, EU, NSF, or appliance integration pathway has been defined.
  6. Real-world data agreement — no lab bench measurements, no empirical calibration curves, no third-party test data
  ingested.

  C. Nearest demonstrable milestone

  [COMPLETE as of 2026-04-13] — Three-baseline benchmark evaluation run and passed on ppo_cce_v2.

  Results:
  - PPO mean η_cascade: 1.000 (submicron benchmark regime, d_p=1.0µm)
  - Heuristic mean η_cascade: 0.511 (pump=0.7 exceeds DEP threshold regardless of voltage)
  - Random mean η_cascade: 0.535 (~50% of steps above threshold)
  - PPO/Random ratio: 1.87× (PASS, threshold >1.5×)
  - PPO/Heuristic ratio: 1.96× (PASS, threshold >1.2×)

  Current nearest next milestone:
  1. Per-stage η breakdown audit — confirm PPO's η_cascade=1.000 is not residual RT saturation. Inspect per-stage
     η_s1/η_s2/η_s3 at PPO's operating point (pump_cmd ≤ 0.22, V=500V, d_p=1.0µm). Low effort, high credibility value.
  2. Update _PPO_CCE_MODEL_PATH in app.py:25 from ppo_cce_v1.zip to ppo_cce_v2.zip so the API serves the correct model.
  3. Hardware characterization protocol — a single bench session (ΔP vs Q sweep, cone geometry, post-backflush recovery)
     converts the model from physics-plausible to calibrated. Requires a physical device.

  D. Minimum proof required for external credibility

  Technical credibility (no hardware required):
  - [DONE] Three-baseline evaluation on ppo_cce_v2.zip — PPO superiority confirmed (1.87× random, 1.96× heuristic)
  - [DONE] Telemetry stale-read bug in /api/run ppo_cce path — fixed (Task F)
  - [OPEN] Per-stage η audit at PPO operating point — confirm η_cascade=1.000 is genuine nDEP capture, not saturation
  - [OPEN] Update app.py:25 _PPO_CCE_MODEL_PATH from ppo_cce_v1.zip to ppo_cce_v2.zip
  - [OPEN] Run eval_ppo_cce.py on additional disturbance scenarios (two exist; high-fouling and high-flow still need building)
  - [OPEN] Publish η_nominal with proper reference conditions in any external-facing material (never a bare percentage)

  Scientific credibility (requires hardware):
  - A single physical device characterization session that provides: cone geometry measurements, ΔP vs Q curve at known
  fouling states, post-backflush recovery curve
  - This single session would let the model be calibrated and allow performance claims to be grounded

  Commercial credibility (requires both):
  - Physical device + one design partner from laundry appliance or wastewater processing sector testing the device in their
  environment

  E. Most plausible first product wedge based on current technical state

  B2B: OEM integration into laundry appliance manufacturers for a smart microplastic filtration module.

  Rationale from the technical state:
  - The device targets laundry outflow at 12–15 L/min continuous — this is the outflow spec of a standard residential
  washing machine drain
  - The physics simulate a handheld/inline device that processes that outflow
  - The RL controller (once evaluated and M6-ready) is designed for autonomous operation without user intervention —
  matching OEM "fit and forget" requirements
  - Dense-phase capture (PET, PA, PVC) covers the microplastics most likely to be regulated first (synthetic fiber fragments
   from polyester/nylon clothing)
  - The digital twin accelerates time-to-specification for the physical device — reducing OEM integration risk

  Alternative wedge: Industrial water treatment pre-screening at laundry facilities (hotels, hospitals) where higher flow
  rates and regulatory pressure make the investment case clearer.

  F. What type of customer or design partner would make sense first

  1. Appliance OEM R&D lab — A washing machine manufacturer (e.g., Miele, Bosch, LG, Samsung) that has an active
  sustainability engineering program. They can characterize a physical device in their test facility, providing the
  calibration data the model needs while evaluating integration feasibility.
  2. University textile/water research lab — A lab already studying microplastic shedding from fabrics would be a
  low-friction design partner. They have the characterization equipment (particle counters, turbidity meters, flow benches)
  needed to calibrate the model without a full industrial partnership.
  3. Municipal wastewater technology testbed — Organizations like water utilities with pilot testing programs for new
  filtration technologies. They are willing to run small-scale devices and provide structured data, which is exactly what
  HydrOS needs for its first hardware calibration.

  G. What technical unknowns still block company-level positioning

  1. Physical device geometry — Until a device is built and measured, the model cannot make defensible performance claims.
  Every number in the simulation is a placeholder.
  2. Actual η_nominal at built geometry — The locked reference efficiency (~85.4%) is a model output at assumed geometry.
  The real device at real geometry may be meaningfully different.
  3. Buoyant fraction commercial relevance — If PP/PE exclusion eliminates the plastic types most visible to consumers
  (PP/PE are the majority by count in some laundry effluent studies), the product narrative requires repositioning or a
  second capture mechanism.
  4. Sensor feasibility — The M6 sensor model requires knowing what sensors are physically installable in the device form
  factor. A handheld device has severe space constraints. Differential pressure sensing in a clogging filter is
  straightforward; optical scatter at fine mesh scale may not be.
  5. Energy budget — The electrostatic capture system operates at up to 2.5 kV. For a consumer device, the power draw of the
   high-voltage stage relative to its capture benefit is an unresolved engineering question that the current reward function
   does not adequately model (volt_penalty weight = 0.05 is an arbitrary starting point).
  6. Regulatory pathway duration — NSF 42/53 or equivalent filtration standards take 12–18 months minimum. EU microplastics
  regulation (expected 2025–2027) may create a tailwind or a compliance requirement depending on timing.
  7. DEP flow rate architecture — PPO's learned optimal policy requires Q ≤ 7.7 L/min to activate nDEP. Residential
  washing machines drain at 12–15 L/min. Whether the device controls its own inlet flow (requiring a buffer tank or bypass
  manifold) or operates in a lower-flow regime (requiring a different hydraulic integration) is an unresolved hardware
  architecture question. The current simulation assumes the device has authority over its own inlet flow, which may not be
  physically accurate.
  8. Deployment-representative particle size — The simulation uses d_p=1.0µm (submicron benchmark) or d_p=10µm
  (HydrionEnv). Real laundry outflow contains a full size distribution: fibers 100–5000µm, fragments 10–500µm. The
  dominant captured fraction in a real device will be in size ranges the current benchmark does not represent. Full PSD
  integration is required before any efficiency claim can be described as representative of real laundry outflow.

  ---
  9. EXECUTIVE DELTA SUMMARY

  1. The simulation pipeline is structurally complete and architecturally sound — hydraulics, clogging, backflush,
  electrostatics, particles all implemented with peer-reviewed physics formulas.
  2. No physical device exists. All geometry parameters are design defaults, not measurements. The model describes a
  plausible device, not a characterized one.
  3. [FIXED — Task E] HydrionEnv observation index 3 E_norm/E_field_norm key mismatch has been resolved. env.py:322 now
  writes E_field_norm (v2 key); obs index 3 correctly reflects electrostatic field strength. A regression test guards
  this going forward.
  4. CCE and HydrionEnv are NOT interchangeable despite both having shape (12,) observations — their field semantics differ
  at index 3 and beyond. "obs12_v2 compatible" means shape-compatible, not semantically identical.
  5. [EVALUATED — Task D] ppo_cce_v2 (500k steps, seed=42, submicron regime) has been evaluated against three baselines.
  PPO/Random: 1.87×, PPO/Heuristic: 1.96× — both criteria pass. PPO discovered the DEP flow-rate threshold: operating at
  pump_cmd ≤ 0.22 (Q ≤ 7.7 L/min) activates nDEP capture. This is a genuine RL discovery of device physics, not a
  pre-programmed rule. All results are simulation outputs; ppo_cce_v1 is archived and not a valid benchmark artifact.
  6. The dep_exponent bistable bug is fixed (confirmed in configs/default.yaml:63). Clean-start RL training now produces
  observable fouling.
  7. [COMPLETE] The three-baseline benchmark has been run and passed on ppo_cce_v2. The most important remaining simulation-
  internal step is a per-stage η breakdown audit to confirm PPO's η_cascade=1.000 reflects genuine nDEP capture and not
  a residual saturation signal. The most important system-level step is updating app.py:25 to point to ppo_cce_v2.zip.
  8. Sensor realism (M6) is the critical blocker for any deployment-validity claim. The observation uses truth_state
  (perfect information). A controller trained without sensor noise has zero deployment validity.
  9. The Phase 1 reward in HydrionEnv does not optimize capture efficiency — it optimizes processed flow throughput. A
  policy could score well while capturing poorly. This mismatch between stated mission and reward signal is the most
  important reward gap.
  10. The console is significantly more functional than its own documentation claims — the docs say "static scaffold, no
  live data binding" but the code has full scenario playback, animated physics visualization, and API binding.
  11. Two safety implementations exist (shield.py + shielded_env.py) — functionally operational but a documented duplication
   risk requiring eventual consolidation.
  12. [FIXED — Task F] The telemetry stale-read bug in /api/run ppo_cce path has been resolved. The endpoint now reads
  CCE state directly when in ppo_cce mode. Analytics from ppo_cce API runs are now reliable. Remaining action: update
  _PPO_CCE_MODEL_PATH in app.py:25 from ppo_cce_v1.zip to ppo_cce_v2.zip.
  13. Buoyant plastics (PP/PE) are excluded by design — locked constraint §E. This affects approximately 40–50% of laundry
  microplastic count. The product narrative must either accept this scope or define an upstream treatment pathway.
  14. 88 tests pass with strong coverage across the physics stack, training infrastructure, and validation protocols. The
  test suite is a genuine engineering asset. (One test added in Task E: E_field_norm obs regression.)
  15. The η_nominal reference definition is locked and correct — any public efficiency claim must cite η = XX% @ 10µm / 13.5
   L/min. A bare efficiency percentage is not a valid HydrOS output.
  16. The most important next validation milestone is physical device characterization — a single measurement session (ΔP vs
   Q curve, cone geometry, post-backflush recovery) converts the model from a physics-plausible simulator to a calibrated
  digital twin.
  17. The most important near-term positioning risk is conflating the simulation (which works) with hardware performance
  (which has not been measured). Every external claim must be qualified with the current calibration status.
  18. The most important next design partner characteristic is: access to a particle counter or flow bench with ΔP
  measurement, and willingness to test a small physical prototype. A university textile lab or appliance OEM R&D facility
  both qualify.
  19. A strategist building the company blueprint needs to know: the value proposition today is the digital twin + RL
  framework as a product development acceleration tool, not a deployed filtration system. The path to a deployed system runs
   through M6 (sensors), hardware characterization, and then a design partner integration test.
  20. The PPO benchmark (1.87× random, 1.96× heuristic) is a simulation result under specific conditions: d_p=1.0µm
  monodisperse, 40-second episodes, constant inlet concentration, and the assumption that the device controls its own
  inlet flow. None of these conditions match a real wash cycle. The benchmark demonstrates that RL can discover non-trivial
  device physics — it does not demonstrate that the device achieves any specific capture rate in a real laundry machine.
  21. The two most important simulation-reality boundary gaps not yet in the architecture roadmap: (a) the device needs a
  hardware solution for operating below the DEP threshold flow rate (12–15 L/min inlet vs ≤7.7 L/min PPO optimum), and
  (b) benchmark particle size (1µm) does not represent the dominant laundry microplastic size range (10–5000µm). Both
  require hardware design decisions before the simulation can be used for calibrated performance claims.
  