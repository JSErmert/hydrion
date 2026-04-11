# HydrOS M4 Report — Particle Realism
**Date:** 2026-04-10  
**Author:** HydrOS Co-Orchestrator (Claude Sonnet 4.6)  
**Scope:** Milestone 4 — Density classification, Stokes settling, per-stage size-dependent capture, formal efficiency definition  
**Status:** AUTHORITATIVE — defines M4 implementation scope, role in HydrOS, validation criteria, and forward implications  
**Depends on:** M3 complete (✓ — obs12_v2, E_field_kVm, E_capture_gain, radial field model)

---

## Purpose

This report defines the complete scope of Milestone 4 (M4), its role within HydrOS, and the foundation it lays for future phases. M4 has one primary mission:

> Make particle capture physically meaningful — stratified by density, size, flow rate, and stage — and lock the formal system efficiency definition that all future engineering claims, console displays, and RL objectives will reference.

M4 does not touch hydraulics, clogging, backflush, or the electrostatic field model. Those are stable. M4 extends the particle module — `particles.py` — with the physical reality that different particles behave differently, and that the system's ability to capture them depends on what they are, where they go, and how fast the water is moving.

---

## Entry Conditions

M4 may not begin until all of the following are confirmed:

| Condition | Status |
|---|---|
| ElectrostaticsModel v2: radial field, `E_field_kVm` in truth_state | ✓ M3 |
| `E_capture_gain` ∈ [0, 1] produced by electrostatics, consumed by particles | ✓ M3 |
| obs12_v2 active: index 3 = `E_field_norm` | ✓ M3 |
| Dense-phase scope locked: ρ > 1.0 g/cm³ target, PP/PE excluded | ✓ M1.5 |
| Full test suite: 25/25 passing | ✓ M3 |

---

## What M3 Left Unresolved in Particles

M3 made electrostatics physically meaningful. But after M3, `particles.py` still:

- Operates with a single scalar `capture_eff` — no size dimension, no stage dimension, no density dimension
- Treats all particles identically regardless of whether they are 1 µm or 500 µm
- Applies the same capture efficiency to PP (buoyant) and PET (dense) without distinction
- Produces `particle_capture_eff` as an aggregate scalar with no qualification
- Cannot answer the question: *what fraction of 10 µm dense-phase particles are captured at nominal flow?*

That last question is the definition of system efficiency. Until M4, HydrOS cannot answer it.

---

## 1. What M4 Is — Role in HydrOS

### 1.1 The Transition from Abstraction to Physical Stratification

Before M4, the particle module models capture as:

```python
capture_eff = capture_eff_base + alpha_clog × mesh_avg + alpha_E × E_capture_gain
```

This is a signal-level model. It responds to clogging and electrostatics but has no physical geometry. A 500 µm fragment and a 5 µm fiber receive the same capture treatment. PP pellets and PET fragments are indistinguishable.

After M4, capture is stratified:

- **By density**: Dense (ρ > 1.0) and neutral particles are captured. Buoyant (PP, PE) pass through — tracked, not captured.
- **By size**: Each stage has a distinct capture curve based on its mesh pore size. Stage 1 (500 µm) captures large debris efficiently but passes small particles. Stage 3 (5 µm) captures nearly everything at low flow but degrades at high flow.
- **By flow rate**: Stage 3 capture efficiency decreases measurably above 10 L/min due to reduced residence time and increased drag — a real physical effect that was absent before M4.
- **By stage**: System efficiency is the compound product of three sequential stages, not a single scalar.

### 1.2 The Formal Efficiency Definition

M4 locks the definition that every engineering claim about this device must reference:

```
η_nominal = η_system(d = 10 µm, Q = 13.5 L/min, clean filter, dense-phase particles)
```

This is not an arbitrary number. It is:
- A specific particle size (10 µm — the hardest size for the fine stage to capture reliably)
- A specific flow rate (13.5 L/min — middle of the nominal operating envelope)
- A defined filter state (clean — worst-case for capture, best-case for pressure)
- A defined particle class (dense-phase — the device's stated scope)

Without this definition, any efficiency percentage displayed on the console or cited in a report is ambiguous. "99% capture" means nothing without knowing: 99% of *what*, at *what flow*, in *what condition*.

M4 makes this definition operational — computed in code, locked in constraints, displayed with qualification on the console.

### 1.3 Why M4 Is the Credibility Gate

M1/M1.5 made the hydraulic and fouling backbone credible.  
M3 made the electrostatic field model defensible.  
M4 makes the *output claim* — what the device actually captures — defensible.

HydrOS cannot be presented as a serious microplastic extraction simulator until M4 is complete. Before M4, the system can simulate pressure, fouling, and electrostatics. It cannot say, with physical grounding, what fraction of laundry microplastics it captures or how that changes with operating conditions. M4 is the bridge from physics simulation to engineering performance.

---

## 2. M4 Scope — Four Technical Components

### 2.1 Density Classification

Three density classes are introduced across all PSD bins:

```python
DENSITY_DENSE   = 'dense'    # ρ > 1.0 g/cm³  — PET (1.38), PA (1.14), PVC (1.16–1.58), biofilm-coated
DENSITY_NEUTRAL = 'neutral'  # ρ ≈ 1.0 g/cm³  — weathered, transitional fragments
DENSITY_BUOYANT = 'buoyant'  # ρ < 1.0 g/cm³  — PP (0.91), PE (0.95) — tracked, NOT captured
```

In `particles.py`, concentration is split:

```python
C_in_dense    # concentration of dense-phase particles [0–1 relative]
C_in_neutral  # concentration of neutral-phase particles [0–1 relative]
C_in_buoyant  # buoyant fraction — exits as pass-through directly

C_out = (
    C_in_dense    × (1.0 - capture_eff_dense)
  + C_in_neutral  × (1.0 - capture_eff_neutral)
  + C_in_buoyant  # fully passes through — scope constraint
)
```

`C_in_buoyant` is tracked in truth_state as a measurement of what the system intentionally does not capture. This is not a failure state — it is a scope constraint materialized in the simulation.

Default distribution (YAML-configurable):
- `dense_fraction`: 0.70 — PET/PA/PVC dominant in laundry effluent
- `neutral_fraction`: 0.15 — weathered fragments
- `buoyant_fraction`: 0.15 — PP/PE pass-through

### 2.2 Stokes Settling

Dense particles (ρ > 1.0) settle under gravity. This is a physical reality that assists capture at the outer collection wall of each stage.

```python
def stokes_velocity_ms(rho_p_kgm3: float, d_p_m: float,
                        rho_w: float = 1000.0, mu: float = 1e-3) -> float:
    """Stokes settling velocity [m/s]. Positive = downward (sinking)."""
    return (rho_p_kgm3 - rho_w) * 9.81 * d_p_m**2 / (18.0 * mu)
```

For PET (ρ = 1380 kg/m³) at d = 10 µm:
```
v_s = (1380 - 1000) × 9.81 × (10e-6)² / (18 × 1e-3) ≈ 2.1 × 10⁻⁵ m/s
```

This is small relative to flow velocity but non-negligible over the residence time of a stage. The settling contribution is applied as a `capture_boost_settling` term proportional to `v_s × t_residence / stage_height_m`.

For buoyant particles (PP, PE): `v_s < 0` — they rise, moving *away* from the downward collection tubes. This is the physical reason buoyant-phase capture is out of scope — not just a design choice, but a consequence of gravity.

### 2.3 Per-Stage Size-Dependent Capture Efficiency

Each stage has its own capture curve parameterized by mesh pore size. These replace the single aggregate `capture_eff_base`.

**Stage 1 — 500 µm coarse mesh:**
```python
def capture_eff_s1(d_p_um: float, fouling_s1: float) -> float:
    base = clip((d_p_um / 500.0) ** 1.5, 0.0, 0.99)
    fouling_factor = 1.0 + 0.15 × fouling_s1
    return clip(base × fouling_factor, 0.0, 0.99)
```
Captures particles approaching 500 µm efficiently. Small particles (<50 µm) pass through almost entirely. Fouling slightly improves capture (pore restriction effect).

**Stage 2 — 100 µm medium mesh:**
```python
def capture_eff_s2(d_p_um: float, fouling_s2: float) -> float:
    base = clip((d_p_um / 100.0) ** 1.2, 0.0, 0.98)
    fouling_factor = 1.0 + 0.20 × fouling_s2
    return clip(base × fouling_factor, 0.0, 0.98)
```
More aggressive capture curve. Catches medium fragments efficiently. Fine particles still pass.

**Stage 3 — 5 µm fine pleated cartridge (flow-rate dependent):**
```python
def capture_eff_s3(d_p_um: float, fouling_s3: float, Q_lmin: float) -> float:
    base = clip((d_p_um / 5.0) ** 0.8, 0.0, 0.97)
    flow_penalty = exp(-0.04 × max(Q_lmin - 10.0, 0.0))
    fouling_factor = 1.0 + 0.10 × fouling_s3
    return clip(base × flow_penalty × fouling_factor, 0.0, 0.97)
```
Stage 3 is the primary fine particle capture stage. The `flow_penalty` term activates above 10 L/min and degrades efficiency — this is the physical consequence of reduced residence time in the pleated cartridge at high throughput. At Q = 20 L/min, the penalty factor is approximately `exp(-0.4) ≈ 0.67`.

**System compound efficiency:**
```python
eta_system = 1.0 - (1.0 - eta_s1) × (1.0 - eta_s2) × (1.0 - eta_s3)
```

For a 10 µm particle at Q = 13.5 L/min, clean filter:
- η_s1 ≈ 0.003 (10 µm is tiny relative to 500 µm)
- η_s2 ≈ 0.028 (10 µm relative to 100 µm)
- η_s3 ≈ 0.745 (10 µm relative to 5 µm, flow penalty active)
- η_system ≈ 1 − (0.997 × 0.972 × 0.255) ≈ **0.753**

This is the physics-derived baseline for η_nominal. The actual value will be tuned to calibration data, but the structure — stage 3 as primary capture, flow-rate degradation — is correct.

### 2.4 Formal Efficiency Definition Lock

The following is added to `docs/context/06_LOCKED_SYSTEM_CONSTRAINTS.md` as Section G:

```
## G. System Efficiency Definition

η_nominal = η_system(d = 10 µm, Q = 13.5 L/min, clean filter, dense-phase particles)

This is the reference efficiency for:
- Console display
- Engineering claims
- RL reward shaping (M6)
- Bench validation targets

Display format: "η = XX% @ 10µm / 13.5 L/min"

Never display a bare efficiency percentage without this qualification.
"99%" without context is not a valid system output.
```

---

## 3. Truth State and Observation Updates

### 3.1 New truth_state keys

```python
# Density classification
C_in_dense          # dense-phase input concentration [0–1]
C_in_neutral        # neutral-phase input concentration [0–1]
C_in_buoyant        # buoyant pass-through concentration [0–1]
buoyant_fraction    # fraction of C_in that is buoyant (config value)

# Per-stage capture
capture_eff_s1      # Stage 1 capture efficiency at current PSD/flow/fouling
capture_eff_s2      # Stage 2 capture efficiency
capture_eff_s3      # Stage 3 capture efficiency

# System efficiency
eta_system          # compound system efficiency: 1-(1-s1)(1-s2)(1-s3)
eta_nominal         # η at reference conditions: d=10µm, Q=13.5, clean, dense
```

These are truth_state additions — not observation vector extensions. The observation vector stays at 12D (obs12_v2). The per-stage and density values are available to the validation system, reward shaping, and console telemetry but do not change the RL interface.

### 3.2 Observation schema — no change

obs12_v2 is unchanged. `particle_capture_eff` at index 5 transitions semantically to represent the aggregate dense-phase capture efficiency — the same index, updated meaning, no schema bump.

If a future milestone extends the observation to include per-stage capture or density fractions, that is `obs14_v1` or higher. M4 does not trigger this.

---

## 4. File Changes

### 4.1 `hydrion/physics/particles.py`

- Add `dense_fraction`, `neutral_fraction`, `buoyant_fraction` to `ParticleParams`
- Add `stage_height_m` for Stokes settling (placeholder geometry parameter)
- Add `stokes_velocity_ms()` utility function
- Add `capture_eff_s1()`, `capture_eff_s2()`, `capture_eff_s3()` stage capture functions
- Update `update()`: split `C_in` into density fractions, compute per-stage capture, compute compound `eta_system`, compute `eta_nominal` at reference conditions
- Write all new truth_state keys listed in §3.1

### 4.2 `configs/default.yaml`

Add under `particles:`:

```yaml
particles:
  C_in_base: 0.7

  # Density classification (Option A — dense-phase scope, locked §E)
  dense_fraction:   0.70    # PET, PA, PVC, biofilm-coated — primary target
  neutral_fraction: 0.15    # weathered, transitional
  buoyant_fraction: 0.15    # PP, PE — pass-through, tracked not captured

  # Stage geometry for Stokes settling
  stage_height_m:   0.05    # [m] representative stage height — placeholder

  # Stokes settling target polymer densities [kg/m³]
  rho_dense_kgm3:   1380.0  # PET reference density
  rho_water_kgm3:   1000.0  # water at ~20°C
  mu_water_Pas:     1.0e-3  # dynamic viscosity [Pa·s]

  # Per-stage fouling coupling to capture
  fouling_gain_s1:  0.15    # capture gain per unit fouling (Stage 1)
  fouling_gain_s2:  0.20    # capture gain per unit fouling (Stage 2)
  fouling_gain_s3:  0.10    # capture gain per unit fouling (Stage 3)

  # Stage 3 flow-rate penalty
  s3_flow_penalty_coeff: 0.04   # exp(-coeff × max(Q-10, 0)) degradation
  s3_flow_onset_lmin:    10.0   # flow rate above which penalty activates

  # Reference conditions for eta_nominal (locked §G)
  eta_ref_d_um:     10.0    # [µm] reference particle size
  eta_ref_Q_lmin:   13.5    # [L/min] reference flow rate

  psd:
    enabled: false
    mode: hybrid
  shape:
    enabled: false
    fiber_fraction: 1.0
```

### 4.3 `docs/context/06_LOCKED_SYSTEM_CONSTRAINTS.md`

Add Section G: System Efficiency Definition (see §2.4).

### 4.4 `docs/calibration/M2-2.5_CALIBRATION_PARAMETER_REGISTER.md`

Add all new particle parameters with evidence class `Placeholder` or `Proxy` as appropriate.

### 4.5 `docs/context/04_CURRENT_ENGINE_STATUS.md`

Update §8 (Particle Module) on M4 completion.

---

## 5. Validation Criteria

All criteria must pass before M4 is considered complete.

### 5.1 Density classification

| Test | Expected |
|---|---|
| `C_in_buoyant` exits as full pass-through | `C_out_buoyant = C_in_buoyant` (no capture) |
| `buoyant_fraction` + `dense_fraction` + `neutral_fraction` = 1.0 | no leakage |
| `C_in_dense > 0` in truth_state | dense path active |

### 5.2 Stokes settling

| Test | Expected |
|---|---|
| PET (ρ=1380) at d=10µm: `v_s > 0` | sinking — assists collection |
| PP (ρ=910) at d=10µm: `v_s < 0` | rising — physically correct, passes through |
| `capture_boost_settling > 0` for dense particles at nominal flow | settling contributes |

### 5.3 Per-stage capture physics

| Test | Expected |
|---|---|
| d=10µm, clean, Q=13.5: `eta_s1 < 0.05` | coarse mesh passes fine particles |
| d=10µm, clean, Q=13.5: `eta_s3 > 0.5` | fine mesh captures majority |
| `eta_s3(Q=5) > eta_s3(Q=20)` at same fouling, same d | flow-rate degradation confirmed |
| `eta_system > eta_s3` | compound stages improve on single-stage |

### 5.4 Formal efficiency definition

| Test | Expected |
|---|---|
| `eta_nominal` computed at d=10µm, Q=13.5, clean, dense | scalar ∈ [0, 1] |
| `eta_nominal` stable across episodes (deterministic) | no random variation |
| Console display shows `"η @ 10µm / 13.5 L/min"` qualification | never bare percentage |

### 5.5 Regression

| Test | Expected |
|---|---|
| Full test suite | 25/25 pass (minimum); new M4 tests added |
| Hydraulics outputs unchanged | dp_s3 >> dp_s2 >> dp_s1 preserved |
| Clogging outputs unchanged | fouling grows from zero, sum ≤ 1.0 |
| Electrostatics outputs unchanged | E_field_kVm, E_capture_gain stable |
| obs12_v2 schema unchanged | shape (12,), index 3 = E_field_norm |

---

## 6. What M4 Does NOT Do

M4 is contained. The following are explicitly excluded:

| Item | Deferred to |
|---|---|
| Reward function update to use `eta_nominal` | M6 |
| Sensor modeling of particle concentration (turbidity calibration) | M5 |
| PSD bins with explicit bin edges and per-bin concentrations (full PSD) | M4.5 / M5 |
| Fiber-vs-fragment shape-dependent capture behavior | M4.5 |
| Conductivity effects on particle charge state | M3.5+ |
| Upstream buoyant-phase treatment module | Out of scope |
| Extending observation vector beyond 12D | post-M4 if required |
| Console schematic correction | Phase 4 (already unblocked by M3) |

---

## 7. M4 Exit Conditions

M4 is complete when all of the following are confirmed:

- [ ] `C_in_dense`, `C_in_neutral`, `C_in_buoyant` tracked separately in truth_state
- [ ] `C_in_buoyant` exits as full pass-through — no capture applied
- [ ] `stokes_velocity_ms()` implemented and correctly signed (dense > 0, buoyant < 0)
- [ ] `capture_eff_s1()`, `capture_eff_s2()`, `capture_eff_s3()` implemented with correct pore-size physics
- [ ] `eta_s3` decreases measurably at Q > 15 L/min vs Q = 5 L/min
- [ ] `eta_system = 1 − (1−η_s1)(1−η_s2)(1−η_s3)` implemented
- [ ] `eta_nominal` computed and stored in truth_state at reference conditions
- [ ] Section G (efficiency definition) locked in `06_LOCKED_SYSTEM_CONSTRAINTS.md`
- [ ] `default.yaml` updated with all new particle parameters
- [ ] `CALIBRATION_PARAMETER_REGISTER.md` updated with new M4 parameters
- [ ] Full test suite passing with new M4-specific tests added
- [ ] `04_CURRENT_ENGINE_STATUS.md` updated to reflect M4 state

---

## 8. M4's Role Across HydrOS — What It Changes

M4 is not a feature addition. It is a *realism layer* that changes what can be said about the system.

Before M4, HydrOS can say:
> "The system generates pressure and fouling that respond to flow and backflush."

After M4, HydrOS can say:
> "The system captures X% of dense-phase particles above 10 µm at nominal flow, degrading to Y% at peak flow, and recovering to Z% after backflush — and these numbers are physically derived."

This is the difference between a physics scaffold and a physics argument.

### Impact on the RL agent

Pre-M4: the agent optimizes for processed throughput and pressure safety. It has no visibility into *what* is being captured or whether the capture is size-appropriate.

Post-M4: `eta_system` and `capture_eff_s3` are available in truth_state. The reward function can be updated (M6) to weight capture quality — specifically `eta_nominal` — alongside throughput. An agent that sacrifices fine-particle capture efficiency for throughput will be penalized explicitly. An agent that manages S3 fouling carefully to preserve low-flow efficiency will be rewarded.

### Impact on validation

The M4 efficiency definition creates a concrete validation target: bench-test the device at d=10µm, Q=13.5 L/min with clean filter, measure actual capture, and compare to `eta_nominal`. Every M4 parameter can now be tightened against a specific, reproducible measurement.

### Impact on the console

The console gains a meaningful efficiency display. Previously, `particle_capture_eff` was an aggregate scalar with no qualification. Post-M4, the display becomes:

```
η = 75%  @ 10µm / 13.5 L/min
```

With real-time `eta_system` updating as fouling changes, flow varies, or voltage toggles. The number means something.

---

## 9. Forward View — M4 as Foundation for Post-M4 and Future Hydrion

*This section is forward-looking and secondary. M4 scope above is the governing specification.*

### 9.1 What M4 Establishes That Post-M4 Depends On

M4 creates three structural foundations that subsequent milestones require:

**Foundation 1 — Density class as a first-class attribute**  
Once `C_in_dense`, `C_in_neutral`, `C_in_buoyant` exist as distinct truth_state values, future modules can treat them independently. Sensor modeling (M5) can calibrate turbidity signal separately for each class — fiber-dominated vs fragment-dominated signals have different optical signatures. The RL agent can eventually optimize for dense-phase capture specifically, not aggregate capture.

**Foundation 2 — Per-stage capture as observable physics**  
`capture_eff_s1`, `capture_eff_s2`, `capture_eff_s3` in truth_state make stage-specific behavior visible. This enables stage-targeted backflush logic: if S3 capture efficiency degrades while S1 and S2 are clean, the system can trigger a targeted S3 backflush rather than a full three-stage burst. That is a more efficient maintenance strategy and directly flows from having stage-resolved capture data.

**Foundation 3 — η_nominal as the anchor scalar**  
Every future efficiency claim — in engineering reports, console telemetry, RL reward, bench validation — references the same definition. This prevents the fragmentation problem where different modules use different efficiency measures that cannot be compared. M6 reward shaping, console display, and validation benchmarks all read from the same source.

### 9.2 M4.5 — The Natural Next Step

M4's size-dependent capture curves use representative bulk particle sizes. A full particle size distribution (PSD) — with explicit bin edges, lognormal or measured distributions, and per-bin concentrations — is a natural extension. This is M4.5 scope:

- Activate `psd.enabled = true` in YAML
- Compute `eta_s1(d)`, `eta_s2(d)`, `eta_s3(d)` across all PSD bins
- Produce `eta_system(PSD)` as a mass-weighted average over the distribution
- Update `eta_nominal` to reference the standard distribution at reference conditions

M4 does not implement this — it creates the per-stage functions that M4.5 will integrate over.

### 9.3 M5 — Sensor Realism Against M4 Truth

M5 introduces physically realistic sensors. The most important M5 dependency is that turbidity and scatter signals must be calibrated against a known particle composition. M4 provides that composition via `C_in_dense`, `C_in_neutral`, `C_in_buoyant`, and the density class fractions.

Without M4's density classification, M5 turbidity modeling is unconstrained — it can be calibrated to any signal. With M4, the calibration target is specific: turbidity should track `C_out` weighted by the optical cross-sections of the dense and neutral fractions. Buoyant particles in pass-through would affect turbidity differently than captured dense particles affecting downstream concentration.

M4 does not implement sensors. It provides the composition ground truth that M5 sensors will be calibrated against.

### 9.4 M6 — Reward Alignment

The current reward function (M1 design) rewards processed throughput and penalizes pressure and fouling. It does not reward capture quality directly. M6 will introduce a multi-objective reward that includes `eta_nominal` as a primary signal.

M4 is the prerequisite: `eta_nominal` must be physically derived before it can be meaningfully rewarded. Rewarding a placeholder capture scalar would train the agent to optimize a fiction. M4 makes `eta_nominal` a trustworthy number.

The M6 reward extension is anticipated as:

```python
reward += w_capture × eta_nominal   # reward capture quality at reference conditions
reward -= w_eta_s3_degradation × max(0, eta_nominal_clean - eta_nominal)
# penalize degradation from clean-state reference
```

This requires M4's `eta_nominal` computation to be stable, deterministic, and physically grounded.

### 9.5 Future Hydrion Iterations — What M4 Enables

Beyond the milestone roadmap, M4's particle stratification has implications for hardware design decisions:

**Stage targeting:** If per-stage capture data from simulation matches bench data, the simulation can guide hardware changes — e.g., is increasing the S3 effective area worth the pressure increase? Is a thicker Stage 1 mesh better or worse for S2/S3 capture? These are simulation-answerable questions after M4.

**Operating envelope optimization:** The flow-rate degradation term in `capture_eff_s3` directly maps to an optimal operating point question: what flow rate maximizes `eta_nominal × q_processed_lmin`? This is a real design question the simulation can answer quantitatively after M4.

**Buoyant-phase context:** `C_in_buoyant` tracked as pass-through quantifies the size of the buoyant-phase problem. If laundry effluent is 40% PP/PE by count, the device captures at most 60% of all particles regardless of how well it performs on the dense fraction. This number has implications for how the device is positioned in any product claim. M4 makes that number computable, even if the device itself cannot address it.

**RL agent specialization:** A sufficiently trained agent with M4-grounded rewards may discover operating strategies that are not obvious from first principles — e.g., deliberately allowing S1 fouling to improve its capture contribution to fine-particle retention, or modulating voltage to compensate for high-flow capture degradation at S3. These strategies emerge from physically grounded interactions that only exist after M4.

---

## Final Statement

M4 is the milestone that makes HydrOS's output claim defensible.

Everything before M4 — hydraulics, fouling, backflush, electrostatics — is the engine. M4 is the first milestone where the engine produces a number that means something in the real world: a capture efficiency for a specific particle, at a specific flow, in a specific filter state.

That number — `η_nominal` — is the anchor of every future engineering claim about this device.

> Physics first. Code second. UI last.

M4 completes the physics.
