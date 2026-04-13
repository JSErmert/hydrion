# Milestone 1 Design Record
# Hydraulic + Fouling + Backflush Realism Backbone

**Status**: COMPLETE — merged to branch `HydrOS-x-Claude-Code`, PR open against `main`
**Date**: 2026-04-09
**Commit**: `1bc1f5f` (implementation) + `b38d006` (pre-merge docs patches)
**Branch**: `HydrOS-x-Claude-Code`

---

## Objective

Establish a physically meaningful interaction between flow, pressure, clogging,
and backflush recovery. This is the foundation layer on which all subsequent
realism milestones are built.

---

## Scope Locked at Design Time

Milestone 1 covers exactly:
- Hydraulics: pump curve, area-normalized resistance, bypass
- Clogging: decomposed fouling (cake/bridge/pore), irreversible fraction
- Backflush: pulse state machine, partial recovery, diminishing returns
- Validation: 6 new tests covering all above
- Reward: interim 5-term reward to expose hydraulic/fouling tradeoffs to RL

Explicitly out of scope for M1:
- Electrostatics expansion (M3)
- Console redesign / Firebase / mobile (separate track)
- Particle model updates (M4)
- Sensor noise/drift (M5)

---

## Implementation Guardrails Applied

These guardrails were set before coding began and held throughout:

1. **Area normalization is first-pass**: `k_eff = k_base × (A_s3/A_si)` is
   documented as a calibration approximation. Base resistances unchanged.
   All area factors are YAML-exposed for re-tuning.

2. **maintenance_required is telemetry only**: The 70% threshold flag is computed
   in `_update_normalized_state()` and written to truth_state. It is explicitly
   excluded from the reward function. Reward remains continuous.

3. **Capture efficiency curve is YAML-parameterized**: The non-monotonic formula
   `baseline + gain × n × (1−n)^exponent` is fully exposed in `configs/default.yaml`
   under the `clogging:` section. No hardcoded physics constants in the model.

4. **Validation file style-aligned**: `milestone1_validation.py` follows the same
   conventions as the existing suite: `run_*() → Dict`, `argparse main()`,
   `--config` flag, `--test` flag for single-test invocation.

5. **Observation schema immutable**: `obs12_v1` (12D) was not modified. No new
   observation channels were added or reordered.

6. **truth_state / sensor_state separation preserved**: All physics models write
   only to truth_state. sensor_state is untouched by any M1 module.

---

## Key Design Decisions

### D1 — Pump Curve: Quadratic Intersection

**Decision**: Solve the pump-system curve intersection as a proper quadratic
(`u² + β·u − 1 = 0`) rather than the original two-step approximation.

**Reason**: The approximation had a zero-collapse bug where Q_raw ≥ Q_max caused
`q_ratio → 1.0 → P_avail → 0 → Q_in = 0`. P_in became non-monotone with pump_cmd.
The quadratic solution is the correct physics; P_in and Q_in now increase
monotonically with pump_cmd at all clog levels.

**Formula**:
```
P_pump = P_max_eff × (1 − u²)   [pump curve]
P_sys  = R_forward × Q_max × u  [system curve]

β = R_forward × Q_max_m3s / P_max_eff
u = (−β + √(β² + 4)) / 2        u ∈ [0, 1] by construction
```

---

### D2 — Area-Normalized Clog Sensitivity

**Decision**: Scale clog-to-resistance coefficient by `A_s3/A_si` (Stage 3 as
reference, area 900 cm²).

**Physical basis**: Resistance increase per unit fouling ∝ 1/area at fixed
pore-resistance density.

**Consequence**: Stage 1 (120 cm²) has 7.5× higher clog sensitivity than Stage 3.
Under fouling, Stage 1 ΔP dominates, not Stage 3. This is correct from a resistance
physics standpoint but may invert expected operational behavior (Stage 3 is where
microplastics physically accumulate fastest). Flagged for M2 calibration.

**Scale factors**:
- Stage 1: `k_eff = k_base × (900/120) ≈ 7.5×`
- Stage 2: `k_eff = k_base × (900/220) ≈ 4.1×`
- Stage 3: `k_eff = k_base × 1.0` (reference)

---

### D3 — Bypass Logic with Hysteresis

**Decision**: Passive bypass activates when `P_tentative > 65 kPa`, deactivates
when `P_tentative < 65 kPa × 0.90 = 58.5 kPa`. Hysteresis band prevents
step-level oscillation.

**Parameters (YAML-driven)**:
- `bypass_pressure_threshold_pa: 65000` (~81% of P_max = 80 kPa)
- `bypass_flow_fraction: 0.30` (30% diverted around filter)
- `bypass_hysteresis_fraction: 0.90`

**Risk**: Threshold is set relative to P_max_Pa. If P_max_Pa is recalibrated, bypass
behavior silently shifts. The threshold should eventually be a hardware-fixed spec.

---

### D4 — Decomposed Fouling Model

**Decision**: Replace scalar `Mc_i` loading with per-component primary state:
`cake_si`, `bridge_si`, `pore_si`, `irreversible_si`.

**Component weights per stage** (first-pass physical intuition):
- Stage 1 (coarse, 500 µm): bridge-dominant (60% bridge, 20% cake, 20% pore)
- Stage 2 (medium, 100 µm): mixed (45% bridge, 30% cake, 25% pore)
- Stage 3 (fine, 5 µm): cake-dominant (55% cake, 25% bridge, 20% pore)

**Aggregate compatibility**: `n_i = fouling_frac_si`, `Mc_i = fouling_frac_si × Mc_i_max`.
All pre-existing Mc/n fields preserved and recomputed each step.

---

### D5 — Backflush Recovery Coefficients

**Decision**: Recovery applies per-component with differential recoverability:
- Cake: 35% (surface deposit, most accessible to reverse flow)
- Bridge: 20% (structural entanglement, partial disruption)
- Pore: 8% (internal restriction, least accessible)

**Diminishing returns**: `recovery_scale = bf_source_eff × factor^max(n_bursts−1, 0)`
First burst gets full scale. Each subsequent burst: 80% of previous.

**Source efficiency**: 0.70 for autonomous mode (recirculated effluent). Clean water
service mode would use 1.0.

---

### D6 — Bistable Deposition Kinetics (Known Issue)

**Decision**: Use `dep_exponent = 2.0` (autocatalytic / quadratic in fouling_frac).

**Consequence** (discovered during validation):
The formula `dep ∝ (ff + ε)^2` combined with shear `∝ ff` creates bistable kinetics.
Each stage has an unstable equilibrium at:
```
ff_u = shear_coeff / (dep_rate × dep_base × Q_ref)
```

At default parameters:
- Stage 1: ff_u ≈ 0.667
- Stage 2: ff_u ≈ 0.417
- Stage 3: ff_u ≈ 0.222

Below ff_u: fouling returns to zero (self-cleaning).
Above ff_u: fouling accelerates to saturation.

**RL Training Impact**: Episodes starting from clean reset (`ff = 0`) will never
observe fouling development. The fouling penalty term in the reward is effectively
always 0 in clean-start training.

**Resolution (M1.5 sprint)**:
Change `dep_exponent: 2.0 → 1.0` in `configs/default.yaml`. No code changes needed.
With `dep_exponent=1`, the steady-state fouling exceeds 1.0 (saturates via clip),
producing monotone growth from any initial condition including clean reset.

---

### D7 — Milestone 1 Reward (5-term, interim)

**Decision**: Replace the pre-existing 3-term reward with a 5-term version that
exposes the new hydraulic and fouling signals.

```python
reward = (
    w_processed_flow   × (q_processed / Q_nominal_max)
  − w_pressure_penalty × max(0, pressure − 0.50)²
  − w_fouling_penalty  × max(0, mesh_avg − 0.40)
  − w_bypass_penalty   × (q_bypass / Q_nominal_max)
  − w_backflush_cost   × bf_active
)
```

**Key choices**:
- Flow term uses `q_processed` (filtered throughput), not `q_in` (includes bypass)
- Pressure penalty is quadratic and only activates above 50% normalized pressure
- Fouling penalty is linear above 40% average loading
- maintenance_required explicitly excluded (keeps reward continuous)
- All weights in `configs/default.yaml` under `reward:` section

**Reward range**: approximately [−0.85, +2.67] per step (asymmetric, uncapped upward).
Q_nominal_max = 15 L/min (nominal ceiling). Flow > 15 produces reward > w_processed_flow.

---

## Files Modified

| File | Change type |
|---|---|
| `configs/default.yaml` | +6 new sections; `max_Q_Lmin` bug fix (50→20) |
| `hydrion/state/init.py` | +~40 new truth_state fields |
| `hydrion/physics/hydraulics.py` | Full rewrite to v2 (API unchanged) |
| `hydrion/physics/clogging.py` | Full rewrite to v3 (API unchanged) |
| `hydrion/env.py` | Backflush state machine, M1 reward, extended logging |
| `hydrion/validation/milestone1_validation.py` | New file, 6 tests |

---

## Validation Results

All 10 validation tests pass:

| Test | Module | Status |
|---|---|---|
| pressure_flow_sweep | M1 new | PASS |
| fouling_nonlinearity | M1 new | PASS |
| backflush_recovery | M1 new | PASS |
| diminishing_returns | M1 new | PASS |
| bypass_activation | M1 new | PASS |
| nan_bounded_regression | M1 new | PASS |
| stress_matrix | Pre-existing | PASS |
| envelope_sweep | Pre-existing | PASS |
| mass_balance_test | Pre-existing | PASS |
| recovery_latency_test | Pre-existing | PASS |

---

## Open Issues for M1.5 Calibration Sprint

| ID | Issue | Fix | Effort |
|---|---|---|---|
| R1 | Bistable kinetics prevents RL training from clean reset | `dep_exponent: 2.0 → 1.0` in YAML | 1 line |
| C2 | Component sum can exceed `fouling_frac_si` at extreme params | Normalize components after clip in `_update_stage()` | ~5 lines |
| R3 | Area normalization inverts Stage 3 dominance intuition | Calibrate against hardware pressure drop data | Lab sprint |
| A3 | Bypass threshold coupled to P_max_Pa, not hardware spec | Decouple in YAML (add `bypass_pressure_pa_hardware_fixed: true`) | M2 |

---

## What Was Not Changed

- `obs12_v1` observation schema (12D, schema locked)
- truth_state / sensor_state boundary
- Physics pipeline execution order
- Public API of any physics model
- Pre-existing aggregate compatibility fields (Mc1/2/3, n1/2/3, mesh_loading_avg)
- Pre-existing validation tests (all still pass)
