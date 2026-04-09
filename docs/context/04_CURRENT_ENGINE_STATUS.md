# Current Engine Status

This document defines the **actual state of the HydrOS / Hydrion system as implemented today**.

It must be treated as:
- ground truth of current capabilities
- boundary of existing functionality
- reference for gap analysis

Do not assume features exist unless explicitly listed here.

---

# 1. Core Engine Status

## Environment

The system is implemented as a Gymnasium-compatible environment:

- `hydrion/env.py` (HydrionEnv)

### Current capabilities

- Continuous control environment
- Deterministic step loop
- Modular pipeline execution
- Stable action/observation interfaces
- Episode tracking and logging

---

# 2. Action Interface

## Current action space (4D continuous)

Action vector:


[valve_cmd, pump_cmd, bf_cmd, node_voltage_cmd]


### Properties

- Range: [0, 1]
- Applied directly to `truth_state`
- Used by:
  - hydraulics
  - electrostatics
  - clogging (indirectly)
  - backflush logic

### Status

- Stable
- Hardware-aligned
- No abstraction gap

---

# 3. Observation Interface

## Current observation space (12D)

Defined in:


hydrion/sensors/sensor_fusion.py


### Current ordering


0 flow
1 pressure
2 clog
3 E_norm
4 C_out
5 particle_capture_eff
6 valve_cmd
7 pump_cmd
8 bf_cmd
9 node_voltage_cmd
10 sensor_turbidity
11 sensor_scatter


### Properties

- Normalized to [0, 1]
- Deterministic construction
- Stable schema (`obs12_v1`)
- Used for RL and future UI binding

### Status

- Strong foundation
- Must remain unchanged unless versioned

---

# 4. Physics Pipeline

## Current execution order


Hydraulics
→ Clogging
→ Electrostatics
→ Particles
→ Sensors
→ Observation


### Status

- Explicitly defined
- Modular
- Correctly sequenced for current abstraction level

---

# 5. Hydraulics Module

**[UPDATED: Milestone 1 — 2026-04-09]**

## Implementation


hydrion/physics/hydraulics.py  (HydraulicsModel v2)


### Current behavior

- Solves pump-system operating point via quadratic intersection (monotone Q_in and P_in)
- Area-normalized clog sensitivity: `k_eff = k_base × (A_s3/A_si)`
- Passive bypass: activates above 65 kPa, hysteresis band prevents oscillation
- Splits Q_in into `q_processed_lmin` + `q_bypass_lmin`
- Computes explicit per-stage ΔP: `dp_stage1_pa`, `dp_stage2_pa`, `dp_stage3_pa`
- Writes to truth_state; sensor_state untouched

### Outputs

- `Q_out_Lmin`, `q_in_lmin`, `q_processed_lmin`, `q_bypass_lmin`
- `P_in`, `P_m1`, `P_m2`, `P_m3`, `P_out`
- `dp_stage1_pa`, `dp_stage2_pa`, `dp_stage3_pa`, `dp_total_pa`
- `bypass_active`

### Strengths

- Physically correct pump curve (quadratic intersection)
- Bypass prevents unrealistic pressure excursions
- Area normalization correctly scales resistance per unit fouling
- All parameters YAML-exposed; no hardcoded physics constants

### Limitations

- Not calibrated to real laundry flow profiles
- No transient surge modeling
- No temperature / viscosity variation
- Bypass threshold implicitly coupled to P_max_Pa (see audit issue A3)
- Area normalization inverts Stage 3 ΔP dominance intuition (see audit issue R3)

---

# 6. Clogging Module

**[UPDATED: Milestone 1 — 2026-04-09]**

## Implementation


hydrion/physics/clogging.py  (CloggingModel v3)


### Current behavior

- Decomposed fouling model with 4 primary state variables per stage:
  - `cake_si`, `bridge_si`, `pore_si` (recoverable components)
  - `irreversible_si` (permanent accumulation above 70% threshold)
- Aggregate compatibility fields maintained: `Mc1/2/3`, `n1/2/3`, `mesh_loading_avg`
- Non-monotonic capture efficiency curve: `baseline + gain × n × (1−n)^exponent`
- Nonlinear deposition: `dep ∝ (ff + ε)^dep_exponent`
- Passive shear removal: `∝ Q_in / Q_ref`

### KNOWN CALIBRATION ISSUE (M1.5)

`dep_exponent = 2.0` creates bistable kinetics. Each stage has an unstable fixed point:

```
ff_u = shear_coeff / (dep_rate × dep_base × Q_ref)
```

At default params: Stage 1 ≈ 0.667, Stage 2 ≈ 0.417, Stage 3 ≈ 0.222.

Below ff_u → self-cleans to zero. Above ff_u → accelerates to saturation.

**Consequence**: Clean-start RL training will never observe fouling. Fix: `dep_exponent: 2.0 → 1.0` in YAML (M1.5 sprint).

### Strengths

- Physically decomposed fouling (cake/bridge/pore) is hardware-aligned
- Irreversible fraction models permanent mesh degradation
- All component weights and kinetics parameters are YAML-exposed
- Full backward compatibility with pre-M1 observation schema

### Limitations

- dep_exponent=2 blocks clean-start fouling growth (M1.5 fix pending)
- Component weights are first-pass estimates, not hardware-validated
- Component sum can exceed fouling_frac at extreme params (C2, ~5 lines to fix)

---

# 7. Electrostatics Module

## Implementation


hydrion/physics/electrostatics.py


### Current behavior

- Maps node_voltage_cmd to:
  - V_node
  - E_field
  - E_norm

- Influences particle capture indirectly

### Strengths

- Stable abstraction
- Integrates cleanly with particle model

### Limitations

- Not physically grounded yet
- No conductivity dependence
- No particle-size dependence
- No spatial capture logic
- No separation between:
  - inlet conditioning
  - lower-node capture

---

# 8. Particle Module

## Implementation


hydrion/physics/particles.py


### Current behavior

- Tracks:
  - C_in
  - C_out
  - particle_capture_eff
  - C_fibers

- Supports:
  - PSD bins
  - fiber fraction
  - per-bin concentrations

### Strengths

- Mass conservation aware
- PSD-ready
- Shape-aware foundation exists

### Limitations

- Capture is still abstracted
- Not strongly tied to mesh geometry
- Limited coupling to electrostatics realism
- No strong size-selective calibration yet

---

# 9. Sensor System

## Implementation


hydrion/sensors/optical.py


### Current sensors

- turbidity
- scatter
- optional camera proxy

### Strengths

- Correct separation from truth_state
- Noise injection present
- Integrated into observation

### Limitations

- No differential pressure sensor
- No flow sensor
- No drift modeling
- No fouling modeling
- No calibration bias
- No latency modeling
- Camera not yet meaningful (AI abstraction missing)

---

# 10. Safety System

## Implementation


hydrion/wrappers/shielded_env.py
hydrion/safety/shield.py


### Current behavior

- Pre-action filtering:
  - clipping
  - rate limiting
  - sanity checks

- Post-step:
  - penalties
  - constraint detection
  - optional termination

### Strengths

- Correct architectural placement
- Works with RL training

### Limitations

- Multiple implementations (duplication risk)
- Event tracking not formalized
- Some internal mismatches (state access assumptions)

---

# 11. Validation System

## Implementation


hydrion/validation/
tests/


### Current components

- stress matrix
- envelope sweep
- mass balance
- recovery latency

### Strengths

- Strong research foundation
- Config-driven
- Reproducible

### Limitations

- Not fully integrated into training loop
- Not yet tied to realism calibration
- No automated benchmark reporting pipeline

---

# 12. Visualization / Observatory

## Implementation


hydrion/rendering/


### Current features

- episode history
- time-series plots
- anomaly visualization
- frame export
- video generation

### Strengths

- Strong research observability
- Non-intrusive (read-only)

### Limitations

- Not yet integrated with live UI
- No unified telemetry interface

---

# 13. Front-End Console (Phase 1.5)

## Implementation


apps/hydros-console/


### Current state

- React + TypeScript + Vite
- SystemCutaway SVG
- Metric panels
- Layout scaffold

### Limitations

- Static values (no live data)
- No telemetry binding
- No simulation control
- No run history
- No PPO comparison
- No cadence logic

---

# 14. Reward System

**[UPDATED: Milestone 1 — 2026-04-09]**

## Current reward (5-term, interim)

```python
reward = (
    w_processed_flow   × (q_processed / Q_nominal_max)      # +2.0
  − w_pressure_penalty × max(0, pressure − 0.50)²           # −1.0
  − w_fouling_penalty  × max(0, mesh_avg − 0.40)            # −0.5
  − w_bypass_penalty   × (q_bypass / Q_nominal_max)         # −0.3
  − w_backflush_cost   × bf_active                          # −0.1
)
```

Approximate range: [−0.85, +2.67] per step (uncapped upward).

### Key properties

- Uses `q_processed` (filtered throughput), not `q_in` (includes bypass)
- Pressure penalty is quadratic, activates above 50% normalized pressure
- Fouling penalty is linear, activates above 40% average loading
- `maintenance_required` flag explicitly excluded (keeps reward continuous)
- All weights in `configs/default.yaml` under `reward:` section

### Strengths

- Exposes hydraulic/fouling tradeoffs to RL agent
- Penalizes bypass explicitly (unfiltered flow costs reward)
- Penalizes backflush cost (discourages unnecessary triggering)
- Continuous reward surface — no discrete cliffs

### Limitations

- Does NOT prioritize capture efficiency directly (M6)
- Does NOT reflect energy usage (M6)
- Q_nominal_max (15 L/min) means flow > 15 produces reward > w_processed_flow (intentional but worth noting)
- Interim design — will be replaced by multi-objective reward in M6

---

# 15. Summary

## What is strong

- modular architecture
- state separation
- stable observation contract
- validation foundation
- RL-ready structure
- hardware-aligned control inputs

## What is abstract

- electrostatics realism (M3 target)
- sensor realism — no differential pressure sensor, no drift modeling (M5 target)
- reward alignment — capture not yet primary signal (M6 target)
- telemetry integration (console track)

## What was abstract, now concrete (Milestone 1)

- hydraulics: pump curve, area-normalized resistance, bypass logic, per-stage ΔP
- clogging: decomposed fouling (cake/bridge/pore), irreversible fraction, capture efficiency curve
- backflush: pulse state machine, partial recovery, diminishing returns, cooldown

---

# Final Statement

HydrOS is currently:

> A structurally correct, modular, RL-ready digital twin with hydraulic and fouling
realism grounded in Milestone 1. The physics pipeline now captures pump curve behavior,
decomposed fouling, passive bypass, and backflush recovery as physically meaningful
interactions. Electrostatics, sensor realism, and reward alignment remain abstract.

The next phase (M1.5 calibration sprint) must:

- Fix bistable deposition kinetics (`dep_exponent: 2.0 → 1.0`)
- Begin hardware ΔP calibration to resolve Stage 3 dominance question

Then (M3):

- Introduce physically grounded electrostatic conditioning and capture