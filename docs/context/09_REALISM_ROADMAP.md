# Realism Roadmap

This document defines the **ordered implementation roadmap** for evolving HydrOS into a high-fidelity microplastic extraction simulation engine.

This roadmap is **authoritative**.

Claude must:
- follow this order
- avoid skipping stages
- avoid parallel expansion across modules
- focus on fidelity concentration per milestone

---

# Guiding Principle

> Realism is not achieved by adding features everywhere.
> Realism is achieved by deepening truth in the correct order.

---

# System-Level Priority Order

Realism must be built in this sequence:

1. Hydraulics
2. Clogging
3. Backflush
4. Electrostatics
5. Particle selectivity
6. Sensors
7. Reward + control

---

# Milestone 1 — Hydraulic + Fouling Backbone

**STATUS: COMPLETE**
**Date**: 2026-04-09
**Branch**: `HydrOS-x-Claude-Code`
**Commits**: `1bc1f5f` (implementation) + `b38d006` (pre-merge docs patches)
**Validation**: 10/10 tests pass — see `docs/updates/milestone1_design_record.md`

## Delivered

- Pump-system curve intersection (quadratic solve, monotone P_in/Q_in)
- Area-normalized clog sensitivity (`k_eff = k_base × A_s3/A_si`)
- Passive bypass with hysteresis (65 kPa threshold, 90% deactivation band)
- Decomposed fouling: per-stage `cake`, `bridge`, `pore`, `irreversible` state
- Non-monotonic capture efficiency curve (YAML-parameterized)
- 5-term reward exposing hydraulic/fouling tradeoffs to RL
- 6 new validation tests (4 pre-existing also pass)

## Known Issues (M1.5 sprint)

| ID | Issue | Fix |
|---|---|---|
| R1 | dep_exponent=2 → bistable; filter won't foul from clean reset | `dep_exponent: 1.0` in YAML |
| C2 | Component sum can exceed fouling_frac at extreme params | Normalize after clip in `_update_stage()` |
| R3 | Area normalization inverts Stage 3 ΔP dominance | Lab calibration sprint |
| A3 | Bypass threshold coupled to P_max_Pa | Decouple in M2 |

---

# Milestone 1.5 — Calibration Sprint

**STATUS: OPEN**
**Priority**: HIGH — R1 (bistable kinetics) blocks effective RL training

## Scope

1. Fix bistable deposition kinetics: `dep_exponent: 2.0 → 1.0` in `configs/default.yaml`
2. Fix component sum normalization: ~5 lines in `clogging.py` `_update_stage()`
3. Begin hardware ΔP calibration: tune `dep_rate_s1/s2/s3` against physical pressure drop data
4. Validate that clean-start episodes now produce observable fouling growth
5. Verify RL training produces non-degenerate control policies under new kinetcs

## Acceptance Criteria

- `dep_exponent=1`: fouling grows from `ff=0` to saturation within a reasonable episode
- `mesh_loading_avg` reaches 0.70 from clean reset in ≤ 500 steps at nominal flow
- M1 validation suite (10/10) still passes after parameter changes

---

# Milestone 2 — Backflush Dynamics

**STATUS: COMPLETE (merged into Milestone 1 sprint)**
**Date**: 2026-04-09

Delivered in `HydrOS-x-Claude-Code` commit `1bc1f5f`:
- Multi-burst pulse state machine (3 pulses, 0.4 s, 0.25 s spacing, 9 s cooldown)
- Per-component recovery (cake 35%, bridge 20%, pore 8%)
- Diminishing returns (80% factor per successive burst)
- Recirculated effluent source (70% efficiency); clean water mode hook present
- Cooldown prevents unrealistic oscillation

All M2 acceptance criteria satisfied. Backflush is now a physically meaningful maintenance system.

---

# Milestone 3 — Electrostatic Conditioning + Capture

## Objective

Introduce physically grounded electrostatic behavior.

---

## Scope

### Split into two subsystems

#### A. Polarization Ring (Upstream)

- modifies particle charge state
- depends on:
  - residence time
  - voltage
  - conductivity (future hook)

#### B. Lower Collector Node (Primary)

- dominant capture region (~70%)
- increases retention probability
- strongest at fine mesh stage

---

## Modeling Approach

- AC-equivalent / DEP-like abstraction
- no corona discharge baseline
- not purely DC-only model

---

## Add Parameters

- node_voltage (0–1.5 kV nominal)
- field_strength_factor
- capture_gain
- conductivity_attenuation (future)

---

## Acceptance Criteria

- capture improves when electrostatics enabled
- effect depends on flow and residence time
- turning electrostatics off produces measurable drop
- system remains numerically stable

---

# Milestone 4 — Particle Realism

## Objective

Improve representation of microplastics.

---

## Scope

### Particle Taxonomy

Primary classes:
- fibers
- fragments

Subclasses:
- coarse / fine

---

### PSD Improvements

- realistic bin distributions
- inflow variability
- size-dependent capture

---

### Material (future)

- PET
- PP
- PE
- PA

---

## Acceptance Criteria

- capture varies by size class
- PSD shifts across stages
- fiber behavior differs from fragments

---

# Milestone 5 — Sensor Realism

## Objective

Make the system observable in a realistic way.

---

## Scope

Add sensors:

- differential pressure
- flow
- turbidity
- optical scatter
- AI camera proxy

Add realism:

- noise
- drift
- fouling
- bias
- latency
- dropout

---

## Acceptance Criteria

- sensors differ from truth_state
- sensor error behaves realistically
- system remains stable under noise

---

# Milestone 6 — Reward + Control

## Objective

Align RL behavior with real system goals.

---

## Replace reward with multi-objective form:

- capture mass (primary)
- pressure penalty
- clog penalty
- energy usage
- maintenance cost
- smoothness

---

## Acceptance Criteria

- PPO outperforms baseline controller
- agent learns maintenance timing
- system avoids unsafe states
- trade-offs are visible and interpretable

---

# Milestone 7 — Validation Integration

## Objective

Tie realism to measurable correctness.

---

## Extend validation protocol:

- calibration curves
- repeatability checks
- parameter sweeps
- scenario testing

---

## Acceptance Criteria

- each module validated independently
- combined system passes full protocol
- reproducible across seeds

---

# Development Rules

## DO

- implement one module at a time
- validate before moving forward
- preserve architecture
- document parameter meaning

## DO NOT

- expand multiple modules simultaneously
- change observation schema casually
- introduce realism without validation
- optimize before system is physically grounded

---

# Final Directive

Claude must treat this roadmap as:

> The authoritative sequence for transforming HydrOS from a stable simulation into a realistic, physically meaningful system.

Deviation from this order requires explicit justification.