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

## Objective

Establish physically meaningful interaction between:
- flow
- pressure
- clogging
- throughput

This is the **foundation of realism**.

---

## Scope

### Hydraulics

Upgrade current model to include:

- flow envelope constraints:
  - 5 L/min (low)
  - 12–15 L/min (nominal)
  - 20 L/min (transient)

- throughput limitation:
  - handheld system capacity
  - overflow / bypass logic

- stage-area-dependent flow resistance:
  - Stage 1: 120 cm²
  - Stage 2: 220 cm²
  - Stage 3: 900 cm²

---

### Clogging

Replace current abstraction with hybrid model:

1. surface cake accumulation (primary)
2. fiber bridging / entanglement (secondary)
3. pore restriction (tertiary)

Add:

- recoverable clog fraction
- irreversible fouling fraction
- nonlinear degradation near ~70% capacity

---

## Acceptance Criteria

- pressure vs flow curve behaves realistically
- clog increases pressure nonlinearly
- system reaches degraded state near threshold
- no NaNs or instability
- validation protocol passes

---

# Milestone 2 — Backflush Dynamics

## Objective

Convert backflush from abstraction → physically meaningful maintenance system.

---

## Scope

### Backflush Model

Implement:

- multi-burst square pulse:
  - 3 pulses
  - 0.4 s duration
  - 0.25 s spacing
  - 8–10 s cooldown

### Cleaning Behavior

Model:

- partial recovery
- diminishing returns
- irreversible fouling persistence
- stage-specific cleaning efficiency

### Fluid Source

- default: recirculated effluent
- secondary: clean water (service mode)

---

## Acceptance Criteria

- clog decreases during pulses
- recovery is partial, not perfect
- repeated cycles show degradation
- cooldown prevents unrealistic oscillation
- RL can learn maintenance timing

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