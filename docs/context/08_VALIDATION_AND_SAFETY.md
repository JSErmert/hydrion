# Validation and Safety

This document defines the **validation framework and safety doctrine** for HydrOS.

It establishes:
- what constitutes correct behavior
- how realism is measured
- how failures are detected
- how unsafe states are handled
- how validation integrates with development and RL

---

# 1. Validation Philosophy

HydrOS is a **research-grade system**.

This means:

> A feature is not complete when it works.  
> A feature is complete when it is validated.

---

## Core Principle

Validation is not optional.

Every realism upgrade must:
- be testable
- be measurable
- be reproducible
- be stressable

---

# 2. Validation Protocol v2

HydrOS includes a structured validation suite located in:


hydrion/validation/
tests/


---

## Current Components

### 1. Stress Matrix

- runs system under extreme/random conditions
- tests:
  - stability
  - bounded behavior
  - failure modes

---

### 2. Envelope Sweep

- explores action space systematically
- verifies:
  - outputs remain within valid ranges
  - system does not produce NaNs or explosions

---

### 3. Mass Balance Test

- ensures:
  - C_out ≤ C_in
  - capture efficiency is valid
  - conservation principles are respected

---

### 4. Recovery Latency Test

- introduces disturbance
- measures:
  - time to recover to stable operation
  - effectiveness of control / backflush

---

### 5. Scenario Testing (implicit / extendable)

- structured disturbances
- known expected behavior patterns

---

# 3. What “Passing” Means

A system is considered valid only if:

## Stability
- no NaNs
- no divergence
- bounded state variables

## Physical Plausibility
- values remain within realistic ranges
- relationships are consistent (pressure vs flow, clog vs pressure)

## Mass Integrity
- no artificial particle creation
- no negative concentrations
- capture efficiency in [0,1]

## Recovery Behavior
- system recovers after disturbance
- recovery is not instantaneous (must be realistic)

## Repeatability
- same seed → same results

---

# 4. Safety Doctrine

Safety in HydrOS is not embedded in physics.

It is implemented as a **separate control layer**.

---

## Safety Location


hydrion/wrappers/shielded_env.py
hydrion/safety/shield.py


---

## Safety Responsibilities

- prevent unsafe actions
- enforce system constraints
- apply penalties
- terminate if necessary

---

## Safety MUST NOT:

- modify physics directly
- alter truth_state calculations
- bypass pipeline logic

---

## Safety MUST:

- operate on top of environment
- act on actions before execution
- evaluate state after execution

---

# 5. Pre-Action Safety (Projection Layer)

Before actions are applied:

- clip to valid range
- enforce rate limits
- enforce actuator sanity rules

### Example

- prevent pump from exceeding safe ramp rate
- prevent backflush spam
- prevent valve-pump mismatch

---

# 6. Post-Step Safety (Constraint Layer)

After step execution:

### Soft constraints
- pressure exceeds soft threshold → penalty
- clog exceeds soft threshold → penalty

### Hard constraints
- pressure exceeds hard threshold → violation
- clog exceeds critical threshold → violation
- flow collapse under load → violation

---

## Possible Actions

- apply penalty
- project action
- terminate episode

---

# 7. Safety Signals

Safety must expose:

- violation flags
- projected vs raw action
- penalty applied
- reason for violation

These must be accessible via:


info["safety"]


---

# 8. Known Issues (Current Repo)

These must be addressed carefully:

## Issue 1 — Multiple Shield Implementations

- `shield.py`
- `shielded_env.py`

### Risk
- inconsistent behavior
- unclear canonical logic

---

## Issue 2 — State Reference Mismatch

Some safety logic assumes:


env.state


But actual engine uses:


env.truth_state


### Required Fix

Safety must read:


truth_state


not a non-existent state object.

---

## Issue 3 — Event Tracking Missing

Currently:
- safety events are not formalized

Needed:
- structured event logging
- event IDs
- time association

---

# 9. Validation Integration with Development

Every module upgrade must follow:

1. Implement feature
2. Run validation suite
3. Analyze failure modes
4. Adjust parameters
5. Re-run validation
6. Only then proceed

---

# 10. Validation Integration with RL

Validation must precede RL training.

---

## Required before training

- stable simulation
- bounded outputs
- no NaNs
- correct mass behavior

---

## During training

Monitor:

- violation frequency
- recovery time
- instability patterns
- reward consistency

---

## After training

Compare:

- PPO vs baseline
- stability vs performance tradeoff
- maintenance efficiency

---

# 11. Calibration Doctrine

Realism requires calibration.

Each milestone must include:

## Measurable target

Examples:
- pressure vs flow curve
- clogging progression
- recovery after backflush

## Parameter tuning

- adjust YAML parameters
- match expected physical behavior

---

## Future Calibration Path

- lab bench data
- empirical curves
- sensor readings
- flow measurements

---

# 12. Final Rule

> No realism is accepted without validation.

> No validation is meaningful without defined expectations.

HydrOS must remain:

- testable
- falsifiable
- measurable
- interpretable

at every stage of development.