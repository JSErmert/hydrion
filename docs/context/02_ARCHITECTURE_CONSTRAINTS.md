# Architecture Constraints

This document defines the **non-negotiable rules** governing HydrOS.

These constraints must be preserved across all development, refactoring, and realism upgrades.

---

## 1. Truth vs Sensor Separation

### Rule
- `truth_state` is authoritative
- `sensor_state` is observational only

### Implications
- Physics modules write ONLY to `truth_state`
- Sensor modules write ONLY to `sensor_state`
- Sensors MUST NOT modify physical truth

---

## 2. Immutable Pipeline Ordering

The simulation pipeline is fixed:

Hydraulics  
→ Clogging  
→ Electrostatics  
→ Particles  
→ Sensors  
→ Observation  
→ Safety Shield (external wrapper)

### Rule
- This order must NOT change without explicit architectural migration

### Reason
- Defines system semantics
- Ensures reproducibility
- Prevents hidden coupling

---

## 3. Observation Contract Stability

### Current schema
- 12-dimensional observation vector (`obs12_v1`)

### Rule
- No reordering
- No silent additions
- No silent removals

### If change is required
- Create new schema version (e.g., `obs16_v2`)
- Update all dependent systems explicitly

---

## 4. YAML-Driven Configuration

### Rule
- All tunable parameters must be defined in YAML

### Forbidden
- Hardcoded physics constants
- Hidden “magic numbers”

### Required
- Parameter documentation
- Sensible defaults
- Reproducible hashing

---

## 5. Safety Isolation

### Rule
- Safety logic must NOT be embedded in physics modules

### Structure
- Safety exists as wrapper (ShieldedEnv / SafeRLShield)

### Implication
- Physics remains pure
- Safety is modular and testable

---

## 6. Rendering / Console Isolation

### Rule
- UI must be read-only

### Forbidden
- Direct mutation of simulation state
- UI-driven physics changes

### Allowed
- Run commands (reset, step, execute)
- Telemetry consumption

---

## 7. No Architectural Flattening

### Forbidden
- merging modules for convenience
- collapsing physics into a single file
- bypassing module boundaries

---

## 8. Determinism Preservation

### Rule
- Simulation must be reproducible under fixed seed

### Implication
- randomness must be controlled
- noise must be configurable

---

## 9. Realism Discipline

### Rule
- realism must be incremental and validated

### Forbidden
- adding complexity without testability
- introducing physics without calibration path

---

## 10. Development Doctrine

All work must prioritize:

1. correctness
2. clarity
3. stability
4. interpretability
5. scalability

Speed is secondary to architectural integrity.