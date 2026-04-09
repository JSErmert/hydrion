# Repository Map

This document defines the **structural layout of the HydrOS / Hydrion repository**.

It is intended to:
- guide navigation
- identify canonical modules
- prevent duplication of logic
- clarify ownership of system components

Claude must treat this as the **source of truth for file structure**.

---

# 1. Repository Overview

The repository is organized into four primary domains:

1. Core Engine (Hydrion)
2. Front-End Console (HydrOS)
3. Configuration & Validation
4. Training, Evaluation, and Visualization Tools

---

# 2. Core Engine (Hydrion)

## Root


hydrion/


This directory contains the **simulation engine**.

---

## 2.1 Environment


hydrion/env.py


### Responsibility

- main simulation loop
- pipeline orchestration
- action application
- observation generation
- reward calculation
- logging integration

### Notes

- this is the **single most important file**
- defines pipeline ordering
- defines control interface

---

## 2.2 Configuration


hydrion/config.py
configs/


### Responsibility

- YAML loading
- parameter access
- config hashing
- runtime configuration control

### Notes

- ALL parameters must originate from YAML
- no hardcoding allowed

---

## 2.3 State


hydrion/state/


### Responsibility

- initialize truth_state
- initialize sensor_state
- define state structure

### Key Concept

- truth_state = physics truth
- sensor_state = measurement layer

---

## 2.4 Physics Modules


hydrion/physics/


Contains:

- hydraulics.py
- clogging.py
- electrostatics.py
- particles.py

---

### Responsibilities

#### Hydraulics
- flow
- pressure
- stage pressure distribution

#### Clogging
- mesh loading
- clog state
- capture baseline

#### Electrostatics
- node voltage
- field strength
- normalized electrostatic effect

#### Particles
- concentration propagation
- capture efficiency
- PSD and shape handling

---

### Rules

- each module updates only truth_state
- no module may write to sensor_state
- modules communicate only through state

---

## 2.5 Sensors


hydrion/sensors/


Contains:

- optical.py
- sensor_fusion.py

---

### Responsibilities

#### OpticalSensorArray
- turbidity proxy
- scatter proxy
- optional camera proxy

#### Sensor Fusion
- builds observation vector
- defines obs12_v1 schema

---

### Rules

- sensors write ONLY to sensor_state
- observation ordering must remain stable

---

## 2.6 Safety


hydrion/safety/
hydrion/wrappers/


Contains:

- shield.py
- shielded_env.py

---

### Responsibility

- enforce constraints
- filter actions
- apply penalties
- terminate unsafe episodes

---

### Notes

- safety is NOT part of physics
- must remain modular

---

## 2.7 Validation


hydrion/validation/
tests/


Contains:

- stress_matrix.py
- envelope_sweep.py
- mass_balance_test.py
- recovery_latency_test.py

---

### Responsibility

- test system correctness
- enforce invariants
- validate realism behavior

---

## 2.8 Runtime


hydrion/runtime/


Contains:

- run_context.py
- seeding.py

---

### Responsibility

- reproducibility
- run identity
- seed control

---

## 2.9 Logging


hydrion/logging/


### Responsibility

- run metadata logging
- timestep logging
- reproducibility tracking

---

## 2.10 Rendering / Observatory


hydrion/rendering/


Contains:

- observatory.py
- episode_history.py

---

### Responsibility

- visualization support
- time-series analysis
- anomaly detection
- video generation

---

### Notes

- must remain read-only
- no influence on simulation

---

# 3. Front-End Console (HydrOS)

## Location


apps/hydros-console/


---

## Structure


src/
main.tsx
App.tsx
layout/
components/
panels/
system/
styles/


---

## Responsibilities

- display system state
- visualize metrics
- present system cutaway
- show validation indicators

---

## Current Status

- static scaffold
- no live data binding
- no telemetry integration

---

## Future Role

- real-time observability
- mission-control interface
- run comparison system

---

# 4. Configuration & Validation Data


configs/
configs/validation/


---

## Responsibility

- define system parameters
- define validation conditions
- enable reproducibility

---

# 5. Training & Evaluation

## Scripts


train_ppo.py
train_ppo_v15.py
eval_ppo.py


---

## Visualization Tools


run_visual.py
visualize_episode.py
visualize_timeseries.py
make_video.py


---

## Responsibility

- train policies
- evaluate performance
- generate visual outputs

---

# 6. Canonical Files

These files must be treated as **primary sources of truth**:

- `hydrion/env.py`
- `hydrion/sensors/sensor_fusion.py`
- `hydrion/physics/*`
- `hydrion/config.py`
- `configs/default.yaml`

---

# 7. Known Structural Risks

## 1. Safety duplication

- shield.py
- shielded_env.py

Risk:
- inconsistent behavior

---

## 2. Sensor-state mirroring

Sensors may mirror values into truth_state.

Risk:
- violates separation principle

---

## 3. Console duplication risk

Multiple SystemView implementations may exist.

Risk:
- drift in visualization logic

---

# 8. Final Rule

> Every change must respect module ownership.

Before modifying code, identify:

- which module owns the behavior
- which state is being modified
- which pipeline stage is affected

No change should cross boundaries without explicit justification.