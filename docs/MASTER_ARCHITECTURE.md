# HydrOS Master Architecture

This document defines the **top-level architecture** of HydrOS.

It provides:
- system overview
- structural relationships
- execution flow
- development direction

It is the **bridge** between:
- README (what the system is)
- CLAUDE.md (how to behave)
- /docs/context (deep system knowledge)

---

# 1. System Definition

HydrOS is a **research-grade digital twin platform** designed to simulate and optimize a:

> handheld microplastic extraction system for laundry outflow

The system integrates:

- fluid dynamics
- filtration physics
- electrostatic capture
- sensor modeling
- reinforcement learning control
- validation-driven engineering
- real-time observability

---

# 2. System Layers

HydrOS is composed of the following layers:

---

## 2.1 Physical Simulation Layer (Hydrion Engine)

Location:

hydrion/


Modules:
- Hydraulics
- Clogging
- Electrostatics
- Particles

Responsibilities:
- compute system state
- propagate physical interactions
- enforce deterministic behavior

---

## 2.2 Sensor & Observation Layer

Location:

hydrion/sensors/


Components:
- Optical sensors
- Sensor fusion

Responsibilities:
- convert truth_state → sensor_state
- generate observation vector
- simulate measurement limitations

---

## 2.3 Safety & Constraint Layer

Location:

hydrion/safety/
hydrion/wrappers/


Responsibilities:
- enforce operational limits
- filter unsafe actions
- detect violations
- apply penalties / termination

---

## 2.4 Validation Layer

Location:

hydrion/validation/
tests/


Components:
- stress matrix
- envelope sweep
- mass balance
- recovery latency

Responsibilities:
- verify system correctness
- enforce realism constraints
- validate new features

---

## 2.5 Policy / Control Layer

Components:
- PPO training
- baseline controllers

Responsibilities:
- optimize system behavior
- manage trade-offs
- evaluate performance

---

## 2.6 Telemetry Layer

Responsibilities:
- expose simulation state
- provide structured data interface
- support observability and replay

Key concept:
- simulation → telemetry → UI

---

## 2.7 Front-End Console Layer

Location:

apps/hydros-console/


Responsibilities:
- visualize system
- display metrics
- support analysis
- compare runs

Role:
- observer, not controller

---

# 3. Canonical Simulation Pipeline

The system executes in a fixed order:


Hydraulics
→ Clogging
→ Electrostatics
→ Particles
→ Sensors
→ Observation
→ Safety (wrapper)


---

## Rule

This ordering is **immutable**.

It defines:
- system semantics
- data dependencies
- RL behavior

---

# 4. State Architecture

HydrOS maintains strict state separation:

---

## truth_state

- physical reality
- updated by physics modules only

---

## sensor_state

- measurement outputs
- updated by sensors only

---

## observation

- derived from truth + sensor
- fixed 12D vector

---

## action

- 4D control vector
- applied each step

---

## reward

- scalar performance signal
- used for RL optimization

---

## telemetry

- structured snapshot of system state
- consumed by UI

---

# 5. Repository Structure

## Core Engine

hydrion/


## Frontend

apps/hydros-console/


## Configuration

configs/


## Validation

tests/
hydrion/validation/


## Documentation

docs/
docs/context/


---

# 6. Current Development Phase

## Phase 1.5 — Research Console + Realism Backbone

---

## Focus Areas

1. telemetry binding
2. hydraulic realism
3. clogging realism
4. backflush dynamics
5. electrostatic modeling
6. RL benchmarking

---

# 7. Realism Priority Order

All development must follow this sequence:

1. Hydraulics
2. Clogging
3. Backflush
4. Electrostatics
5. Particles
6. Sensors
7. Reward / Control

---

## Rule

No skipping.  
No parallel expansion.

---

# 8. Governance Model

HydrOS operates under the:

> Council of HydrOS

---

## Structure

- Co-Orchestrator
- Implementation Engineer
- Validation Engineer
- Sensor Research Agent
- Visualization Architect
- Refactor Agent
- Policy Agent
- Hardware Translation Agent

---

## Purpose

- maintain discipline
- enforce constraints
- prevent drift
- accelerate correctly

---

# 9. Documentation Hierarchy

Claude and engineers must read in this order:

---

## 1. README.md
System overview

## 2. MASTER_ARCHITECTURE.md
System structure (this file)

## 3. CLAUDE.md
Execution behavior rules

## 4. docs/context/
Detailed system knowledge

---

# 10. Final Directive

HydrOS must be developed as:

- modular
- validated
- constraint-driven
- physically grounded
- hardware-forward

---

## Summary

HydrOS is not a codebase.

It is:

> a structured engineering system designed to simulate, understand, and eventually control a real-world device.