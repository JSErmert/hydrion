# HydrOS / Hydrion

## Research-Grade Digital Twin for Microplastic Extraction

HydrOS is a **modular, physics-driven simulation engine** designed to model and optimize a **handheld microplastic extraction system for laundry outflow**.

It combines:

- multi-stage filtration physics
- electrostatic capture modeling
- sensor realism
- Safe reinforcement learning (RL)
- validation-driven engineering
- mission-control observability

---

# What This Is

HydrOS is:

- a **digital twin**
- a **control system simulator**
- a **research platform**
- a **hardware-forward architecture**

It is designed to evolve toward:

> predictive alignment with real-world lab data and eventual physical deployment

---

# What This Is NOT

- not a toy RL environment
- not a dashboard demo
- not a monolithic simulation
- not a UI-first project

---

# Core Concepts

## Modular Physics Pipeline

Hydrion simulates:

- Hydraulics
- Clogging (tri-stage mesh)
- Electrostatics (field-assisted capture)
- Particles (PSD + shape)
- Sensors (optical + future expansion)

---

## State Separation

- `truth_state` → physical reality
- `sensor_state` → measured reality

---

## Stable Interfaces

- 4D action space
- 12D observation vector (immutable)

---

## Safe RL

- action filtering
- constraint enforcement
- failure detection

---

## Validation First

- stress testing
- envelope sweeps
- mass balance
- recovery dynamics

---

# Repository Structure

See:


docs/context/03_REPO_MAP.md


---

# Current Phase

**Phase 1.5 — Research Console + Realism Backbone**

Focus:

- telemetry binding
- hydraulic + clogging realism
- backflush dynamics
- electrostatic modeling
- RL benchmarking

---

# Development Philosophy

HydrOS is built with:

- modular discipline
- validation-first mindset
- hardware-forward thinking
- controlled iteration

---

# Documentation

Full system context:


docs/context/


Start here:

- 01_SYSTEM_IDENTITY.md
- 02_ARCHITECTURE_CONSTRAINTS.md
- 04_CURRENT_ENGINE_STATUS.md
- 06_LOCKED_SYSTEM_CONSTRAINTS.md
- 09_REALISM_ROADMAP.md

---

# Running the System

(leave this minimal for now — can expand later)

---

# Final Note

This repository is not a generic codebase.

It is:

> a structured engineering system designed to simulate, understand, and eventually control 