# Locked System Constraints

The following values are **locked system-level constraints**.

They are NOT exploratory.

They must be treated as **authoritative inputs** for:
- model design
- parameter initialization
- reward shaping
- module interfaces
- validation targets

---

# A. Laundry Outflow Operating Envelope

## Flow Regimes

- Low-flow: 5 L/min  
- Nominal: 12–15 L/min  
- Peak transient: 20 L/min  

## Throughput Requirement

- Continuous processing: 12–15 L/min  
- Short-duration tolerance: up to 20 L/min  

## Flow Doctrine

- Full-flow processing is baseline  
- Bypass is protective, not primary  
- System must prioritize stable operation under nominal flow  

---

# B. Stage Geometry and Fouling Capacity

## Effective Areas

- Stage 1: 120 cm²  
- Stage 2: 220 cm²  
- Stage 3: 900 cm² (pleated effective area)

## Fouling Threshold

- Maintenance threshold: ~70% capacity

## Interpretation

- Not cosmetic
- Represents degraded hydraulics and recovery
- Must trigger nonlinear system behavior

---

# C. Backflush Dynamics

## Pulse Structure

- 3 pulses
- 0.4 s pulse duration
- 0.25 s spacing

## Cooldown

- 8–10 seconds no retrigger

## Fluid Source

- Default: recirculated filtered effluent  
- Secondary: clean water (service mode only)

## Doctrine

- Autonomous system must not depend on external clean water
- Cleaning effectiveness depends on source fluid

---

# D. Electrostatic Capture

## Voltage

- Nominal: 0–1.5 kV  
- Upper realism bound: 2.5 kV  
- Hard clamp: 3.0 kV  

## Modeling Approach

- AC-equivalent / DEP-like abstraction
- NOT corona-based
- NOT purely DC electrophoresis

## Functional Allocation

- 70% capture: lower collector node  
- 30% capture: inlet polarization ring  

## Interpretation

- Lower node = primary capture region
- Ring = upstream conditioning, not sink

---

# E. Particle Density Scope

## Decision: Option A — Dense-Phase Only

**Locked 2026-04-10**

HydrOS targets **dense microplastics only**:

- Target class: `ρ > 1.0 g/cm³`
- Primary targets: PET (1.38), PA/nylon (1.14), PVC (1.16–1.58), biofilm-coated fragments
- Excluded class: buoyant microplastics — PP (ρ ≈ 0.91), PE (ρ ≈ 0.95)

## Collection Topology

Collection tubes exit downward from the outer wall node of each stage.
Gravity-fed downward path is physically correct for ρ > 1.0 g/cm³ particles.

Buoyant-phase capture (PP, PE) requires separate upstream treatment — **out of scope**.
Dual-path (upward + downward) collection is NOT implemented and NOT planned.

## Simulation Implication

The buoyant fraction (PP, PE) passes through the system uncaptured.
`C_in_buoyant` must be tracked as pass-through in M4 particle module.
System efficiency is defined over the dense-phase fraction only.

---

# F. Realism Prioritization

## Primary

- Pressure vs flow behavior

## Secondary

- Backflush recovery dynamics
- Partial recovery + diminishing returns
- Irreversible fouling accumulation

## Tertiary

- Particle-selective capture (PSD)

---

# Implementation Doctrine

Realism must be built in this order:

1. Hydraulics
2. Fouling
3. Backflush
4. Electrostatics
5. Particle selectivity

---

# Final Rule

Do NOT expand features broadly.

Focus on:

> High-fidelity interaction between flow, clogging, and maintenance first.

Everything else builds on that.