# Realism Target

This document defines the **physical system HydrOS is attempting to simulate**.

It is the anchor for:
- physics design
- model calibration
- system interpretation
- realism evaluation

This document must be treated as:
> the closest representation of the real-world system intent

---
## System Reference

See:

docs/assets/hydros_system_reference.md

# 1. System Context

## Primary Domain

- household laundry outflow
- microfiber-rich wastewater
- variable flow, pressure, and chemical composition

---

## Device Scale

- handheld consumer device
- inline drain attachment
- continuous real-time operation

---

## Operational Mode

- full-flow processing under nominal conditions
- bypass only under transient overload
- autonomous maintenance via backflush

---

# 2. Physical Architecture

## Inlet Stage

- swirl conditioning
- polarization ring (electrostatic preconditioning)

---

## Filtration Stages

### Stage 1 — Coarse Mesh
- ~500 µm
- captures:
  - hair
  - lint
  - large debris

---

### Stage 2 — Medium Mesh
- ~100 µm
- captures:
  - smaller fibers
  - medium fragments

---

### Stage 3 — Fine Pleated Cartridge
- ~5 µm equivalent
- large effective surface area
- wrapped or assisted by electrostatic field

---

## Electrostatic System

- polarization ring at inlet
- dominant collector node at bottom
- optional charged fine mesh
- field-assisted capture, not purely mechanical

---

## Backflush System

- pulse-based cleaning
- multi-burst actuation
- recirculated effluent as primary cleaning fluid
- clean-water mode for service only

---

## Storage System

- detachable reservoir
- stores captured material
- tracks fill level (future realism extension)

---

# 3. Particle Reality

## Primary Classes

- fibers (dominant)
- fragments

---

## Secondary Considerations

- size distribution (PSD)
- shape-dependent behavior
- material properties (future)

---

## Behavior Expectations

- fibers:
  - entangle
  - bridge
  - form surface cake

- fragments:
  - accumulate
  - block pores
  - follow size-dependent capture

---

# 4. Flow Reality

## Envelope

- low: 5 L/min
- nominal: 12–15 L/min
- peak: 20 L/min

---

## Behavior

- variable flow
- transient spikes
- pressure buildup with clogging
- flow reduction under heavy fouling

---

# 5. Clogging Reality

Clogging must include:

- surface cake formation
- fiber bridging
- pore restriction

---

## Recovery

- mostly recoverable via backflush
- partial recovery is realistic
- irreversible fouling must exist under stress

---

# 6. Electrostatic Reality

## Mechanism

- field-assisted capture (DEP-like abstraction)
- charge conditioning at inlet
- dominant capture at lower node

---

## Not Included (baseline)

- corona discharge
- purely DC electrophoresis

---

## Dependencies

- field strength
- particle size
- shape
- residence time
- flow velocity
- conductivity (future)

---

# 7. Sensor Reality

Target sensor system:

- differential pressure sensor
- flow sensor
- turbidity sensor
- optical scatter sensor
- micro-optic AI camera

---

## Sensor Behavior

- noise
- drift
- fouling
- bias
- latency
- dropout

---

# 8. Control Reality

## Control Mode

- continuous real-time control

---

## Control Variables

- pump
- valve
- backflush
- electrostatic field

---

## Optimization Objective

- maximize capture
- maintain safe pressure
- limit clogging
- minimize maintenance
- maintain flow

---

# 9. Realism Definition

HydrOS is considered “high realism” when:

- flow behavior matches expected physical trends
- clogging produces nonlinear pressure response
- backflush produces partial, diminishing recovery
- electrostatic capture meaningfully alters outcomes
- sensors diverge realistically from truth
- system behavior is interpretable and testable

---

# Final Statement

HydrOS is not simulating a generic filter.

It is simulating:

> A physically grounded, multi-stage, electro-assisted microplastic extraction system operating under real-world constraints.