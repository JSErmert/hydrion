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

## Implementation


hydrion/physics/hydraulics.py


### Current behavior

- Computes:
  - flow (`Q_out_Lmin`)
  - pressure (`P_in`, `P_out`, stage pressures)
- Uses:
  - pump_cmd
  - valve_cmd
  - clogging resistance

### Outputs

- Raw values (Pa, L/min)
- Normalized channels:
  - flow
  - pressure

### Strengths

- Stable
- Clean coupling with clogging
- Supports control dynamics

### Limitations

- Not yet calibrated to real laundry flow profiles
- No transient surge modeling
- No bypass modeling
- No temperature / viscosity variation

---

# 6. Clogging Module

## Implementation


hydrion/physics/clogging.py


### Current behavior

- Maintains:
  - Mc1, Mc2, Mc3 (mesh loadings)
  - n1, n2, n3 (normalized clog)
  - mesh_loading_avg

- Updates based on:
  - flow
  - particle load (C_fibers)

### Strengths

- Multi-stage representation
- Stable and continuous
- Works well for RL training

### Limitations

- No explicit surface cake model
- No fiber bridging mechanics
- No pore-blocking dynamics
- No irreversible fouling
- Backflush interaction is simplified

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

## Current reward


r = 2flow - pressure - 0.5clog


### Strengths

- Stable
- Encourages flow + low clog

### Limitations

- Does NOT prioritize capture
- Does NOT reflect maintenance cost
- Does NOT reflect energy usage
- Not aligned with system mission

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

- electrostatics realism
- clogging realism
- backflush dynamics
- sensor realism
- reward alignment
- telemetry integration

---

# Final Statement

HydrOS is currently:

> A structurally correct, modular, RL-ready digital twin with strong foundations,
but still operating at an abstract level in key physical subsystems.

The next phase must focus on:

- increasing realism
- preserving stability
- maintaining modular integrity

without introducing architectural drift.