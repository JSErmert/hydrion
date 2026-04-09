You are operating inside the HydrOS / Hydrion repository on branch:

HydrOS-x-Claude-Code

This is not a generic coding task.
This is a governed engineering environment.

Before taking action, you must read and internalize the following files in this order:

1. README.md
2. docs/MASTER_ARCHITECTURE.md
3. CLAUDE.md
4. docs/context/01_SYSTEM_IDENTITY.md
5. docs/context/02_ARCHITECTURE_CONSTRAINTS.md
6. docs/context/03_REPO_MAP.md
7. docs/context/04_CURRENT_ENGINE_STATUS.md
8. docs/context/05_REALISM_TARGET.md
9. docs/context/06_LOCKED_SYSTEM_CONSTRAINTS.md
10. docs/context/07_TELEMETRY_AND_CONSOLE.md
11. docs/context/08_VALIDATION_AND_SAFETY.md
12. docs/context/09_REALISM_ROADMAP.md
13. docs/context/10_COUNCIL_OF_HYDROS.md

You must treat those files as authoritative doctrine.

---

# Primary Objective

Perform a full architecture audit of the current repo against the context docs, then implement:

## Milestone 1 — Hydraulic + Fouling + Backflush Realism Backbone

This milestone must increase realism in the following order:

1. Hydraulics
2. Clogging
3. Backflush

Electrostatics is explicitly NOT the target of this implementation pass unless needed only as a compatibility-preserving placeholder.

---

# Required Working Style

You must behave as a disciplined co-orchestrated engineering system.

That means:

- reason before coding
- do not guess
- do not flatten architecture
- do not change observation schema casually
- do not bypass truth_state vs sensor_state separation
- do not add broad feature sprawl
- focus on fidelity concentration
- preserve numerical stability for RL

You are not allowed to treat the system image as literal law.
If an image is present in the repo context, treat it as:
- a physical architecture anchor
- a design-intent reference
- not a rigid implementation spec

---

# Audit Phase (must happen before implementation)

First perform an architecture audit and report the following:

## 1. Repo audit
Identify:
- current strengths
- current realism gaps
- any structural contradictions relative to context docs
- any duplicated or conflicting implementations
- current Milestone 1 readiness

## 2. Module-specific audit
For each of these files, explain current behavior, strengths, and limitations relative to Milestone 1:

- hydrion/env.py
- hydrion/physics/hydraulics.py
- hydrion/physics/clogging.py
- hydrion/physics/particles.py
- hydrion/safety/shield.py
- hydrion/wrappers/shielded_env.py
- configs/default.yaml
- relevant validation files in hydrion/validation/

## 3. Milestone 1 delta
State exactly what is missing for Milestone 1 in terms of:
- state variables
- YAML parameters
- equations / model behavior
- backflush event timing
- bypass behavior
- reward alignment
- validation coverage

## 4. Risk analysis
Identify the top risks if Milestone 1 is implemented poorly, especially:
- numerical instability
- architectural drift
- reward misalignment
- over-coupling between modules
- breaking RL compatibility

Do not code until this audit is complete.

---

# Locked System Constraints (authoritative)

Treat the following as system-level constraints, not suggestions:

## A. Laundry Outflow Operating Envelope
- low-flow regime: 5 L/min
- nominal regime: 12–15 L/min
- peak transient regime: 20 L/min

## Throughput Doctrine
- full-flow processing is baseline
- device should continuously handle 12–15 L/min
- short-duration tolerance up to 20 L/min
- passive overflow / bypass is protective only, not primary mode

## B. Stage Geometry
- Stage 1 area: 120 cm^2
- Stage 2 area: 220 cm^2
- Stage 3 effective area: 900 cm^2

## Fouling Threshold
- maintenance-required state begins around 70% equivalent cake / solids holding capacity
- this is a true hydraulic degradation regime shift, not a cosmetic warning

## C. Backflush Doctrine
- multi-burst square-pulse backflush
- 3 pulses
- 0.4 s per pulse
- 0.25 s interpulse spacing
- 8–10 s cooldown / no-retrigger interval

## Backflush Fluid Source
- default autonomous source: recirculated filtered effluent
- clean-water mode exists only for service / calibration
- autonomous system must not assume unlimited external clean water

## D. Realism Prioritization
Primary calibration target:
- pressure-drop vs flow across clean / partial / heavy fouling

Secondary calibration target:
- backflush recovery dynamics
- partial recovery
- diminishing returns
- irreversible fouling accumulation under stress

Tertiary calibration target:
- capture efficiency vs PSD after hydraulic / fouling backbone is stable

## Final priority rule
Hydraulic truth and fouling truth come first.
Backflush truth is part of the same backbone.
Electrostatic truth comes after this milestone.

---

# Milestone 1 Implementation Requirements

Implement Milestone 1 with the following design intent:

## 1. Hydraulics realism
Upgrade hydraulics so pressure-drop depends on:
- incoming flow
- stage area
- stage fouling fraction
- nonlinear degradation past maintenance threshold
- bypass relief when overloaded

Add support for:
- q_in_lmin
- q_processed_lmin
- q_bypass_lmin
- dp_stage1_pa
- dp_stage2_pa
- dp_stage3_pa
- dp_total_pa
- bypass_active

## 2. Clogging realism
Upgrade clogging to represent:
- surface cake accumulation (primary)
- fiber bridging / entanglement (secondary)
- pore restriction (tertiary)

Add distinction between:
- recoverable fouling
- irreversible fouling

Add stage-wise state for each stage and an overall fouling fraction.

The system should exhibit:
- nonlinear degradation near ~70%
- worsening recovery under heavy fouling
- nonzero irreversible accumulation under stress

## 3. Backflush realism
Backflush must no longer be a simple abstraction.
Implement it as a timed event system with:
- burst scheduling
- pulse index
- cooldown logic
- source-fluid-dependent cleaning efficiency
- diminishing returns across pulses and repeated cycles

This should be orchestrated in env.py, while keeping fouling removal logic owned by the appropriate module.

## 4. Reward alignment (Milestone 1 only)
Do not perform full final reward redesign yet.
But update reward enough so the system is no longer purely flow-centric.

Milestone 1 reward should reflect:
- processed flow
- pressure penalty
- fouling penalty
- bypass penalty
- backflush cost

Capture should not yet dominate until later milestones.

## 5. Validation
Add or update validation so Milestone 1 can be defended.

At minimum validate:
- pressure-drop vs flow curves
- fouling growth behavior
- threshold nonlinearities
- partial backflush recovery
- diminishing returns
- bypass activation under overload
- no NaNs / no negative unphysical state

---

# Architectural Constraints (must not be violated)

You must preserve:

- truth_state as authoritative
- sensor_state as observational only
- immutable pipeline ordering
- YAML-driven configuration
- stable 12D observation contract
- safety logic isolated from physics
- frontend / rendering isolation
- modular ownership of behavior

Do not:
- merge physics modules
- casually expand observation schema
- move safety into physics
- introduce hidden coupling
- make UI assumptions inside engine code

---

# Required Output Format

Respond in 3 phases.

## Phase 1 — Audit Report
Provide:
- repo assessment
- module-by-module Milestone 1 gap analysis
- implementation risks
- contradictions or cleanup items that should happen before coding

## Phase 2 — Implementation Plan
Provide:
- exact files to edit
- exact new state variables
- exact YAML additions
- proposed model equations / update rules
- validation additions
- staged implementation order

## Phase 3 — Code Changes
Then implement Milestone 1 directly in the repo.

Requirements:
- keep commits logically grouped if possible
- explain any tradeoffs
- preserve backward compatibility where reasonable
- document new parameters and new truth_state fields
- update tests / validation

---

# Success Condition

Milestone 1 is successful only if:
- the repo remains architecturally clean
- flow / pressure / fouling / recovery interactions become materially more realistic
- validation covers the new realism
- the system remains stable enough for future RL benchmarking

If you discover a contradiction between current code and context docs, do not improvise silently.
State the contradiction explicitly and propose the least destructive resolution.

Begin with Phase 1 only.