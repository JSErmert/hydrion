You are operating inside the HydrOS / Hydrion repository on branch:

HydrOS-x-Claude-Code

This is a governed engineering environment.
Do not behave like a generic coding assistant.

Before taking any action, read and internalize these files in this exact order:

1. README.md
2. docs/MASTER_ARCHITECTURE.md
3. CLAUDE.md
4. docs/15_CLAUDE_INITIALIZATION_CHECKLIST.md

Then read all files in:

5. docs/context/

Required context files include:
- 01_SYSTEM_IDENTITY.md
- 02_ARCHITECTURE_CONSTRAINTS.md
- 03_REPO_MAP.md
- 04_CURRENT_ENGINE_STATUS.md
- 05_REALISM_TARGET.md
- 06_LOCKED_SYSTEM_CONSTRAINTS.md
- 07_TELEMETRY_AND_CONSOLE.md
- 08_VALIDATION_AND_SAFETY.md
- 09_REALISM_ROADMAP.md
- 10_COUNCIL_OF_HYDROS.md
- 11_SECURITY_PROTOCOL.md
- 12_DEBUGGING_AND_SUPERPOWERS.md
- 13_AUTOMATION_AND_TEST_PROTOCOL.md
- 14_PRODUCT_SURFACES_STRATEGY.md

Also read:

6. docs/assets/hydrOS_system_reference.md

A corresponding image file is present and should be treated as:
- a physical system-intent anchor
- a guide to stage hierarchy and hardware direction
- not a rigid implementation law

The image is secondary to:
1. repo code
2. architecture docs
3. locked system constraints

---

# Primary Objective

Perform a full architecture audit of the current repository against the documented HydrOS doctrine and then prepare for:

## Milestone 1 — Hydraulic + Fouling + Backflush Realism Backbone

Do NOT start coding immediately.

Audit first.

---

# Required Doctrine

You must preserve all of the following:

- truth_state is authoritative
- sensor_state is observational only
- immutable simulation pipeline ordering
- stable observation schema unless explicitly versioned
- YAML-driven extensibility only
- safety logic remains outside physics modules
- frontend / telemetry remains read-only observer
- no architectural flattening
- no hidden coupling
- no realism expansion without validation path

Current canonical pipeline:

Hydraulics  
→ Clogging  
→ Electrostatics  
→ Particles  
→ Sensors  
→ Observation  
→ Shield (wrapper)

---

# Locked Milestone 1 Constraints

Treat the following as authoritative:

## Laundry Outflow Operating Envelope
- low-flow regime: 5 L/min
- nominal regime: 12–15 L/min
- peak transient regime: 20 L/min

## Throughput Doctrine
- full-flow processing is baseline
- device should continuously process 12–15 L/min
- short-duration tolerance up to 20 L/min
- passive overflow / bypass is protective only

## Stage Geometry
- Stage 1 area: 120 cm^2
- Stage 2 area: 220 cm^2
- Stage 3 effective area: 900 cm^2

## Fouling Threshold
- maintenance-required state begins around 70% equivalent cake / solids capacity
- this must correspond to real hydraulic degradation and recovery degradation

## Backflush Doctrine
- multi-burst square-pulse actuation
- 3 pulses
- 0.4 s per pulse
- 0.25 s interpulse spacing
- 8–10 s cooldown / no-retrigger interval

## Backflush Fluid Source
- default autonomous source: recirculated filtered effluent
- clean-water mode only for service / calibration

## Realism Priority
1. hydraulic truth
2. fouling truth
3. backflush truth
4. electrostatic truth later
5. particle-selective truth after hydraulic backbone is stable

Primary calibration target:
- pressure-drop versus flow under clean / partially fouled / heavily fouled conditions

Secondary calibration target:
- backflush recovery dynamics
- partial recovery
- diminishing returns
- irreversible fouling under stress

Do not reinterpret these as optional suggestions.

---

# Phase 1 Task Sequence

## Phase 1 — Architecture Audit
Provide:

### A. Strength Report
- what is already strong in the current repo
- what should be preserved

### B. Gap Report
- exact current gaps relative to Milestone 1

### C. Structural Risk Report
- duplicated logic
- hidden assumptions
- unstable abstractions
- reward misalignment
- validation blind spots

### D. Milestone 1 Readiness Assessment
- what is already sufficient
- what must change first

Do not code yet.

---

## Phase 2 — Milestone 1 Design
After the audit, produce:

### A. Files to edit
At minimum evaluate:
- hydrion/env.py
- hydrion/physics/hydraulics.py
- hydrion/physics/clogging.py
- configs/default.yaml
- relevant validation files
- relevant logging / telemetry support

### B. YAML additions
Add only what is needed for:
- flow envelope
- stage geometry
- fouling decomposition
- maintenance threshold
- bypass logic
- backflush timing
- source-fluid cleaning efficiency

### C. State additions
Propose exact truth_state additions for:
- q_in_lmin
- q_processed_lmin
- q_bypass_lmin
- dp_total_pa
- dp_stage1_pa
- dp_stage2_pa
- dp_stage3_pa
- bypass_active
- cake / bridge / pore per stage
- recoverable / irreversible fouling per stage
- fouling fraction per stage
- backflush event state

### D. Equation-level proposal
Use lightweight, RL-stable model forms for:
- stage-area-aware pressure-drop
- nonlinear fouling escalation
- recoverable / irreversible fouling split
- pulse-based backflush recovery
- protective bypass behavior

### E. Validation additions
Define:
- pressure-flow sweep
- fouling nonlinearity test
- backflush recovery test
- diminishing returns test
- bypass activation test
- NaN / bounded-state regression

Do not code yet.

---

## Phase 3 — Implementation
Only after audit + design are complete, implement Milestone 1.

Implementation order must be:

1. YAML additions
2. state additions
3. clogging upgrade
4. hydraulics upgrade
5. backflush event state machine in env.py
6. interim reward update for Milestone 1
7. validation additions
8. logging / telemetry support for new fields

---

# Reward Rule for Milestone 1

Do NOT jump to final capture-dominant reward yet.

Milestone 1 reward should still prioritize hydraulic / fouling realism and include:
- processed flow
- pressure penalty
- fouling penalty
- bypass penalty
- backflush cost

Keep it stable for RL benchmarking.

---

# What Not to Do

Do NOT:
- overhaul electrostatics yet
- add conductivity coupling yet
- redesign the console
- expand the observation schema
- add mobile app logic
- introduce Firebase
- broaden into later milestones

This task is strictly:
- hydraulics
- fouling
- backflush
- validation support
- minimal reward alignment

---

# Required Response Style

Be:
- structured
- technically honest
- architecture-aware
- validation-first
- explicit about uncertainty

Do NOT:
- hand-wave
- hype
- guess without reading code
- skip directly to code

---

# Final Directive

Your job is not to “make it more complex.”

Your job is to make:
- flow
- pressure
- fouling
- maintenance recovery

feel physically alive first,
while preserving modularity, numerical stability, and future RL compatibility.

Begin with Phase 1 only.