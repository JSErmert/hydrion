# Telemetry and Console

This document defines the **Phase 1.5 telemetry and research console doctrine** for HydrOS.

It establishes:
- the role of the front-end console
- the telemetry contract between backend and UI
- live state binding requirements
- cadence logic
- status indicators
- comparison and history requirements

---

# 1. Console Identity

HydrOS Console is NOT a generic dashboard.

It is a:

- research instrument
- observability layer
- mission-control system view
- engineering analysis interface

The console exists to:
- observe
- analyze
- compare
- explain

It does NOT exist to:
- manually override physics
- mutate simulation truth directly
- become the primary control logic

---

# 2. Console Role in System Architecture

The console is a **read-only observer** layered on top of the Hydrion engine.

### Backend role
- produce authoritative simulation truth
- generate measurements
- evaluate safety
- log run history
- expose telemetry

### Frontend role
- bind to telemetry
- display system state
- show derived indicators
- support comparison and inspection

---

# 3. Phase 1.5 Objectives

The current front-end phase is:

> **Front-End Research Console Evolution**

The primary goals are:

1. live metric binding
2. hybrid cadence model
3. system status indicators
4. PPO vs baseline comparison view
5. run-history telemetry interface

---

# 4. Current Console Status

Current implementation exists in:

```text
apps/hydros-console/
Current state
React + TypeScript + Vite scaffold
static SystemCutaway
layout and metric panels
no backend telemetry binding
no run control
no live history
Interpretation

The console is currently a visual shell, not yet a live research instrument.

5. Console Doctrine

The console must preserve the following principles:

1. Read-only interaction

The UI must never directly mutate:

truth_state
sensor_state
internal simulation state
2. Backend truth authority

The backend is the sole source of truth.

3. Hybrid time model

Simulation time and render time are related, but not identical.

4. Hardware-forward presentation

The UI must feel like a real industrial system monitor, not a SaaS dashboard.

5. Interpretation over decoration

Every visual element must correspond to meaningful system behavior.

6. Telemetry Contract

The console should consume a structured telemetry interface.

Core payload types
RunManifest
TelemetryFrame
ShieldEvent
ValidationStatus
EpisodeSummary
TimeseriesChunk
6.1 RunManifest

Purpose:

describe run/session metadata
define schema compatibility
establish timing and policy context
Required fields
run_id
config_hash
seed
engine version
timing info
observation schema
action schema
policy mode
6.2 TelemetryFrame

Purpose:

provide latest live snapshot of system state
Required domains
Time
step_idx
sim_time_s
dt_s
Truth
hydraulics
clogging
electrostatics
particles
storage / fill (if implemented)
Sensors
optical and future sensor readings
Observation
fixed observation vector
Action
raw action
applied action
Reward
current reward
optional reward term breakdown
Shield
active/inactive
last event id
violation risk
Health
stable / unstable
nan_detected
6.3 ShieldEvent

Purpose:

make safety behavior inspectable

Required:

event_id
reason
severity
time
raw action
applied action
relevant snapshot
limits / thresholds
6.4 ValidationStatus

Purpose:

expose correctness indicators to UI

Required:

mass_balance
envelope status
stability warnings
reproducibility flags
6.5 EpisodeSummary

Purpose:

support run history and policy comparison

Required:

run_id
policy mode
total return
capture metrics
violation metrics
recovery metrics
config hash
7. Minimal Endpoint Surface

The baseline console should use a simple API before adding WebSockets.

Recommended endpoints
Run lifecycle
POST /runs
POST /runs/{id}/reset
POST /runs/{id}/step
POST /runs/{id}/execute
Live telemetry
GET /runs/{id}/telemetry/latest
GET /runs/{id}/validation/latest
GET /runs/{id}/shield/events
History
GET /runs
GET /runs/{id}/timeseries
8. Cadence Doctrine

HydrOS uses a hybrid cadence model.

8.1 Simulation time

Simulation time is determined by:

step index
dt

It is authoritative.

8.2 Render time

Render time is determined by:

UI polling interval
animation smoothing
frame interpolation

It is subordinate to simulation truth.

8.3 Rule

The UI must never define simulation state.

It only visualizes it.

8.4 Recommended Phase 1.5 baseline
simulation runs on backend
frontend polls at approximately 100 ms
backend may step 1 or more times per poll
render layer may smooth transitions
smoothing must not fake physical state changes
9. Live Binding Doctrine

The following visual elements must ultimately be bound to real telemetry:

SystemCutaway / SVG
flow
pressure
clog
E_norm
backflush status
storage fill / captured mass (if available)
Core Metrics Panel
capture efficiency
pressure
flow
clog state
maintenance threshold
Validation Panel
mass balance status
stability warnings
shield active state
anomaly state
10. PPO vs Baseline Comparison

The console must support direct comparison between:

learned policy (PPO)
baseline / rules-based controller
Comparison dimensions
capture
pressure stability
clog progression
backflush frequency
violations
recovery time

This must not be an afterthought.
It is part of the research instrument.

11. Run History and Replay

HydrOS Console must eventually support:

list of runs
run selection
replay of previous telemetry
timeseries inspection
summary comparison
12. What the Console Must Never Become

The console must never become:

a control hack panel
a mutation interface for truth_state
a decorative analytics page
a replacement for validation

It must remain:

a clean observer and analyzer of system behavior

13. Phase 1.5 Implementation Order
telemetry state contract
backend endpoint layer
frontend polling hook
live metric binding
hybrid cadence smoothing
status indicators
run history
PPO vs baseline comparison
14. Final Directive

HydrOS Console is the observability layer for future hardware-ready intelligence.

It must be designed with:

clarity
discipline
truth alignment
engineering seriousness

not novelty.