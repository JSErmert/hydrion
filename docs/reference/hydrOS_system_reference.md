# HydrOS System Reference Image

## File


docs/assets/hydros_system_reference.png


---

# Purpose

This image represents the **physical system intent** for HydrOS.

It is used to:

- anchor simulation architecture to real-world structure
- preserve stage hierarchy and flow direction
- guide realism development
- inform module relationships

---

# Interpretation Rules

This image is:

- a **design anchor**
- a **physical reference**
- a **conceptual guide**

This image is NOT:

- a strict implementation specification
- a complete physical model
- a source of exact equations or parameters

---

# System Breakdown (Mapped to HydrOS Modules)

## 1. Inlet Section

### Components
- swirl region
- polarization ring

### Interpretation
- flow conditioning
- particle distribution shaping
- electrostatic preconditioning

### Module Mapping
- Hydraulics
- Electrostatics (conditioning phase)

---

## 2. Stage 1 — Coarse Mesh

### Characteristics
- large pore size (~500 µm)
- low resistance
- captures large debris

### Interpretation
- minimal clog sensitivity
- early-stage load reduction

### Module Mapping
- Hydraulics (low resistance)
- Clogging (low-rate accumulation)

---

## 3. Stage 2 — Medium Mesh

### Characteristics
- medium pore size (~100 µm)
- increased resistance
- fiber interaction begins

### Interpretation
- fiber bridging becomes relevant
- moderate clog sensitivity

### Module Mapping
- Hydraulics
- Clogging (bridge-dominant contribution)

---

## 4. Stage 3 — Fine Pleated Mesh

### Characteristics
- fine filtration (~5 µm equivalent)
- large effective area (~900 cm²)
- highest resistance

### Interpretation
- dominant clog location
- surface cake formation
- electrostatic interaction region

### Module Mapping
- Hydraulics (major pressure drop)
- Clogging (cake-dominant)
- Electrostatics (capture interaction)

---

## 5. Electrostatic System

### Components
- polarization ring (upstream)
- lower collector node (primary)

### Interpretation
- upstream: modifies particle charge state
- downstream: captures particles near fine stage

### Allocation
- ~30% conditioning (inlet)
- ~70% capture (lower node)

### Module Mapping
- Electrostatics

---

## 6. Backflush System

### Components
- annular release channel
- pulse-based actuation

### Interpretation
- removes accumulated fouling
- partial recovery mechanism
- not perfect cleaning

### Module Mapping
- Clogging (recovery)
- Env (event timing + orchestration)

---

## 7. Storage Reservoir

### Components
- detachable capture container

### Interpretation
- accumulated mass tracking
- maintenance trigger

### Module Mapping (future)
- Storage state (not yet implemented fully)

---

## 8. Sensors

### Components
- differential pressure
- flow sensor
- turbidity / optical
- camera (future AI proxy)

### Interpretation
- measurement layer
- imperfect observation of truth

### Module Mapping
- Sensors
- Observation

---

# Critical Insight

The image reinforces the following:

- flow direction matters
- stage hierarchy matters
- physical interactions are sequential
- electrostatics is not isolated — it is coupled to filtration
- clogging is not uniform across stages

---

# Final Directive

When implementing or modifying the system:

- use this image to preserve physical intuition
- do not overfit to visual details
- prioritize consistency with architecture constraints and locked system parameters

This image supports the system.

It does not define it.