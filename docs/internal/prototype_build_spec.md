# HydrOS — Prototype Build Specification (Starter)

**Internal use only. Not for external distribution.**
**Status: Design-default. No physical hardware fabricated as of 2026-04-12.**
**Source: Extracted from repo at commit 634dcb4 on branch explore/conical-cascade-arch.**

Parameter status markers:
  [ASSUMED]       — value chosen by designer; not measured; plausible but unvalidated
  [LOCKED]        — fixed by physics model or external standard; not a free parameter
  [UNKNOWN]       — no value chosen; must be determined before hardware fabrication
  [MUST-MEASURE]  — must be characterized from actual hardware before simulation is valid

---

## 1. Device Overall Geometry

| Parameter | Value | Status | Notes |
|-----------|-------|--------|-------|
| Device type | Conical cascade, 3 stages in series | [ASSUMED] | S1 coarse → S2 medium → S3 fine |
| Overall housing diameter | ~80 mm at inlet | [ASSUMED] | Matches S1 D_in |
| Overall housing length | ~240 mm | [ASSUMED] | Sum of cone axial lengths |
| Working fluid | Water | [LOCKED] | MU_WATER = 1e-3 Pa·s, RHO_WATER = 1000 kg/m³ |
| Operating temperature | 20°C (293.15 K) | [ASSUMED] | All viscosity/diffusivity values use this |
| Target particle species | PP, PE, PET | [ASSUMED] | Literature-calibrated CM factors |

---

## 2. Stage Geometry

### Stage 1 (S1 — Coarse Pre-filter)

| Parameter | Value | Status | Notes |
|-----------|-------|--------|-------|
| Inlet diameter D_in | 80 mm | [ASSUMED] | `D_in_m = 0.080` |
| Apex diameter D_tip | 20 mm | [ASSUMED] | `D_tip_m = 0.020` |
| Axial cone length L_cone | 120 mm | [ASSUMED] | `L_cone_m = 0.120` |
| Half-angle | ~9.5° | [LOCKED] | Derived: arctan((40-10)/120) |
| Slant length | ~124 mm | [LOCKED] | Derived: sqrt(120²+30²) |
| Cone wall material | Woven mesh | [ASSUMED] | Deep-bed RT model applies |

### Stage 2 (S2 — Medium Filter)

| Parameter | Value | Status | Notes |
|-----------|-------|--------|-------|
| Inlet diameter D_in | 40 mm | [ASSUMED] | `D_in_m = 0.040` |
| Apex diameter D_tip | 10 mm | [ASSUMED] | `D_tip_m = 0.010` |
| Axial cone length L_cone | 80 mm | [ASSUMED] | `L_cone_m = 0.080` |
| Half-angle | ~9.5° | [LOCKED] | Derived |
| Bed depth model | Single-screen | [ASSUMED] | bed_depth_m = d_c = 50 µm |
| Cone wall material | Woven mesh (flat screen) | [ASSUMED] | Single-screen RT model |

### Stage 3 (S3 — Fine Membrane)

| Parameter | Value | Status | Notes |
|-----------|-------|--------|-------|
| Inlet diameter D_in | 20 mm | [ASSUMED] | `D_in_m = 0.020` |
| Apex diameter D_tip | 4 mm | [ASSUMED] | `D_tip_m = 0.004` |
| Axial cone length L_cone | 40 mm | [ASSUMED] | `L_cone_m = 0.040` |
| Half-angle | ~9.5° | [LOCKED] | Derived |
| Bed depth model | Single-screen (membrane) | [ASSUMED] | bed_depth_m = d_c = 1.5 µm |
| Cone wall material | Microporous membrane | [ASSUMED] | Not woven mesh — treat constants with caution |
| S3 area mean | ~1.63 cm² | [LOCKED] | Derived from D_in and D_tip |

---

## 3. Mesh / Filtration Specifications

### S1 Mesh

| Parameter | Value | Status | Notes |
|-----------|-------|--------|-------|
| Pore opening | 500 µm | [ASSUMED] | `opening_um = 500.0` |
| Wire diameter d_w | 125 µm | [ASSUMED] | `d_w_um = 125.0`; d_w = opening/4 (balanced weave) |
| Collector diameter d_c | 125 µm | [ASSUMED] | Equal to d_w for woven mesh |
| Mesh solidity α | ~0.36 | [LOCKED] | Derived from d_w/pitch formula |
| Material | [UNKNOWN] | | Stainless, nylon, PP? Hamaker constant depends on this |

### S2 Mesh

| Parameter | Value | Status | Notes |
|-----------|-------|--------|-------|
| Pore opening | 100 µm | [ASSUMED] | `opening_um = 100.0` |
| Wire diameter d_w | 50 µm | [ASSUMED] | `d_w_um = 50.0` |
| Collector diameter d_c | 50 µm | [ASSUMED] | Equal to d_w |
| Mesh solidity α | ~0.41 | [LOCKED] | Derived |
| Material | [UNKNOWN] | | |

### S3 Membrane

| Parameter | Value | Status | Notes |
|-----------|-------|--------|-------|
| Pore opening | 5 µm | [ASSUMED] | `opening_um = 5.0` |
| Wire/fiber diameter d_w | 1.5 µm | [ASSUMED] | `d_w_um = 1.5` |
| Collector diameter d_c | 1.5 µm | [ASSUMED] | Pore-scale feature size |
| Mesh solidity α | ~0.41 | [LOCKED] | Derived |
| Membrane type | [UNKNOWN] | | Track-etched? Asymmetric? Pore tortuosity unknown |
| Porosity | [MUST-MEASURE] | | Actual porosity affects pressure drop |
| Hamaker constant | 0.5×10⁻²⁰ J | [ASSUMED] | Literature estimate for polymer-water |

---

## 4. Electrode / DEP Specifications

### S1 Electrode

| Parameter | Value | Status | Notes |
|-----------|-------|--------|-------|
| Voltage (design max) | 500 V | [ASSUMED] | `voltage_V = 500.0` |
| Electrode gap | 5 mm | [ASSUMED] | `electrode_gap_m = 5e-3` |
| Tip radius | 0.5 mm | [ASSUMED] | `tip_radius_m = 0.5e-3` — macroscale electrode |
| Frequency | 100 kHz | [ASSUMED] | Polarization zone config; above MW crossover |
| Electrode geometry | [UNKNOWN] | | Ring? Array? Wire? |

### S2 Electrode

| Parameter | Value | Status | Notes |
|-----------|-------|--------|-------|
| Voltage (design max) | 500 V | [ASSUMED] | |
| Electrode gap | 3 mm | [ASSUMED] | `electrode_gap_m = 3e-3` |
| Tip radius | 0.2 mm | [ASSUMED] | |
| Electrode geometry | [UNKNOWN] | | |

### S3 Electrode (iDEP — membrane-integrated)

| Parameter | Value | Status | Notes |
|-----------|-------|--------|-------|
| Voltage (design max) | 500 V | [ASSUMED] | |
| Electrode gap | 1 mm | [ASSUMED] | `electrode_gap_m = 1e-3` |
| Tip radius | 3 µm | [ASSUMED] | iDEP pore-scale electrode; critical for v_crit |
| Electrode geometry | [UNKNOWN] | | Must be integrated into membrane pore structure |
| Fabrication method | [UNKNOWN] | | Sputtered coating? Conductive membrane? |
| Grad_E² at 500V | ~7.3×10¹⁴ V²/m³ | [LOCKED] | Derived from tip/gap geometry |

---

## 5. Flow / Hydraulic Assumptions

| Parameter | Value | Status | Notes |
|-----------|-------|--------|-------|
| Nominal flow rate | ~10 L/min | [ASSUMED] | Hydraulics model default Q |
| DEP activation threshold | Q ≤ 7.7 L/min | [LOCKED] | Derived: v_crit(S3, 500V) = 789 mm/s |
| Max design pressure | 80 kPa | [ASSUMED] | Safety shield hard limit |
| Pump type | [UNKNOWN] | | Centrifugal? Peristaltic? |
| Pump max flow | ~15-20 L/min | [ASSUMED] | OBS_Q_MAX = 25 L/min in obs |
| Valve type | [UNKNOWN] | | Modulating or on/off? |

---

## 6. Backflush System

| Parameter | Value | Status | Notes |
|-----------|-------|--------|-------|
| Backflush burst duration | 1.7 s | [ASSUMED] | 3 pulses × (0.4s ON + 0.25s OFF) |
| Pulse count per burst | 3 | [ASSUMED] | `_bf_n_pulses = 3` |
| Pulse duration | 0.4 s | [ASSUMED] | `_bf_pulse_dur = 0.4` |
| Interpulse interval | 0.25 s | [ASSUMED] | `_bf_interpulse = 0.25` |
| Cooldown after burst | 9 s | [ASSUMED] | `_bf_cooldown = 9.0` |
| Channel drain rate | 20%/step | [ASSUMED] | `_FLUSH_DRAIN_RATE = 0.20` |
| Backflush fluid source | [UNKNOWN] | | Same water reversed? Separate fluid? |
| Cleaning efficiency | 70% | [ASSUMED] | `bf_source_efficiency = 0.70` |

---

## 7. Accumulation / Storage

| Parameter | Value | Status | Notes |
|-----------|-------|--------|-------|
| Collection channel capacity | ~0.4 L per stage | [ASSUMED] | `_CHANNEL_CAPACITY_M3 = 4e-4` |
| Storage chamber capacity | ~12 L | [ASSUMED] | `_STORAGE_CAPACITY_M3 = 1.2e-2` |
| Number of collection channels | 3 (one per stage) | [ASSUMED] | |
| Storage swap mechanism | [UNKNOWN] | | |

---

## 8. Sensor Placeholders

| Sensor | Model Status | Notes |
|--------|-------------|-------|
| Turbidity | [ASSUMED] — M4 empirical model | No physical sensor spec |
| Light scatter | [ASSUMED] — M4 empirical model | No physical sensor spec |
| Pressure sensor | [ASSUMED] — dp_total_pa from hydraulics | Must-measure: actual dead-band, range |
| Flow meter | [ASSUMED] — q_processed_lmin from hydraulics | Must-measure: type, accuracy |
| Fouling indicator | [ASSUMED] — clogging model output | No physical sensor; derived from ΔP |
| E-field sensor | [ASSUMED] — derived from V_node cmd | No feedback sensing |

---

## 9. Parameters Requiring Measurement Before Simulation Validation

**Critical (simulation validity):**
1. S3 pore size distribution (actual) — [MUST-MEASURE]
2. S3 membrane porosity and tortuosity — [MUST-MEASURE]
3. S2 and S3 mesh wire diameter and solidity (actual, from manufacturer spec or SEM) — [MUST-MEASURE]
4. Electrode tip geometry and effective grad_E² at S3 — [MUST-MEASURE]
5. Hamaker constants for actual mesh/membrane materials in water — [MUST-MEASURE]
6. Actual pressure-flow characteristic of the assembled device — [MUST-MEASURE]
7. CM factors at the operating frequency (100 kHz) vs. literature values — [MUST-MEASURE]

**Important (model accuracy):**
8. Fouling rate constant and cake/bridge/pore ratios for real MPs — [MUST-MEASURE]
9. Backflush effectiveness (actual cleaning fraction per burst) — [MUST-MEASURE]
10. Collection channel geometry and drain rate — [MUST-MEASURE]

---

## 10. First Hardware Milestone Targets

Before any simulation-hardware comparison is valid, the following must be characterized:

| Test | Measurement | Validates |
|------|-------------|-----------|
| Pressure-flow sweep | ΔP(Q) for each stage and assembled cascade | Hydraulics model |
| Particle capture test | eta(Q) at fixed V for 1 µm and 10 µm calibration particles | RT + nDEP model |
| DEP threshold test | Capture fraction vs. Q at V=500V | DEP force balance |
| Fouling progression | ΔP vs. accumulated particle load | Clogging model |
| Backflush recovery | Pressure recovery vs. burst count | Backflush model |

---

*Source code references: `hydrion/environments/conical_cascade_env.py`,*
*`hydrion/physics/m5/capture_rt.py`, `hydrion/physics/m5/dep_ndep.py`,*
*`hydrion/physics/m5/conical_stage.py`.*

*Last generated: 2026-04-12.*
