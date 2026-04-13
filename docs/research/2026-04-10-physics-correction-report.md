# HydrOS Physics Correction Report
**Date:** 2026-04-10  
**Author:** HydrOS Co-Orchestrator (Claude Sonnet 4.6)  
**Scope:** Full system audit — hardware architecture, simulation engine, schematic representation  
**Status:** AUTHORITATIVE — feeds M1.5 sprint planning and M3–M4 milestone scope revision

---

## Purpose

This report documents every physics and engineering gap identified in HydrOS that prevents the system from being presented as a faithful model of real hardware. It distinguishes between:

- **Architectural errors** — wrong physics direction, structural mismatch with hardware
- **Specification gaps** — correct intent, missing quantification
- **Known calibration debt** — already tracked, confirmed here for completeness
- **Deferred realism** — intentionally abstract per the roadmap, not an error

For each finding, the report states: what is wrong, why it matters, what must change, and where in the codebase/milestone roadmap that change belongs.

---

## 1. Electrode Geometry — Axial vs Radial

### Classification: Architectural Error

### What the current model does

`electrostatics.py` models the collection electrode as a **point on the downstream (axial) face** of each stage. The electric field is computed as:

```python
E_field = V_node / gap_m    # gap_m = 0.01 m (fixed, 1 cm)
```

This is a scalar — no spatial position, no direction. The field acts as a uniform scalar modifier on particle capture efficiency, applied identically at all radial positions within the stage.

### Why this is wrong

A downstream axial electrode creates a field vector pointing **along the flow axis** — parallel to the direction of water travel. For electrostatically enhanced mesh filtration, the required field is **radial** — normal to the mesh surface, pointing from the central counter-electrode (axis) outward to the outer cylindrical wall.

The physical device architecture requires:
- **Counter-electrode:** central rod or axis of the cylindrical stage housing
- **Collection electrode:** outer cylindrical wall of each stage
- **Field vector:** radial, from axis outward — this is what drives dielectrophoretic capture toward the mesh surface and ultimately toward the wall

An axial field does not push particles toward the mesh. It pushes them downstream, potentially assisting bypass rather than capture.

### What the locked constraints say (06_LOCKED_SYSTEM_CONSTRAINTS.md)

```
Functional Allocation:
- 70% capture: lower collector node
- 30% capture: inlet polarization ring
```

The "lower collector node" is a wall-mounted electrode on the **outer** cylindrical wall at the lower collection region — not the downstream face. The inlet polarization ring is a separate upstream conditioning electrode. Neither is axial in the sense currently modeled.

### Impact

| System | Impact |
|---|---|
| Simulation physics | Field drives capture in wrong direction; capture enhancement from E-field is physically unjustified |
| RL training | Agent learns to use node voltage as a generic "boost" with no spatial meaning |
| Hardware transfer | Physical device with radial electrode geometry will behave differently from model predictions |
| Schematic (SVG) | Electrostatic nodes shown on right-wall (axial face) of each stage — incorrect |

### Required changes

**Hardware/Architecture:**
- Establish outer cylindrical wall as the collection electrode for all three stages
- Establish central axis rod as the counter-electrode
- Define the "lower collector node" as a contact point on the outer wall at the lower collection zone (not the downstream face)
- Maintain the 70/30 functional allocation: 70% outer wall node, 30% inlet polarization ring

**Simulation engine (`electrostatics.py`, M3):**
- Replace scalar `E_field = V / gap` with a radial field model: `E_r(r) = V / (r × ln(r_outer/r_inner))`
- Split into two sub-components: `InletPolarizationRing` (30%) and `OuterWallCollectorNode` (70%)
- Make capture gain dependent on radial position — strongest at outer mesh surface
- Couple to residence time (flow rate dependency): lower flow = longer exposure to field = higher capture

**Schematic (MachineCore.tsx):**
- Move electrostatic node indicators from right wall of each stage to the **top and bottom outer wall** (outer cylindrical surface in cross-section)
- Mesh fan lines should converge toward the outer wall, not the right-wall axial face
- In cross-section: mesh lines fan from the central pipe outward to the top/bottom edges of the cylinder rectangle

---

## 2. Particle Buoyancy — Not Modeled

### Classification: Architectural Error

### What the current model does

`particles.py` treats all particles as density-neutral. There is no buoyancy term:

```python
C_out = C_in * (1.0 - capture_eff)   # Purely filtration-based. No density separation.
```

Collection tubes in the hardware design route particles **downward** (from electrostatic node → collection tube drops → horizontal manifold → rises into storage reservoir from below). This gravity-fed topology assumes all particles are denser than water.

### Why this is wrong

The microplastics most commonly found in laundry effluent are:

| Material | Density (g/cm³) | Behavior in water | Prevalence in laundry |
|---|---|---|---|
| PP (polypropylene) | 0.90–0.91 | **Floats** | Very high (fleece, sportswear) |
| PE (polyethylene) | 0.94–0.96 | **Floats** | High |
| PET (polyester) | 1.38–1.41 | Sinks | Very high (polyester garments) |
| PA (polyamide/nylon) | 1.13–1.15 | Sinks slightly | High |
| PVC | 1.16–1.58 | Sinks | Low in laundry |
| Biofilm-coated PP/PE | ~1.01–1.10 | Sinks/neutral | Occurs with aged plastic |

PP and PE — two of the most abundant microplastic types in laundry water — are less dense than water. They **float**. Once released from the electrostatic node, they would migrate **upward**, not downward. A downward gravity-fed collection tube would not capture them.

PET microfibers from polyester garments do sink, as do PA fibers. So the current design captures the denser subset correctly but misses the buoyant subset entirely.

### Impact

| System | Impact |
|---|---|
| Collection topology | Downward tubes only capture dense plastics; buoyant plastics return to flow |
| Stage loading | Buoyant particles accumulate at upper portions of each stage, not lower; fouling is asymmetric |
| Capture efficiency | True whole-system capture efficiency is lower than modeled for PP/PE fraction |
| Hardware design | Collection tube direction must accommodate both dense (downward) and buoyant (upward) pathways, or device scope must be explicitly constrained to dense plastics |

### Required changes

**Product/hardware decision (required before engineering correction):**

Choose one of two defensible positions:

**Option A — Constrained scope:**  
HydrOS targets **dense microplastics only** (PET, PA, PVC, biofilm-coated fragments, ρ > 1.0 g/cm³). The gravity-fed downward collection topology is correct for this subset. Explicitly document this as a system scope constraint in `06_LOCKED_SYSTEM_CONSTRAINTS.md`. Add a stated exclusion: "System does not capture buoyant-phase microplastics (PP, PE, ρ < 1.0 g/cm³) without additional upstream treatment."

**Option B — Full-spectrum capture:**  
Add an upward collection pathway for buoyant particles. Each electrostatic node has two exits: downward collection tube (dense phase) and upward collection tube (buoyant phase), both routing to the annular storage reservoir. The reservoir stores both streams in the outer annular ring. This is mechanically more complex but physically complete.

**Simulation engine (`particles.py`, M4):**
- Add density class to PSD bins: `rho_class` ∈ {buoyant, neutral, dense}
- Apply gravitational drift term: `v_grav = (ρ_p - ρ_w) × g × d_p² / (18 × μ)` (Stokes settling velocity)
- For buoyant particles: drift is upward, reduces downward collection efficiency
- Add `buoyant_fraction` to observation vector (proportion of C_in that is buoyant-phase)
- Capture efficiency should be disaggregated: `capture_eff_dense` and `capture_eff_buoyant`

**Schematic (MachineCore.tsx):**
- If Option A: add a label/badge indicating "DENSE-PHASE TARGET" on the machine view
- If Option B: show dual-exit node topology — downward tube (dense) and upward tube (buoyant)

---

## 3. E-Field Expressed as Percentage — No Physical Units

### Classification: Specification Gap

### What the current model does

```python
E_norm_ref: float = 3e5       # [V/m] — arbitrary reference
E_norm = abs(E_field) / E_norm_ref   # dimensionless, clipped to [0.0, 2.0]
```

The UI displays `E_norm` as "E-FIELD 72%" or similar. The physical field strength in V/m is never exposed.

Additionally, `electrostatics.py` defaults `V_max = 3000 V`, which **exceeds the locked upper realism bound of 2.5 kV** defined in `06_LOCKED_SYSTEM_CONSTRAINTS.md`.

### Why this is wrong

"72% E-field" has no physical referent. It cannot be compared to hardware specifications, literature values, or safety thresholds. The normalization reference (3×10⁵ V/m) is arbitrary — it happens to correspond to `V_max / gap_m` at the current defaults, meaning the percentage is really just a proxy for "fraction of maximum voltage command," not a physical field quantity.

For context: electrostatically enhanced water filtration systems typically operate at **10–300 kV/m** field strengths depending on electrode geometry and particle target. A 1 cm electrode gap at 1.5 kV produces 150 kV/m. This number is meaningful and verifiable. "72%" is not.

The `V_max = 3000 V` hardcode also violates the locked constraint (`hard clamp: 3.0 kV` is the absolute maximum, `upper realism bound: 2.5 kV`). The system should target 2.5 kV as the operational ceiling with 3.0 kV as a hard clamp — not treat 3.0 kV as the normalized reference point.

### Required changes

**Simulation engine (`electrostatics.py`, M3):**
- Replace `E_norm` with `E_field_kVm` (field strength in kV/m) in truth_state
- Set `V_max = 2500 V` (upper realism bound); add `V_hard_clamp = 3000 V` as a safety limit
- Compute field as: `E_field_kVm = V_node / (gap_m × 1000)` initially; replace with radial model when electrode geometry is corrected (see Issue 1)
- Document the expected operating range: "DEP capture onset ~50 kV/m; nominal operating range 100–200 kV/m at 1.5 kV across 8–15 mm gap"

**Observation schema:**
- Replace `E_norm` (obs index 3) with `E_field_norm` normalized against the physically meaningful `V_max_realism = 2500 V`, not an arbitrary reference
- Version the observation schema if E_norm replacement is not backward compatible: `obs12_v2`

**Frontend:**
- Display as "E-FIELD 145 kV/m" not "E-FIELD 72%"
- Keep a normalized display for the efficiency gauge (0–100% of `V_max_realism`) but label it as voltage-fraction, not field percentage

---

## 4. Efficiency — Unqualified Single Scalar

### Classification: Specification Gap + Deferred Realism (M4)

### What the current model does

```python
capture_eff = (
    capture_eff_baseline              # 0.80
    + capture_eff_gain                # 0.12
    * mesh_loading_avg
    * (1.0 - mesh_loading_avg) ** 0.5
)
# Output: single scalar ∈ [0.30, 0.98]
```

This is a single number, applied uniformly across all particle sizes, all flow rates, and all three stages. The UI displays it as `efficiencyPct`, typically shown as "82%" or "99%".

### Why this is insufficient

Real filtration efficiency is a function of:

1. **Particle size** — A 5 µm pleated cartridge captures >95% of particles ≥10 µm but only ~30–50% of 1 µm particles. Coarse mesh (500 µm) has near-zero capture for sub-50 µm particles.
2. **Flow rate** — Higher flow = shorter residence time = lower capture, especially at fine stages. A 5 µm cartridge at 5 L/min may achieve 85% capture; at 20 L/min it may drop to 55%.
3. **Stage** — S1 captures large particles and has little effect on fine ones. S3 captures fine particles but is destroyed if S1/S2 let through coarse load.
4. **Filter state (fouling level)** — The current model captures this, but a fouled filter at high efficiency is physically different from a clean filter at equal efficiency: the pores are partially blocked, increasing ΔP and reducing capture uniformity.

Displaying "82% efficiency" without qualification is like displaying "28 MPG" without specifying highway or city. It is meaningful in context, misleading without it.

### Required changes

**Simulation engine (`clogging.py` + `particles.py`, M4):**
- Implement per-stage capture efficiency: `capture_eff_s1`, `capture_eff_s2`, `capture_eff_s3`
- Each stage has its own efficiency curve parameterized by: fouling level, flow rate, and particle size class
- Stage 1 efficiency curve: high for >100 µm, low for <50 µm
- Stage 3 efficiency curve: flow-rate dependent, high residence time = high efficiency
- Overall system efficiency = `1 - (1 - η_s1)(1 - η_s2)(1 - η_s3)` per size bin

**Efficiency definition (formally locked):**
```
HydrOS system efficiency is defined as:
  η_system(d, Q) = fraction of particles of diameter d captured
                   at volumetric flow rate Q through all three stages
                   at current fouling state.

  η_nominal = η_system(d=10 µm, Q=13.5 L/min, clean filter)
```

This definition must be documented in `06_LOCKED_SYSTEM_CONSTRAINTS.md`.

**Frontend:**
- Show per-stage fouling (already shown) alongside per-stage capture efficiency (needs M4)
- Display overall efficiency with qualification: "82% @ 10µm / 13.5 L/min" or similar
- "99%" should never appear as an unqualified number on a hardware-facing display

---

## 5. Pressure Drop — Area Normalization Inverts Stage 3 Dominance

### Classification: Known Calibration Debt (R3, already tracked)

### What the current model does

```python
k_m1_eff = k_m1_clog * (A_s3_ref / A_s1)   # 900/120 = 7.5x
k_m2_eff = k_m2_clog * (A_s3_ref / A_s2)   # 900/220 = 4.1x
k_m3_eff = k_m3_clog * 1.0                  # reference stage
```

After area normalization, the effective resistance ordering becomes:
- S2 highest (2.0e7 × 4.1 = 8.2e7)
- S1 second (1.0e7 × 7.5 = 7.5e7)
- S3 lowest (4.0e7 × 1.0 = 4.0e7)

### Why this is wrong

The area normalization was intended to scale **fouling sensitivity** (how much a unit of fouling increases resistance). Instead it is being applied to **base resistance**, which inverts physical intuition.

Physical pressure drop ordering for a clean filter:
- S3 (5 µm pleated cartridge, 900 cm² effective area) — **dominant ΔP source** — fine pores, even with large area
- S2 (100 µm mesh, 220 cm²) — moderate ΔP
- S1 (500 µm coarse mesh, 120 cm²) — lowest ΔP

A pleated cartridge's large surface area exists precisely to reduce ΔP for fine pores, but it still dominates relative to coarse stages. The model correctly gives S3 a higher base resistance (4.0e7 vs 1.0e7 for S1), but the area normalization then reduces S3 relative to S1, reversing the physical ordering.

This is explicitly documented as audit issue R3 in `04_CURRENT_ENGINE_STATUS.md`. It is listed as an M1.5 calibration task.

### Required changes

**Simulation engine (`hydraulics.py`, M1.5):**
- Decouple area normalization from base resistance: apply it only to the fouling-coupled term, not to `R_base`
- Rewrite resistance as: `R_total_si = R_base_si + R_fouling_si × (A_s3/A_si) × ff_si`
- This preserves the physical intuition that S3 has the highest base resistance while still scaling fouling sensitivity inversely with area
- Validate: at clean state (ff=0), S3 ΔP > S2 ΔP > S1 ΔP at nominal flow (13.5 L/min)
- Validate: total clean-state ΔP at 13.5 L/min falls in physically credible range (~25–50 kPa for this filter class)

---

## 6. Secondary Findings

### 6.1 Bistable Fouling Kinetics (R1) — Already Fixed in YAML

**Status: Confirmed fix applied, pending validation**

`dep_exponent = 2` creates bistable kinetics that prevent fouling from growing from a clean state. Fix (`dep_exponent: 1.0`) is in `configs/default.yaml` as an M1.5 action. Confirmed correct. Acceptance criterion: `mesh_loading_avg` reaches 0.70 from clean reset within ≤500 steps at nominal flow.

### 6.2 Voltage Hard Clamp Violation

`electrostatics.py` defaults `V_max = 3000 V`. Per `06_LOCKED_SYSTEM_CONSTRAINTS.md`, the **upper realism bound is 2.5 kV**; the **hard clamp is 3.0 kV**. Using 3.0 kV as the normalized reference means the system routinely operates at the hard clamp ceiling, not within the nominal operating range.

**Fix:** Set `V_max = 2500` (realism ceiling) and add `V_hard_clamp = 3000` as an absolute safety limit with a warning log when approached.

### 6.3 Safety Duplication

Two safety implementations exist: `hydrion/safety/shield.py` and `hydrion/wrappers/shielded_env.py`. The canonical logic is unclear. This creates ambiguity about which system is authoritative for pre-action filtering and post-step constraint detection. Resolve by designating one canonical module and deleting or explicitly subordinating the other.

### 6.4 No Differential Pressure Sensor in Sensor Model

`04_CURRENT_ENGINE_STATUS.md` explicitly lists "No differential pressure sensor" as a limitation. The per-stage ΔP values exist in `truth_state` but are not exposed through the sensor model to the observation vector. This means the RL agent cannot observe pressure directly — it is inferred from the `pressure` observation which is a total system pressure, not per-stage ΔP.

For M5, per-stage ΔP sensors are essential for realistic agent behavior: real maintenance controllers make backflush decisions based on ΔP thresholds, not aggregate fouling estimates.

### 6.5 Reward Does Not Include Capture Efficiency

The current reward incentivizes **throughput** (`q_processed`) but not **capture quality**. A system with high flow and low capture efficiency earns as much reward as a system with high flow and high capture efficiency. The physical goal of HydrOS is to remove microplastics — this is not yet in the reward signal.

This is the M6 multi-objective reward redesign. It is not a bug, but the implication is important: the RL agent trained on the current reward is not learning to maximize microplastic removal. It is learning to maximize filtered volume under pressure/fouling constraints.

---

## 7. Schematic Representation Corrections

These corrections apply to `apps/hydros-console/src/components/MachineCore.tsx` and the visual mockups.

### 7.1 Electrostatic Node Position

**Current (incorrect):** Nodes shown on the right (downstream axial) wall of each stage cylinder.

**Correct:** Nodes shown on the **outer cylindrical wall** of each stage — the top and bottom edges of the cylinder rectangle in cross-section. In the SVG, this means node indicators at the top-right and bottom-right corners of each stage rectangle, representing the outer wall contact point.

**Mesh fan lines:** Must converge **outward** (toward the top/bottom walls) from the central pipe axis, not rightward toward the downstream face. In the corrected cross-section:

```
TOP WALL (collection surface)
  ↖ mesh fan lines radiating FROM center axis TO outer wall
  ●  node contact point (outer wall, lower collection zone)
  ↙ mesh fan lines (mirrored below center)
BOTTOM WALL (collection surface)
```

### 7.2 Collection Tube Direction

**Depends on hardware decision (Issue 2, Option A vs B):**

- **Option A (dense-phase only):** Collection tubes exit from the **bottom** of the outer cylindrical wall, drop to the manifold, rise into storage. Current topology is correct for this subset.
- **Option B (full-spectrum):** Dual exits — downward tube (dense) and upward tube (buoyant) from the outer wall node. Both route to the annular reservoir.

### 7.3 E-Field Display

Replace "E-FIELD 72%" with "E-FIELD 145 kV/m" (or the computed dimensional value). Display the normalized voltage percentage separately as "V_NODE 60%" if needed for gauge display.

### 7.4 Efficiency Display Qualification

Replace bare "82%" with "82% η@10µm" or equivalent. The efficiency gauge on the right advisory panel should display the qualified value with particle size reference once M4 is implemented. Pre-M4, add a subtitle below the efficiency gauge: "est. aggregate / unqualified."

---

## 8. Impact on Milestone Roadmap

### M1.5 (Current Sprint — Calibration)

| Item | Type | Action |
|---|---|---|
| dep_exponent bistable fix | Already tracked | Confirm YAML fix, validate |
| Component sum overflow (C2) | Already tracked | 5-line fix in clogging.py |
| Pressure drop area normalization (R3) | Already tracked | Decouple base R from fouling normalization |
| Voltage V_max clamp violation | **New** | Set V_max=2500, add V_hard_clamp=3000 |
| Bypass threshold coupling (A3) | Already tracked | Decouple in M1.5 or M2 |

### M3 (Electrostatic Conditioning)

The M3 scope must now explicitly include the electrode geometry correction. The roadmap already calls for splitting into InletPolarizationRing + LowerCollectorNode, but it must now specify:

- LowerCollectorNode is a **radial outer-wall electrode**, not an axial downstream electrode
- Field is computed radially: `E_r(r) = V / (r × ln(r_out/r_in))`
- Capture efficiency is position-dependent: strongest at outer mesh surface, weaker near axis
- E-field output is in **kV/m**, not a dimensionless percentage
- V_max realism bound enforced at 2500 V (V_hard_clamp at 3000 V)

### M4 (Particle Realism)

M4 scope must now include the buoyancy correction. The roadmap calls for "realistic bin distributions, inflow variability, size-dependent capture" — this must now also include:

- **Density classification** of PSD bins: buoyant (ρ < 1.0), neutral (ρ ≈ 1.0), dense (ρ > 1.0)
- **Stokes drift velocity** per bin applied to vertical transport
- **Buoyancy-dependent collection path** (requires hardware decision from Issue 2 first)
- The formal efficiency definition (Issue 4) is fully implemented in M4

### M6 (Reward + Control)

The multi-objective reward must include:
- `capture_mass_rate` as the primary signal (per-size-bin capture, not just flow throughput)
- This is only possible after M4 implements size-dependent capture
- Until M4, reward remains throughput-dominant by necessity

---

## 9. Recommended Action Sequence

### Immediate (before next implementation session)

1. **Hardware decision on buoyancy scope (Issue 2):** Choose Option A (dense-phase constraint) or Option B (dual-path full-spectrum). This decision gates the collection tube topology in both hardware and simulation.

2. **Schematic correction decision:** Once electrode geometry and collection direction are settled, the machine view SVG redesign can begin. The J/K mockups need to be revised to show:
   - Radial mesh fans (center-to-outer-wall, not left-wall-to-right-wall)
   - Node indicators on outer cylindrical wall
   - Correct collection tube direction based on Option A or B choice

### M1.5 Sprint (in order)

3. `dep_exponent: 1.0` — confirm YAML, run validation suite
4. Component sum normalization — 5-line fix in `clogging.py`
5. `R_base` / fouling normalization decoupling in `hydraulics.py`
6. `V_max = 2500`, `V_hard_clamp = 3000` in `electrostatics.py`
7. Safety implementation deduplication (designate canonical module)

### M3 (expanded scope)

8. Split electrostatics into `InletPolarizationRing` + `OuterWallCollectorNode`
9. Implement radial field model: `E_r(r)` with proper geometry
10. Replace `E_norm` with `E_field_kVm` in truth_state and observation
11. Version observation schema: `obs12_v2` if E_norm is replaced
12. Validate: capture efficiency increases with voltage, strongest at outer mesh

### M4 (expanded scope)

13. Add density classification to PSD bins
14. Implement Stokes drift for buoyancy-affected bins
15. Implement per-stage, per-size-class capture efficiency curves
16. Formally lock efficiency definition in `06_LOCKED_SYSTEM_CONSTRAINTS.md`
17. Validate: PET fibers captured at higher rate than PP fibers at same diameter

---

## 10. Summary

HydrOS is a well-structured digital twin with a correct hydraulic and fouling backbone. It is not physically defensible at the hardware level in its current form due to five concrete problems:

| # | Problem | Severity | Milestone |
|---|---|---|---|
| 1 | Electrode geometry: axial instead of radial | **Critical** | M3 |
| 2 | Particle buoyancy not modeled | **Critical** | M4 + hardware decision |
| 3 | E-field in dimensionless percentage, V_max exceeds realism bound | High | M3 |
| 4 | Efficiency undefined by size, flow, or stage | High | M4 |
| 5 | Pressure drop area normalization inverts S3 dominance | Medium | M1.5 |

None of these problems break the current simulation as a training environment. The RL agent can still learn policies. But the system cannot be presented as a faithful model of real hardware — or used to make predictions about real hardware behavior — until these corrections are made.

The schematic in `MachineCore.tsx` and all visual mockups (J, K) also require revision to reflect the corrected electrode geometry and collection topology once the hardware decision on buoyancy scope is made.

---

*This report supersedes informal notes in `04_CURRENT_ENGINE_STATUS.md` regarding audit issues R1, C2, R3, A3. Those issues are confirmed here with additional context. Issues 1–3 above (electrode geometry, buoyancy, E-field units) are new findings not previously tracked.*
