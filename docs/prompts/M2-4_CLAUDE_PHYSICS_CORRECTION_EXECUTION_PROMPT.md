# CLAUDE PHYSICS CORRECTION EXECUTION PROMPT
# Milestones M1.5 → M3 → M4

You are executing physics corrections to HydrOS / Hydrion.

This is NOT a research task.
This is NOT a feature addition.

This is a **physics grounding operation** — correcting five confirmed errors
that prevent HydrOS from being presented as real hardware.

Read `docs/research/2026-04-10-physics-correction-report.md` before acting.

---

# Locked Decision: Dense-Phase Scope (Option A)

HydrOS targets **dense microplastics only**:

```
ρ_particle > 1.0 g/cm³
Primary targets: PET, PA, PVC, biofilm-coated fragments
Excluded: PP (ρ ≈ 0.91), PE (ρ ≈ 0.95) — buoyant phase, out of scope
```

This is a **system scope constraint**, not a limitation to fix later.

You MUST add this to `docs/context/06_LOCKED_SYSTEM_CONSTRAINTS.md`:

```
## F. Particle Density Scope

Target class: dense microplastics (ρ > 1.0 g/cm³)
Excluded class: buoyant microplastics (PP, PE, ρ < 1.0 g/cm³)

Collection topology: gravity-fed downward from outer wall node.
Buoyant-phase capture requires separate upstream treatment — out of scope.
```

Do NOT implement dual-path (Option B) collection. It is not selected.

---

# Execution Order

You MUST follow this sequence. No parallel work across phases.

```
Phase 1: M1.5 Calibration Fixes       (unblocks RL training + validates backbone)
Phase 2: M3 Electrostatic Correction  (unblocks M4 + grounds electrode geometry)
Phase 3: M4 Particle Realism          (requires M3 + density scope decision)
Phase 4: Schematic Correction         (requires M3 complete)
```

---

# Phase 1 — M1.5 Calibration Fixes

## 1.1 Bistable Kinetics (R1)

**File:** `configs/default.yaml`

Confirm this is applied:
```yaml
dep_exponent: 1.0   # was 2.0 — bistable; filter won't foul from clean reset
```

**Validation:** `mesh_loading_avg` reaches 0.70 from clean reset (ff=0)
within ≤500 steps at Q=13.5 L/min.

Run full M1 test suite (10/10 must pass).

---

## 1.2 Component Sum Overflow (C2)

**File:** `hydrion/physics/clogging.py`, method `_update_stage()`

After computing `cake_si`, `bridge_si`, `pore_si`, add normalization:

```python
# Normalize so component sum never exceeds fouling_frac
component_sum = cake_si + bridge_si + pore_si
if component_sum > fouling_frac:
    scale = fouling_frac / component_sum
    cake_si   *= scale
    bridge_si *= scale
    pore_si   *= scale
```

Apply before writing to state.

---

## 1.3 Pressure Drop: Decouple Base R from Area Normalization (R3)

**File:** `hydrion/physics/hydraulics.py`

**Current (wrong):** Area normalization applied to base resistance, inverts S3 dominance.

**Correct:** Area normalization applies to fouling sensitivity only — not base resistance.

```python
# R_total per stage = base resistance + fouling-coupled term scaled by area
R_s1 = R_m1_base + k_m1_clog * (A_ref / A_s1) * ff_s1
R_s2 = R_m2_base + k_m2_clog * (A_ref / A_s2) * ff_s2
R_s3 = R_m3_base + k_m3_clog * 1.0             * ff_s3
```

**Validation (clean filter, Q=13.5 L/min):**
- dp_stage3 > dp_stage2 > dp_stage1
- dp_total ∈ [25, 50] kPa

---

## 1.4 Voltage Hard Clamp (New Finding)

**File:** `hydrion/physics/electrostatics.py`

V_max = 3000 V violates the locked realism bound of 2500 V.

```python
# Replace:
V_max: float = 3000.0

# With:
V_max_realism: float = 2500.0    # upper operational bound (06_LOCKED_SYSTEM_CONSTRAINTS.md)
V_hard_clamp:  float = 3000.0    # absolute safety ceiling

# Apply:
V_node = np.clip(node_voltage_cmd * V_max_realism, 0.0, V_hard_clamp)
```

Log a warning if V_node approaches V_hard_clamp.

---

## 1.5 Safety Module Deduplication

**Files:** `hydrion/safety/shield.py`, `hydrion/wrappers/shielded_env.py`

Designate `hydrion/safety/shield.py` as canonical.

`shielded_env.py` must delegate to `shield.py` — no duplicated logic.

Document which module owns: pre-action filtering, rate limiting, post-step penalties.

---

## Phase 1 Done When

- [ ] M1 validation suite: 10/10 pass
- [ ] dep_exponent=1: fouling grows from zero to 0.70 within 500 steps
- [ ] dp_stage3 > dp_stage2 > dp_stage1 at clean state, nominal flow
- [ ] dp_total ∈ [25, 50] kPa at clean, Q=13.5 L/min
- [ ] V_max_realism = 2500 enforced; V_hard_clamp = 3000 as ceiling
- [ ] Safety: one canonical module, no duplication
- [ ] `04_CURRENT_ENGINE_STATUS.md` updated

---

# Phase 2 — M3 Electrostatic Correction

## 2.1 Electrode Architecture

Replace the single axial-face electrode with two physically grounded subsystems.

**Current (wrong):** One scalar node, downstream axial face, field = V/gap, position-agnostic.

**Required:** Two subsystems with radial field geometry.

---

### SubSystem A: InletPolarizationRing (30% capture contribution)

- Location: upstream face of each stage
- Function: conditions particle charge state — increases dielectrophoretic response
- Depends on: voltage, residence time
- Model:
  ```python
  t_residence_s = stage_volume_L / max(Q_proc_lmin / 60.0, eps)
  charge_factor = ring_weight * np.tanh(
      (V_node / V_ring_ref) * (t_residence_s / t_E_ref_s)
  )
  ```
- Output: `charge_factor` ∈ [0, 1] applied to capture_eff_gain in particles module

---

### SubSystem B: OuterWallCollectorNode (70% capture contribution)

- Location: outer cylindrical wall of each stage (the collection surface)
- Function: primary electrostatic capture via radial field
- Field geometry: **RADIAL** — not axial

  ```python
  # Cylindrical radial field: E_r(r) = V / (r * ln(r_outer / r_inner))
  # At the collection wall (r = r_outer):
  r_inner: float = 0.005   # [m] central counter-electrode radius
  r_outer: float = 0.040   # [m] outer collection wall radius

  E_r_wall_Vm  = V_node / (r_outer * np.log(r_outer / r_inner))
  E_field_kVm  = E_r_wall_Vm / 1000.0
  ```

- Capture gain: proportional to E_field_kVm, coupled to residence time:
  ```python
  node_capture_gain = node_weight * alpha_E * E_field_kVm * np.tanh(
      t_residence_s / t_E_ref_s
  )
  ```

---

## 2.2 Replace E_norm with E_field_kVm

**Current observation schema (obs12_v1), index 3:** `E_norm` — dimensionless [0, 2.0]

**New (obs12_v2), index 3:** `E_field_norm` — `E_field_kVm / E_field_kVm_max`

```python
# Compute max field at V_max_realism for normalization reference
E_field_kVm_max = (V_max_realism / 1000.0) / (r_outer * np.log(r_outer / r_inner))

# In sensor_fusion.py:
E_field_norm = np.clip(ts['E_field_kVm'] / p.E_field_kVm_max, 0.0, 1.0)
```

**Version the schema:**
- Update schema label from `obs12_v1` → `obs12_v2`
- Update `04_CURRENT_ENGINE_STATUS.md`

---

## 2.3 YAML Parameters for M3

Add to `configs/default.yaml` under `electrostatics:`:

```yaml
electrostatics:
  V_max_realism:  2500.0   # [V]   upper operational bound (locked constraint)
  V_hard_clamp:   3000.0   # [V]   absolute safety ceiling

  r_inner_m:      0.005    # [m]   counter-electrode (central rod) radius
  r_outer_m:      0.040    # [m]   outer collection wall radius

  t_E_ref_s:      2.0      # [s]   residence time reference for capture saturation
  V_ring_ref:     500.0    # [V]   polarization ring conditioning reference

  ring_weight:    0.30     # share of E-field capture from inlet ring
  node_weight:    0.70     # share of E-field capture from outer wall node

  alpha_E:        0.08     # [1/(kV/m)] capture gain per unit field (placeholder)
```

Document all new parameters in `docs/calibration/M2-2.5_CALIBRATION_PARAMETER_REGISTER.md`
with evidence class: Placeholder (to be replaced by bench measurement in M3.5).

---

## Phase 2 Done When

- [ ] `electrostatics.py` uses radial field model: `E_r(r)` not `V/gap`
- [ ] `InletPolarizationRing` and `OuterWallCollectorNode` are separate documented classes
- [ ] `E_field_kVm` stored in `truth_state`
- [ ] `E_norm` replaced by `E_field_norm` in observation vector
- [ ] Schema versioned: `obs12_v2`
- [ ] Validation: disabling voltage measurably reduces `particle_capture_eff`
- [ ] Validation: doubling flow rate reduces capture (residence time effect)
- [ ] Validation: V_node never exceeds V_hard_clamp = 3000 V
- [ ] M1 validation suite still passes
- [ ] `04_CURRENT_ENGINE_STATUS.md` updated

---

# Phase 3 — M4 Particle Realism

## 3.1 Density Classification — Dense-Phase Scope

Add density class to PSD bins. Three classes:

```python
DENSITY_DENSE   = 'dense'    # ρ > 1.0 g/cm³  — PET, PA, PVC, biofilm-coated
DENSITY_NEUTRAL = 'neutral'  # ρ ≈ 1.0 g/cm³  — weathered, transitional
DENSITY_BUOYANT = 'buoyant'  # ρ < 1.0 g/cm³  — PP, PE (tracked, NOT captured)
```

Device captures dense + neutral only. Buoyant fraction is tracked as pass-through.

In `particles.py`:
```python
C_in_dense    # concentration of dense-phase particles [mg/L]
C_in_neutral  # concentration of neutral-phase particles [mg/L]
C_in_buoyant  # buoyant fraction — tracked, exits as C_out directly

C_out = (
    C_in_dense    * (1.0 - capture_eff_dense)
  + C_in_neutral  * (1.0 - capture_eff_neutral)
  + C_in_buoyant  # fully passes through — scope constraint
)
```

---

## 3.2 Stokes Settling for Dense Particles

Add settling velocity:

```python
def stokes_velocity_ms(rho_p_kgm3, d_p_m, rho_w=1000.0, mu=1e-3):
    """Stokes settling velocity [m/s]. Positive = downward (sinking)."""
    g = 9.81
    return (rho_p_kgm3 - rho_w) * g * d_p_m**2 / (18.0 * mu)
```

For dense particles (ρ > 1.0): `v_s > 0` → settling assists downward collection.
Apply a `capture_boost_settling` term proportional to Stokes velocity relative to
the flow velocity in the collection zone.

---

## 3.3 Per-Stage Size-Dependent Capture Efficiency

Each stage has its own capture curve parameterized by mesh pore size.

```python
def capture_eff_s1(d_p_um: float, fouling_s1: float) -> float:
    """Stage 1: 500 µm coarse mesh."""
    base = np.clip((d_p_um / 500.0) ** 1.5, 0.0, 0.99)
    fouling_factor = 1.0 + 0.15 * fouling_s1
    return float(np.clip(base * fouling_factor, 0.0, 0.99))

def capture_eff_s2(d_p_um: float, fouling_s2: float) -> float:
    """Stage 2: 100 µm medium mesh."""
    base = np.clip((d_p_um / 100.0) ** 1.2, 0.0, 0.98)
    fouling_factor = 1.0 + 0.20 * fouling_s2
    return float(np.clip(base * fouling_factor, 0.0, 0.98))

def capture_eff_s3(d_p_um: float, fouling_s3: float, Q_lmin: float) -> float:
    """Stage 3: 5 µm pleated cartridge — flow-rate dependent."""
    base = np.clip((d_p_um / 5.0) ** 0.8, 0.0, 0.97)
    flow_penalty = np.exp(-0.04 * max(Q_lmin - 10.0, 0.0))
    fouling_factor = 1.0 + 0.10 * fouling_s3
    return float(np.clip(base * flow_penalty * fouling_factor, 0.0, 0.97))
```

System efficiency per size bin:
```python
eta_system = 1.0 - (1.0 - eta_s1) * (1.0 - eta_s2) * (1.0 - eta_s3)
```

---

## 3.4 Formal Efficiency Definition

Lock in `docs/context/06_LOCKED_SYSTEM_CONSTRAINTS.md`:

```
## G. System Efficiency Definition

η_nominal = η_system(d = 10 µm, Q = 13.5 L/min, clean filter, dense-phase particles)

This is the reference efficiency displayed on the console.

Display format: "η = 82% @ 10µm / 13.5 L/min"

Never display a bare efficiency percentage without qualification.
The display label "99%" without context is not a valid system output.
```

---

## 3.5 Observation Schema Update

Maintain backward compatibility where possible.

`particle_capture_eff` (index 5) transitions to `capture_eff_dense` — semantically the
same for the dense-phase-scoped system.

Add to truth_state (not necessarily observation):
```python
capture_eff_s1      # per-stage capture, dense-phase
capture_eff_s2
capture_eff_s3
buoyant_fraction    # fraction of C_in that is buoyant (pass-through, not captured)
```

If observation vector is extended beyond 12D, version schema: `obs14_v1`.

---

## Phase 3 Done When

- [ ] PSD bins carry density class attribute
- [ ] `C_in_dense`, `C_in_neutral`, `C_in_buoyant` tracked separately in truth_state
- [ ] Buoyant fraction exits as pass-through — not captured
- [ ] Per-stage capture efficiency: S1, S2, S3 each have distinct curves
- [ ] S3 capture efficiency decreases measurably at Q > 15 L/min
- [ ] System efficiency = 1 − (1−η_s1)(1−η_s2)(1−η_s3)
- [ ] Formal efficiency definition locked in `06_LOCKED_SYSTEM_CONSTRAINTS.md`
- [ ] Dense-phase scope constraint locked in `06_LOCKED_SYSTEM_CONSTRAINTS.md`
- [ ] M1 validation suite still passes
- [ ] `04_CURRENT_ENGINE_STATUS.md` updated

---

# Phase 4 — Schematic Correction (MachineCore.tsx)

Execute AFTER Phase 2 (M3) is complete.

## 4.1 Electrostatic Node Position

**Current (wrong):** Node on right (downstream axial) wall of each stage rectangle.

**Correct:** Node on **outer cylindrical wall** — bottom edge of each stage rectangle
in cross-section (lower collection zone).

For each stage (SVG coordinates):
- Primary node: `(CX, CYL_BOTTOM)` — outer wall, lower collection zone
- Secondary ring indicator: top-center of stage `(CX, CYL_Y)`, smaller and dimmer
  (represents the inlet polarization ring)

---

## 4.2 Mesh Fan Lines — Radial, Not Axial

**Current (wrong):** Lines fan from left wall (x ≈ 110) to right-wall node.

**Correct:** Lines fan from the **central pipe axis** outward to the **outer
cylindrical wall** (top and bottom edges of each stage rectangle).

Cross-section appearance:

```
TOP WALL      y = CYL_Y
   ╲  ╲  ╲        mesh lines from center axis → top outer wall
──────────────     PIPE AXIS  y = PIPE_CY
   ╱  ╱  ╱        mesh lines from center axis → bottom outer wall
BOTTOM WALL   y = CYL_BOTTOM
```

In SVG, per stage (S1 example, CX=142, PIPE_CY=133):
```xml
<!-- Upper fan: center axis → top outer wall -->
<line x1="142" y1="133" x2="118" y2="74"  stroke="#38BDF8" .../>
<line x1="142" y1="133" x2="142" y2="74"  stroke="#38BDF8" .../>
<line x1="142" y1="133" x2="166" y2="74"  stroke="#38BDF8" .../>
<!-- Lower fan (mirrored) -->
<line x1="142" y1="133" x2="118" y2="192" stroke="#38BDF8" .../>
<line x1="142" y1="133" x2="142" y2="192" stroke="#38BDF8" .../>
<line x1="142" y1="133" x2="166" y2="192" stroke="#38BDF8" .../>
```

---

## 4.3 Collection Tubes — Dense Phase Only, Downward

Collection tubes exit from the **bottom outer wall** of each stage.
Dense-phase only. No upward path.

- Entry: `(CX, CYL_BOTTOM)` — bottom outer wall of each stage
- Drop: vertical tube to manifold at `y = 220`
- Manifold: horizontal `y = 220`, from S1 CX to storage CX
- Rise: vertical into annular storage bottom

Render as double-wall pipe cross-sections (as built in Option K mockup).

---

## 4.4 E-Field Display Units

**File:** `TopTelemetryBand.tsx`, `RightAdvisoryPanel.tsx`

```typescript
// Before:
`${(state.eFieldNorm * 100).toFixed(0)}%`

// After:
`${state.eFieldKVm.toFixed(0)} kV/m`
```

Add `eFieldKVm: number` to `HydrosDisplayState` type.
Map from `truth_state.E_field_kVm` in `displayStateMapper.ts`.

---

## 4.5 Efficiency Display Qualification

**File:** `RightAdvisoryPanel.tsx`

Below efficiency percentage, add qualification line:

```tsx
{state != null && (
  <div style={{ fontSize: 7, color: '#475569', marginTop: 2 }}>
    η @ 10µm / 13.5 L/min
  </div>
)}
```

Pre-M4 (before per-size efficiency is implemented): display `est. aggregate` subtitle.

---

## 4.6 Dense-Phase Scope Badge

Add to MachineCore schematic, subtle position (bottom-left of center panel):

```tsx
<text style={{ fontSize: 6, fill: '#1E3A5F', letterSpacing: '0.1em' }}>
  DENSE-PHASE TARGET  ρ > 1.0 g/cm³
</text>
```

Must not dominate the visual. It is a specification label, not a headline.

---

## Phase 4 Done When

- [ ] Node indicators on outer cylindrical wall (bottom of each stage)
- [ ] Mesh fan lines radiate from center axis outward (not left-wall to right-wall)
- [ ] Collection tubes exit from bottom outer wall, downward to manifold
- [ ] E-field displays in kV/m
- [ ] Efficiency displays with size/flow qualification
- [ ] Dense-phase badge visible on machine view
- [ ] Visual reads as physically correct radial-field cross-section

---

# What You Must NOT Do

- Do NOT implement buoyant-phase collection — Option B is rejected
- Do NOT start M3 before Phase 1 validation passes
- Do NOT start M4 before Phase 2 (M3) is complete
- Do NOT start schematic corrections before Phase 2 is complete
- Do NOT change observation schema ordering without versioning
- Do NOT alter truth_state / sensor_state separation
- Do NOT change the reward function — that is M6 scope
- Do NOT add YAML parameters without updating CALIBRATION_PARAMETER_REGISTER.md
- Do NOT promote placeholder parameter values without physical evidence

---

# Commit Convention

```
fix(physics/M1.5): <description>
fix(physics/M3): <description>
fix(physics/M4): <description>
fix(schematic): <description>
```

Update `04_CURRENT_ENGINE_STATUS.md` at the end of each phase.

---

# Final Standard

Every change must answer:

> "Does this make HydrOS behave more like the real device?"

If the answer is no, or requires speculation, do not apply it.

HydrOS is not being optimized.
HydrOS is being grounded.

**Physics first. Code second. UI last.**
