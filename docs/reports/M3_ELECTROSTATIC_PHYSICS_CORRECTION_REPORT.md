# HydrOS M3 Report — Electrostatic Physics Correction
**Date:** 2026-04-10  
**Author:** HydrOS Co-Orchestrator (Claude Sonnet 4.6)  
**Scope:** Milestone 3 — Electrostatic subsystem grounding, radial field model, observation schema v2  
**Status:** AUTHORITATIVE — defines M3 implementation scope, validation criteria, and entry/exit conditions  
**Depends on:** M1.5 complete (✓), Phase 1 validation suite passing (✓)

---

## Purpose

This report defines the complete engineering scope of Milestone 3 (M3). M3 has one primary mission:

> Replace the placeholder single-node axial electrode model with two physically grounded subsystems that correctly represent the electrostatic architecture of the HydrOS device.

M3 does not touch hydraulics, clogging, backflush, or particle realism. Those are either already grounded (M1/M1.5) or belong to M4. M3 is a contained, physics-correct replacement of one module — `electrostatics.py` — with a model that matches hardware intent.

---

## Entry Conditions

M3 may not begin until all of the following are confirmed:

| Condition | Status |
|---|---|
| `dep_exponent = 1.0` in YAML (R1 fix) | ✓ M1.5 |
| Component sum normalization in `_update_stage()` (C2 fix) | ✓ M1.5 |
| k_m3_clog >> k_m2_clog >> k_m1_clog, no area scaling (R3 fix) | ✓ M1.5 |
| V_max_realism = 2500 V, V_hard_clamp = 3000 V | ✓ M1.5 |
| Dense-phase scope locked (Option A) | ✓ M1.5 |
| Full test suite: 25/25 passing | ✓ M1.5 |

---

## What M1.5 Left Unresolved in Electrostatics

The M1.5 sprint corrected the voltage bounds. It did not correct the electrode architecture.

After M1.5, `electrostatics.py` still:

- Models a single downstream axial-face electrode
- Computes field as `E_field = V_node / gap_m` (fixed 0.01 m scalar gap)
- Applies capture gain uniformly with no geometry, no position, no residence time
- Reports `E_norm` — a dimensionless percentage relative to an arbitrary 3×10⁵ V/m reference

This is physically indefensible and cannot be used to make hardware-level claims about capture behavior.

---

## 1. Diagnosis — Current Electrostatic Model

### 1.1 Wrong Field Geometry

**Current code:**
```python
E_field = V_node / max(p.gap_m, p.eps)   # gap_m = 0.01 m
E_norm  = clip(abs(E_field) / E_norm_ref, 0.0, 2.0)
```

This produces a scalar uniform field with no physical geometry. The gap (0.01 m) is a placeholder with no hardware basis.

**Why it is wrong:**  
An axial-face electrode creates a field vector along the **flow axis** (parallel to water travel direction). For electrostatic mesh filtration, the required field is **radial** — normal to the mesh surface, from the central counter-electrode outward to the outer cylindrical collection wall. An axial field does not drive particles toward the mesh or toward the collection wall. It drives them downstream.

### 1.2 No Residence Time Coupling

Electrostatic capture depends on how long a particle spends in the field. At high flow rates, particles transit faster and capture efficiency drops. The current model has no residence time term — it applies the same gain regardless of whether Q = 5 L/min or Q = 20 L/min.

### 1.3 No 30/70 Functional Split

The locked system constraint (06_LOCKED_SYSTEM_CONSTRAINTS.md §D) specifies:
- **70% capture:** lower outer-wall collector node
- **30% capture:** inlet polarization ring (upstream charge conditioning)

The current model has one node. There is no representation of the ring.

### 1.4 E_norm is Not Physical

`E_norm` is a dimensionless percentage normalized to an arbitrary reference. It carries no units, no hardware meaning, and cannot be used to reason about field strength at the collection surface. The observation vector must expose a physically meaningful quantity.

---

## 2. M3 Solution — Two-Subsystem Radial Model

M3 replaces the single axial node with two physically grounded subsystems that together implement the 30/70 functional allocation.

---

### 2.1 SubSystem A — InletPolarizationRing

**Hardware role:** Upstream face of each stage. Conditions the charge state of particles before they contact the mesh. Increases dielectrophoretic susceptibility.

**What it is NOT:** It is not a capture sink. Particles do not deposit here. It modifies the capture gain of SubSystem B downstream.

**Model:**

```python
t_residence_s = stage_volume_L / max(Q_proc_lmin / 60.0, eps)

charge_factor = ring_weight * tanh(
    (V_node / V_ring_ref) * (t_residence_s / t_E_ref_s)
)
# charge_factor ∈ [0, ring_weight]
```

**Parameters:**

| Parameter | YAML key | Value | Evidence |
|---|---|---|---|
| Ring capture weight | `electrostatics.ring_weight` | 0.30 | Locked constraint §D |
| Conditioning voltage reference | `electrostatics.V_ring_ref` | 500.0 V | Placeholder |
| Residence time reference | `electrostatics.t_E_ref_s` | 2.0 s | Placeholder |

**Output:** `charge_factor` — fed into `capture_eff_gain` in the particles module.

---

### 2.2 SubSystem B — OuterWallCollectorNode

**Hardware role:** Outer cylindrical wall of each stage. Primary capture region. Particles driven radially outward by the electrostatic field deposit on the collection surface and drain downward through collection tubes.

**Field geometry:** Radial cylindrical field (not axial).

For a concentric cylindrical capacitor:
```
E_r(r) = V / (r × ln(r_outer / r_inner))
```

At the collection wall (`r = r_outer`):
```python
E_r_wall_Vm = V_node / (r_outer * log(r_outer / r_inner))
E_field_kVm = E_r_wall_Vm / 1000.0
```

This is the field **at the surface where capture happens** — the physically meaningful quantity.

**Parameters:**

| Parameter | YAML key | Value | Evidence |
|---|---|---|---|
| Counter-electrode radius | `electrostatics.r_inner_m` | 0.005 m | Placeholder — central rod estimate |
| Collection wall radius | `electrostatics.r_outer_m` | 0.040 m | Placeholder — stage housing estimate |
| Node capture weight | `electrostatics.node_weight` | 0.70 | Locked constraint §D |
| Capture gain per kV/m | `electrostatics.alpha_E` | 0.08 | Placeholder — bench calibration target |
| Residence time reference | `electrostatics.t_E_ref_s` | 2.0 s | Shared with ring |

**Capture gain model:**

```python
node_capture_gain = node_weight * alpha_E * E_field_kVm * tanh(
    t_residence_s / t_E_ref_s
)
```

**Output:** `node_capture_gain` — added to `capture_eff_gain` in the particles module alongside `charge_factor`.

---

### 2.3 Combined Capture Effect

Total electrostatic contribution to capture efficiency:

```python
E_capture_gain = charge_factor + node_capture_gain
# Applied in particles.py as additive gain on base capture_eff
```

When `V_node = 0`: both terms zero, electrostatics disabled, capture falls to mechanical-only baseline.  
When `Q` increases: `t_residence_s` decreases, `tanh` output decreases, capture gain drops — physically correct.  
When `V_node = V_max_realism (2500 V)`: system operates at maximum designed capture contribution.

---

## 3. E_field_kVm — Replacing E_norm

### 3.1 What changes

| | Current (obs12_v1) | M3 output (obs12_v2) |
|---|---|---|
| truth_state key | `E_field` [V/m], `E_norm` [dimensionless] | `E_field_kVm` [kV/m] |
| Observation index 3 | `E_norm` ∈ [0, 2.0] (dimensionless) | `E_field_norm` = `E_field_kVm / E_field_kVm_max` ∈ [0, 1] |
| Schema label | `obs12_v1` | `obs12_v2` |

### 3.2 Normalization reference

```python
# Compute at V_max_realism for consistent ceiling
E_field_kVm_max = (V_max_realism / 1000.0) / (r_outer * log(r_outer / r_inner))

# Normalize for observation
E_field_norm = clip(E_field_kVm / E_field_kVm_max, 0.0, 1.0)
```

With default geometry: at `V = 2500 V`, `r_inner = 0.005 m`, `r_outer = 0.040 m`:
```
ln(0.040 / 0.005) = ln(8) ≈ 2.079
E_r_wall = 2500 / (0.040 × 2.079) ≈ 30,100 V/m ≈ 30.1 kV/m
```

`E_field_kVm_max ≈ 30.1 kV/m` at nominal operating voltage.  
`E_field_norm = 1.0` at `V_max_realism`.

### 3.3 Schema versioning contract

`obs12_v2` differs from `obs12_v1` only at index 3. All other indices unchanged. Any consumer of the observation vector must detect schema version before reading index 3.

---

## 4. File Changes

### 4.1 `hydrion/physics/electrostatics.py`

**Complete replacement:**

- `ElectrostaticsParams` dataclass: remove `gap_m`, `E_norm_ref`; add `r_inner_m`, `r_outer_m`, `t_E_ref_s`, `V_ring_ref`, `ring_weight`, `node_weight`, `alpha_E`
- `ElectrostaticsModel.update()`: implement radial field computation, InletPolarizationRing model, OuterWallCollectorNode model
- New truth_state keys: `V_node`, `E_field_kVm`, `charge_factor`, `node_capture_gain`, `E_capture_gain`
- Remove: `E_field` [V/m], `E_norm` [dimensionless]

### 4.2 `hydrion/sensors/sensor_fusion.py`

- Index 3: replace `E_norm` → `E_field_norm`
- Update schema label: `obs12_v1` → `obs12_v2`
- Document the version change in module docstring

### 4.3 `configs/default.yaml`

Replace `electrostatics:` section:

```yaml
electrostatics:
  V_max_realism:  2500.0   # [V]   upper operational bound (locked constraint)
  V_hard_clamp:   3000.0   # [V]   absolute safety ceiling

  r_inner_m:      0.005    # [m]   counter-electrode (central rod) radius
  r_outer_m:      0.040    # [m]   outer collection wall radius

  t_E_ref_s:      2.0      # [s]   residence time saturation reference
  V_ring_ref:     500.0    # [V]   ring conditioning voltage reference

  ring_weight:    0.30     # share of E capture from inlet ring (locked §D)
  node_weight:    0.70     # share of E capture from outer wall node (locked §D)

  alpha_E:        0.08     # [1/(kV/m)] capture gain per unit field (placeholder)
```

### 4.4 `docs/calibration/M2-2.5_CALIBRATION_PARAMETER_REGISTER.md`

Add all new parameters (§5.6) with evidence class `Placeholder`, tied to electrostatic bench test validation hook.

### 4.5 `docs/context/04_CURRENT_ENGINE_STATUS.md`

Update §7 (Electrostatics Module) on M3 completion:
- Document new subsystem architecture
- Mark E_norm → E_field_kVm
- Mark obs12_v2 active

---

## 5. Particle Module Coupling

M3 modifies how `electrostatics.py` feeds into `particles.py`.

**Current coupling:** `E_norm` passed as a scalar multiplier.

**M3 coupling:** `E_capture_gain` = `charge_factor + node_capture_gain` passed as additive gain on the base capture efficiency curve in `particles.py`.

The particles module does not need to know about the internal geometry. It consumes `E_capture_gain` ∈ [0, 1] as a physics-computed gain signal.

No changes to the particles module in M3. M4 will extend it with per-stage, per-size capture curves.

---

## 6. Validation Criteria

All criteria must pass before M3 is considered complete.

### 6.1 Field physics

| Test | Expected |
|---|---|
| `V_node = 0` → `E_field_kVm = 0`, `E_capture_gain = 0` | capture = mechanical baseline only |
| `V_node = V_max_realism` → `E_field_kVm ≈ 30.1 kV/m` | within ±5% of analytical value |
| `V_node` never exceeds `V_hard_clamp = 3000 V` under any `node_cmd` | hard clamp enforced |
| `E_field_norm ∈ [0, 1]` at all operating voltages | no out-of-bounds |

### 6.2 Capture behavior

| Test | Expected |
|---|---|
| Voltage ON vs OFF: measurable difference in `particle_capture_eff` | Δeff > 0.05 at Q=13.5, fouling=0 |
| Q=5 vs Q=20 at same voltage: residence time effect | `capture_eff(Q=5) > capture_eff(Q=20)` |
| `charge_factor` increases with voltage | monotone in V ∈ [0, V_max_realism] |
| `node_capture_gain` increases with E_field_kVm | monotone in V |

### 6.3 Observation schema

| Test | Expected |
|---|---|
| Observation index 3 is `E_field_norm` ∈ [0, 1] | not `E_norm` ∈ [0, 2] |
| Schema version = `obs12_v2` | confirmed in sensor_fusion.py |
| Indices 0–2, 4–11 unchanged | no regression |

### 6.4 Regression

| Test | Expected |
|---|---|
| Full test suite (25 tests) | 25/25 pass |
| Hydraulics pipeline outputs unchanged | dp_s3 >> dp_s2 >> dp_s1 preserved |
| Clogging pipeline outputs unchanged | fouling grows from zero, sum ≤ 1.0 |

---

## 7. What M3 Does NOT Do

M3 is scoped tightly. The following are explicitly excluded:

| Item | Deferred to |
|---|---|
| Per-stage separate electrostatic parameters | M3.5 or M4 calibration |
| Conductivity dependence of field | M3.5 or M4 |
| Particle-size dependence of E-capture | M4 |
| Per-stage size-dependent capture curves | M4 |
| Density classification (dense / buoyant) | M4 |
| Stokes settling | M4 |
| Console schematic correction (radial fans, outer-wall nodes) | Phase 4 — after M3 complete |
| Reward function update | M6 |

M3 does not extend the observation vector beyond 12D. The schema change is a replacement at index 3, not an extension.

---

## 8. M3 Exit Conditions

M3 is complete when all of the following are confirmed:

- [ ] `electrostatics.py` uses radial cylindrical field: `E_r(r) = V / (r × ln(r_outer/r_inner))`
- [ ] `InletPolarizationRing` and `OuterWallCollectorNode` are implemented as separate, documented subsystems
- [ ] `E_field_kVm` is stored in `truth_state`
- [ ] `E_norm` replaced by `E_field_norm` in observation vector (index 3)
- [ ] Schema versioned: `obs12_v2` labeled in `sensor_fusion.py`
- [ ] `default.yaml` updated with all M3 electrostatic parameters
- [ ] `CALIBRATION_PARAMETER_REGISTER.md` updated with new parameters
- [ ] Validation: disabling voltage measurably reduces `particle_capture_eff`
- [ ] Validation: higher flow rate reduces capture (residence time effect confirmed)
- [ ] Validation: `V_node` never exceeds `V_hard_clamp = 3000 V`
- [ ] Full test suite: 25/25 pass
- [ ] `04_CURRENT_ENGINE_STATUS.md` updated to reflect M3 state

---

## 9. What M3 Unlocks

| Unlocks | Why |
|---|---|
| **M4 — Particle Realism** | M4's per-stage capture curves couple into `E_capture_gain`; M3 must be stable first |
| **Phase 4 — Schematic Correction** | React schematic corrections require `E_field_kVm` in truth_state and `obs12_v2` in observation |
| **Console E-field display** | `eFieldKVm` telemetry panel requires `E_field_kVm` from M3 |
| **Console efficiency qualification** | Pre-M4, efficiency display carries `est. aggregate` tag; M4 replaces this with formal definition |

---

## Final Statement

M3 is not a feature addition. It is a physics correction.

The current electrostatic model cannot be defended as hardware-faithful. An axial scalar field, a fixed 1 cm gap, and a dimensionless E_norm percentage are engineering placeholders — useful for RL scaffolding but not for physical grounding.

M3 replaces this with:
- A cylindrical radial field model grounded in the concentric electrode geometry of the physical device
- Two subsystems that respect the 30/70 functional allocation locked in the system constraints
- A residence-time-dependent capture model that correctly penalizes high flow rates
- A physical unit (`E_field_kVm`) in place of a dimensionless proxy

After M3, the electrostatic subsystem can be presented as physically defensible.

> Physics first. Code second. UI last.
