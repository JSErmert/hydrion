# M5 Console — Conical Cascade Design Spec

**Visual Baseline:** `machine-core-v4.html` — BASELINE 1 (locked 2026-04-11)
**Baseline path:** `.superpowers/brainstorm/221-1775895481/content/machine-core-v4.html`
**Revision policy:** Baseline1 is the build target. Future visual revisions (Baseline2+) are permitted. Changes affecting device geometry or apex node positions require a corresponding env parameter update. Pure aesthetic changes (colors, labels, proportions) are decoupled from the physics engine.

---

## 1. Device Geometry

### Outer Housing
- Constant-diameter tube throughout
- Bore: y=64–244 in diagram space (180 diagram units = full bore height)
- Housing runs x=36–674 (stages + polarization zone)
- Flat top at y=64 continuous from IN tube through all zones to OUT tube

### Zones left → right

| Zone | x range | Type | Notes |
|---|---|---|---|
| IN tube | x=4–36 | Fixed | Full-bore feed pipe, matching housing cross-section |
| Polarization zone | x=36–118 | Fixed | Pre-stage electrostatic field conditioning |
| S1 — Coarse | x=118–298 | Fixed | 500 µm coarse weave, apex at (296, 243) |
| S2 — Medium | x=306–486 | Fixed | 100 µm medium weave, apex at (484, 243) |
| S3 — Fine | x=494–674 | Fixed | 5 µm microporous membrane, apex at (672, 243) |
| Transition buffer | x=674–714 | Fixed | Collection tube termination zone, bore unobstructed |
| Snap-off seam | x=714 | Boundary | Fixed/Detachable division |
| Storage chamber | x=714–886 | Detachable | Bore + particle accumulation bottom ring |
| OUT tube | x=886–934 | Fixed | Full-bore exit pipe, matching housing cross-section |

### Cone geometry (bent mesh)
Each stage uses a cubic bezier from flat at the housing top wall to apex at the housing floor:

- **S1:** `M(118,64) C(195,64)(292,96)(296,243)` — mesh wall; inner ribbon offset +8px
- **S2:** `M(306,64) C(383,64)(480,96)(484,243)`
- **S3:** `M(494,64) C(571,64)(668,96)(672,243)`

Clean water zone: above mesh (passes through).
Concentration zone: below mesh (particle-laden, guided to apex by nDEP + gravity).

### Inlet/exit face rings
Every bore-to-bore junction has a paired ellipse (outer rx=7 ry=90, inner rx=3 ry=90) showing the 3D opening:
- cx=118 (POL zone → S1)
- cx=306 (S1 → S2)
- cx=494 (S2 → S3)
- cx=674 (S3 → transition buffer entry)
- cx=714 (transition buffer exit → storage chamber bore)

---

## 2. Electrostatic Nodes

One apex node per stage. Renders as a glow ellipse + circle + ⊕ symbol at the apex coordinate.

| Stage | Node position | Relative brightness |
|---|---|---|
| S1 | (296, 243) | Dim — particle load highest here, field calibrated softest |
| S2 | (484, 243) | Medium |
| S3 | (672, 243) | Brightest — finest filtration, highest field strength |

Physics: **nDEP (negative dielectrophoresis)**. Particles experience force away from field maxima, directed toward apex by geometry. Clausius-Mossotti (CM) factors confirmed for PP, PE, PET (RSC 2025). See `docs/context/` M5 physics grounding.

---

## 3. Collection Architecture

### Collection channels
Three nested horizontal channels running the full device length (x=118–674), below the housing:

| Channel | y range | Center y | Stage | Pore size |
|---|---|---|---|---|
| S1 | 252–268 | 260 | Coarse | 500 µm |
| S2 | 274–290 | 282 | Medium | 100 µm |
| S3 | 296–312 | 304 | Fine | 5 µm |

Each channel has:
- A dark background rect with stage-color stroke
- A particle fill gradient rect (grows from apex rightward as particles accumulate)
- An in-tube label: "Sx COLLECTION · [pore size] stage"

### Hydraulic flush inlets
Located at x=86–104 (left cap of each channel), one per channel, color-matched to stage:
- S1: orange (#FB923C)
- S2: amber (#FBBF24)
- S3: cyan (#38BDF8)

Function: hydraulic pressure inlet. Pushes accumulated tube contents L→R into the transition buffer zone on command. Activated per-channel or collectively prior to M1 detachment.

### Ejection pipes (node → channel)
Double-stroke pipes (dark outer stroke-width=7, colored inner stroke-width=3.5) running vertically from each apex node down to its collection channel:
- S1: (296, 244) → (296, 252)
- S2: (484, 244) → (484, 274)
- S3: (672, 244) → (672, 296)

### Channel → transition buffer pipes
Horizontal double-stroke pipes from x=673 to x=714 (snap-off seam), terminating with a filled dot at the seam face. Three pipes at y=260, y=282, y=304. Terminal dots sit AT x=714 — insinuating feed-through into the storage chamber.

---

## 4. Transition Buffer Zone (FIXED, x=674–714)

A narrow (40 diagram-unit) cylindrical section fixed to the device body. Functions:
- Receives all three collection channel outputs
- Bore (y=64–244) passes through completely unobstructed — clean water flow unaffected
- Particle accumulation from channels pools in the bottom portion (y=244–328)
- Acts as the mechanical anchor for collection tubes — tubes do NOT reach into the detachable storage chamber

No labels or annotations inside this zone. The snap-off seam at x=714 (dashed amber line) is its right boundary.

---

## 5. Storage Chamber (DETACHABLE, x=714–886)

### Structure
- Flat top at y=64 (matches housing — no top ring)
- Bore: y=64–244 (clean water passes through unobstructed)
- Bottom ring (particle accumulation): y=244–328
- Labels inside bottom ring: "STORAGE CHAMBER" + "particle accumulation"

### Detachment mechanism
**Snap-off seam at x=714** (dashed amber vertical line). Two swap modes:

| Mode | Sequence | Use case |
|---|---|---|
| M1 | flush → detach | Normal swap. bf_cmd clears apex nodes → channels → buffer; then detach storage chamber. |
| M2 | detach → install fresh → bf_cmd | Emergency. Detach first (full capacity), install clean chamber, then run bf_cmd to clear tubes into fresh chamber. |

### Fill-level sensor
Vertical bar in bottom ring right interior (x=847–858, y=253–319):
- Green fill indicates current accumulation level
- Amber threshold marker at 80% fill
- At threshold: trigger M1 swap recommendation as observation flag

---

## 6. Physics Bindings (M5 Engine)

All physics implemented in `hydrion/physics/m5/`. This spec defines the **geometry contract** that the env must honor.

| Parameter | Value | Source |
|---|---|---|
| Capture formula | RT 1976 (radial trajectory) | Peer-reviewed |
| DEP mode | Negative (nDEP) | RSC 2025 confirmed |
| CM factors | PP, PE, PET material-specific | RSC 2025 |
| Stage pore sizes | 500 µm / 100 µm / 5 µm | [DESIGN_DEFAULT — wire diameters not yet specified to physical units] |
| Mesh wire diameter | [DESIGN_DEFAULT] | To be specified in Baseline2 or physics calibration pass |
| Tube inner diameter | [DESIGN_DEFAULT] | To be specified in physics calibration pass |

---

## 7. Observation Space (12-dim)

| Index | Variable | Description |
|---|---|---|
| 0–2 | `q_stage[0..2]` | Volumetric flow rate per stage (L/min) |
| 3–5 | `c_particle[0..2]` | Particle concentration per stage (relative, 0–1) |
| 6–8 | `node_occupancy[0..2]` | Apex node occupancy per stage (0–1) |
| 9 | `fill_level` | Storage chamber fill level (0–1, threshold at 0.8) |
| 10 | `clog_state` | Clogging severity (0–1) |
| 11 | `bf_active` | bf_cmd currently active (0 or 1) |

---

## 8. Console Layer (React Component)

The `machine-core-v4.html` SVG becomes a live React component bound to the env observation vector. Rendering rules:

| Element | Driven by |
|---|---|
| Particle stream opacity + dot density | `c_particle[stage]` |
| Collection channel fill gradient width | `node_occupancy[stage]` |
| Fill sensor green bar height | `fill_level` |
| Amber threshold marker + ⚠ label | Activates at `fill_level >= 0.8` |
| nDEP field line opacity | Active when `bf_active == 0` |
| Node glow intensity | `node_occupancy[stage]` |
| Flush inlet highlight | Per-channel flush command active |

Component path: `apps/hydros-console/src/components/MachineCoreView.tsx`

---

## 9. [DESIGN_DEFAULT] Items

Items flagged as design defaults — physically plausible but not yet calibrated to real hardware specs. Must be resolved before M5 validation pass:

1. **Mesh wire diameters** — coarse/medium/fine weave wire gauge
2. **Collection tube inner diameter** — physical dimensions of S1/S2/S3 channels
3. **Transition buffer volume** — derived from physical tube diameter × buffer length
4. **Storage chamber capacity** — physical volume of bottom ring (determines fill-level calibration)
