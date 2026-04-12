# Per-Particle Physics Engine — Design Spec

**Date:** 2026-04-11
**Branch:** explore/conical-cascade-arch
**Status:** Approved for implementation

---

## Goal

Replace the decorative CSS particle animation in `ConicalCascadeView.tsx` with a physics-accurate per-particle trajectory system. The `ParticleDynamicsEngine` is a standalone Python module that integrates particle motion under fluid drag, nDEP force, and gravity/buoyancy at sub-step resolution. It is the single source of truth for particle physics. The console renders from its output — no physics in TypeScript.

This system enables:
- Trajectory-level RL analysis (why did this action change capture outcomes?)
- Cross-validation of the RT 1976 aggregate model against per-particle integration
- Research export of full trajectory data for hardware design studies
- Physically interpretable visualization (trajectories derived from forces, not scripted arcs)

---

## Architecture

Three layers. Physics authority flows downward. No physics crosses the Python→TypeScript boundary.

```
┌─────────────────────────────────────────────────────┐
│  Physics Layer (Python)                             │
│  field_models.py       — analytical_conical_field   │
│  particle_dynamics.py  — ParticleDynamicsEngine     │
└──────────────────────┬──────────────────────────────┘
                       │ ParticleTrajectory[]
┌──────────────────────▼──────────────────────────────┐
│  Environment Layer (Python)                         │
│  conical_cascade_env.py — calls engine per step     │
│                         — stores minimal runtime    │
│                         — logs full trajectories    │
│                           only when flag=True       │
└──────────────────────┬──────────────────────────────┘
                       │ [{x, y, status, species}] per step
┌──────────────────────▼──────────────────────────────┐
│  Console Layer (TypeScript)                         │
│  displayStateMapper.ts — maps to SVG coordinates   │
│  ConicalCascadeView.tsx — renders circles, no CSS  │
│                           animation, no physics     │
└─────────────────────────────────────────────────────┘
```

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `hydrion/physics/m5/field_models.py` | `analytical_conical_field` factory — returns `field_fn(x_norm, r_norm) → grad_E2` |
| Create | `hydrion/physics/m5/particle_dynamics.py` | `ParticleDynamicsEngine`, `SimParticle`, `ParticleTrajectory` |
| Modify | `hydrion/environments/conical_cascade_env.py` | Call engine once per step, store runtime state, opt-in spine logging |
| Modify | `hydrion/service/app.py` | Add `particle_streams` to step payload |
| Modify | `apps/hydros-console/src/scenarios/displayStateMapper.ts` | Add `particleStreams` to `HydrosDisplayState` |
| Modify | `apps/hydros-console/src/components/ConicalCascadeView.tsx` | Replace `AnimatedParticleStream` with `ParticleStreamRenderer` |

No other files change. `hydrion/visual_sampling/particle_sampler.py` is preserved but superseded — do not delete, do not call.

---

## Section 1 — field_models.py

### Purpose

Provides the default `field_fn(x_norm, r_norm) → grad_E2` callable for the engine. Analytically models how ∇|E|² varies through a conical stage. FEM-swappable: any callable with the same signature can replace it.

### Coordinate system

- `x_norm ∈ [0, 1]` — axial position along the cone (0 = inlet, 1 = apex)
- `r_norm ∈ [0, 1]` — radial position normalized to local cone radius at `x_norm` (0 = axis, 1 = wall)

### Formula

```python
R(x_norm)            = R_in - (R_in - R_tip) * x_norm
concentration(x_norm) = (R_tip / R(x_norm)) ** n_field_conc
wall_enhancement(r_norm) = 1.0 + beta_r * r_norm**2

grad_E2(x_norm, r_norm) = grad_E2_apex
                         * concentration(x_norm)
                         * wall_enhancement(r_norm)
```

**Physical basis:**
- `concentration`: field lines concentrate as cone cross-section narrows (flux conservation). Exponent `n_field_conc` is 3–4 depending on assumptions about electrode geometry self-similarity. Default 4.
- `wall_enhancement`: cone wall is the high-field electrode surface. Field is stronger near the wall, driving nDEP repulsion inward.

### Factory signature

```python
def analytical_conical_field(
    stage: ConicalStageSpec,
    beta_r: float = 1.5,            # [DESIGN_DEFAULT] wall enhancement factor
    n_field_conc: int = 4,          # [DESIGN_DEFAULT] area-scaling exponent (2–6)
) -> Callable[[float, float], float]:
    """
    Returns a callable field_fn(x_norm, r_norm) -> grad_E2 [V²/m³].

    STOKES REGIME ASSUMPTION: this field is used to compute terminal velocity
    contributions. Valid for Re_p << 1 (d_p ~ 10-100 µm in water at mm/s).

    DESIGN_DEFAULT: beta_r=1.5 and n_field_conc=4 are approximate.
    Replace with FEM-calibrated values before hardware comparison.
    The interface (callable signature) does not change when constants are updated.
    """
```

### FEM swap pattern (future)

```python
def fem_field_from_table(
    table: np.ndarray,          # shape (Nx, Nr), values = grad_E2
    x_edges: np.ndarray,        # x_norm bin edges
    r_edges: np.ndarray,        # r_norm bin edges
) -> Callable[[float, float], float]:
    interp = RegularGridInterpolator((x_edges, r_edges), table)
    def field_fn(x_norm: float, r_norm: float) -> float:
        return float(interp([[x_norm, r_norm]]))
    return field_fn
```

Engine interface unchanged when FEM data is substituted.

---

## Section 2 — particle_dynamics.py

### Stokes-regime assumption (must be documented in module docstring)

At `d_p ~ 10–100 µm` in water at `mm/s` velocities, particle Reynolds number `Re_p = ρ_m * v * d_p / μ ~ 10⁻³`. Inertia is negligible. Particle velocity at each instant equals the sum of:

```
v_total = v_fluid + v_DEP + v_gravity
```

where:
- `v_fluid` — contributes directly (no force conversion)
- `v_DEP = F_DEP / (3π μ d_p)` — Stokes terminal velocity
- `v_gravity = F_grav / (3π μ d_p) = (ρ_p − ρ_m) g d_p² / (18μ)` — Stokes settling velocity

All three terms are dimensionally consistent `[m/s]` before summing. **This assumption must be stated explicitly in the module docstring and each force method.**

### Data structures

```python
@dataclass
class InputParticle:
    """Caller-provided particle specification. Engine does not generate these."""
    particle_id: str
    species: str        # "PP" | "PE" | "PET"
    d_p_m: float        # diameter [m]

@dataclass
class SimParticle:
    """Internal integration state. One instance per substep per particle."""
    particle_id: str
    species: str
    d_p_m: float
    x_norm: float       # axial [0=inlet, 1=apex]
    r_norm: float       # radial [0=axis, 1=local wall]
    vx: float           # axial velocity [m/s]
    vr: float           # radial velocity [m/s]
    status: str         # "in_transit" | "near_wall" | "captured" | "escaped"

@dataclass
class ParticleTrajectory:
    """Full integration record. Returned by engine. Used for research export."""
    particle_id: str
    species: str
    d_p_m: float
    stage_idx: int
    positions: list[tuple[float, float]]   # (x_norm, r_norm) per substep
    final_status: str
    captured_at_substep: int | None
```

### Fluid velocity model

```python
# Cone geometry
R(x_norm) = R_in - (R_in - R_tip) * x_norm
A(x_norm) = π * R(x_norm)**2
dR_dx     = -(R_in - R_tip) / L_cone      # constant (linear taper)

# Mean axial velocity — continuity: ∂(A·v)/∂x = 0
v_mean(x_norm) = Q / A(x_norm)             # increases toward apex

# Parabolic cross-profile (Poiseuille approximation, slowly-varying cone)
v_axial(x_norm, r_norm) = 2 * v_mean(x_norm) * (1 - r_norm**2)

# Radial drift from cone narrowing (incompressibility, first-order approx)
v_radial(x_norm, r_norm) = -(dR_dx / R(x_norm)) * v_axial(x_norm, r_norm) * r_norm
```

Note: Poiseuille fully-developed assumption breaks down in rapidly converging regions. Valid for half-angles ≤ 20°. Documented in code as approximation.

### nDEP force → terminal velocity

```python
# DEP force (radial component only — cylindrical approximation)
F_DEP = dep_force_N(r_p, Re_K, field_fn(x_norm, r_norm))
# Re[K] < 0 for PP/PE/PET → F_DEP < 0 → toward axis (inward)

v_DEP_radial = F_DEP / (3 * π * μ * d_p)   # Stokes terminal velocity
v_DEP_axial  = 0.0                           # DEP is radially directed
```

### Gravity/buoyancy → terminal velocity

```python
# Stokes settling velocity (signed: negative = upward for buoyant particles)
v_gravity = (rho_p - rho_water) * G_ACC * d_p**2 / (18 * mu)
# Sign convention (horizontal cone, flow axis = x, gravity = physical vertical = r direction):
# PP/PE: rho_p < rho_water → v_gravity < 0 → radially inward (toward axis, buoyant rise)
# PET:   rho_p > rho_water → v_gravity > 0 → radially outward (toward wall, sedimentation)
# r_norm increases toward wall, so positive v_gravity drives particle toward wall
```

### Integration (Euler, Phase 1)

```python
dt_sub = dt_sim / n_substeps

x_new = x + (v_axial + v_DEP_axial) * dt_sub
r_new = r + (v_radial + v_DEP_radial + v_gravity) * dt_sub
r_new = clip(r_new, 0.0, 1.0)   # enforce cone boundary
```

Upgrade path to RK2 is mechanical — same force functions, different integration step.

### Capture and escape logic

Four states:

| State | Condition |
|-------|-----------|
| `in_transit` | Default — no condition met |
| `near_wall` | `r_norm >= (1.0 - ε_wall)` where `ε_wall = 0.05` |
| `captured` | See below |
| `escaped` | `x_norm >= 1.0` |

**Captured — two independent mechanisms:**

1. **Apex trap (nDEP primary mechanism):**
   - `x_norm >= 0.90 AND r_norm <= 0.25`
   - Particle has converged to within 25% of the axis in the apex zone
   - Physically: reached the field minimum where converging particles accumulate

2. **RT mesh contact (mechanical filtration):**
   - `r_norm >= (1.0 - ε_wall)` — wall contact
   - AND `d_p_um > mesh.opening_um` — particle physically cannot pass through mesh
   - Size gate only. No force condition. This is mechanical interception.

**near_wall without RT capture:**
- Particle is at wall but `d_p_um <= mesh.opening_um`
- Continues in `near_wall` state — may slide along wall, be captured at apex, or escape
- nDEP force at this position will be at its maximum (wall_enhancement peak) — strong inward push

**Note:** `v_r_toward_wall > 0` is NOT a capture indicator. In an nDEP device, wall-directed radial velocity means DEP is insufficient — this is the failure mode, not the capture mode.

### Engine class

```python
class ParticleDynamicsEngine:
    """
    Standalone per-particle trajectory integrator.

    STOKES REGIME: All non-fluid forces converted to terminal velocity via
    v = F / (3π μ d_p). Valid for Re_p << 1 (micron-scale particles in water).

    Caller provides:
    - particle set (InputParticle[]) — engine has no generation logic
    - stage geometry (ConicalStageSpec)
    - field callable (field_fn(x_norm, r_norm) → grad_E2)
    - flow rate Q [m³/s]
    - sim timestep dt [s] and n_substeps

    Returns ParticleTrajectory[] per stage.
    """

    def integrate(
        self,
        particles: list[InputParticle],
        stage: ConicalStageSpec,
        stage_idx: int,
        Q_m3s: float,
        field_fn: Callable[[float, float], float],
        dt_sim: float,
        n_substeps: int = 100,
    ) -> list[ParticleTrajectory]:
        ...
```

---

## Section 3 — conical_cascade_env.py Integration

### Default particle set (Phase 1)

```python
_DEFAULT_PARTICLES = [
    InputParticle("pp-median",  "PP",  d_p_m=25e-6),
    InputParticle("pe-median",  "PE",  d_p_m=25e-6),
    InputParticle("pet-median", "PET", d_p_m=25e-6),
]
```

One particle per species at median diameter. Caller can override via env constructor.

### Per-step call

```python
# In ConicalCascadeEnv.__init__:
self._particle_engine = ParticleDynamicsEngine()
self._particle_set    = particles or _DEFAULT_PARTICLES
self._field_fns       = [analytical_conical_field(stg) for stg in self._stages]
self._log_trajectories: bool = log_trajectories  # constructor flag, default False

# In ConicalCascadeEnv.step():
all_trajectories = []
for i, stage in enumerate(self._stages):
    trajs = self._particle_engine.integrate(
        particles  = self._particle_set,
        stage      = stage,
        stage_idx  = i,
        Q_m3s      = float(self._state["Q_m3s"]),
        field_fn   = self._field_fns[i],
        dt_sim     = self._dt,
        n_substeps = 100,
    )
    all_trajectories.extend(trajs)
```

### Runtime state (minimal — console-facing)

Written to `self._state` every step. Rewritten completely each step. No history.

```python
# Per stage (s1, s2, s3):
self._state["particle_positions_s1"] = [
    {"x_norm": t.positions[-1][0], "r_norm": t.positions[-1][1],
     "status": t.final_status, "species": t.species}
    for t in trajs_s1
]
self._state["captured_pp_s1"]  = sum(1 for t in trajs_s1 if t.species == "PP"  and t.final_status == "captured")
self._state["captured_pet_s1"] = sum(1 for t in trajs_s1 if t.species == "PET" and t.final_status == "captured")
# ... repeat for s2, s3
```

### Research logging (opt-in)

```python
if self._log_trajectories:
    artifacts.append_trajectories(run_dir, step_idx, all_trajectories)
# Default: False. Full ParticleTrajectory arrays never touch runtime state.
```

---

## Section 4 — API Payload

Added to the existing step payload dict in `app.py`:

```python
"particle_streams": {
    "s1": [
        {"x_norm": float, "r_norm": float, "status": str, "species": str},
        ...
    ],
    "s2": [...],
    "s3": [...],
}
```

Python sends cone-local coordinates `(x_norm, r_norm)`. **SVG coordinate conversion happens in TypeScript**, using the STAGES geometry constants that already live in `ConicalCascadeView.tsx`. Python has no knowledge of SVG coordinate space — duplicating those constants in Python would be fragile.

Coordinate mapping in TypeScript (displayStateMapper.ts or ConicalCascadeView.tsx):

```typescript
function coneToSVG(xNorm: number, rNorm: number, stg: Stage): { x: number; y: number } {
  const x = stg.xStart + xNorm * (stg.apexX - stg.xStart);
  // r_norm=0 (axis) → y=154 (centreline); r_norm=1 (wall) → y=apexY
  const y = 154 + rNorm * (stg.apexY - 154);
  return { x, y };
}
```

---

## Section 5 — Console Integration

### HydrosDisplayState addition (displayStateMapper.ts)

```typescript
interface ParticlePoint {
  x: number;        // SVG coordinate — converted from x_norm in mapper
  y: number;        // SVG coordinate — converted from r_norm in mapper
  status: string;   // "in_transit" | "near_wall" | "captured" | "escaped"
  species: string;  // "PP" | "PE" | "PET"
}

interface ParticleStreams {
  s1: ParticlePoint[];
  s2: ParticlePoint[];
  s3: ParticlePoint[];
}

// Added to HydrosDisplayState:
particleStreams: ParticleStreams | null;
```

### ConicalCascadeView.tsx

Replace `AnimatedParticleStream` with `ParticleStreamRenderer`:

```tsx
interface ParticleStreamRendererProps {
  points: ParticlePoint[];
  stageColor: string;
}

function ParticleStreamRenderer({ points, stageColor }: ParticleStreamRendererProps) {
  if (!points.length) return null;
  return (
    <>
      {points.map((p, i) => (
        <circle
          key={i}
          cx={p.x}
          cy={p.y}
          r={STATUS_RADIUS[p.status] ?? 2}
          fill={STATUS_COLOR[p.status] ?? stageColor}
          opacity={STATUS_OPACITY[p.status] ?? 0.7}
        />
      ))}
    </>
  );
}

const STATUS_RADIUS:  Record<string, number> = { captured: 3.5, near_wall: 2.5, in_transit: 2, escaped: 1.5 };
const STATUS_COLOR:   Record<string, string> = { captured: '#ffdd44', near_wall: '#ff7744', escaped: '#888888' };
const STATUS_OPACITY: Record<string, number> = { captured: 0.9, near_wall: 0.7, in_transit: 0.6, escaped: 0.3 };
```

No CSS keyframes. No animation state. No physics. Positions update at the existing API poll rate.

---

## Design Defaults — Calibration Required Before Hardware Comparison

| Parameter | Location | Value | Flag | Calibration path |
|-----------|----------|-------|------|------------------|
| `n_field_conc` | `field_models.py` | 4 | [DESIGN_DEFAULT] | FEM simulation of cone electrode geometry |
| `beta_r` | `field_models.py` | 1.5 | [DESIGN_DEFAULT] | FEM or experimental field mapping |
| `grad_E2_apex` | `DEPConfig` (existing) | Hemisphere-on-post estimate | [DESIGN_DEFAULT] | FEM or voltage sweep calibration |
| `ε_wall` | `particle_dynamics.py` | 0.05 | [DESIGN_DEFAULT] | Calibrate to observed near-wall behavior |
| Apex trap threshold | `particle_dynamics.py` | `x_norm >= 0.90, r_norm <= 0.25` | [DESIGN_DEFAULT] | Calibrate to cone geometry and observed trapping zone |
| `n_substeps` | `conical_cascade_env.py` | 100 | Tunable | Convergence test: vary until trajectories stabilise |

---

## Success Conditions

- Trajectories are deterministic: same particle set + same physics state → same trajectories
- Captured fraction from engine is consistent with (but not necessarily equal to) RT 1976 aggregate `eta_stage`
- PP particles drift upward (buoyancy), PET drift downward — visible species separation
- At high flow, more particles escape before reaching apex — flow-speed dependence visible
- At backflush (`backflush > 0.5`), particles move right-to-left — direction reversal correct
- Full trajectory logging produces importable data for external analysis (numpy/pandas)
- Console renders from Python output only — confirmed by removing API call and observing static display
