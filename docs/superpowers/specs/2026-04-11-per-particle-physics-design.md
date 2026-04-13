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
                       │ [{x_norm, r_norm, status, species}] per step (stage-local)
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

### Coordinate note — r_norm is cone-local, not world-space vertical

`r_norm` represents fractional radial distance from the flow axis to the local cone wall. It is **not** a world-space vertical position. Gravity and buoyancy forces are projected into this reduced radial direction as a modeling approximation, valid when the cone flow axis is horizontal so that physical vertical ≈ radial. This approximation is exact for a horizontal cone; it breaks down for arbitrary cone orientations. This must be documented in code at the point where `v_gravity` is added to `vr`.

### Status semantics — stage-local vs device-level

All statuses are **stage-local** unless otherwise noted.

| Status | Scope | Terminal? | Meaning |
|--------|-------|-----------|---------|
| `in_transit` | Stage-local | No — transient | Particle actively being integrated |
| `near_wall` | Stage-local | No — transient | Particle within `ε_wall` of local wall; capture evaluated next substep |
| `captured` | Stage-local | Yes | Particle captured in this stage; does not proceed to next stage |
| `passed` | Stage-local | Yes | Particle exited this stage at apex (x_norm ≥ 1.0); routes to next stage |

Device-level outcome: a particle that `passed` all three stages has **escaped the cascade**. This is tracked by the environment (Section 3), not by the engine.

`ParticleTrajectory.final_status` is always one of `captured` or `passed`. `in_transit` and `near_wall` are only valid in `SimParticle.status` (internal integration state). They must not appear as a `final_status`.

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
    """Internal integration state. One instance per substep per particle.
    status is stage-local and transient during integration."""
    particle_id: str
    species: str
    d_p_m: float
    x_norm: float       # axial [0=inlet, 1=apex]
    r_norm: float       # cone-local radial [0=axis, 1=local wall] — NOT world-space vertical
    vx: float           # axial velocity [m/s]
    vr: float           # radial velocity [m/s]
    status: str         # "in_transit" | "near_wall" (transient) | "captured" | "passed" (terminal)

@dataclass
class ParticleTrajectory:
    """Full integration record for one particle through one stage.
    Returned by engine per stage. Used for research export.
    final_status is always 'captured' or 'passed' — never 'in_transit' or 'near_wall'."""
    particle_id: str
    species: str
    d_p_m: float
    stage_idx: int
    positions: list[tuple[float, float]]   # (x_norm, r_norm) per substep
    final_status: str                      # "captured" | "passed" — stage-local terminal outcome
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

`near_wall` is a **transient geometric condition**, not a terminal outcome. A particle in `near_wall` continues integrating. It transitions to `captured` if a capture condition is met, or to `passed` if it exits the stage. `near_wall` must never appear as `final_status`.

| State | Type | Condition |
|-------|------|-----------|
| `in_transit` | Transient | Default — no condition met |
| `near_wall` | Transient | `r_norm >= (1.0 - ε_wall)` where `ε_wall = 0.05` |
| `captured` | Terminal | See below — stage-local |
| `passed` | Terminal | `x_norm >= 1.0` — exited stage; routes to next stage |

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
- Continues integrating in `near_wall` state — nDEP at this position is at maximum wall_enhancement, pushing inward
- May transition to `captured` (apex trap), remain `near_wall`, or reach `passed`

**Note:** `v_r_toward_wall > 0` is NOT a capture indicator. In an nDEP device, wall-directed radial velocity means DEP is insufficient — this is the failure mode, not the capture mode.

### Backflush mode

When `backflush > 0.5` (env state), the engine receives a negative flow rate signal. Phase 1 rules:

1. **Axial velocity reversed:** `v_mean(x_norm) = -|Q_backflush| / A(x_norm)` — particles move toward inlet (decreasing x_norm)
2. **Field stays active:** Voltage applied during flush to maintain nDEP repulsion from wall; reduces re-deposition of flushed particles onto walls
3. **Capture logic suspended:** No new captures during backflush. A particle cannot be captured while backflush is active — it is being flushed outward.
4. **Already-captured particles remain captured:** Detachment model deferred to a future phase. Phase 1 treats captured particles as permanently fixed once captured.

Terminal condition during backflush: `x_norm <= 0.0` → particle has exited the inlet end → status `passed` (flushed out). This routes to the run's waste stream, not the next stage.

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

### Per-step call — cascade routing

Each stage receives only particles that passed (were not captured by) the previous stage. The output of one stage is the input of the next. This is not implicit — it is the explicit routing rule.

```python
# In ConicalCascadeEnv.__init__:
self._particle_engine = ParticleDynamicsEngine()
self._particle_set    = particles or _DEFAULT_PARTICLES
self._field_fns       = [analytical_conical_field(stg) for stg in self._stages]
self._log_trajectories: bool = log_trajectories  # constructor flag, default False

# In ConicalCascadeEnv.step():
all_trajectories = []
active_particles = list(self._particle_set)  # full set enters S1

for i, stage in enumerate(self._stages):
    if not active_particles:
        break  # all particles captured upstream — no work to do

    trajs = self._particle_engine.integrate(
        particles  = active_particles,          # survivors from previous stage only
        stage      = stage,
        stage_idx  = i,
        Q_m3s      = float(self._state["Q_m3s"]),
        field_fn   = self._field_fns[i],
        dt_sim     = self._dt,
        n_substeps = 100,
    )
    all_trajectories.extend(trajs)

    # Route survivors: only particles that passed this stage proceed to the next
    active_particles = [
        InputParticle(t.particle_id, t.species, t.d_p_m)
        for t in trajs if t.final_status == "passed"
    ]

# Particles remaining in active_particles after S3 have escaped the full cascade
escaped_device = active_particles
```

### Runtime state (minimal — console-facing)

Written to `self._state` every step. Rewritten completely each step. No history.

```python
# Per stage (s1, s2, s3) — stage-local statuses ("captured" | "passed"):
self._state["particle_positions_s1"] = [
    {"x_norm": t.positions[-1][0], "r_norm": t.positions[-1][1],
     "status": t.final_status, "species": t.species}
    for t in trajs_s1
]
self._state["captured_pp_s1"]  = sum(1 for t in trajs_s1 if t.species == "PP"  and t.final_status == "captured")
self._state["captured_pet_s1"] = sum(1 for t in trajs_s1 if t.species == "PET" and t.final_status == "captured")
# ... repeat for s2, s3

# Device-level escaped count (passed all three stages):
self._state["escaped_device_pp"]  = sum(1 for p in escaped_device if p.species == "PP")
self._state["escaped_device_pet"] = sum(1 for p in escaped_device if p.species == "PET")
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

**Species/status visual encoding rule:**

Species separation is a primary research output. The renderer must encode both dimensions independently:
- **Species → hue.** The color of a particle identifies what it is.
- **Status → radius and opacity.** The emphasis indicates what is happening to it.

Do not use status-only coloring — it would make PP and PET indistinguishable, defeating the purpose of showing species separation.

```tsx
interface ParticleStreamRendererProps {
  points: ParticlePoint[];
}

// Species hue — identifies the particle type
const SPECIES_HUE: Record<string, string> = {
  PP:  '#7fff7f',   // green   — buoyant, low density
  PE:  '#4a9eff',   // blue    — buoyant, slightly denser than PP
  PET: '#ff9966',   // orange  — sinking, high density
};

// Status modifies radius and opacity only — not hue
const STATUS_RADIUS:  Record<string, number> = { captured: 3.5, near_wall: 2.5, in_transit: 2.0, passed: 1.5 };
const STATUS_OPACITY: Record<string, number> = { captured: 0.95, near_wall: 0.75, in_transit: 0.6, passed: 0.25 };

function ParticleStreamRenderer({ points }: ParticleStreamRendererProps) {
  if (!points.length) return null;
  return (
    <>
      {points.map((p, i) => (
        <circle
          key={i}
          cx={p.x}
          cy={p.y}
          r={STATUS_RADIUS[p.status] ?? 2}
          fill={SPECIES_HUE[p.species] ?? '#aaaaaa'}
          opacity={STATUS_OPACITY[p.status] ?? 0.6}
        />
      ))}
    </>
  );
}
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
| `n_substeps` | `conical_cascade_env.py` | 100 | Tunable | Convergence criterion: run at 50, 100, 200. Accept n if all particle endpoint positions satisfy \|x_norm(n) − x_norm(2n)\| < 0.01 AND capture outcomes (captured/passed) agree for all particles. Default 100 is not arbitrary — it passes this criterion for the design-default geometry and flow range. Must be re-validated when geometry changes. |

---

## Success Conditions

- Trajectories are deterministic: same particle set + same physics state → same trajectories
- Captured fraction from engine is consistent with (but not necessarily equal to) RT 1976 aggregate `eta_stage`
- PP particles drift upward (buoyancy), PET drift downward — visible species separation
- At high flow, more particles escape before reaching apex — flow-speed dependence visible
- At backflush (`backflush > 0.5`), axial fluid velocity is negative — particles move toward inlet, no new captures occur
- n_substeps convergence verified: capture outcomes agree between n=100 and n=200 for all particles in default geometry
- Full trajectory logging produces importable data for external analysis (numpy/pandas)
- Console renders from Python output only — confirmed by removing API call and observing static display
