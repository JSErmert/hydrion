# Per-Particle Physics Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace decorative CSS particle animation with a physics-accurate `ParticleDynamicsEngine` that integrates per-particle trajectories under fluid drag, nDEP force, and gravity/buoyancy. The console renders from Python output — no physics in TypeScript.

**Architecture:** Three layers. `field_models.py` returns a swappable field callable. `particle_dynamics.py` integrates particles through one stage under Stokes-regime force superposition (Euler, 100 substeps). `conical_cascade_env.py` calls the engine for each of the three stages in cascade order (only `passed` particles advance). Final positions are written to `_state["particle_streams"]`, extracted by the scenario runner, and delivered to TypeScript as `ScenarioStepRecord.particleStreams`. TypeScript converts `(x_norm, r_norm)` to SVG coordinates and renders `<circle>` elements — no CSS animation, no browser physics.

**Tech Stack:** Python 3.11, numpy, dataclasses; FastAPI; TypeScript/React, SVG

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `hydrion/physics/m5/field_models.py` | `analytical_conical_field` factory — `field_fn(x_norm, r_norm) → grad_E2`; FEM swap pattern |
| Create | `hydrion/physics/m5/particle_dynamics.py` | `InputParticle`, `SimParticle`, `ParticleTrajectory` dataclasses; force helper functions; `ParticleDynamicsEngine.integrate()` |
| Create | `tests/test_field_models.py` | Unit tests for field model |
| Create | `tests/test_particle_dynamics.py` | Unit + integration tests for engine |
| Modify | `hydrion/environments/conical_cascade_env.py` | Engine instantiation; cascade routing loop; `particle_streams` in `_state`; `truth_state`/`sensor_state`/`_update_normalized_state` compatibility properties |
| Modify | `hydrion/scenarios/types.py` | Add `particle_streams` field to `ScenarioStepRecord` |
| Modify | `hydrion/scenarios/runner.py` | Extract `particle_streams` from truth dict; set on `ScenarioStepRecord` |
| Modify | `hydrion/service/app.py` | Use `ConicalCascadeEnv` in `/api/scenarios/run` |
| Modify | `apps/hydros-console/src/api/types.ts` | Add `particleStreams?` to `ScenarioStepRecord` |
| Modify | `apps/hydros-console/src/scenarios/displayStateMapper.ts` | Add `ParticlePoint`, `ParticleStreams`, `particleStreams` to `HydrosDisplayState`; `coneToSVG` conversion |
| Modify | `apps/hydros-console/src/components/ConicalCascadeView.tsx` | Replace `AnimatedParticleStream` with `ParticleStreamRenderer`; species hue + status radius/opacity encoding |

No other files change. `hydrion/visual_sampling/particle_sampler.py` is preserved but not called.

---

## Context for implementers

**Existing infrastructure usable without modification:**

- `dep_force_N(r_m, Re_K, grad_E2)` — `hydrion/physics/m5/dep_ndep.py:83`
- `CM_PP`, `CM_PE`, `CM_PET` (≈ −0.480, −0.479, −0.472) — `hydrion/physics/m5/materials.py:122`
- `PP`, `PE`, `PET` (with `rho_kgm3`) — `hydrion/physics/m5/materials.py:48`
- `EPS_0`, `EPS_R_WATER`, `MU_WATER`, `RHO_WATER`, `G_ACC` — `hydrion/physics/m5/materials.py`
- `ConicalStageSpec` (with `D_in_m`, `D_tip_m`, `L_cone_m`, `dep.grad_E2`, `mesh.opening_um`) — `hydrion/physics/m5/conical_stage.py:35`
- `_default_stages()` — `hydrion/environments/conical_cascade_env.py:53` — returns the three default `ConicalStageSpec` instances used by tests

**Design defaults — must be flagged in comments:**

| Parameter | Value | Flag |
|-----------|-------|------|
| `beta_r` | 1.5 | `[DESIGN_DEFAULT]` |
| `n_field_conc` | 4 | `[DESIGN_DEFAULT]` |
| `EPSILON_WALL` | 0.05 | `[DESIGN_DEFAULT]` |
| Apex trap | `x_norm >= 0.90, r_norm <= 0.25` | `[DESIGN_DEFAULT]` |
| `n_substeps` | 100 | Tunable — passes convergence criterion for default geometry |

---

## Task 1: field_models.py

**Files:**
- Create: `hydrion/physics/m5/field_models.py`
- Create: `tests/test_field_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_field_models.py
import pytest
from hydrion.environments.conical_cascade_env import _default_stages
# Import will fail until field_models.py is created


def test_import():
    from hydrion.physics.m5.field_models import analytical_conical_field
    assert callable(analytical_conical_field)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_field_models.py::test_import -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Create field_models.py**

```python
# hydrion/physics/m5/field_models.py
"""
M5 electric field models — grad_E2 callables for the ParticleDynamicsEngine.

STOKES REGIME ASSUMPTION: These field models compute ∇|E|² used to derive nDEP
terminal velocity via v_DEP = F_DEP / (3π μ d_p). Valid for Re_p << 1
(d_p ~ 10–100 µm in water at mm/s).
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .conical_stage import ConicalStageSpec


def analytical_conical_field(
    stage: ConicalStageSpec,
    beta_r: float = 1.5,        # [DESIGN_DEFAULT] wall enhancement factor
    n_field_conc: int = 4,      # [DESIGN_DEFAULT] flux-concentration exponent (2–6)
) -> Callable[[float, float], float]:
    """
    Returns field_fn(x_norm, r_norm) -> grad_E2 [V²/m³].

    Coordinate system:
        x_norm in [0, 1] — axial (0=inlet, 1=apex)
        r_norm in [0, 1] — radial normalized to local cone radius (0=axis, 1=wall)

    Physics:
        R(x_norm)             = R_in - (R_in - R_tip) * x_norm
        concentration(x_norm) = (R_tip / R(x_norm)) ** n_field_conc
        wall_enhancement(r)   = 1.0 + beta_r * r_norm**2
        grad_E2               = grad_E2_apex * concentration * wall_enhancement

    At x_norm=1.0, r_norm=0.0: grad_E2 == stage.dep.grad_E2 (apex on axis).

    STOKES REGIME ASSUMPTION: valid for Re_p << 1 (micron-scale particles in water).

    [DESIGN_DEFAULT] beta_r=1.5, n_field_conc=4 — replace with FEM-calibrated values
    before hardware comparison. The callable interface does not change when constants
    are updated.

    Args:
        stage:        ConicalStageSpec (R_in, R_tip from D_in_m/D_tip_m, dep.grad_E2)
        beta_r:       wall enhancement shape factor [DESIGN_DEFAULT]
        n_field_conc: flux-concentration exponent [DESIGN_DEFAULT]

    Returns:
        Callable[[float, float], float]: field_fn(x_norm, r_norm) -> grad_E2
    """
    R_in         = stage.D_in_m  / 2.0
    R_tip        = stage.D_tip_m / 2.0
    grad_E2_apex = stage.dep.grad_E2

    def field_fn(x_norm: float, r_norm: float) -> float:
        R_x           = R_in - (R_in - R_tip) * x_norm
        concentration = (R_tip / max(R_x, 1e-12)) ** n_field_conc
        wall_enh      = 1.0 + beta_r * r_norm ** 2
        return float(grad_E2_apex * concentration * wall_enh)

    return field_fn


def fem_field_from_table(
    table: np.ndarray,
    x_edges: np.ndarray,
    r_edges: np.ndarray,
) -> Callable[[float, float], float]:
    """
    FEM field model — constructs field_fn from a 2D lookup table.

    Drop-in replacement for analytical_conical_field. Engine interface unchanged.

    Args:
        table:   shape (Nx, Nr), values = grad_E2 [V²/m³]
        x_edges: x_norm grid coordinates (length Nx)
        r_edges: r_norm grid coordinates (length Nr)

    Returns:
        Callable[[float, float], float]: field_fn(x_norm, r_norm) -> grad_E2
    """
    try:
        from scipy.interpolate import RegularGridInterpolator
    except ImportError as e:  # pragma: no cover
        raise ImportError("scipy is required for fem_field_from_table") from e

    interp = RegularGridInterpolator(
        (x_edges, r_edges), table, method="linear",
        bounds_error=False, fill_value=0.0,
    )

    def field_fn(x_norm: float, r_norm: float) -> float:
        return float(interp([[x_norm, r_norm]]))

    return field_fn
```

- [ ] **Step 4: Write full test suite**

```python
# tests/test_field_models.py
import math
import pytest
from hydrion.environments.conical_cascade_env import _default_stages
from hydrion.physics.m5.field_models import analytical_conical_field


@pytest.fixture
def stage_s1():
    return _default_stages()[0]   # D_in=80mm, D_tip=20mm, L=120mm


def test_import():
    assert callable(analytical_conical_field)


def test_apex_on_axis_equals_grad_E2_apex(stage_s1):
    """
    At x_norm=1.0 (apex), r_norm=0.0 (axis):
        R(1.0) = R_tip  → concentration = (R_tip/R_tip)^4 = 1.0
        wall_enh(0.0)   = 1 + beta_r*0 = 1.0
        → grad_E2 = grad_E2_apex exactly.
    """
    field_fn = analytical_conical_field(stage_s1)
    result = field_fn(1.0, 0.0)
    assert result == pytest.approx(stage_s1.dep.grad_E2, rel=1e-6)


def test_increases_toward_apex(stage_s1):
    """Concentration factor must increase monotonically toward apex at r_norm=0."""
    field_fn = analytical_conical_field(stage_s1)
    v01 = field_fn(0.1, 0.0)
    v05 = field_fn(0.5, 0.0)
    v09 = field_fn(0.9, 0.0)
    assert v01 < v05 < v09, (
        f"grad_E2 must increase toward apex. Got {v01:.3e} < {v05:.3e} < {v09:.3e}"
    )


def test_increases_toward_wall(stage_s1):
    """Wall enhancement must increase monotonically toward wall at fixed x_norm."""
    field_fn = analytical_conical_field(stage_s1)
    v02 = field_fn(0.5, 0.2)
    v05 = field_fn(0.5, 0.5)
    v09 = field_fn(0.5, 0.9)
    assert v02 < v05 < v09, (
        f"grad_E2 must increase toward wall. Got {v02:.3e} < {v05:.3e} < {v09:.3e}"
    )


def test_n_field_conc_parameter(stage_s1):
    """Higher n_field_conc increases field concentration toward apex."""
    fn4 = analytical_conical_field(stage_s1, n_field_conc=4)
    fn6 = analytical_conical_field(stage_s1, n_field_conc=6)
    # At x_norm=0.5, fn6 should concentrate more than fn4
    assert fn6(0.5, 0.0) > fn4(0.5, 0.0)


def test_beta_r_parameter(stage_s1):
    """Higher beta_r increases wall enhancement."""
    fn_low  = analytical_conical_field(stage_s1, beta_r=0.5)
    fn_high = analytical_conical_field(stage_s1, beta_r=3.0)
    assert fn_high(0.5, 0.8) > fn_low(0.5, 0.8)


def test_returns_positive_values(stage_s1):
    """grad_E2 must be positive everywhere."""
    field_fn = analytical_conical_field(stage_s1)
    for x in [0.0, 0.3, 0.7, 1.0]:
        for r in [0.0, 0.3, 0.7, 1.0]:
            assert field_fn(x, r) > 0, f"grad_E2 must be positive at ({x}, {r})"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_field_models.py -v`
Expected: All 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add hydrion/physics/m5/field_models.py tests/test_field_models.py
git commit -m "feat(m5): add analytical_conical_field factory (field_models.py)"
```

---

## Task 2: particle_dynamics.py — data structures and force helpers

**Files:**
- Create: `hydrion/physics/m5/particle_dynamics.py`
- Create: `tests/test_particle_dynamics.py` (force-function tests only)

- [ ] **Step 1: Write the failing force-function tests**

```python
# tests/test_particle_dynamics.py
import math
import pytest

# These imports will fail until particle_dynamics.py is created
from hydrion.physics.m5.particle_dynamics import (
    InputParticle, SimParticle, ParticleTrajectory,
    _fluid_velocity, _dep_radial_velocity, _gravity_radial_velocity,
)
from hydrion.physics.m5.materials import MU_WATER, RHO_WATER
from hydrion.environments.conical_cascade_env import _default_stages
from hydrion.physics.m5.field_models import analytical_conical_field


def test_import_dataclasses():
    p = InputParticle(particle_id="pp-1", species="PP", d_p_m=25e-6)
    assert p.species == "PP"
    assert p.d_p_m == pytest.approx(25e-6)


def test_dep_radial_velocity_is_negative_for_pp(stage_s1_fixture):
    """PP has Re[K] < 0 → nDEP force is negative → radially inward (v < 0)."""
    stage = stage_s1_fixture
    field_fn = analytical_conical_field(stage)
    v = _dep_radial_velocity(0.5, 0.5, 25e-6, "PP", field_fn)
    assert v < 0, f"PP nDEP must be inward (negative), got {v:.3e}"


def test_dep_radial_velocity_is_negative_for_pet(stage_s1_fixture):
    """PET has Re[K] < 0 → nDEP is also inward (all three polymers are nDEP)."""
    stage = stage_s1_fixture
    field_fn = analytical_conical_field(stage)
    v = _dep_radial_velocity(0.5, 0.5, 25e-6, "PET", field_fn)
    assert v < 0, f"PET nDEP must be inward (negative), got {v:.3e}"


def test_gravity_pp_is_negative():
    """PP ρ_p=910 < ρ_water=1000 → buoyant → v_gravity < 0 → toward axis."""
    v = _gravity_radial_velocity(25e-6, "PP")
    assert v < 0, f"PP buoyancy must be negative (toward axis), got {v:.3e}"


def test_gravity_pet_is_positive():
    """PET ρ_p=1380 > ρ_water=1000 → sinking → v_gravity > 0 → toward wall."""
    v = _gravity_radial_velocity(25e-6, "PET")
    assert v > 0, f"PET sedimentation must be positive (toward wall), got {v:.3e}"


def test_fluid_axial_zero_at_wall(stage_s1_fixture):
    """Poiseuille: no-slip at r_norm=1.0 → v_axial = 0."""
    stage = stage_s1_fixture
    R_in  = stage.D_in_m  / 2.0
    R_tip = stage.D_tip_m / 2.0
    Q_m3s = 10.0 / 60000.0  # 10 L/min
    v_ax, _ = _fluid_velocity(0.5, 1.0, Q_m3s, R_in, R_tip, stage.L_cone_m)
    assert abs(v_ax) < 1e-10, f"v_axial must be 0 at wall, got {v_ax:.3e}"


def test_fluid_axial_increases_toward_apex(stage_s1_fixture):
    """Mean velocity increases as cone narrows toward apex (continuity)."""
    stage = stage_s1_fixture
    R_in  = stage.D_in_m  / 2.0
    R_tip = stage.D_tip_m / 2.0
    Q_m3s = 10.0 / 60000.0
    v01, _ = _fluid_velocity(0.1, 0.0, Q_m3s, R_in, R_tip, stage.L_cone_m)
    v09, _ = _fluid_velocity(0.9, 0.0, Q_m3s, R_in, R_tip, stage.L_cone_m)
    assert v01 < v09, f"v_axial must increase toward apex. Got {v01:.3e} vs {v09:.3e}"


@pytest.fixture
def stage_s1_fixture():
    return _default_stages()[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_particle_dynamics.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create particle_dynamics.py (data structures + force helpers only)**

```python
# hydrion/physics/m5/particle_dynamics.py
"""
M5 per-particle trajectory integrator — ParticleDynamicsEngine.

STOKES REGIME ASSUMPTION: At d_p ~ 10–100 µm in water at mm/s, particle Reynolds
number Re_p = ρ_m v d_p / μ ~ 10⁻³. Inertia is negligible. Particle velocity at
each instant:
    v_total = v_fluid + v_DEP + v_gravity
where v_DEP and v_gravity are Stokes terminal velocities (v = F / 3πμd_p).

COORDINATE NOTE: r_norm is cone-local (0 = axis, 1 = local wall at x_norm), NOT
world-space vertical. Gravity is projected into the radial direction — exact for
a horizontal cone axis, approximate otherwise.

STATUS SEMANTICS (stage-local):
    'in_transit' — transient: particle is integrating
    'near_wall'  — transient: r_norm >= (1 - EPSILON_WALL); capture evaluated
    'captured'   — terminal: captured in this stage; does not advance
    'passed'     — terminal: exited at apex (x_norm >= 1.0); routes to next stage

    ParticleTrajectory.final_status is always 'captured' or 'passed'.
    'in_transit' and 'near_wall' are internal states only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .materials import (
    MU_WATER, RHO_WATER, G_ACC,
    PP, PE, PET,
    CM_PP, CM_PE, CM_PET,
    EPS_R_WATER,
)
from .dep_ndep import dep_force_N
from .conical_stage import ConicalStageSpec

# Capture geometry constants — [DESIGN_DEFAULT]
EPSILON_WALL    = 0.05   # near-wall band width [DESIGN_DEFAULT]
APEX_X_THRESH   = 0.90   # apex trap axial entry point [DESIGN_DEFAULT]
APEX_R_THRESH   = 0.25   # apex trap radial radius [DESIGN_DEFAULT]

# Pre-computed CM factors (see materials.py for source citations)
_CM  = {"PP": CM_PP,          "PE": CM_PE,          "PET": CM_PET}
_RHO = {"PP": PP.rho_kgm3,    "PE": PE.rho_kgm3,    "PET": PET.rho_kgm3}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InputParticle:
    """Caller-provided particle specification. Engine generates no particles."""
    particle_id: str
    species: str    # "PP" | "PE" | "PET"
    d_p_m: float    # diameter [m]


@dataclass
class SimParticle:
    """Internal integration state. Rebuilt each substep.
    status is stage-local and transient during integration."""
    particle_id: str
    species: str
    d_p_m: float
    x_norm: float   # axial [0=inlet, 1=apex]
    r_norm: float   # cone-local radial [0=axis, 1=local wall] — NOT world-space vertical
    vx: float       # axial velocity [m/s]
    vr: float       # radial velocity [m/s]
    status: str     # "in_transit" | "near_wall" | "captured" | "passed"


@dataclass
class ParticleTrajectory:
    """Full integration record for one particle through one stage.
    final_status is always 'captured' or 'passed' — never 'in_transit' or 'near_wall'."""
    particle_id: str
    species: str
    d_p_m: float
    stage_idx: int
    positions: list[tuple[float, float]]  # (x_norm, r_norm) per substep
    final_status: str                     # "captured" | "passed" — stage-local terminal
    captured_at_substep: Optional[int]    # substep index when captured, or None


# ---------------------------------------------------------------------------
# Force helper functions (module-level, importable for testing)
# ---------------------------------------------------------------------------

def _fluid_velocity(
    x_norm: float,
    r_norm: float,
    Q_m3s: float,
    R_in: float,
    R_tip: float,
    L_cone: float,
    mu: float = MU_WATER,
) -> tuple[float, float]:
    """
    Parabolic Poiseuille axial velocity + radial drift in a slowly-varying cone.

    STOKES REGIME: Poiseuille approximation valid for half-angles <= 20°.
    Continuity: v_mean(x) = Q / A(x). No-slip: v_axial(x, r_norm=1.0) = 0.

    Returns:
        (v_axial, v_radial) [m/s]
        v_axial  > 0 for Q > 0 (particle moves toward apex)
        v_radial is typically negative (inward) due to cone narrowing
    """
    R_x   = R_in - (R_in - R_tip) * x_norm
    A_x   = math.pi * R_x ** 2
    v_mean = Q_m3s / max(A_x, 1e-12)
    v_axial = 2.0 * v_mean * (1.0 - r_norm ** 2)   # Poiseuille profile
    dR_dx   = -(R_in - R_tip) / max(L_cone, 1e-12)  # constant taper
    v_radial = -(dR_dx / max(R_x, 1e-12)) * v_axial * r_norm
    return v_axial, v_radial


def _dep_radial_velocity(
    x_norm: float,
    r_norm: float,
    d_p_m: float,
    species: str,
    field_fn: Callable[[float, float], float],
    mu: float = MU_WATER,
    eps_r_medium: float = EPS_R_WATER,
) -> float:
    """
    nDEP Stokes terminal velocity — radial component only.

    DEP force is radially directed (cylindrical approximation). Axial DEP = 0.

    STOKES REGIME: v_DEP = F_DEP / (3π μ d_p). Valid for Re_p << 1.

    Returns:
        v_DEP_radial [m/s] — negative for nDEP (toward axis, away from high-field wall)
    """
    r_p   = d_p_m / 2.0
    Re_K  = _CM[species]
    grad_e2 = field_fn(x_norm, r_norm)
    F_DEP = dep_force_N(r_p, Re_K, grad_e2, eps_r_medium)
    return F_DEP / (3.0 * math.pi * mu * d_p_m)


def _gravity_radial_velocity(
    d_p_m: float,
    species: str,
    mu: float = MU_WATER,
    rho_medium: float = RHO_WATER,
    g: float = G_ACC,
) -> float:
    """
    Stokes settling / buoyancy velocity projected into cone-local radial direction.

    Approximation: gravity ≈ radial. Exact for horizontal cone axis.

    STOKES REGIME: v_grav = (ρ_p - ρ_m) g d_p² / (18μ).

    Sign convention (r_norm increases toward wall):
        PP/PE: rho_p < rho_water → v_gravity < 0 → toward axis (buoyant)
        PET:   rho_p > rho_water → v_gravity > 0 → toward wall (sinks)
    """
    rho_p = _RHO[species]
    return (rho_p - rho_medium) * g * d_p_m ** 2 / (18.0 * mu)


# ---------------------------------------------------------------------------
# Capture predicates
# ---------------------------------------------------------------------------

def _is_apex_captured(p: SimParticle) -> bool:
    """Apex trap — nDEP primary mechanism. Particle converged to field minimum."""
    return p.x_norm >= APEX_X_THRESH and p.r_norm <= APEX_R_THRESH


def _is_rt_captured(p: SimParticle, mesh_opening_um: float) -> bool:
    """
    RT mesh — mechanical filtration. Size-gated only. No force condition.
    A particle at the wall passes through the mesh if d_p <= opening.
    """
    d_p_um = p.d_p_m * 1e6
    return p.r_norm >= (1.0 - EPSILON_WALL) and d_p_um > mesh_opening_um


# ---------------------------------------------------------------------------
# Engine (integrate() is in Task 3)
# ---------------------------------------------------------------------------

class ParticleDynamicsEngine:
    """
    Standalone per-particle trajectory integrator.

    STOKES REGIME: All non-fluid forces converted to terminal velocity via
    v = F / (3π μ d_p). Valid for Re_p << 1 (micron-scale particles in water).

    Caller provides:
        particles   — list[InputParticle] (engine generates no particles)
        stage       — ConicalStageSpec (geometry + mesh + dep)
        field_fn    — Callable[[float, float], float]: (x_norm, r_norm) → grad_E2
        Q_m3s       — volumetric flow rate [m³/s]
        dt_sim      — simulation timestep [s]
        n_substeps  — default 100, passes convergence criterion for default geometry

    Returns list[ParticleTrajectory] per integrate() call — one per input particle.
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
        backflush: bool = False,
    ) -> list[ParticleTrajectory]:
        # Implemented in Task 3
        raise NotImplementedError("Task 3: implement integrate()")
```

- [ ] **Step 4: Run force-function tests**

Run: `pytest tests/test_particle_dynamics.py -k "not integrate" -v`
Expected: All 7 force-function tests PASS. `integrate` test skipped or not yet written.

- [ ] **Step 5: Commit**

```bash
git add hydrion/physics/m5/particle_dynamics.py tests/test_particle_dynamics.py
git commit -m "feat(m5): particle_dynamics data structures and force helpers"
```

---

## Task 3: particle_dynamics.py — integration loop and capture logic

**Files:**
- Modify: `hydrion/physics/m5/particle_dynamics.py` (fill in `integrate()`)
- Modify: `tests/test_particle_dynamics.py` (add integration tests)

- [ ] **Step 1: Write integration tests (they fail until integrate() is implemented)**

Add these tests to `tests/test_particle_dynamics.py`:

```python
# Add these imports at the top of test_particle_dynamics.py
from hydrion.physics.m5.particle_dynamics import ParticleDynamicsEngine

# Add fixture for engine
@pytest.fixture
def engine():
    return ParticleDynamicsEngine()


# Add these tests:

def test_integrate_returns_one_trajectory_per_particle(engine, stage_s1_fixture):
    stage = stage_s1_fixture
    field_fn = analytical_conical_field(stage)
    particles = [
        InputParticle("pp-1", "PP",  25e-6),
        InputParticle("pe-1", "PE",  25e-6),
        InputParticle("pet-1","PET", 25e-6),
    ]
    trajs = engine.integrate(particles, stage, 0, 10.0/60000.0, field_fn, dt_sim=1.0)
    assert len(trajs) == 3


def test_final_status_is_terminal(engine, stage_s1_fixture):
    """final_status must always be 'captured' or 'passed' — never 'in_transit' or 'near_wall'."""
    stage = stage_s1_fixture
    field_fn = analytical_conical_field(stage)
    particles = [InputParticle(f"p{i}", sp, 25e-6) for i, sp in enumerate(["PP","PE","PET"])]
    trajs = engine.integrate(particles, stage, 0, 10.0/60000.0, field_fn, dt_sim=1.0)
    for t in trajs:
        assert t.final_status in ("captured", "passed"), (
            f"final_status must be terminal, got '{t.final_status}' for {t.species}"
        )


def test_deterministic(engine, stage_s1_fixture):
    """Same inputs must produce identical trajectories (no randomness)."""
    stage = stage_s1_fixture
    field_fn = analytical_conical_field(stage)
    particles = [InputParticle("pp-1", "PP", 25e-6)]
    Q = 10.0 / 60000.0

    trajs_a = engine.integrate(particles, stage, 0, Q, field_fn, dt_sim=1.0)
    trajs_b = engine.integrate(particles, stage, 0, Q, field_fn, dt_sim=1.0)

    assert trajs_a[0].final_status == trajs_b[0].final_status
    assert trajs_a[0].positions[-1] == trajs_b[0].positions[-1]


def test_pp_drifts_inward_relative_to_pet(engine, stage_s1_fixture):
    """
    PP is buoyant (floats toward axis) and has same nDEP as PET.
    At low flow, both captured. At high flow where some escape:
    PP should end at lower r_norm than PET (due to buoyancy assisting inward drift).
    """
    stage = stage_s1_fixture
    field_fn = analytical_conical_field(stage)
    # Use high flow to ensure escaped particles (species separation visible in passed set)
    Q_high = 20.0 / 60000.0   # 20 L/min — above typical capture range
    pp  = InputParticle("pp-1",  "PP",  25e-6)
    pet = InputParticle("pet-1", "PET", 25e-6)
    trajs = engine.integrate([pp, pet], stage, 0, Q_high, field_fn, dt_sim=1.0)
    t_pp  = next(t for t in trajs if t.species == "PP")
    t_pet = next(t for t in trajs if t.species == "PET")
    # Final r_norm: PP should be lower (toward axis) than PET (toward wall)
    r_pp  = t_pp.positions[-1][1]
    r_pet = t_pet.positions[-1][1]
    assert r_pp < r_pet, (
        f"PP (buoyant) should end closer to axis than PET (dense). "
        f"r_PP={r_pp:.3f} r_PET={r_pet:.3f}"
    )


def test_high_flow_increases_passed_fraction(engine, stage_s1_fixture):
    """At higher flow, fewer particles should be captured (more pass through)."""
    stage = stage_s1_fixture
    field_fn = analytical_conical_field(stage)
    particles = [InputParticle(f"pp-{i}", "PP", 25e-6) for i in range(3)]

    Q_low  = 5.0  / 60000.0
    Q_high = 18.0 / 60000.0

    trajs_low  = engine.integrate(particles, stage, 0, Q_low,  field_fn, dt_sim=1.0)
    trajs_high = engine.integrate(particles, stage, 0, Q_high, field_fn, dt_sim=1.0)

    captured_low  = sum(1 for t in trajs_low  if t.final_status == "captured")
    captured_high = sum(1 for t in trajs_high if t.final_status == "captured")

    # Note: if all are captured at low flow and all pass at high, this holds.
    # If Q_low is also too high to capture, test still passes (captured_low >= captured_high).
    assert captured_low >= captured_high, (
        f"Higher flow should not increase captures. "
        f"low={captured_low} high={captured_high}"
    )


def test_convergence_n100_vs_n200(engine, stage_s1_fixture):
    """
    Substep convergence: capture outcomes must agree between n=100 and n=200.
    Final positions must agree within 0.01. This verifies the default n_substeps=100
    is adequate for the design-default geometry.
    """
    stage = stage_s1_fixture
    field_fn = analytical_conical_field(stage)
    Q = 10.0 / 60000.0
    particles = [
        InputParticle("pp-1",  "PP",  25e-6),
        InputParticle("pe-1",  "PE",  25e-6),
        InputParticle("pet-1", "PET", 25e-6),
    ]

    trajs_100 = engine.integrate(particles, stage, 0, Q, field_fn, dt_sim=1.0, n_substeps=100)
    trajs_200 = engine.integrate(particles, stage, 0, Q, field_fn, dt_sim=1.0, n_substeps=200)

    for t100, t200 in zip(trajs_100, trajs_200):
        assert t100.final_status == t200.final_status, (
            f"Outcome diverges between n=100 and n=200 for {t100.species}: "
            f"{t100.final_status} vs {t200.final_status}"
        )
        x100, r100 = t100.positions[-1]
        x200, r200 = t200.positions[-1]
        assert abs(x100 - x200) < 0.01, (
            f"x_norm endpoint divergence > 0.01 for {t100.species}: {x100:.4f} vs {x200:.4f}"
        )
        assert abs(r100 - r200) < 0.01, (
            f"r_norm endpoint divergence > 0.01 for {t100.species}: {r100:.4f} vs {r200:.4f}"
        )


def test_backflush_no_captures(engine, stage_s1_fixture):
    """During backflush, capture logic is suspended — no particle may be captured."""
    stage = stage_s1_fixture
    field_fn = analytical_conical_field(stage)
    Q = 10.0 / 60000.0
    particles = [InputParticle(f"p{i}", sp, 25e-6) for i, sp in enumerate(["PP","PE","PET"])]
    trajs = engine.integrate(particles, stage, 0, Q, field_fn, dt_sim=1.0, backflush=True)
    for t in trajs:
        assert t.final_status == "passed", (
            f"No captures during backflush. Got '{t.final_status}' for {t.species}"
        )
```

- [ ] **Step 2: Run tests to verify they fail on NotImplementedError**

Run: `pytest tests/test_particle_dynamics.py -v`
Expected: Force-function tests PASS; integration tests FAIL with `NotImplementedError`

- [ ] **Step 3: Implement integrate() in particle_dynamics.py**

Replace the `integrate()` stub in `ParticleDynamicsEngine` with:

```python
def integrate(
    self,
    particles: list[InputParticle],
    stage: ConicalStageSpec,
    stage_idx: int,
    Q_m3s: float,
    field_fn: Callable[[float, float], float],
    dt_sim: float,
    n_substeps: int = 100,
    backflush: bool = False,
) -> list[ParticleTrajectory]:
    """
    Euler integration of all particles through one conical stage.

    Each particle starts at (x_norm=0.0, r_norm=0.5) — stage inlet, mid-radius.
    Substep dt = dt_sim / n_substeps.

    Upgrade path: replace Euler step with RK2 using same _fluid_velocity,
    _dep_radial_velocity, _gravity_radial_velocity functions.

    Backflush mode:
        - Q_m3s negated → axial velocity reversed (particles move toward inlet)
        - Field stays active (nDEP prevents wall re-deposition)
        - Capture logic suspended (no new captures)
        - Exit at x_norm <= 0.0 → status 'passed' (flushed to waste stream)
    """
    R_in  = stage.D_in_m  / 2.0
    R_tip = stage.D_tip_m / 2.0
    L     = stage.L_cone_m
    mesh_opening_um = stage.mesh.opening_um

    # Negative Q reverses axial flow; DEP field unchanged
    Q_eff  = -abs(Q_m3s) if backflush else Q_m3s
    dt_sub = dt_sim / max(n_substeps, 1)

    trajectories: list[ParticleTrajectory] = []

    for inp in particles:
        p = SimParticle(
            particle_id=inp.particle_id,
            species=inp.species,
            d_p_m=inp.d_p_m,
            x_norm=0.0,
            r_norm=0.5,   # [DESIGN_DEFAULT] mid-radius starting position
            vx=0.0,
            vr=0.0,
            status="in_transit",
        )
        positions: list[tuple[float, float]] = [(p.x_norm, p.r_norm)]
        captured_at: Optional[int] = None

        for sub_idx in range(n_substeps):
            if p.status in ("captured", "passed"):
                break

            # Force superposition — Stokes regime (see module docstring)
            v_ax, v_rad = _fluid_velocity(
                p.x_norm, p.r_norm, Q_eff, R_in, R_tip, L
            )
            v_dep_r  = _dep_radial_velocity(
                p.x_norm, p.r_norm, p.d_p_m, p.species, field_fn
            )
            # Gravity projected into radial direction (see COORDINATE NOTE)
            v_grav_r = _gravity_radial_velocity(p.d_p_m, p.species)

            # Euler integration
            p.x_norm += v_ax  * dt_sub
            p.r_norm += (v_rad + v_dep_r + v_grav_r) * dt_sub
            p.r_norm  = float(np.clip(p.r_norm, 0.0, 1.0))  # enforce cone boundary

            positions.append((p.x_norm, p.r_norm))

            if backflush:
                if p.x_norm <= 0.0:
                    p.status = "passed"   # flushed out inlet end
                    break
            else:
                # Apex trap (nDEP primary mechanism)
                if _is_apex_captured(p):
                    p.status = "captured"
                    captured_at = sub_idx + 1
                    break
                # RT mesh filtration (size-gated)
                if _is_rt_captured(p, mesh_opening_um):
                    p.status = "captured"
                    captured_at = sub_idx + 1
                    break
                # Near-wall transient state
                p.status = "near_wall" if p.r_norm >= (1.0 - EPSILON_WALL) else "in_transit"
                # Stage exit
                if p.x_norm >= 1.0:
                    p.status = "passed"
                    break

        # Guarantee terminal final_status (ran out of substeps → treat as passed)
        if p.status in ("in_transit", "near_wall"):
            p.status = "passed"

        trajectories.append(ParticleTrajectory(
            particle_id=p.particle_id,
            species=p.species,
            d_p_m=p.d_p_m,
            stage_idx=stage_idx,
            positions=positions,
            final_status=p.status,
            captured_at_substep=captured_at,
        ))

    return trajectories
```

- [ ] **Step 4: Run all particle_dynamics tests**

Run: `pytest tests/test_particle_dynamics.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add hydrion/physics/m5/particle_dynamics.py tests/test_particle_dynamics.py
git commit -m "feat(m5): ParticleDynamicsEngine.integrate() — Euler Stokes regime"
```

---

## Task 4: conical_cascade_env.py — engine integration and cascade routing

**Files:**
- Modify: `hydrion/environments/conical_cascade_env.py`
- Create: `tests/test_particle_engine_env.py`

- [ ] **Step 1: Write failing env tests**

```python
# tests/test_particle_engine_env.py
import json
import numpy as np
import pytest
from hydrion.environments.conical_cascade_env import ConicalCascadeEnv


@pytest.fixture
def env():
    e = ConicalCascadeEnv(config_path="configs/default.yaml", seed=0)
    e.reset(seed=0)
    return e


def test_particle_streams_in_state_after_step(env):
    """After step(), _state must contain 'particle_streams' with s1/s2/s3 keys."""
    action = np.array([0.5, 0.5, 0.0, 0.8], dtype=np.float32)
    env.step(action)
    assert "particle_streams" in env._state, "particle_streams missing from _state"
    ps = env._state["particle_streams"]
    assert "s1" in ps and "s2" in ps and "s3" in ps


def test_particle_streams_have_required_fields(env):
    """Each particle point must have x_norm, r_norm, status, species."""
    action = np.array([0.5, 0.5, 0.0, 0.8], dtype=np.float32)
    env.step(action)
    ps = env._state["particle_streams"]
    for key in ("s1", "s2", "s3"):
        for pt in ps[key]:
            assert "x_norm"  in pt, f"{key} point missing x_norm"
            assert "r_norm"  in pt, f"{key} point missing r_norm"
            assert "status"  in pt, f"{key} point missing status"
            assert "species" in pt, f"{key} point missing species"
            assert pt["status"]  in ("captured", "passed")
            assert pt["species"] in ("PP", "PE", "PET")


def test_particle_streams_json_serializable(env):
    """particle_streams must be JSON-serializable (required for API payload)."""
    action = np.array([0.5, 0.5, 0.0, 0.8], dtype=np.float32)
    env.step(action)
    ps = env._state["particle_streams"]
    json_str = json.dumps(ps)
    assert len(json_str) > 0


def test_truth_state_property(env):
    """truth_state property must return _state (same dict object)."""
    assert env.truth_state is env._state


def test_sensor_state_property(env):
    """sensor_state property must return an empty dict."""
    assert isinstance(env.sensor_state, dict)
    assert len(env.sensor_state) == 0


def test_cascade_routing_s2_receives_only_passed(env):
    """
    Particles in s2 must have been 'passed' from s1 — cascade routing.
    If all 3 particles are captured in s1, s2 and s3 must be empty.
    """
    # Use low flow to maximize capture in s1
    action = np.array([0.3, 0.3, 0.0, 1.0], dtype=np.float32)
    env.step(action)
    ps = env._state["particle_streams"]
    n_s1 = len(ps["s1"])
    n_s2 = len(ps["s2"])
    # If s1 captured all, s2 is empty. If s1 passed some, s2 has those.
    # Either way, s2 count <= s1 count (cascade can only reduce, never add particles)
    assert n_s2 <= n_s1, f"s2 ({n_s2}) cannot have more particles than s1 ({n_s1})"


def test_backflush_no_captures_in_streams(env):
    """During backflush, all particle_streams entries must have status='passed'."""
    action = np.array([0.5, 0.5, 1.0, 0.8], dtype=np.float32)  # bf_cmd=1.0
    env.step(action)
    ps = env._state["particle_streams"]
    for key in ("s1", "s2", "s3"):
        for pt in ps[key]:
            assert pt["status"] == "passed", (
                f"No captures during backflush. Got {pt['status']} in {key}"
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_particle_engine_env.py -v`
Expected: FAIL — `particle_streams` not yet in `_state`, `truth_state` property missing

- [ ] **Step 3: Modify conical_cascade_env.py — add imports**

Add to the imports section at the top of `conical_cascade_env.py`:

```python
from ..physics.m5.field_models import analytical_conical_field
from ..physics.m5.particle_dynamics import (
    InputParticle, ParticleDynamicsEngine,
)
```

- [ ] **Step 4: Add _DEFAULT_PARTICLES and modify __init__**

Add `_DEFAULT_PARTICLES` constant after the existing `_FLUSH_DRAIN_RATE` line:

```python
# ---------------------------------------------------------------------------
# Default particle set — one representative particle per species (Phase 1)
# ---------------------------------------------------------------------------
_DEFAULT_PARTICLES: list[InputParticle] = [
    InputParticle("pp-median",  "PP",  d_p_m=25e-6),
    InputParticle("pe-median",  "PE",  d_p_m=25e-6),
    InputParticle("pet-median", "PET", d_p_m=25e-6),
]
```

Add `particles` and `log_trajectories` to `ConicalCascadeEnv.__init__` signature:

```python
def __init__(
    self,
    config_path: str = "configs/default.yaml",
    stages: list[ConicalStageSpec] | None = None,
    pol_zone: PolarizationZone | None = None,
    d_p_um: float = 10.0,
    seed: int | None = None,
    render_mode=None,
    particles: list[InputParticle] | None = None,   # NEW
    log_trajectories: bool = False,                  # NEW
):
```

Add to `__init__` body, after `self._dt = ...`:

```python
# Particle dynamics engine
self._particle_engine  = ParticleDynamicsEngine()
self._particle_set     = particles if particles is not None else list(_DEFAULT_PARTICLES)
self._log_trajectories = log_trajectories
```

- [ ] **Step 5: Add compatibility properties and method to ConicalCascadeEnv**

Add these three members to `ConicalCascadeEnv` (after `_info`):

```python
# ------------------------------------------------------------------
# ScenarioRunner compatibility — mirrors HydrionEnv interface
# ------------------------------------------------------------------

@property
def truth_state(self) -> dict:
    """
    Expose _state as truth_state for ScenarioRunner compatibility.
    ConicalCascadeEnv uses _state as its authoritative physics state
    (no sensor layer). Returns the live dict — mutations are reflected.
    """
    return self._state

@property
def sensor_state(self) -> dict:
    """
    Return empty dict — ConicalCascadeEnv has no sensor noise layer.
    Required by ScenarioRunner for ScenarioStepRecord.sensorState.
    """
    return {}

def _update_normalized_state(self) -> None:
    """
    Sync clogging model internal state into _state.
    Called by apply_initial_state() after writing initial fouling.
    """
    self._state.update(self.clogging._state)
```

- [ ] **Step 6: Add cascade routing loop inside step()**

In `step()`, after the existing `self.clogging.update(...)` and before the polarization zone block, insert:

```python
        # ── Particle dynamics engine — cascade routing ─────────────────────
        # Each stage receives only particles that passed through previous stages.
        # Voltage-scaled stages used so field_fn reflects the applied voltage.
        bf_active = bf_cmd > 0.5

        trajs_per_stage: list[list] = [[], [], []]
        active_particles = list(self._particle_set)

        for i, stg in enumerate(stages):  # `stages` is already voltage-scaled above
            if not active_particles:
                break
            field_fn_i = analytical_conical_field(stg)
            trajs = self._particle_engine.integrate(
                particles  = active_particles,
                stage      = stg,
                stage_idx  = i,
                Q_m3s      = Q_m3s,
                field_fn   = field_fn_i,
                dt_sim     = self._dt,
                n_substeps = 100,
                backflush  = bf_active,
            )
            trajs_per_stage[i] = trajs
            # Cascade routing: only 'passed' particles enter the next stage
            active_particles = [
                InputParticle(t.particle_id, t.species, t.d_p_m)
                for t in trajs if t.final_status == "passed"
            ]

        # Particles still in active_particles after S3 escaped the full cascade
        escaped_device = active_particles

        # Write particle_streams — final position of each particle (current step only)
        def _make_stream(trajs_list: list) -> list[dict]:
            return [
                {
                    "x_norm":  t.positions[-1][0],
                    "r_norm":  t.positions[-1][1],
                    "status":  t.final_status,
                    "species": t.species,
                }
                for t in trajs_list
            ]

        self._state["particle_streams"] = {
            "s1": _make_stream(trajs_per_stage[0]),
            "s2": _make_stream(trajs_per_stage[1]),
            "s3": _make_stream(trajs_per_stage[2]),
        }

        # Per-stage capture counts
        for i in range(3):
            lb = f"s{i + 1}"
            tl = trajs_per_stage[i]
            self._state[f"captured_pp_{lb}"]  = sum(1 for t in tl if t.species == "PP"  and t.final_status == "captured")
            self._state[f"captured_pe_{lb}"]  = sum(1 for t in tl if t.species == "PE"  and t.final_status == "captured")
            self._state[f"captured_pet_{lb}"] = sum(1 for t in tl if t.species == "PET" and t.final_status == "captured")

        # Device-level escape counts
        self._state["escaped_device_pp"]  = sum(1 for p in escaped_device if p.species == "PP")
        self._state["escaped_device_pet"] = sum(1 for p in escaped_device if p.species == "PET")

        # Research logging (opt-in — full ParticleTrajectory arrays)
        # if self._log_trajectories:
        #     artifacts.append_trajectories(run_dir, step_idx, all_trajs)  # Phase 2
```

Also add `particle_streams` initialization to `reset()`, after the existing `self._state["bf_active"] = 0.0` line:

```python
        self._state["particle_streams"] = None
```

- [ ] **Step 7: Run env tests**

Run: `pytest tests/test_particle_engine_env.py tests/test_conical_cascade_env.py -v`
Expected: All tests PASS (including the original accumulation tests)

- [ ] **Step 8: Commit**

```bash
git add hydrion/environments/conical_cascade_env.py tests/test_particle_engine_env.py
git commit -m "feat(m5): wire ParticleDynamicsEngine into ConicalCascadeEnv with cascade routing"
```

---

## Task 5: Service layer — wire ConicalCascadeEnv through scenario runner

**Files:**
- Modify: `hydrion/scenarios/types.py`
- Modify: `hydrion/scenarios/runner.py`
- Modify: `hydrion/service/app.py`
- Create: `tests/test_particle_scenario_runner.py`

- [ ] **Step 1: Write failing runner test**

```python
# tests/test_particle_scenario_runner.py
import pytest
from hydrion.scenarios.runner import ScenarioRunner, load_scenario
from hydrion.environments.conical_cascade_env import ConicalCascadeEnv


def test_scenario_runner_produces_particle_streams():
    """
    ScenarioRunner with ConicalCascadeEnv must populate particleStreams
    on each ScenarioStepRecord after the first step.
    """
    env = ConicalCascadeEnv(config_path="configs/default.yaml")
    runner = ScenarioRunner(env)
    scenario = load_scenario("hydrion/scenarios/examples/baseline_nominal.yaml")
    history = runner.run(scenario)

    # At least one step should have particle_streams
    found = False
    for step in history.steps:
        if step.particle_streams is not None:
            found = True
            ps = step.particle_streams
            assert "s1" in ps and "s2" in ps and "s3" in ps
            # At least some particles must appear
            total = len(ps["s1"]) + len(ps["s2"]) + len(ps["s3"])
            assert total > 0, "particle_streams must have at least one particle"
            break

    assert found, "No step contained particle_streams — check CCE._state['particle_streams']"


def test_scenario_step_dict_has_particleStreams():
    """ScenarioStepRecord.to_dict() must include 'particleStreams' key."""
    env = ConicalCascadeEnv(config_path="configs/default.yaml")
    runner = ScenarioRunner(env)
    scenario = load_scenario("hydrion/scenarios/examples/baseline_nominal.yaml")
    history = runner.run(scenario)
    d = history.to_dict()
    step_dict = d["steps"][1]   # step 0 may have None (before first integration)
    assert "particleStreams" in step_dict
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_particle_scenario_runner.py -v`
Expected: FAIL — `ScenarioStepRecord` has no `particle_streams` attribute

- [ ] **Step 3: Add particle_streams field to ScenarioStepRecord in types.py**

In `hydrion/scenarios/types.py`, modify `ScenarioStepRecord`:

```python
@dataclass
class ScenarioStepRecord:
    t: float
    stepIndex: int
    scenarioInputs: Dict[str, Any]
    truthState: Dict[str, Any]
    sensorState: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]
    particle_streams: Optional[Dict[str, list]] = None   # NEW — from ParticleDynamicsEngine

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "stepIndex": self.stepIndex,
            "scenarioInputs": self.scenarioInputs,
            "truthState": self.truthState,
            "sensorState": self.sensorState,
            "reward": self.reward,
            "done": self.done,
            "info": self.info,
            "particleStreams": self.particle_streams,    # NEW
        }
```

Add `Optional` to the imports at the top of `types.py`:
```python
from typing import Any, Dict, List, Optional
```

- [ ] **Step 4: Extract particle_streams in runner.py**

In `hydrion/scenarios/runner.py`, after the `truth = {k: ...}` comprehension, add:

```python
            # Extract particle_streams before building truthState.
            # particle_streams is a non-numeric nested dict — kept separate to
            # preserve truthState as Record<string, number> in the TypeScript layer.
            particle_streams = truth.pop("particle_streams", None)
```

And modify the `history.steps.append(...)` call to pass `particle_streams`:

```python
            history.steps.append(ScenarioStepRecord(
                t=round(t, 6),
                stepIndex=step_idx,
                scenarioInputs={
                    "flowLmin": round(flow, 4),
                    "particleDensity": round(density, 4),
                    "activeDisturbances": [
                        {"type": d.type, "intensity": d.intensity} for d in active_dist
                    ],
                },
                truthState=truth,
                sensorState=sensor,
                reward=float(reward),
                done=bool(terminated or truncated),
                info=info,
                particle_streams=particle_streams,   # NEW
            ))
```

- [ ] **Step 5: Change /api/scenarios/run to use ConicalCascadeEnv**

In `hydrion/service/app.py`, add the import:

```python
from hydrion.environments.conical_cascade_env import ConicalCascadeEnv
```

In the `/api/scenarios/run` endpoint, change:

```python
    env = HydrionEnv(config_path="configs/default.yaml", auto_reset=False)
```

to:

```python
    env = ConicalCascadeEnv(config_path="configs/default.yaml")
```

- [ ] **Step 6: Run all service layer tests**

Run: `pytest tests/test_particle_scenario_runner.py tests/test_conical_cascade_env.py tests/test_particle_engine_env.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add hydrion/scenarios/types.py hydrion/scenarios/runner.py hydrion/service/app.py tests/test_particle_scenario_runner.py
git commit -m "feat(m5): wire ConicalCascadeEnv into scenario runner; propagate particle_streams to API"
```

---

## Task 6: TypeScript — displayStateMapper and ParticleStreamRenderer

**Files:**
- Modify: `apps/hydros-console/src/api/types.ts`
- Modify: `apps/hydros-console/src/scenarios/displayStateMapper.ts`
- Modify: `apps/hydros-console/src/components/ConicalCascadeView.tsx`

- [ ] **Step 1: Add particleStreams to TypeScript ScenarioStepRecord in api/types.ts**

In `apps/hydros-console/src/api/types.ts`, add `particleStreams?` to `ScenarioStepRecord`:

```typescript
/** Raw particle position from Python engine — cone-local coordinates */
interface ParticlePointRaw {
  x_norm: number;
  r_norm: number;
  status: string;   // "captured" | "passed"
  species: string;  // "PP" | "PE" | "PET"
}

/** One simulation step from a scenario execution. */
export interface ScenarioStepRecord {
  t: number;
  stepIndex: number;
  scenarioInputs: ScenarioInputsRecord;
  /** Physics truth state from env.truth_state — authoritative, not observed. */
  truthState: Record<string, number>;
  /** Sensor state from env.sensor_state — observational, may contain noise. */
  sensorState: Record<string, number>;
  reward: number;
  done: boolean;
  info: Record<string, unknown>;
  /** Per-stage particle positions from ParticleDynamicsEngine — cone-local coords */
  particleStreams?: {
    s1: ParticlePointRaw[];
    s2: ParticlePointRaw[];
    s3: ParticlePointRaw[];
  };
}
```

Note: `ParticlePointRaw` contains `x_norm, r_norm` (not SVG coords). The mapper converts to SVG space.

- [ ] **Step 2: Add ParticlePoint, ParticleStreams, and particleStreams to displayStateMapper.ts**

In `apps/hydros-console/src/scenarios/displayStateMapper.ts`, add after the imports:

```typescript
// ---------------------------------------------------------------------------
// Particle stream types
// ---------------------------------------------------------------------------

export interface ParticlePoint {
  x: number;        // SVG coordinate (converted from x_norm)
  y: number;        // SVG coordinate (converted from r_norm)
  status: string;   // "captured" | "passed"
  species: string;  // "PP" | "PE" | "PET"
}

export interface ParticleStreams {
  s1: ParticlePoint[];
  s2: ParticlePoint[];
  s3: ParticlePoint[];
}

// Stage geometry for (x_norm, r_norm) → SVG coordinate conversion.
// These constants must match ConicalCascadeView.tsx STAGES exactly.
// If the geometry changes, update both files.
const _CY = 154;  // device centreline y (matches ConicalCascadeView CY)
const _STAGE_GEOM = [
  { xStart: 118, apexX: 296, apexY: 243 },  // S1
  { xStart: 306, apexX: 484, apexY: 243 },  // S2
  { xStart: 494, apexX: 672, apexY: 243 },  // S3
] as const;

function coneToSVG(
  xNorm: number,
  rNorm: number,
  stageIdx: number,
): { x: number; y: number } {
  const stg = _STAGE_GEOM[stageIdx];
  return {
    x: stg.xStart + xNorm * (stg.apexX - stg.xStart),
    y: _CY + rNorm * (stg.apexY - _CY),
  };
}

function mapParticleStream(
  raw: Array<{ x_norm: number; r_norm: number; status: string; species: string }> | undefined,
  stageIdx: number,
): ParticlePoint[] {
  if (!raw || raw.length === 0) return [];
  return raw.map(p => ({
    ...coneToSVG(p.x_norm, p.r_norm, stageIdx),
    status:  p.status,
    species: p.species,
  }));
}
```

Add `particleStreams: ParticleStreams | null` to `HydrosDisplayState`:

```typescript
export interface HydrosDisplayState {
  // ... existing fields ...

  // Per-particle physics streams (from ParticleDynamicsEngine)
  particleStreams: ParticleStreams | null;
}
```

Add mapping in `mapStepRecordToDisplayState`:

```typescript
  // At the end of the return object in mapStepRecordToDisplayState:
  particleStreams: step.particleStreams
    ? {
        s1: mapParticleStream(step.particleStreams.s1, 0),
        s2: mapParticleStream(step.particleStreams.s2, 1),
        s3: mapParticleStream(step.particleStreams.s3, 2),
      }
    : null,
```

- [ ] **Step 3: Build frontend and verify TypeScript compiles**

Run: `cd apps/hydros-console && npx tsc --noEmit`
Expected: Zero type errors

- [ ] **Step 4: Replace AnimatedParticleStream with ParticleStreamRenderer in ConicalCascadeView.tsx**

In `apps/hydros-console/src/components/ConicalCascadeView.tsx`:

**a)** Add `ParticlePoint` and `ParticleStreams` import from displayStateMapper at the top:

```typescript
import type { HydrosDisplayState, ParticlePoint } from '../scenarios/displayStateMapper';
```

**b)** Add `ParticleStreamRenderer` component (add it near where `AnimatedParticleStream` currently is — around line 179):

```typescript
// ── Physics-accurate particle renderer — replaces CSS animation ─────────
//
// Species → hue: identifies what the particle is (primary research signal)
// Status  → radius and opacity: indicates what is happening (secondary)
// No CSS keyframes. No animation state. Positions update at API poll rate.

const SPECIES_HUE: Record<string, string> = {
  PP:  '#7fff7f',   // green  — buoyant, low density
  PE:  '#4a9eff',   // blue   — buoyant, slightly denser than PP
  PET: '#ff9966',   // orange — sinking, high density
};

const STATUS_RADIUS: Record<string, number> = {
  captured:  3.5,
  near_wall: 2.5,
  in_transit: 2.0,
  passed:    1.5,
};

const STATUS_OPACITY: Record<string, number> = {
  captured:  0.95,
  near_wall: 0.75,
  in_transit: 0.60,
  passed:    0.25,
};

interface ParticleStreamRendererProps {
  points: ParticlePoint[];
}

function ParticleStreamRenderer({ points }: ParticleStreamRendererProps) {
  if (points.length === 0) return null;
  return (
    <>
      {points.map((p, i) => (
        <circle
          key={i}
          cx={p.x}
          cy={p.y}
          r={STATUS_RADIUS[p.status]  ?? 2.0}
          fill={SPECIES_HUE[p.species] ?? '#aaaaaa'}
          opacity={STATUS_OPACITY[p.status] ?? 0.6}
        />
      ))}
    </>
  );
}
```

**c)** Update `ConicalCascadeView` component props to accept `particleStreams`. Find the props interface (it accepts `HydrosDisplayState`) and verify `particleStreams` flows through.

**d)** In the SVG render body of `ConicalCascadeView`, find the `AnimatedParticleStream` usage for each stage (there should be 3 — one per stage). Replace each with `ParticleStreamRenderer`. The current pattern looks like:

```tsx
<AnimatedParticleStream
  stageIdx={i}
  xStart={stg.xStart}
  ...
/>
```

Replace all three with:

```tsx
{/* Particle streams: rendered from Python physics output, no browser physics */}
{state.particleStreams && (
  <ParticleStreamRenderer
    points={state.particleStreams[`s${i + 1}` as 's1' | 's2' | 's3']}
  />
)}
```

Note: if `AnimatedParticleStream` is called inside a `STAGES.map()` loop, replace the entire AnimatedParticleStream call. If called three times individually, replace each.

**e)** Remove the `AnimatedParticleStream`, `AnimatedParticleStreamProps`, `buildParticles`, `particleKeyframes`, and `AnimParticle` definitions — they are no longer needed.

- [ ] **Step 5: Build the frontend**

Run: `cd apps/hydros-console && npm run build`
Expected: Build succeeds, zero errors

- [ ] **Step 6: Run full test suite**

Run: `pytest -v`
Expected: All Python tests PASS

- [ ] **Step 7: Commit**

```bash
git add apps/hydros-console/src/api/types.ts \
        apps/hydros-console/src/scenarios/displayStateMapper.ts \
        apps/hydros-console/src/components/ConicalCascadeView.tsx
git commit -m "feat(console): ParticleStreamRenderer replaces CSS animation — physics positions from Python"
```

---

## Self-Review Checklist

Run after all tasks complete before offering the finishing-branch step.

### Spec coverage

| Spec requirement | Task that implements it |
|-----------------|------------------------|
| `analytical_conical_field` factory with `beta_r`, `n_field_conc` | Task 1 |
| FEM swap pattern (`fem_field_from_table`) | Task 1 |
| `InputParticle`, `SimParticle`, `ParticleTrajectory` dataclasses | Task 2 |
| Stokes regime assumption documented in module docstring | Task 2 |
| Force superposition: fluid + nDEP + gravity | Task 2 |
| `v_DEP = F_DEP / (3π μ d_p)` using existing `dep_force_N` | Task 2 |
| Gravity sign: PP < 0 (buoyant), PET > 0 (sinking) | Task 2 |
| Euler integration with `r_norm = clip(r_new, 0, 1)` | Task 3 |
| Apex trap: `x_norm >= 0.90 AND r_norm <= 0.25` | Task 3 |
| RT capture: size-gated only (`d_p_um > mesh.opening_um`) | Task 3 |
| `near_wall` is transient — `final_status` never `in_transit`/`near_wall` | Task 3 |
| Backflush: axial reversed, field active, capture suspended | Task 3 |
| `passed` particles cascade from S1 → S2 → S3 | Task 4 |
| `particle_streams` written to `_state` every step | Task 4 |
| `truth_state` property + `sensor_state` property on CCE | Task 4 |
| `particle_streams` extracted from truthState, kept separate | Task 5 |
| `ConicalCascadeEnv` used in `/api/scenarios/run` | Task 5 |
| `coneToSVG` conversion in mapper using STAGES geometry | Task 6 |
| Species → hue, status → radius/opacity (not status → hue) | Task 6 |
| CSS animation and `AnimatedParticleStream` removed | Task 6 |
| `n_substeps` convergence verified at n=100 vs n=200 | Task 3 |

### Placeholder scan

Every code block in this plan contains complete, runnable code. Verify no `NotImplementedError` remains in production code after Task 3.

### Type consistency

- `InputParticle` created in Task 2, used by `integrate()` (Task 3), `_DEFAULT_PARTICLES` (Task 4), and cascade routing (Task 4) — same type throughout.
- `ParticleTrajectory.final_status` checked for `"passed"` in cascade routing (Task 4) — matches `"passed"` terminal status in `integrate()` (Task 3).
- `particle_streams` keys `"s1"/"s2"/"s3"` written in Python (Task 4), read in runner (Task 5), typed in TypeScript as `s1/s2/s3` (Task 6) — consistent.
- `coneToSVG` stage index 0/1/2 matches `STAGES[0]/[1]/[2]` in `ConicalCascadeView.tsx` — both use 0-based indexing.
