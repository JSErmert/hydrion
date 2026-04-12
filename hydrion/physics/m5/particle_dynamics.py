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
    v_radial = (dR_dx / max(R_x, 1e-12)) * v_axial * r_norm
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
# Engine stub — integrate() implemented in Task 3
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
