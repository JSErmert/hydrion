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
