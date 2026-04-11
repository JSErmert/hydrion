"""
M5 conical stage — single cone section combining RT filtration + nDEP.

Physical model:
    A conical section tapers from inlet diameter D_in to apex D_tip over
    axial length L_cone. The mesh lines the cone wall surface.

    nDEP action (cone wall = high-field):
        Particles repelled from wall → converge toward central axis.
        At apex: field minimum → top-hat trap captures converged particles.

    RT action (mesh on cone wall):
        As particles traverse the cone, interception capture occurs.
        Effective face velocity increases toward apex (area decreases).
        RT efficiency computed at mean face velocity.

    Combined (independent mechanisms):
        η_stage = 1 − (1 − η_RT)(1 − η_DEP)

    Density split (from RT gravity term):
        PP/PE: N_G < 0 → reduced capture — buoyant escape is physics, not a flag
        PET:   N_G > 0 → enhanced capture
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .capture_rt import MeshSpec, rt_single_collector, stage_capture_efficiency
from .dep_ndep   import DEPConfig, ndep_capture_probability, v_critical_ms
from .materials  import PolymerProps, MU_WATER


@dataclass
class ConicalStageSpec:
    """Geometry and operating parameters for one conical capture stage."""
    label: str
    mesh: MeshSpec
    dep: DEPConfig
    D_in_m: float           # inlet cone diameter [m]
    D_tip_m: float          # apex diameter [m]
    L_cone_m: float         # axial length of cone section [m]
    half_angle_deg: float = field(init=False)

    def __post_init__(self) -> None:
        r_in  = self.D_in_m  / 2.0
        r_tip = self.D_tip_m / 2.0
        self.half_angle_deg = float(
            np.degrees(np.arctan2(r_in - r_tip, max(self.L_cone_m, 1e-12)))
        )

    @property
    def area_in_m2(self) -> float:
        return np.pi * (self.D_in_m / 2.0) ** 2

    @property
    def area_tip_m2(self) -> float:
        return np.pi * (self.D_tip_m / 2.0) ** 2

    @property
    def area_mean_m2(self) -> float:
        return (self.area_in_m2 + self.area_tip_m2) / 2.0

    @property
    def slant_length_m(self) -> float:
        """Cone wall slant length — used as bed length in RT efficiency."""
        r_in  = self.D_in_m  / 2.0
        r_tip = self.D_tip_m / 2.0
        return float(np.hypot(self.L_cone_m, r_in - r_tip))


def stage_capture(
    stage: ConicalStageSpec,
    polymer: PolymerProps,
    Re_K: float,
    Q_m3s: float,
    d_p_m: float,
    fouling_frac: float = 0.0,
    rho_m: float = 1000.0,
    mu: float = MU_WATER,
    T_K: float = 293.15,
) -> dict:
    """
    Capture efficiency for one conical stage.

    Combines:
        η_RT  — RT single-collector efficiency over cone slant length
        η_DEP — nDEP trapping probability at mean face velocity

    Fouling increases effective mesh solidity slightly (deposited material
    narrows pore → improves interception). Coupling coefficient is
    [DESIGN_DEFAULT] — requires empirical calibration.

    Returns dict:
        eta_stage, eta_RT, eta_DEP, U_mean, v_crit,
        fouling_boost, half_angle_deg, + full RT diagnostics.
    """
    # Face velocity at mean cone cross-section
    U_mean = float(Q_m3s) / max(stage.area_mean_m2, 1e-12)

    # RT capture over cone slant length
    rt = rt_single_collector(
        d_p_m=d_p_m,
        mesh=stage.mesh,
        U_ms=U_mean,
        rho_p=polymer.rho_kgm3,
        rho_m=rho_m,
        mu=mu,
        H=polymer.hamaker_J,
        T_K=T_K,
    )
    eta_RT = stage_capture_efficiency(rt["eta_0"], stage.mesh, stage.slant_length_m)

    # nDEP capture probability
    r_m    = d_p_m / 2.0
    eta_DEP = ndep_capture_probability(
        U_ms=U_mean,
        r_m=r_m,
        Re_K=Re_K,
        grad_E2=stage.dep.grad_E2,
        eps_r_medium=stage.dep.eps_r_medium,
        mu=mu,
    )

    # Fouling coupling — [DESIGN_DEFAULT] coefficient 0.10
    fouling_boost = float(np.clip(0.10 * fouling_frac * eta_RT, 0.0, 0.05))

    # Combined (independent mechanisms)
    eta_stage = float(np.clip(
        1.0 - (1.0 - eta_RT - fouling_boost) * (1.0 - eta_DEP),
        0.0, 1.0,
    ))

    v_crit = v_critical_ms(r_m, Re_K, stage.dep.grad_E2, stage.dep.eps_r_medium, mu)

    return dict(
        eta_stage=eta_stage,
        eta_RT=float(eta_RT),
        eta_DEP=float(eta_DEP),
        U_mean=float(U_mean),
        v_crit=float(v_crit),
        fouling_boost=float(fouling_boost),
        half_angle_deg=stage.half_angle_deg,
        **{f"rt_{k}": v for k, v in rt.items()},
    )


def cascade_capture(
    stages: list[ConicalStageSpec],
    polymer: PolymerProps,
    Re_K: float,
    Q_m3s: float,
    d_p_m: float,
    fouling_fracs: list[float] | None = None,
    rho_m: float = 1000.0,
    mu: float = MU_WATER,
    T_K: float = 293.15,
) -> dict:
    """
    Compound capture efficiency through a cascade of conical stages.

        η_cascade = 1 − ∏(1 − η_i)

    Returns per-stage breakdown and compound efficiency.
    """
    n = len(stages)
    ff_list = fouling_fracs if fouling_fracs and len(fouling_fracs) == n else [0.0] * n

    per_stage = []
    survival  = 1.0   # fraction not yet captured

    for i, stg in enumerate(stages):
        result = stage_capture(
            stage=stg,
            polymer=polymer,
            Re_K=Re_K,
            Q_m3s=Q_m3s,
            d_p_m=d_p_m,
            fouling_frac=ff_list[i],
            rho_m=rho_m,
            mu=mu,
            T_K=T_K,
        )
        survival *= (1.0 - result["eta_stage"])
        per_stage.append(result)

    eta_cascade = float(1.0 - survival)

    return dict(
        eta_cascade=eta_cascade,
        survival=float(survival),
        per_stage=per_stage,
        n_stages=n,
    )
