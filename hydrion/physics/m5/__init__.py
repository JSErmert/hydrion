"""
M5 physics package — research-grounded conical cascade model.

All constants carry inline SOURCE citations to peer-reviewed literature.
See docs/context/M5_PHYSICS_BASIS.md for full reference list.

Modules:
    materials      — polymer/fluid constants (ε_r, σ, ρ, H, CM factor)
    capture_rt     — Rajagopalan-Tien (1976) liquid-phase filtration
    dep_ndep       — nDEP force, v_critical, Maxwell-Wagner relaxation
    conical_stage  — single conical capture stage (RT + nDEP combined)
    polarization   — inlet polarization zone diagnostics
"""
from .materials     import PP, PE, PET, EPS_R_WATER, RHO_WATER, MU_WATER, cm_factor
from .capture_rt    import MeshSpec, MESH_S1, MESH_S2, MESH_S3_MEMBRANE, rt_single_collector, stage_capture_efficiency
from .dep_ndep      import DEPConfig, dep_force_N, v_critical_ms, ndep_capture_probability
from .conical_stage import ConicalStageSpec, stage_capture, cascade_capture
from .polarization  import PolarizationZone

__all__ = [
    "PP", "PE", "PET",
    "EPS_R_WATER", "RHO_WATER", "MU_WATER",
    "cm_factor",
    "MeshSpec", "MESH_S1", "MESH_S2", "MESH_S3_MEMBRANE",
    "rt_single_collector", "stage_capture_efficiency",
    "DEPConfig", "dep_force_N", "v_critical_ms", "ndep_capture_probability",
    "ConicalStageSpec", "stage_capture", "cascade_capture",
    "PolarizationZone",
]
