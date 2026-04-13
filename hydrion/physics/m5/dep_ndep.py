"""
M5 dielectrophoresis — nDEP physics for microplastics in water.

All three target polymers (PP, PE, PET) exhibit NEGATIVE DEP in water.
Re[K] ≈ −0.47 to −0.50 across DC to tens of MHz.

Physical consequence for conical cascade device:
    - Cone wall = HIGH E-field region
    - nDEP pushes particles AWAY from cone wall → TOWARD central axis (low-field)
    - Apex trap captures particles converging at the field minimum
    - This INVERTS the M4 assumption (M4 used attraction to node — incorrect)

Sources:
    [PG1992]  Pethig et al. J. Phys. D 25:881 (1992). DOI:10.1088/0022-3727/25/5/022
    [GV2002]  Gascoyne & Vykoukal. Electrophoresis 23:1973 (2002).
    [RSC2025] RSC Advances/PMC (2025) — nDEP confirmed on PP, PE, PET, PVC
              fragments (25–50 µm) in water using FISHBONE-and-funnel electrode.
    [PMC5507] Lapizco-Encinas group (PMC5507384) — triangular insulator geometry
              gives ~50× improvement in ∇|E|² vs circular posts.

DEP force:
    F_DEP = 2π ε_m ε_0 r³ Re[K] ∇|E|²      (negative for nDEP)

Critical velocity (force balance with Stokes drag):
    v_crit = ε_m ε_0 r² |Re[K]| ∇|E|² / (3μ)
    Capture when U_face < v_crit.
    This REPLACES the empirical exp(−0.04 × ΔQ) flow penalty in M4.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .materials import EPS_0, EPS_R_WATER, MU_WATER


@dataclass
class DEPConfig:
    """
    Electrostatic field configuration for one conical stage.

    grad_E2 is the dominant design parameter. The hemisphere-on-post
    approximation gives a rough estimate; replace with FEM for precision.
    """
    voltage_V: float            # applied voltage [V]
    electrode_gap_m: float      # gap between electrodes [m]
    tip_radius_m: float         # apex tip radius [m] — controls field enhancement
    eps_r_medium: float = EPS_R_WATER
    mu: float = MU_WATER

    @property
    def E_mean_Vm(self) -> float:
        """Mean electric field [V/m]."""
        return self.voltage_V / max(self.electrode_gap_m, 1e-12)

    @property
    def field_enhancement(self) -> float:
        """
        Approximate field enhancement factor β at cone tip.
            β ≈ electrode_gap / tip_radius   (hemisphere-on-post)

        Triangular/conical geometries give ≈50× better ∇|E|² than circular.
        Source: [PMC5507] Lapizco-Encinas group — iDEP insulator shape study.
        """
        return self.electrode_gap_m / max(self.tip_radius_m, 1e-12)

    @property
    def grad_E2(self) -> float:
        """
        Estimated ∇|E|² at cone tip [V²/m³].
            ≈ (β × E_mean)² / electrode_gap

        APPROXIMATE — replace with FEM simulation value for precision.
        """
        E_tip = self.field_enhancement * self.E_mean_Vm
        return E_tip**2 / max(self.electrode_gap_m, 1e-12)


# ---------------------------------------------------------------------------
# Core DEP functions
# ---------------------------------------------------------------------------

def dep_force_N(
    r_m: float,
    Re_K: float,
    grad_E2: float,
    eps_r_medium: float = EPS_R_WATER,
) -> float:
    """
    DEP force on a spherical particle [N].

        F_DEP = 2π ε_m ε_0 r³ Re[K] ∇|E|²

    Sign:
        Re[K] < 0 (nDEP, all three polymers in water) → F_DEP < 0
        Negative force = repulsion from high-field (cone wall) toward axis.
    """
    eps_m = eps_r_medium * EPS_0
    return 2.0 * np.pi * eps_m * float(r_m)**3 * float(Re_K) * float(grad_E2)


def v_critical_ms(
    r_m: float,
    Re_K: float,
    grad_E2: float,
    eps_r_medium: float = EPS_R_WATER,
    mu: float = MU_WATER,
) -> float:
    """
    Critical face velocity for DEP capture [m/s].

    Force balance: F_DEP ≥ F_drag (Stokes)
        6π μ r v_crit = 2π ε_m ε_0 r³ |Re[K]| ∇|E|²
        v_crit = ε_m ε_0 r² |Re[K]| ∇|E|² / (3μ)

    Particles at U_face > v_crit are swept past despite nDEP.
    This gives a physics-derived flow penalty — no empirical coefficient needed.
    """
    eps_m = eps_r_medium * EPS_0
    return eps_m * float(r_m)**2 * abs(Re_K) * float(grad_E2) / (3.0 * mu)


def maxwell_wagner_relaxation(
    eps_r_p: float,
    eps_r_m: float,
    sigma_p: float,
    sigma_m: float,
) -> float:
    """
    Maxwell-Wagner relaxation time τ_MW [s].

        τ_MW = ε_0(ε_p + 2ε_m) / (σ_p + 2σ_m)

    For PP in water (deionised): τ_MW ≈ 0.7 µs.
    Polarisation is quasi-instantaneous at device timescales (ms–s).
    """
    numerator   = EPS_0 * (eps_r_p + 2.0 * eps_r_m)
    denominator = sigma_p + 2.0 * sigma_m
    return numerator / max(denominator, 1e-30)


def ndep_capture_probability(
    U_ms: float,
    r_m: float,
    Re_K: float,
    grad_E2: float,
    eps_r_medium: float = EPS_R_WATER,
    mu: float = MU_WATER,
) -> float:
    """
    Probability of nDEP capture [0, 1] at a given face velocity.

    Smooth sigmoid transition around v_critical:
        P_DEP = 1 / (1 + exp(k × (U − v_crit) / v_crit))    k = 10

    At U << v_crit: P_DEP → 1.0   (full DEP capture)
    At U >> v_crit: P_DEP → 0.0   (swept past)
    ±30% of v_crit spans the transition (avoids hard discontinuity at boundary).
    """
    v_crit = v_critical_ms(r_m, Re_K, grad_E2, eps_r_medium, mu)
    if v_crit <= 0.0:
        return 0.0
    k = 10.0
    x = k * (float(U_ms) - v_crit) / v_crit
    return float(1.0 / (1.0 + np.exp(np.clip(x, -50.0, 50.0))))
