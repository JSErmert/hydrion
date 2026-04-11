"""
M5 material constants — polymer and fluid properties.

PEER-REVIEWED SOURCES ONLY. No commercial databases.

Sources:
    [B1999]  Brandrup, Immergut, Grulke. Polymer Handbook 4th ed. Wiley, 1999.
             Section VI: Electrical Properties of Polymers.
    [KL1987] Ku, Liepins. Electrical Properties of Polymers. Hanser, 1987.
    [N2002]  Neagu et al. J. Appl. Phys. 92:6365 (2002). DOI:10.1063/1.1518784
    [I1980]  Ieda. IEEE Trans. Elec. Insul. 15:206 (1980). DOI:10.1109/TEI.1980.298314
    [M1994]  Mizutani. IEEE TDEI 1:923 (1994). DOI:10.1109/94.329804
    [F1995]  Fernández et al. J. Phys. Chem. Ref. Data 24:33 (1995). DOI:10.1063/1.555977
             IAPWS 1997 standard — ε_r(20°C, 0.1 MPa) = 80.20 ± 0.03
    [V1972]  Visser (1972) — Hamaker constant lower bound for polyolefin/water
    [G1981]  Gregory (1981) — refined Hamaker estimate, lower end recommended for DLVO
    [PG1992] Pethig et al. J. Phys. D 25:881 (1992). DOI:10.1088/0022-3727/25/5/022
    [GV2002] Gascoyne & Vykoukal. Electrophoresis 23:1973 (2002).
    [RSC2025] RSC Advances/PMC (2025) — nDEP on PP, PE, PET, PVC fragments 25-50µm in water
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# ---------------------------------------------------------------------------
# Universal physical constants (CODATA 2018)
# ---------------------------------------------------------------------------
EPS_0 = 8.854187817e-12   # F/m  — vacuum permittivity
K_B   = 1.380649e-23      # J/K  — Boltzmann constant
G_ACC = 9.80665           # m/s² — standard gravity


@dataclass(frozen=True)
class PolymerProps:
    """Dielectric and physical properties of a polymer species."""
    name: str
    eps_r: float          # relative permittivity              [dimensionless]
    sigma_Sm: float       # bulk electrical conductivity       [S/m]
    rho_kgm3: float       # density                            [kg/m³]
    hamaker_J: float      # Hamaker constant in water          [J]


# ---------------------------------------------------------------------------
# Polymer registry
# ---------------------------------------------------------------------------

PP = PolymerProps(
    name="PP",
    eps_r=2.2,            # SOURCE: [B1999] 2.2–2.3 at 1 kHz–1 MHz, 23°C, isotactic
    sigma_Sm=1e-15,       # SOURCE: [I1980] ρ_v ~ 10^15–10^16 Ω·cm → σ ~ 10^-15–10^-16 S/m
    rho_kgm3=910.0,       # isotactic PP — standard reference
    hamaker_J=0.5e-20,    # SOURCE: [V1972][G1981] conservative lower bound, polyolefin/water
)

PE = PolymerProps(
    name="PE",
    eps_r=2.3,            # SOURCE: [B1999] HDPE 2.30–2.35 at 1 kHz–1 MHz, 23°C
    sigma_Sm=1e-15,       # SOURCE: [I1980][M1994] same order as PP
    rho_kgm3=940.0,       # HDPE reference density
    hamaker_J=0.5e-20,    # SOURCE: [V1972][G1981]
)

PET = PolymerProps(
    name="PET",
    eps_r=3.3,            # SOURCE: [N2002] ε_r ≈ 3.2–3.3 at 1 kHz, 20°C (single primary paper)
    sigma_Sm=1e-14,       # SOURCE: [N2002] inferred from dielectric loss — FLAG: moisture-sensitive,
                          #         single source. Use range 1e-13–1e-15 for sensitivity.
    rho_kgm3=1380.0,      # semi-crystalline PET — standard reference
    hamaker_J=0.5e-20,    # conservative; PET slightly higher than polyolefins, same order of magnitude
)

# Polymer registry for iteration
POLYMERS: dict[str, PolymerProps] = {"PP": PP, "PE": PE, "PET": PET}

# ---------------------------------------------------------------------------
# Water properties at 20°C
# ---------------------------------------------------------------------------
EPS_R_WATER = 80.20      # SOURCE: [F1995] IAPWS 1997 — ε_r(20°C, 0.1 MPa) = 80.20 ± 0.03
SIGMA_DI_SM = 5.5e-6     # S/m — deionised water (standard electrochemistry)
RHO_WATER   = 1000.0     # kg/m³ at 20°C
MU_WATER    = 1.0e-3     # Pa·s — dynamic viscosity at 20°C


# ---------------------------------------------------------------------------
# Clausius-Mossotti factor
# ---------------------------------------------------------------------------

def cm_factor(eps_r_particle: float, eps_r_medium: float = EPS_R_WATER) -> float:
    """
    Real part of the Clausius-Mossotti factor (high-frequency permittivity limit).

        Re[K(ω→∞)] = (ε_p − ε_m) / (ε_p + 2ε_m)

    All three target polymers yield Re[K] < 0 in water (nDEP):
        PP  → Re[K] ≈ −0.480   SOURCE: [PG1992][GV2002][RSC2025]
        PE  → Re[K] ≈ −0.479
        PET → Re[K] ≈ −0.472

    DC limit: Re[K] → −0.500 for all three (σ_polymer << σ_water).
    """
    return (eps_r_particle - eps_r_medium) / (eps_r_particle + 2.0 * eps_r_medium)


def brownian_diffusivity(
    d_p_m: float,
    T_K: float = 293.15,
    mu: float = MU_WATER,
) -> float:
    """
    Stokes-Einstein diffusion coefficient [m²/s].

        D = k_B T / (3π μ d_p)

    Cunningham slip correction C_c = 1.0 (valid for d_p >> mean free path in liquid).
    For d_p = 10 µm in water: D ≈ 4.3×10⁻¹⁴ m²/s — diffusion is negligible.
    """
    return K_B * T_K / (3.0 * np.pi * mu * d_p_m)


# Pre-computed CM factors at 20°C water
CM_PP  = cm_factor(PP.eps_r)    # ≈ −0.480
CM_PE  = cm_factor(PE.eps_r)    # ≈ −0.479
CM_PET = cm_factor(PET.eps_r)   # ≈ −0.472
