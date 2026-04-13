"""
M5 polarization zone — inlet particle dipole induction.

The polarization zone is the first section particles encounter.
It establishes the induced dipole moment before any conical stage.

Physics:
    Induced dipole: p_ind = 4π ε_m ε_0 r³ K(ω) E
    Maxwell-Wagner relaxation: τ_MW = ε_0(ε_p + 2ε_m) / (σ_p + 2σ_m)

    For PP in deionised water: τ_MW ≈ 0.7 µs
    → Polarisation is quasi-instantaneous at device timescales (ms–s).
    → The polarisation zone is NOT a rate-limiting step.

    Its role in the simulation:
        1. Marks the voltage-on entry point in state
        2. Computes τ_MW as a diagnostic for each polymer
        3. Confirms nDEP regime at operating frequency
        4. Flags any approach toward crossover frequency

Crossover note:
    Below the Maxwell-Wagner crossover frequency, electric double-layer
    (EDL) effects can induce transient pDEP for PS-like particles.
    Above ~10 kHz in deionised water, nDEP dominates for PP/PE/PET.
    Source: Pethig et al. J. Phys. D 25:881 (1992). DOI:10.1088/0022-3727/25/5/022
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .materials import EPS_0, EPS_R_WATER, PolymerProps, cm_factor
from .dep_ndep   import maxwell_wagner_relaxation


@dataclass
class PolarizationZone:
    """Inlet polarization zone configuration."""
    length_m: float         # axial length [m]
    E_field_Vm: float       # uniform field strength [V/m]
    frequency_Hz: float     # AC frequency; 0 = DC
    area_m2: float          # cross-sectional area at inlet [m²]
    eps_r_medium: float = EPS_R_WATER
    sigma_medium: float = 5.5e-6  # deionised water [S/m]

    def characterise(self, polymer: PolymerProps) -> dict:
        """
        Polarization diagnostics for one polymer species.

        Returns:
            tau_MW_s         — Maxwell-Wagner relaxation time [s]
            f_crossover_Hz   — frequency at which K(ω) changes sign
            is_ndep_regime   — True if operating conditions give nDEP
            Re_K             — Clausius-Mossotti factor (high-freq limit)
            dipole_norm      — |p_ind| per unit r³ per unit E [C·m / m³ / (V/m)]
            polarised_within — τ_MW << t_res? (bool)
        """
        tau_MW = maxwell_wagner_relaxation(
            eps_r_p=polymer.eps_r,
            eps_r_m=self.eps_r_medium,
            sigma_p=polymer.sigma_Sm,
            sigma_m=self.sigma_medium,
        )
        f_crossover = 1.0 / (2.0 * np.pi * tau_MW) if tau_MW > 0 else np.inf

        # nDEP regime: DC with σ_polymer < σ_water, or AC above crossover
        is_ndep = (
            (self.frequency_Hz == 0 and polymer.sigma_Sm < self.sigma_medium)
            or (self.frequency_Hz > 0 and self.frequency_Hz > f_crossover)
        )

        Re_K = cm_factor(polymer.eps_r, self.eps_r_medium)
        eps_m = self.eps_r_medium * EPS_0
        dipole_norm = 4.0 * np.pi * eps_m * abs(Re_K)  # per m³ volume, per V/m

        return dict(
            tau_MW_s=tau_MW,
            f_crossover_Hz=f_crossover,
            is_ndep_regime=is_ndep,
            Re_K=Re_K,
            dipole_norm=dipole_norm,
        )

    def residence_time_s(self, Q_m3s: float) -> float:
        """Particle transit time through polarization zone [s]."""
        U = float(Q_m3s) / max(self.area_m2, 1e-12)
        return self.length_m / max(U, 1e-12)

    def is_fully_polarised(self, polymer: PolymerProps, Q_m3s: float) -> bool:
        """
        True if t_residence > 10 × τ_MW (particle fully polarised before cone entry).
        """
        info  = self.characterise(polymer)
        t_res = self.residence_time_s(Q_m3s)
        return t_res > 10.0 * info["tau_MW_s"]

    def state_dict(self, polymer: PolymerProps, Q_m3s: float) -> dict:
        """Flat dict for writing into simulation state."""
        info  = self.characterise(polymer)
        t_res = self.residence_time_s(Q_m3s)
        return dict(
            pol_tau_MW_us=info["tau_MW_s"] * 1e6,
            pol_f_crossover_kHz=info["f_crossover_Hz"] / 1e3,
            pol_is_ndep=float(info["is_ndep_regime"]),
            pol_Re_K=info["Re_K"],
            pol_t_res_ms=t_res * 1e3,
            pol_fully_polarised=float(self.is_fully_polarised(polymer, Q_m3s)),
        )
