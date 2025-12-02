# hydrion/physics/particles.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np


# We model 5 particle size bins that approximate real microfiber / MP spectra:
#
#   0: ultra      < 5 µm        (ultra-fine microplastics)
#   1: fine       5–20 µm       (fine fibers / fragments)
#   2: small      20–50 µm      (short fibers)
#   3: medium     50–100 µm     (common fibers)
#   4: large      100–500 µm    (large fibers / chunks)
#
BIN_LABELS = ("ultra", "fine", "small", "medium", "large")
BIN_COUNT = len(BIN_LABELS)


@dataclass
class ParticleParams:
    """
    Multi-bin particle transport model with per-bin charge.

    Operates in dimensionless units:
        C_in, C_out in [0, 1] (relative concentration)

    We still expose aggregate C_in / C_out / particle_capture_eff to the rest of
    the environment for backward compatibility, but internally we maintain a
    5-bin size distribution and bin-specific capture + charge behavior.
    """

    # Baseline upstream total concentration (dimensionless)
    C_in_base: float = 0.7

    # How much clogging boosts capture (shared scalar)
    alpha_clog: float = 0.3

    # Base ES coupling strength (shared scalar, scaled per bin + charge)
    alpha_E: float = 0.4

    # Global capture clamps
    capture_floor: float = 0.05
    capture_ceiling: float = 0.999

    eps: float = 1e-8

    # Fraction of total C_in in each bin (must sum to 1.0 after normalization)
    bin_fractions: Sequence[float] = (0.10, 0.20, 0.30, 0.25, 0.15)

    # Mesh sensitivity per bin: larger fibers are more mesh-capturable
    # (multiplier applied to baseline mesh capture)
    mesh_sensitivity: Sequence[float] = (0.3, 0.6, 0.9, 1.1, 1.3)

    # Extra clog boost per bin: smaller bins benefit more when meshes clog
    clog_boost: Sequence[float] = (0.30, 0.25, 0.20, 0.10, 0.05)

    # ES sensitivity per bin: ultra/fine are most influenced by electrostatics
    es_sensitivity: Sequence[float] = (1.5, 1.2, 0.8, 0.3, 0.0)

    # ---- Charge model parameters ------------------------------------------

    # Base rate at which polarization increases charge (per second-equivalent)
    charge_gain_base: float = 1.0

    # Exponential decay rate of charge when not strongly polarized [1/s]
    charge_decay_rate: float = 0.3

    # Max normalized charge (0–1)
    charge_max: float = 1.0

    # Per-bin charge sensitivity to polarization (ultra/fine charge more easily)
    charge_sensitivity: Sequence[float] = (1.4, 1.2, 0.8, 0.4, 0.2)

    # How strongly charge amplifies ES influence on capture
    es_charge_coupling: float = 1.0


class ParticleModel:
    """
    ParticleModel v2.1 — 5-bin size-resolved aggregate model with charge.

    Reads:
        mesh_loading_avg   from CloggingModel (0–1)
        capture_eff        from CloggingModel (baseline mesh stack capture)
        E_norm             from ElectrostaticsModel (0–1 normalized field)
        polarization_level (optional; if missing, we fall back to E_norm)
        C_in               (optional, if already in shared state)

    Writes (aggregate, for backward compatibility):
        C_in               normalized upstream *total* concentration
        C_out              downstream *total* concentration
        particle_capture_eff  = 1 - C_out / max(C_in, eps)

    Additionally (for sensors / visualization / analysis), writes per-bin:
        C_in_<label>
        C_out_<label>
        capture_eff_<label>
        charge_<label>

    And aggregate charge statistics:
        charge_mean        mean charge over bins, weighted by C_in
    """

    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        p_raw = cfg.get("particles", {}) if isinstance(cfg, dict) else {}

        # Basic scalar params
        C_in_base = float(p_raw.get("C_in_base", 0.7))
        alpha_clog = float(p_raw.get("alpha_clog", 0.3))
        alpha_E = float(p_raw.get("alpha_E", 0.4))
        capture_floor = float(p_raw.get("capture_floor", 0.05))
        capture_ceiling = float(p_raw.get("capture_ceiling", 0.999))
        eps = float(p_raw.get("eps", 1e-8))

        # Distribution and sensitivities
        bin_fractions = p_raw.get("bin_fractions", (0.10, 0.20, 0.30, 0.25, 0.15))
        mesh_sensitivity = p_raw.get(
            "mesh_sensitivity", (0.3, 0.6, 0.9, 1.1, 1.3)
        )
        clog_boost = p_raw.get("clog_boost", (0.30, 0.25, 0.20, 0.10, 0.05))
        es_sensitivity = p_raw.get("es_sensitivity", (1.5, 1.2, 0.8, 0.3, 0.0))

        # Charge model parameters (allow override from config)
        charge_gain_base = float(p_raw.get("charge_gain_base", 1.0))
        charge_decay_rate = float(p_raw.get("charge_decay_rate", 0.3))
        charge_max = float(p_raw.get("charge_max", 1.0))
        charge_sensitivity = p_raw.get(
            "charge_sensitivity", (1.4, 1.2, 0.8, 0.4, 0.2)
        )
        es_charge_coupling = float(p_raw.get("es_charge_coupling", 1.0))

        self.params = ParticleParams(
            C_in_base=C_in_base,
            alpha_clog=alpha_clog,
            alpha_E=alpha_E,
            capture_floor=capture_floor,
            capture_ceiling=capture_ceiling,
            eps=eps,
            bin_fractions=tuple(bin_fractions),
            mesh_sensitivity=tuple(mesh_sensitivity),
            clog_boost=tuple(clog_boost),
            es_sensitivity=tuple(es_sensitivity),
            charge_gain_base=charge_gain_base,
            charge_decay_rate=charge_decay_rate,
            charge_max=charge_max,
            charge_sensitivity=tuple(charge_sensitivity),
            es_charge_coupling=es_charge_coupling,
        )

        # Precompute normalized arrays for faster use in update()
        self._bin_frac = self._normalize_array(self.params.bin_fractions)
        self._mesh_sens = np.asarray(self.params.mesh_sensitivity, dtype=float)
        self._clog_boost = np.asarray(self.params.clog_boost, dtype=float)
        self._es_sens = np.asarray(self.params.es_sensitivity, dtype=float)
        self._charge_sens = np.asarray(self.params.charge_sensitivity, dtype=float)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_array(seq: Sequence[float]) -> np.ndarray:
        arr = np.asarray(seq, dtype=float)
        total = float(arr.sum())
        if total <= 0.0:
            # Fallback to uniform if misconfigured
            arr[:] = 1.0 / len(arr)
            return arr
        return arr / total

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #

    def reset(self, state: Dict[str, float]) -> None:
        """
        Initialize particle-related fields in the shared state dict.
        """
        p = self.params

        C_in_total = p.C_in_base
        C_out_total = p.C_in_base

        state["C_in"] = C_in_total
        state["C_out"] = C_out_total
        state["particle_capture_eff"] = 0.0

        # Initialize per-bin values for completeness
        C_in_bins = C_in_total * self._bin_frac
        C_out_bins = C_out_total * self._bin_frac
        capture_bins = np.zeros(BIN_COUNT, dtype=float)
        charge_bins = np.zeros(BIN_COUNT, dtype=float)

        for i, label in enumerate(BIN_LABELS):
            state[f"C_in_{label}"] = float(C_in_bins[i])
            state[f"C_out_{label}"] = float(C_out_bins[i])
            state[f"capture_eff_{label}"] = float(capture_bins[i])
            state[f"charge_{label}"] = float(charge_bins[i])

        state["charge_mean"] = 0.0

    def update(
        self,
        state: Dict[str, float],
        dt: float,
        clogging_model: Any | None = None,
        electrostatics_model: Any | None = None,
    ) -> None:
        """
        Advance the 5-bin particle + charge model by one time step.

        dt is currently small (~0.1) and used for charge dynamics. Capture
        itself is treated quasi-steadily (per-step).
        """
        p = self.params
        eps = p.eps

        # --- 1. Total upstream concentration ----------------------------- #
        C_in_total = float(state.get("C_in", p.C_in_base))
        C_in_total = max(C_in_total, 0.0)

        if C_in_total <= eps:
            # Degenerate case: nothing to transport.
            state["C_in"] = 0.0
            state["C_out"] = 0.0
            state["particle_capture_eff"] = 0.0
            for label in BIN_LABELS:
                state[f"C_in_{label}"] = 0.0
                state[f"C_out_{label}"] = 0.0
                state[f"capture_eff_{label}"] = 0.0
                state[f"charge_{label}"] = 0.0
            state["charge_mean"] = 0.0
            return

        # Size distribution (fixed fractions for now; can be made dynamic later)
        C_in_bins = C_in_total * self._bin_frac  # shape (5,)

        # --- 2. Read clogging / electrostatics context ------------------- #
        mesh_avg = float(state.get("mesh_loading_avg", 0.0))
        capture_eff_stack = float(state.get("capture_eff", 0.8))
        if clogging_model is not None:
            c_state = clogging_model.get_state()
            mesh_avg = float(c_state.get("mesh_loading_avg", mesh_avg))
            capture_eff_stack = float(c_state.get("capture_eff", capture_eff_stack))

        mesh_avg = float(np.clip(mesh_avg, 0.0, 1.0))
        capture_eff_stack = float(np.clip(capture_eff_stack, 0.0, 1.0))

        E_norm = 0.0
        if electrostatics_model is not None:
            E_norm = float(electrostatics_model.get_state().get("E_norm", 0.0))
        else:
            E_norm = float(state.get("E_norm", 0.0))
        E_norm = float(np.clip(E_norm, 0.0, 1.0))

        # Polarization level: if a dedicated model writes this, use it;
        # otherwise, approximate with E_norm for now.
        pol_level = float(state.get("polarization_level", E_norm))
        pol_level = float(np.clip(pol_level, 0.0, 1.0))

        # --- 3. Read/advance per-bin charge ------------------------------ #
        #
        # Simple first-order dynamics:
        #   dQ/dt = gain(pol_level) - decay * Q
        #
        # where gain is stronger for smaller bins via charge_sensitivity.
        #
        charge_bins = np.zeros(BIN_COUNT, dtype=float)
        for i, label in enumerate(BIN_LABELS):
            Q_prev = float(state.get(f"charge_{label}", 0.0))
            charge_bins[i] = Q_prev

        # Gain term (more polarization → faster charging; per-bin sensitivity)
        gain = (
            p.charge_gain_base
            * pol_level
            * self._charge_sens
        )  # shape (5,)

        # Decay term (relaxes toward zero when pol_level is low)
        decay = p.charge_decay_rate

        # Euler step
        charge_bins = charge_bins + dt * (gain - decay * charge_bins)

        # Clamp to [0, charge_max]
        charge_bins = np.clip(charge_bins, 0.0, p.charge_max)

        # --- 4. Per-bin capture calculation ------------------------------ #
        #
        # We interpret "capture_eff_stack" from CloggingModel as the effective
        # capture for a mid-sized bin, then scale it per bin based on:
        #   - mesh_sensitivity (large fibers are easier to catch)
        #   - clog_boost * mesh_avg (clogging helps more for small bins)
        #   - es_sensitivity * alpha_E * E_norm
        #   - charge_bins * es_charge_coupling  (charged particles feel ES more)
        #
        mesh_base = capture_eff_stack * self._mesh_sens
        clog_term = mesh_avg * self._clog_boost

        # Effective ES field per bin, amplified by charge
        E_effective = E_norm * (1.0 + p.es_charge_coupling * charge_bins)
        es_term = p.alpha_E * E_effective * self._es_sens

        capture_bins = mesh_base + p.alpha_clog * clog_term + es_term
        capture_bins = np.clip(capture_bins, p.capture_floor, p.capture_ceiling)

        # --- 5. Apply capture to each bin -------------------------------- #
        C_out_bins = C_in_bins * (1.0 - capture_bins)

        # Aggregate back to totals for backward compatibility
        C_out_total = float(C_out_bins.sum())
        capture_eff_total = 1.0 - C_out_total / max(C_in_total, eps)
        capture_eff_total = float(
            np.clip(capture_eff_total, p.capture_floor, p.capture_ceiling)
        )

        # Aggregate charge statistic (C_in-weighted mean)
        charge_mean = float(
            np.sum(charge_bins * C_in_bins) / max(C_in_bins.sum(), eps)
        )

        # --- 6. Write back into shared state ----------------------------- #
        state["C_in"] = float(C_in_total)
        state["C_out"] = float(C_out_total)
        state["particle_capture_eff"] = float(capture_eff_total)
        state["charge_mean"] = float(charge_mean)

        for i, label in enumerate(BIN_LABELS):
            state[f"C_in_{label}"] = float(C_in_bins[i])
            state[f"C_out_{label}"] = float(C_out_bins[i])
            state[f"capture_eff_{label}"] = float(capture_bins[i])
            state[f"charge_{label}"] = float(charge_bins[i])
