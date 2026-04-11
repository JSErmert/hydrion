# hydrion/physics/particles.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

try:
    from scipy.stats import lognorm
except ImportError:
    lognorm = None


@dataclass
class ParticleParams:
    """
    Aggregate particle transport model.

    Operates in dimensionless units:
        C_in, C_out in [0, 1] (relative concentration)
    """

    C_in_base: float = 0.7          # baseline upstream particle concentration
    alpha_clog: float = 0.3         # how much clogging boosts capture
    alpha_E: float = 0.4            # how much electrostatics boosts capture
    capture_floor: float = 0.3      # min capture efficiency
    capture_ceiling: float = 0.99   # max capture efficiency
    eps: float = 1e-8


def _parse_psd_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Parse PSD config with backward-compatible defaults."""
    psd = raw.get("psd") or {}
    return {
        "enabled": bool(psd.get("enabled", False)),
        "mode": str(psd.get("mode", "hybrid")),
        "parametric": psd.get("parametric") or {},
        "bins": psd.get("bins") or [],
        "bin_edges_um": psd.get("bin_edges_um") or [],
    }


def _parse_shape_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Parse shape config with backward-compatible defaults."""
    shape = raw.get("shape") or {}
    return {
        "enabled": bool(shape.get("enabled", False)),
        "fiber_fraction": float(shape.get("fiber_fraction", 1.0)),
    }


def _compute_bin_weights(
    mode: str,
    parametric: Dict[str, Any],
    bins: List[Dict[str, Any]],
    bin_edges_um: List[float],
) -> List[float]:
    """
    Compute normalized bin weights. Returns list of weights summing to 1.0.
    """
    # bins mode: explicit w_in per bin
    if (mode == "bins" or (mode == "hybrid" and bins and all("w_in" in b for b in bins))) and bins:
        weights = [float(b.get("w_in", 0.0)) for b in bins]
        total = sum(weights)
        if total <= 0:
            return [1.0]  # fallback
        return [w / total for w in weights]

    # parametric / hybrid: compute from distribution over bin_edges
    if mode in ("parametric", "hybrid") and parametric:
        dist = str(parametric.get("distribution", "lognormal")).lower()
        mean_um = float(parametric.get("mean_um", 5.0))
        std_um = float(parametric.get("std_um", 2.0))

        edges = bin_edges_um if bin_edges_um else [0.1, 1.0, 10.0, 100.0]
        if len(edges) < 2:
            return [1.0]

        if dist == "lognormal" and lognorm is not None:
            # lognorm(s, scale=exp(mean_ln), loc=0): s=sigma_ln, scale=exp(mu_ln)
            sigma_ln = np.sqrt(np.log(1.0 + (std_um / max(mean_um, 1e-9)) ** 2))
            mu_ln = np.log(mean_um) - 0.5 * sigma_ln ** 2
            cdf_vals = lognorm.cdf(edges, s=sigma_ln, scale=np.exp(mu_ln))
            weights = np.diff(cdf_vals)
            weights = np.clip(weights, 0.0, 1.0)
            total = float(np.sum(weights))
            if total <= 0:
                weights = np.ones(len(weights)) / len(weights)
            else:
                weights = weights / total
            return [float(w) for w in weights]

        # Fallback: uniform over bins
        n = len(edges) - 1
        return [1.0 / n] * n

    return [1.0]


class ParticleModel:
    """
    ParticleModel v2

    Reads:
        mesh_loading_avg   from CloggingModel
        capture_eff        from CloggingModel (baseline)
        E_norm             from ElectrostaticsModel

    Writes:
        C_in               normalized upstream concentration
        C_out              downstream concentration
        particle_capture_eff
        C_fibers           (when PSD/shape enabled) fiber fraction of C_in for clogging
        fiber_fraction     (when shape enabled) config value for logging
        C_bins / C_L,C_M,C_S (when PSD enabled) per-bin concentrations for logging only
    """

    def __init__(self, cfg: Any | None = None) -> None:
        p_raw: Dict[str, Any] = {}
        if cfg is not None and hasattr(cfg, "raw"):
            p_raw = getattr(cfg, "raw", {}).get("particles", {}) or {}

        self.params = ParticleParams(
            C_in_base=float(p_raw.get("C_in_base", 0.7)),
            alpha_clog=float(p_raw.get("alpha_clog", 0.3)),
            alpha_E=float(p_raw.get("alpha_E", 0.4)),
            capture_floor=float(p_raw.get("capture_floor", 0.3)),
            capture_ceiling=float(p_raw.get("capture_ceiling", 0.99)),
            eps=float(p_raw.get("eps", 1e-8)),
        )

        self._psd_cfg = _parse_psd_config(p_raw)
        self._shape_cfg = _parse_shape_config(p_raw)
        self._bin_weights: List[float] = []
        if self._psd_cfg["enabled"]:
            self._bin_weights = _compute_bin_weights(
                mode=self._psd_cfg["mode"],
                parametric=self._psd_cfg["parametric"],
                bins=self._psd_cfg["bins"],
                bin_edges_um=self._psd_cfg["bin_edges_um"],
            )

    def reset(self, state: Dict[str, float]) -> None:
        state["C_in"] = self.params.C_in_base
        state["C_out"] = self.params.C_in_base
        state["particle_capture_eff"] = 0.0
        state["C_fibers"] = 1.0  # clogging default; overwritten in update when PSD enabled

        if self._psd_cfg["enabled"] and self._bin_weights:
            for i in range(len(self._bin_weights)):
                state[f"C_in_bin_{i}"] = 0.0
                state[f"C_out_bin_{i}"] = 0.0
            if len(self._bin_weights) == 3:
                state["C_L"], state["C_M"], state["C_S"] = 0.0, 0.0, 0.0

        if self._shape_cfg["enabled"]:
            state["fiber_fraction"] = self._shape_cfg["fiber_fraction"]

    def update(
        self,
        state: Dict[str, float],
        dt: float,
        clogging_model: Any | None = None,
        electrostatics_model: Any | None = None,
    ) -> None:
        p = self.params

        C_in = float(state.get("C_in", p.C_in_base))

        mesh_avg = float(state.get("mesh_loading_avg", 0.0))
        capture_eff_base = float(state.get("capture_eff", 0.8))
        E_capture_gain = 0.0
        if electrostatics_model is not None:
            E_capture_gain = float(electrostatics_model.get_state().get("E_capture_gain", 0.0))
        else:
            E_capture_gain = float(state.get("E_capture_gain", 0.0))

        # Boost capture efficiency with clogging + electrostatics
        # E_capture_gain in [0, 1] — normalised signal from ElectrostaticsModel v2
        capture_eff = capture_eff_base + p.alpha_clog * mesh_avg + p.alpha_E * E_capture_gain
        capture_eff = float(
            np.clip(capture_eff, p.capture_floor, p.capture_ceiling)
        )

        C_out = C_in * (1.0 - capture_eff)

        # C_fibers for clogging: fiber fraction of C_in (used by CloggingModel)
        if self._psd_cfg["enabled"] and self._shape_cfg["enabled"]:
            fiber_frac = self._shape_cfg["fiber_fraction"]
            C_fibers = fiber_frac * C_in
        else:
            C_fibers = 1.0

        state["C_in"] = C_in
        state["C_out"] = C_out
        state["particle_capture_eff"] = capture_eff
        state["C_fibers"] = float(np.clip(C_fibers, 0.0, 1.0))

        # Per-bin concentrations for logging/validation only (do not affect physics)
        if self._psd_cfg["enabled"] and self._bin_weights:
            for i, w in enumerate(self._bin_weights):
                C_in_i = w * C_in
                C_out_i = C_in_i * (1.0 - capture_eff)
                state[f"C_in_bin_{i}"] = float(C_in_i)
                state[f"C_out_bin_{i}"] = float(C_out_i)
            if len(self._bin_weights) == 3:
                # L=large (highest bin), M=medium, S=small (lowest bin)
                state["C_L"] = state["C_out_bin_2"]
                state["C_M"] = state["C_out_bin_1"]
                state["C_S"] = state["C_out_bin_0"]
