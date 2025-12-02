# hydrion/physics/particles.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import numpy as np


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


class ParticleModel:
    """
    ParticleModel v1

    Reads:
        mesh_loading_avg   from CloggingModel
        capture_eff        from CloggingModel (baseline)
        E_norm             from ElectrostaticsModel

    Writes:
        C_in               normalized upstream concentration
        C_out              downstream concentration
        particle_capture_eff
    """

    def __init__(self, cfg: Any | None = None) -> None:
        p_raw: Dict[str, float] = {}
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

    def reset(self, state: Dict[str, float]) -> None:
        state["C_in"] = self.params.C_in_base
        state["C_out"] = self.params.C_in_base
        state["particle_capture_eff"] = 0.0

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
        E_norm = 0.0
        if electrostatics_model is not None:
            E_norm = float(electrostatics_model.get_state().get("E_norm", 0.0))
        else:
            E_norm = float(state.get("E_norm", 0.0))

        # Boost capture efficiency with clogging + electrostatics
        capture_eff = capture_eff_base + p.alpha_clog * mesh_avg + p.alpha_E * E_norm
        capture_eff = float(
            np.clip(capture_eff, p.capture_floor, p.capture_ceiling)
        )

        C_out = C_in * (1.0 - capture_eff)

        state["C_in"] = C_in
        state["C_out"] = C_out
        state["particle_capture_eff"] = capture_eff
