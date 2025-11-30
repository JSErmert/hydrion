from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class CloggingParams:
    """
    Tunable parameters for the tri-mesh clogging model.

    The intent is to keep this v1 model:
    - simple enough for fast RL rollouts
    - but shaped so that clogging is slow at first and accelerates
      as the meshes approach saturation.

    Units:
    - Mass-like quantities Mc* are in arbitrary units (AU).
    - Flow is assumed to be in L/min, consistent with HydraulicsModel.
    """

    # Maximum "mass" / loading per mesh (arbitrary units, but internally consistent)
    Mc1_max: float = 1.0
    Mc2_max: float = 1.0
    Mc3_max: float = 1.0

    # Baseline deposition coefficient [AU / (L/min * s)]
    dep_base: float = 1e-3

    # How strongly deposition increases with normalized clog level
    dep_exponent: float = 2.0

    # Shear-induced removal coefficient [1 / s]
    shear_coeff: float = 5e-3

    # Flow at which shear removal becomes significant [L/min]
    shear_Q_ref: float = 15.0

    # Small floor to avoid division by zero
    eps: float = 1e-8


class CloggingModel:
    """
    CloggingModel v1

    This class is intentionally self-contained and stateless w.r.t. RL code –
    it only talks to the environment via a shared `state` dict and a minimal
    public API:

        - reset(state: Dict[str, float]) -> None
        - update(state: Dict[str, float], dt: float) -> None
        - get_state() -> Dict[str, float]

    Expected fields written into the shared state:

        Mc1, Mc2, Mc3      : mesh clog "mass" (AU)
        Mc1_max, ...       : saturation levels (copied from params)
        n1, n2, n3         : normalized clog in [0, 1]
        mesh_loading_avg   : average of (n1, n2, n3) in [0, 1]
        capture_eff        : effective capture efficiency in [0, 1]

    HydraulicsModel reads the *instantaneous* clog state via either:

        - this model's `get_state()` (preferred), or
        - the shared env state dict (for logging / plotting).

    Design notes
    ------------

    - Deposition is driven by the current forward flow (Q_out_Lmin) and
      increases nonlinearly with existing clog (self-accelerating behavior).
    - Shear removal counteracts clogging at high flows, preventing the meshes
      from saturating too quickly during aggressive backflush.
    - v1 does *not* explicitly model particle size spectra; that is deferred
      to a future ParticleModel. Here, we operate at an aggregate level.
    """

    def __init__(self, cfg: Any | None = None) -> None:
        # Try to pull parameters from a HydrionConfig-like object, but also
        # work directly with a raw dict for simplicity.
        c_raw: Dict[str, float] = {}
        if cfg is not None:
            if hasattr(cfg, "raw"):
                # e.g. HydrionConfig(raw=...)
                c_raw = getattr(cfg, "raw", {}).get("clogging", {}) or {}
            elif isinstance(cfg, dict):
                c_raw = cfg.get("clogging", {}) or {}

        self.params = CloggingParams(
            Mc1_max=c_raw.get("Mc1_max", 1.0),
            Mc2_max=c_raw.get("Mc2_max", 1.0),
            Mc3_max=c_raw.get("Mc3_max", 1.0),
            dep_base=c_raw.get("dep_base", 1e-3),
            dep_exponent=c_raw.get("dep_exponent", 2.0),
            shear_coeff=c_raw.get("shear_coeff", 5e-3),
            shear_Q_ref=c_raw.get("shear_Q_ref", 15.0),
            eps=c_raw.get("eps", 1e-8),
        )

        # Internal state cache (primarily for debugging / logging)
        self._state: Dict[str, float] = {
            "Mc1": 0.0,
            "Mc2": 0.0,
            "Mc3": 0.0,
            "Mc1_max": self.params.Mc1_max,
            "Mc2_max": self.params.Mc2_max,
            "Mc3_max": self.params.Mc3_max,
            "n1": 0.0,
            "n2": 0.0,
            "n3": 0.0,
            "mesh_loading_avg": 0.0,
            "capture_eff": 0.0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self, state: Dict[str, float]) -> None:
        """
        Reset clogging state to a clean condition and write initial values
        into the shared env state dict.
        """
        self._state.update(
            Mc1=0.0,
            Mc2=0.0,
            Mc3=0.0,
            n1=0.0,
            n2=0.0,
            n3=0.0,
            mesh_loading_avg=0.0,
            capture_eff=0.0,
        )
        # Keep Mc*_max from params in case config changed
        self._state["Mc1_max"] = self.params.Mc1_max
        self._state["Mc2_max"] = self.params.Mc2_max
        self._state["Mc3_max"] = self.params.Mc3_max

        state.update(self._state)

    def update(self, state: Dict[str, float], dt: float) -> None:
        """
        Advance clogging dynamics by one time step.

        Parameters
        ----------
        state
            Shared environment state dict. This function reads:
                - "Q_out_Lmin" if present (forward flow),
                  otherwise falls back to "flow" or 0.0.
            and writes updated clogging fields (Mc*, n*, mesh_loading_avg,
            capture_eff).
        dt
            Simulation time step [s].
        """
        p = self.params

        # --- 1. Determine the driving flow ---------------------------------
        Q_Lmin = float(
            state.get(
                "Q_out_Lmin",
                state.get("flow", 0.0),
            )
        )
        Q_Lmin = max(Q_Lmin, 0.0)

        # --- 2. Current clog masses ----------------------------------------
        Mc1 = float(self._state.get("Mc1", 0.0))
        Mc2 = float(self._state.get("Mc2", 0.0))
        Mc3 = float(self._state.get("Mc3", 0.0))

        # --- 3. Deposition term --------------------------------------------
        # v1: assume a unit-normalized particle concentration; this can be
        # extended later with a ParticleModel that writes "C_fibers" into state.
        C_fibers = float(state.get("C_fibers", 1.0))

        # Normalized existing clog level drives superlinear acceleration.
        # To avoid zero exponent issues, we use (n + eps)^dep_exponent.
        n1 = Mc1 / max(p.Mc1_max, p.eps)
        n2 = Mc2 / max(p.Mc2_max, p.eps)
        n3 = Mc3 / max(p.Mc3_max, p.eps)

        n1 = np.clip(n1, 0.0, 1.0)
        n2 = np.clip(n2, 0.0, 1.0)
        n3 = np.clip(n3, 0.0, 1.0)

        dep_factor1 = (n1 + p.eps) ** p.dep_exponent
        dep_factor2 = (n2 + p.eps) ** p.dep_exponent
        dep_factor3 = (n3 + p.eps) ** p.dep_exponent

        dM_dep1 = p.dep_base * Q_Lmin * C_fibers * dep_factor1 * dt
        dM_dep2 = p.dep_base * Q_Lmin * C_fibers * dep_factor2 * dt
        dM_dep3 = p.dep_base * Q_Lmin * C_fibers * dep_factor3 * dt

        # --- 4. Shear removal term -----------------------------------------
        shear_scale = (Q_Lmin / max(p.shear_Q_ref, p.eps))
        shear_scale = max(shear_scale, 0.0)

        dM_shear1 = -p.shear_coeff * shear_scale * Mc1 * dt
        dM_shear2 = -p.shear_coeff * shear_scale * Mc2 * dt
        dM_shear3 = -p.shear_coeff * shear_scale * Mc3 * dt

        # --- 5. Integrate and saturate -------------------------------------
        Mc1 = np.clip(Mc1 + dM_dep1 + dM_shear1, 0.0, p.Mc1_max)
        Mc2 = np.clip(Mc2 + dM_dep2 + dM_shear2, 0.0, p.Mc2_max)
        Mc3 = np.clip(Mc3 + dM_dep3 + dM_shear3, 0.0, p.Mc3_max)

        # Re-compute normalized clog
        n1 = float(np.clip(Mc1 / max(p.Mc1_max, p.eps), 0.0, 1.0))
        n2 = float(np.clip(Mc2 / max(p.Mc2_max, p.eps), 0.0, 1.0))
        n3 = float(np.clip(Mc3 / max(p.Mc3_max, p.eps), 0.0, 1.0))

        mesh_loading_avg = (n1 + n2 + n3) / 3.0

        # --- 6. Effective capture efficiency -------------------------------
        # v1: assume baseline 80% capture, rising modestly with clog level
        # until saturation, where flow begins to collapse anyway.
        capture_eff = 0.80 + 0.15 * mesh_loading_avg
        capture_eff = float(np.clip(capture_eff, 0.0, 1.0))

        self._state.update(
            Mc1=float(Mc1),
            Mc2=float(Mc2),
            Mc3=float(Mc3),
            Mc1_max=p.Mc1_max,
            Mc2_max=p.Mc2_max,
            Mc3_max=p.Mc3_max,
            n1=n1,
            n2=n2,
            n3=n3,
            mesh_loading_avg=float(mesh_loading_avg),
            capture_eff=capture_eff,
        )

        # Push into shared state dict for visibility to other subsystems
        state.update(self._state)

    def get_state(self) -> Dict[str, float]:
        """Return a shallow copy of the current clogging state."""
        return dict(self._state)
