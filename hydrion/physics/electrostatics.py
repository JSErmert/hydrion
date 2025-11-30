# hydrion/physics/electrostatics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import numpy as np


@dataclass
class ElectrostaticsParams:
    """
    Simple 1-node electrostatic model standing in for a set of charged meshes.

    V_node tracks a commanded voltage with first-order dynamics and leakage.
    E_field_mag is proportional to |V_node|.
    """

    V_max: float = 3000.0      # [V] max supply voltage
    tau_charge: float = 0.5    # [s] time constant to reach command
    leak_rate: float = 0.1     # [1/s] passive decay
    gap_m: float = 0.01        # [m] characteristic gap -> E = V / gap
    E_norm_ref: float = 3e5    # [V/m] normalize to get E_norm ~ O(1)
    eps: float = 1e-8


class ElectrostaticsModel:
    """
    ElectrostaticsModel v1

    Reads from env state / actions:
        node_voltage_cmd in [0, 1]
        Q_out_Lmin (optional, for future coupling)

    Writes:
        V_node      [V]
        E_field     [V/m]
        E_norm      dimensionless (roughly 0-1)
    """

    def __init__(self, cfg: Any | None = None) -> None:
        e_raw: Dict[str, float] = {}
        if cfg is not None and hasattr(cfg, "raw"):
            e_raw = getattr(cfg, "raw", {}).get("electrostatics", {}) or {}

        self.params = ElectrostaticsParams(
            V_max=float(e_raw.get("V_max", 3000.0)),
            tau_charge=float(e_raw.get("tau_charge", 0.5)),
            leak_rate=float(e_raw.get("leak_rate", 0.1)),
            gap_m=float(e_raw.get("gap_m", 0.01)),
            E_norm_ref=float(e_raw.get("E_norm_ref", 3e5)),
            eps=float(e_raw.get("eps", 1e-8)),
        )

        self.state: Dict[str, float] = {
            "V_node": 0.0,
            "E_field": 0.0,
            "E_norm": 0.0,
        }

    # ------------------------------------------------------------------
    def reset(self, state: Dict[str, float]) -> None:
        self.state.update(V_node=0.0, E_field=0.0, E_norm=0.0)
        state.update(self.state)

    def get_state(self) -> Dict[str, float]:
        return dict(self.state)

    def update(self, state: Dict[str, float], dt: float, node_cmd: float) -> None:
        p = self.params

        node_cmd = float(np.clip(node_cmd, 0.0, 1.0))

        V_node = float(self.state.get("V_node", 0.0))
        V_target = node_cmd * p.V_max

        # First-order approach to V_target with leakage
        dV = ((V_target - V_node) / max(p.tau_charge, p.eps) - p.leak_rate * V_node) * dt
        V_node = V_node + dV

        # Electric field and normalized strength
        E_field = V_node / max(p.gap_m, p.eps)
        E_norm = np.clip(abs(E_field) / max(p.E_norm_ref, p.eps), 0.0, 2.0)

        self.state.update(
            V_node=float(V_node),
            E_field=float(E_field),
            E_norm=float(E_norm),
        )

        state.update(self.state)
