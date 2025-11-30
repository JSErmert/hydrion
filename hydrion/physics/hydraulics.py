from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class HydraulicsParams:
    """
    Tunable hydraulic parameters.

    All pressures in Pa, flow in L/min (external API), internal computation
    uses m³/s where needed.
    """
    Q_max_Lmin: float = 20.0
    P_max_Pa: float = 80_000.0

    # Base hydraulic resistances (Pa / (m³/s))
    R_pipe: float = 2.0e7
    R_valve_min: float = 0.5e7
    R_valve_max: float = 8.0e7

    # Mesh base & clog coefficients
    R_m1_base: float = 1.0e7
    R_m2_base: float = 2.0e7
    R_m3_base: float = 4.0e7
    k_m1_clog: float = 3.0e7
    k_m2_clog: float = 4.0e7
    k_m3_clog: float = 5.0e7

    # Backflush diversion coefficient
    bf_flow_penalty: float = 0.8

    # Numerical limits
    min_Q_Lmin: float = 0.0
    max_Q_Lmin: float = 50.0


class HydraulicsModel:
    """
    HydraulicsModel v1 with corrected config loading.

    Fixes:
    - Prevents dict objects from being passed as parameters.
    - Ensures all hydraulics parameters are cast to float.
    """

    def __init__(self, cfg: Any):

        # Extract hydraulics section safely
        if hasattr(cfg, "raw") and isinstance(cfg.raw, dict):
            h_raw = cfg.raw.get("hydraulics", {}) or {}
        else:
            h_raw = {}

        # Safe float extractor
        def getf(key: str, default: float):
            val = h_raw.get(key, default)
            try:
                return float(val)
            except Exception:
                return default

        # Build parameter set safely
        self.params = HydraulicsParams(
            Q_max_Lmin=getf("Q_max_Lmin", 20.0),
            P_max_Pa=getf("P_max_Pa", 80_000.0),

            R_pipe=getf("R_pipe", 2.0e7),
            R_valve_min=getf("R_valve_min", 0.5e7),
            R_valve_max=getf("R_valve_max", 8.0e7),

            R_m1_base=getf("R_m1_base", 1.0e7),
            R_m2_base=getf("R_m2_base", 2.0e7),
            R_m3_base=getf("R_m3_base", 4.0e7),

            k_m1_clog=getf("k_m1_clog", 3.0e7),
            k_m2_clog=getf("k_m2_clog", 4.0e7),
            k_m3_clog=getf("k_m3_clog", 5.0e7),

            bf_flow_penalty=getf("bf_flow_penalty", 0.8),
            min_Q_Lmin=getf("min_Q_Lmin", 0.0),
            max_Q_Lmin=getf("max_Q_Lmin", 50.0),
        )

        self.state: Dict[str, float] = {}

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------
    def reset(self):
        self.state = {
            "Q_out_Lmin": 0.0,
            "P_in": 0.0,
            "P_m1": 0.0,
            "P_m2": 0.0,
            "P_m3": 0.0,
            "P_out": 0.0,
        }

    def get_state(self) -> Dict[str, float]:
        return dict(self.state)

    # ----------------------------------------------------------------------
    # MAIN HYDRAULIC UPDATE
    # ----------------------------------------------------------------------
    def update(
        self,
        state: Dict[str, Any],
        dt: float,
        action: np.ndarray,
        clogging_model: Any | None = None,
    ):

        p = self.params

        # ---------------------------------------------------------------
        # 1. Decode Actions
        # ---------------------------------------------------------------
        valve_open = float(np.clip(action[0], 0.0, 1.0))
        pump_cmd = float(np.clip(action[1], 0.0, 1.0))
        bf_cmd = float(np.clip(action[2], 0.0, 1.0))

        # ---------------------------------------------------------------
        # 2. Read clogging state
        # ---------------------------------------------------------------
        Mc1 = Mc2 = Mc3 = 0.0
        Mc1_max = Mc2_max = Mc3_max = 1.0

        if clogging_model is not None:
            cs = clogging_model.get_state()
            Mc1 = float(cs.get("Mc1", 0.0))
            Mc2 = float(cs.get("Mc2", 0.0))
            Mc3 = float(cs.get("Mc3", 0.0))
            Mc1_max = float(cs.get("Mc1_max", 1.0))
            Mc2_max = float(cs.get("Mc2_max", 1.0))
            Mc3_max = float(cs.get("Mc3_max", 1.0))

        # Normalize clog levels
        n1 = np.clip(Mc1 / max(Mc1_max, 1e-6), 0.0, 1.0)
        n2 = np.clip(Mc2 / max(Mc2_max, 1e-6), 0.0, 1.0)
        n3 = np.clip(Mc3 / max(Mc3_max, 1e-6), 0.0, 1.0)

        # Mesh resistances
        R_m1 = p.R_m1_base + p.k_m1_clog * n1
        R_m2 = p.R_m2_base + p.k_m2_clog * n2
        R_m3 = p.R_m3_base + p.k_m3_clog * n3

        # ---------------------------------------------------------------
        # 3. Valve resistance
        # ---------------------------------------------------------------
        R_valve = p.R_valve_min + (1.0 - valve_open) * (p.R_valve_max - p.R_valve_min)

        # ---------------------------------------------------------------
        # 4. Total forward resistance
        # ---------------------------------------------------------------
        R_forward = p.R_pipe + R_valve + R_m1 + R_m2 + R_m3
        R_forward = max(R_forward, 1e-6)

        # ---------------------------------------------------------------
        # 5. Pump curve + flow calculation
        # ---------------------------------------------------------------
        P_max_eff = p.P_max_Pa * pump_cmd

        # linear guess
        Q_guess = P_max_eff / R_forward * 1000 * 60
        Q_guess = float(np.clip(Q_guess, p.min_Q_Lmin, p.Q_max_Lmin))

        # non-linear correction
        q_ratio = np.clip(Q_guess / max(p.Q_max_Lmin, 1e-6), 0.0, 1.0)
        P_avail = P_max_eff * (1.0 - q_ratio**2)
        Q_new = P_avail / R_forward * 1000 * 60
        Q_guess = float(np.clip(Q_new, p.min_Q_Lmin, p.Q_max_Lmin))

        # Backflush reduction
        diversion = np.clip(p.bf_flow_penalty * bf_cmd, 0.0, 1.0)
        Q_out = Q_guess * (1.0 - diversion)
        Q_out = float(np.clip(Q_out, p.min_Q_Lmin, p.max_Q_Lmin))

        # ---------------------------------------------------------------
        # 6. Pressure distribution
        # ---------------------------------------------------------------
        Q_m3s = Q_guess / 1000 / 60
        dP_total = R_forward * Q_m3s

        R_list = np.array([p.R_pipe, R_valve, R_m1, R_m2, R_m3])
        weights = R_list / np.sum(R_list)

        dP_pipe, dP_valve, dP_m1, dP_m2, dP_m3 = dP_total * weights

        P_out = 0.0
        P_m3 = P_out + dP_m3
        P_m2 = P_m3 + dP_m2
        P_m1 = P_m2 + dP_m1
        P_in = P_m1 + dP_valve + dP_pipe

        # ---------------------------------------------------------------
        # 7. Update state and shared env dict
        # ---------------------------------------------------------------
        self.state.update(
            Q_out_Lmin=Q_out,
            P_in=float(P_in),
            P_m1=float(P_m1),
            P_m2=float(P_m2),
            P_m3=float(P_m3),
            P_out=float(P_out),
        )

        state.update(self.state)
