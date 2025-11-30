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
    Q_max_Lmin: float = 20.0      # Max forward flow (clean, valve fully open)
    P_max_Pa: float = 80_000.0    # Pump max pressure (at zero flow)

    # Base hydraulic resistances (Pa / (m³/s))
    R_pipe: float = 2.0e7         # Piping & fittings resistance
    R_valve_min: float = 0.5e7    # Valve resistance when fully open
    R_valve_max: float = 8.0e7    # Valve resistance when almost closed

    # Mesh base & clog coefficients – each mesh adds clog-dependent resistance
    R_m1_base: float = 1.0e7
    R_m2_base: float = 2.0e7
    R_m3_base: float = 4.0e7
    k_m1_clog: float = 3.0e7      # Pa / (m³/s) per normalized clog
    k_m2_clog: float = 4.0e7
    k_m3_clog: float = 5.0e7

    # Backflush impact
    bf_flow_penalty: float = 0.8  # bf=1 diverts 80% of forward flow

    # Numerical / safety
    min_Q_Lmin: float = 0.0
    max_Q_Lmin: float = 50.0      # hard upper clamp to avoid explosions


class HydraulicsModel:
    """
    HydraulicsModel v1

    Models:
    - Pump curve (nonlinear, pressure vs. flow)
    - Valve-dependent resistance
    - Three mesh elements with clog-dependent resistance
    - Pipe resistance
    - Simple backflush diversion reducing forward flow

    State keys (written into the shared env state dict):
        Q_out_Lmin   Forward flow after meshes [L/min]
        P_in         Pressure at pump outlet [Pa]
        P_m1         Pressure after valve / before mesh 1 [Pa]
        P_m2         Pressure between mesh 1 and 2 [Pa]
        P_m3         Pressure between mesh 2 and 3 [Pa]
        P_out        Pressure at outlet (reference, ~0 Pa)

    Inputs expected on update():
        action[0] -> valve_open in [0,1]
        action[1] -> pump_cmd   in [0,1]
        action[2] -> bf_cmd     in [0,1]

    Clog inputs are read from the clogging model via its state:
        Mc1, Mc2, Mc3 (clog mass at each mesh, arbitrary units)
        Mc1_max, Mc2_max, Mc3_max (for normalization, optional)

    This is intentionally “v1 realistic”: simple but physically meaningful,
    easy to extend later with more detailed fluid models.
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        # Try to pull parameters from cfg.raw["hydraulics"], else fall back
        h_raw = getattr(cfg, "raw", {}).get("hydraulics", {}) if hasattr(cfg, "raw") else {}
        self.params = HydraulicsParams(
            Q_max_Lmin=h_raw.get("Q_max_Lmin", getattr(getattr(cfg, "hydraulics", cfg), "Q_max_Lmin", 20.0)),
            P_max_Pa=h_raw.get("P_max_Pa", getattr(getattr(cfg, "hydraulics", cfg), "P_max_Pa", 80_000.0)),
            R_pipe=h_raw.get("R_pipe", 2.0e7),
            R_valve_min=h_raw.get("R_valve_min", 0.5e7),
            R_valve_max=h_raw.get("R_valve_max", 8.0e7),
            R_m1_base=h_raw.get("R_m1_base", 1.0e7),
            R_m2_base=h_raw.get("R_m2_base", 2.0e7),
            R_m3_base=h_raw.get("R_m3_base", 4.0e7),
            k_m1_clog=h_raw.get("k_m1_clog", 3.0e7),
            k_m2_clog=h_raw.get("k_m2_clog", 4.0e7),
            k_m3_clog=h_raw.get("k_m3_clog", 5.0e7),
            bf_flow_penalty=h_raw.get("bf_flow_penalty", 0.8),
            min_Q_Lmin=h_raw.get("min_Q_Lmin", 0.0),
            max_Q_Lmin=h_raw.get("max_Q_Lmin", 50.0),
        )

        # Internal state cache (for debugging / plotting)
        self.state: Dict[str, float] = {}

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def reset(self):
        """Reset hydraulics to nominal startup state."""
        self.state = {
            "Q_out_Lmin": 0.0,
            "P_in": 0.0,
            "P_m1": 0.0,
            "P_m2": 0.0,
            "P_m3": 0.0,
            "P_out": 0.0,
        }

    def get_state(self) -> Dict[str, float]:
        """Return a shallow copy of the current hydraulic state."""
        return dict(self.state)

    def update(
        self,
        state: Dict[str, Any],
        dt: float,
        action: np.ndarray,
        clogging_model: Any | None = None,
    ):
        """
        Advance hydraulic state by one step and write results into `state`.

        Parameters
        ----------
        state : dict
            Shared environment state dict (will be updated in-place).
        dt : float
            Timestep (seconds).
        action : np.ndarray
            RL action vector: [valve_open, pump_cmd, bf_cmd, node_voltage].
        clogging_model : object or None
            Object with get_state() returning Mc1, Mc2, Mc3; if None, assumes
            no clogging.
        """
        p = self.params

        # --- 1. Decode actions ------------------------------------------------
        valve_open = float(np.clip(action[0], 0.0, 1.0))   # 0 = closed, 1 = fully open
        pump_cmd   = float(np.clip(action[1], 0.0, 1.0))   # 0 = off,    1 = max pressure
        bf_cmd     = float(np.clip(action[2], 0.0, 1.0))   # 0 = no bf,  1 = max diversion

        # --- 2. Clog-dependent mesh resistances ------------------------------
        Mc1 = Mc2 = Mc3 = 0.0
        Mc1_max = Mc2_max = Mc3_max = 1.0

        if clogging_model is not None:
            cstate = clogging_model.get_state()
            Mc1 = float(cstate.get("Mc1", 0.0))
            Mc2 = float(cstate.get("Mc2", 0.0))
            Mc3 = float(cstate.get("Mc3", 0.0))
            Mc1_max = float(cstate.get("Mc1_max", 1.0))
            Mc2_max = float(cstate.get("Mc2_max", 1.0))
            Mc3_max = float(cstate.get("Mc3_max", 1.0))

        # Normalize clog to [0,1] (saturated)
        n1 = np.clip(Mc1 / max(Mc1_max, 1e-6), 0.0, 1.0)
        n2 = np.clip(Mc2 / max(Mc2_max, 1e-6), 0.0, 1.0)
        n3 = np.clip(Mc3 / max(Mc3_max, 1e-6), 0.0, 1.0)

        R_m1 = p.R_m1_base + p.k_m1_clog * n1
        R_m2 = p.R_m2_base + p.k_m2_clog * n2
        R_m3 = p.R_m3_base + p.k_m3_clog * n3

        # --- 3. Valve resistance ---------------------------------------------
        # Fully open → low resistance; closed → high
        R_valve = p.R_valve_min + (1.0 - valve_open) * (p.R_valve_max - p.R_valve_min)

        # --- 4. Total forward resistance -------------------------------------
        R_forward = p.R_pipe + R_valve + R_m1 + R_m2 + R_m3
        R_forward = max(R_forward, 1e-6)  # safety

        # --- 5. Pump curve & flow solution -----------------------------------
        # Nonlinear pump: available pressure drops with flow.
        #
        # We approximate:
        #   P_pump_available(Q) = P_max * (1 - (Q / Q_max)^2)_+
        #
        # Here we invert the relationship analytically for approximate solution.
        # For v1 realism, we solve fixed point with one iteration.

        # Start with linear guess: Q_lin = (P_max * pump_cmd) / R_forward
        P_max_eff = p.P_max_Pa * pump_cmd
        Q_guess_Lmin = P_max_eff / R_forward * 1000.0 * 60.0  # (Pa / (Pa/(m^3/s))) → m³/s → L/min

        # Clamp to [0, Q_max]
        Q_guess_Lmin = float(np.clip(Q_guess_Lmin, p.min_Q_Lmin, p.Q_max_Lmin))

        # Refine with one fixed-point iteration using nonlinear curve
        for _ in range(1):
            q_ratio = np.clip(Q_guess_Lmin / max(p.Q_max_Lmin, 1e-6), 0.0, 1.0)
            P_avail = P_max_eff * (1.0 - q_ratio**2)
            Q_new_Lmin = P_avail / R_forward * 1000.0 * 60.0
            Q_guess_Lmin = float(np.clip(Q_new_Lmin, p.min_Q_Lmin, p.Q_max_Lmin))

        Q_forward_Lmin = Q_guess_Lmin

        # --- 6. Backflush diversion ------------------------------------------
        # Backflush diverts a fraction of forward flow away from meshes.
        diversion = p.bf_flow_penalty * bf_cmd
        diversion = np.clip(diversion, 0.0, 1.0)
        Q_out_Lmin = Q_forward_Lmin * (1.0 - diversion)

        # Hard safety clamp
        Q_out_Lmin = float(np.clip(Q_out_Lmin, p.min_Q_Lmin, p.max_Q_Lmin))

        # --- 7. Pressure distribution across elements ------------------------
        # Total pressure drop needed for Q_forward:
        Q_m3s = Q_forward_Lmin / 1000.0 / 60.0
        dP_total = R_forward * Q_m3s

        # Normalize resistances to allocate drops
        R_list = np.array([p.R_pipe, R_valve, R_m1, R_m2, R_m3], dtype=float)
        R_sum = float(np.sum(R_list))
        if R_sum <= 0.0:
            weights = np.zeros_like(R_list)
        else:
            weights = R_list / R_sum

        dP_list = dP_total * weights
        dP_pipe, dP_valve, dP_m1, dP_m2, dP_m3 = dP_list

        # Node pressures (assuming outlet reference at ~0 Pa)
        P_out = 0.0
        P_m3 = P_out + dP_m3
        P_m2 = P_m3 + dP_m2
        P_m1 = P_m2 + dP_m1
        P_after_valve = P_m1
        P_in = P_after_valve + dP_valve + dP_pipe  # pump outlet

        # --- 8. Update internal + shared state -------------------------------
        self.state.update(
            Q_out_Lmin=Q_out_Lmin,
            P_in=float(P_in),
            P_m1=float(P_m1),
            P_m2=float(P_m2),
            P_m3=float(P_m3),
            P_out=float(P_out),
        )

        # Write into shared env state
        state["Q_out_Lmin"] = Q_out_Lmin
        state["P_in"] = float(P_in)
        state["P_m1"] = float(P_m1)
        state["P_m2"] = float(P_m2)
        state["P_m3"] = float(P_m3)
        state["P_out"] = float(P_out)
