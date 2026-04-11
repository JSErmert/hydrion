"""
ConicalCascadeEnv — M5 exploration environment.

Parallel alternative to HydrionEnv using research-grounded M5 physics:
    - Rajagopalan-Tien (1976) liquid filtration (replaces power-law capture)
    - nDEP force balance with literature CM factors (replaces empirical E_gain)
    - Density split from RT gravity term (replaces hard-coded buoyant_fraction)
    - Physics-derived flow penalty from v_critical (replaces exp(-0.04 × ΔQ))
    - Polarization zone diagnostics at device inlet

Observation space and action space mirror HydrionEnv (obs12_v2 compatible)
so both environments can be run on identical inputs for direct comparison.

Device geometry (conical cascade):
    Three conical stages in series with graduated pore sizes.
    Particles converge to center axis via nDEP, captured at apex trap.
    Stage order: S1 (coarse) → S2 (medium) → S3 (fine).

Usage:
    baseline = HydrionEnv(config_path="configs/default.yaml")
    explore  = ConicalCascadeEnv(config_path="configs/default.yaml")

    obs_b, _ = baseline.reset(seed=42)
    obs_e, _ = explore.reset(seed=42)
    # Run with identical actions, compare metrics
"""
from __future__ import annotations

import numpy as np
import yaml
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Any

from ..physics.m5 import (
    PP, PE, PET,
    MeshSpec, MESH_S1, MESH_S2, MESH_S3_MEMBRANE,
    DEPConfig, cm_factor,
    ConicalStageSpec, cascade_capture,
    PolarizationZone,
    MU_WATER, RHO_WATER, EPS_R_WATER,
)
from ..physics.m5.conical_stage import stage_capture
from ..physics.hydraulics import HydraulicsModel
from ..physics.clogging   import CloggingModel


# ---------------------------------------------------------------------------
# Default device specification — [DESIGN_DEFAULT] where not yet measured
# ---------------------------------------------------------------------------

def _default_stages() -> list[ConicalStageSpec]:
    """
    Three-stage conical cascade with graduated pore sizes.
    All geometry values are [DESIGN_DEFAULT] — replace with physical device specs.
    """
    # Stage voltage / field config — [DESIGN_DEFAULT] 500V, 5mm gap, 0.5mm tip
    dep_s1 = DEPConfig(voltage_V=500.0, electrode_gap_m=5e-3, tip_radius_m=0.5e-3)
    dep_s2 = DEPConfig(voltage_V=500.0, electrode_gap_m=3e-3, tip_radius_m=0.2e-3)
    dep_s3 = DEPConfig(voltage_V=500.0, electrode_gap_m=1e-3, tip_radius_m=0.05e-3)

    return [
        ConicalStageSpec(
            label="S1_coarse",
            mesh=MESH_S1,
            dep=dep_s1,
            D_in_m=0.080, D_tip_m=0.020, L_cone_m=0.120,  # [DESIGN_DEFAULT]
        ),
        ConicalStageSpec(
            label="S2_medium",
            mesh=MESH_S2,
            dep=dep_s2,
            D_in_m=0.040, D_tip_m=0.010, L_cone_m=0.080,  # [DESIGN_DEFAULT]
        ),
        ConicalStageSpec(
            label="S3_fine",
            mesh=MESH_S3_MEMBRANE,
            dep=dep_s3,
            D_in_m=0.020, D_tip_m=0.004, L_cone_m=0.040,  # [DESIGN_DEFAULT]
        ),
    ]


def _default_pol_zone() -> PolarizationZone:
    return PolarizationZone(
        length_m=0.030,
        E_field_Vm=1e4,
        frequency_Hz=100e3,   # 100 kHz — above Maxwell-Wagner crossover
        area_m2=np.pi * (0.040)**2,  # [DESIGN_DEFAULT] matches S1 inlet diameter
    )


# ---------------------------------------------------------------------------
# Polymer mixture (mirrors M4 density classification)
# ---------------------------------------------------------------------------

POLYMER_MIX = {
    "PP":  {"props": PP,  "fraction": 0.08, "Re_K": cm_factor(PP.eps_r)},   # buoyant
    "PE":  {"props": PE,  "fraction": 0.07, "Re_K": cm_factor(PE.eps_r)},   # buoyant
    "PET": {"props": PET, "fraction": 0.70, "Re_K": cm_factor(PET.eps_r)},  # sinking
    # remaining 0.15: neutral/weathered — modelled as average of PP and PET
}

# ---------------------------------------------------------------------------
# Accumulation model constants — [DESIGN_DEFAULT], replace with physical specs
# ---------------------------------------------------------------------------
_CHANNEL_CAPACITY_M3 = 4e-4   # ~0.4 L per collection channel
_STORAGE_CAPACITY_M3 = 1.2e-2  # ~12 L detachable storage chamber
_FLUSH_DRAIN_RATE    = 0.20    # fraction of channel fill drained per bf step


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ConicalCascadeEnv(gym.Env):
    """
    M5 conical cascade exploration environment.

    Observation: 12-dim vector (obs12_v2 compatible with HydrionEnv).
    Action:      4-dim continuous [valve, pump, backflush, voltage_norm].

    Index  Key              Units
    -----  ---------------  --------
    0      q_in             L/min
    1      delta_p          kPa
    2      fouling_mean     [0,1]
    3      eta_cascade      [0,1]   ← replaces particle_capture_eff
    4      C_in             [0,1]
    5      C_out            [0,1]
    6      E_field_norm     [0,1]
    7      v_crit_norm      [0,1]   ← NEW: DEP critical velocity (normalised)
    8      step_norm        [0,1]
    9      bf_active        {0,1}
    10     eta_PP           [0,1]   ← NEW: PP-specific capture (buoyant species)
    11     eta_PET          [0,1]   ← NEW: PET-specific capture (dense species)
    """

    metadata = {"render_modes": []}

    # Observation normalisation bounds (same as HydrionEnv where shared)
    OBS_Q_MAX    = 25.0   # L/min
    OBS_DP_MAX   = 150.0  # kPa
    OBS_VCRIT_MAX = 0.10  # m/s

    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        stages: list[ConicalStageSpec] | None = None,
        pol_zone: PolarizationZone | None = None,
        d_p_um: float = 10.0,
        seed: int | None = None,
        render_mode=None,
    ):
        super().__init__()
        self.render_mode = render_mode

        # Load config for hydraulics / clogging (reuse existing models)
        try:
            with open(config_path) as f:
                raw_cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raw_cfg = {}

        from ..config import HydrionConfig
        self.cfg = HydrionConfig(raw_cfg)

        self.stages    = stages   or _default_stages()
        self.pol_zone  = pol_zone or _default_pol_zone()
        self.d_p_m     = d_p_um * 1e-6

        # Reuse M4 hydraulics and clogging — physics already validated
        self.hydraulics = HydraulicsModel(self.cfg)
        self.clogging   = CloggingModel(self.cfg)

        # Spaces — obs12_v2 compatible
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )

        self._state: dict[str, float] = {}
        self._step = 0
        self._max_steps = int(raw_cfg.get("max_steps", 300))
        self._dt = float(raw_cfg.get("dt", 0.1))

        # Accumulation state — persists across steps, resets on env.reset()
        self._storage_fill: float       = 0.0
        self._channel_fill: list[float] = [0.0, 0.0, 0.0]

        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step = 0
        self._state = {}
        self._storage_fill = 0.0
        self._channel_fill = [0.0, 0.0, 0.0]

        self.hydraulics.reset()
        self._state.update(self.hydraulics.state)
        self.clogging.reset(self._state)

        self._state["C_in"]  = 0.7
        self._state["C_out"] = 0.7
        self._state["eta_cascade"]     = 0.0
        self._state["eta_PP"]          = 0.0
        self._state["eta_PET"]         = 0.0
        self._state["v_crit_s3"]       = 0.0
        self._state["bf_active"]       = 0.0
        self._state["voltage_norm"]    = 0.8
        self._state["storage_fill"]    = 0.0
        self._state["channel_fill_s1"] = 0.0
        self._state["channel_fill_s2"] = 0.0
        self._state["channel_fill_s3"] = 0.0

        return self._obs(), {}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, 0.0, 1.0).astype(float)
        valve_cmd, pump_cmd, bf_cmd, volt_cmd = action

        # Update hydraulics and clogging (existing M4 models)
        self._state["valve_cmd"]   = float(valve_cmd)
        self._state["pump_cmd"]    = float(pump_cmd)
        self._state["bf_cmd"]      = float(bf_cmd)
        self._state["voltage_norm"] = float(volt_cmd)

        self.hydraulics.update(self._state, dt=self._dt, action=action, clogging_model=self.clogging)
        self._state.update(self.hydraulics.state)
        self.clogging.update(self._state, self._dt)

        # M5 capture physics
        Q_lmin  = float(self._state.get("q_processed_lmin", 10.0))
        Q_m3s   = Q_lmin / 60000.0  # L/min → m³/s

        ff = [
            float(self._state.get("fouling_frac_s1", 0.0)),
            float(self._state.get("fouling_frac_s2", 0.0)),
            float(self._state.get("fouling_frac_s3", 0.0)),
        ]

        # Scale DEP voltage by action
        stages = self._voltage_scaled_stages(volt_cmd)

        # Per-polymer cascade capture
        results: dict[str, Any] = {}
        for pname, pmix in POLYMER_MIX.items():
            r = cascade_capture(
                stages=stages,
                polymer=pmix["props"],
                Re_K=pmix["Re_K"],
                Q_m3s=Q_m3s,
                d_p_m=self.d_p_m,
                fouling_fracs=ff,
                rho_m=RHO_WATER,
                mu=MU_WATER,
            )
            results[pname] = r

        # Weighted compound capture efficiency
        eta_weighted = 0.0
        for pname, pmix in POLYMER_MIX.items():
            eta_weighted += pmix["fraction"] * results[pname]["eta_cascade"]
        # Remaining neutral fraction: average of PP and PET capture
        neutral_eta = (results["PP"]["eta_cascade"] + results["PET"]["eta_cascade"]) / 2.0
        eta_cascade = float(np.clip(eta_weighted + 0.15 * neutral_eta, 0.0, 1.0))

        # C_out: weighted sum of species that escape
        C_in = float(self._state.get("C_in", 0.7))
        C_out = C_in * (
            POLYMER_MIX["PP"]["fraction"]  * (1.0 - results["PP"]["eta_cascade"])
            + POLYMER_MIX["PE"]["fraction"]  * (1.0 - results["PE"]["eta_cascade"])
            + POLYMER_MIX["PET"]["fraction"] * (1.0 - results["PET"]["eta_cascade"])
            + 0.15 * (1.0 - neutral_eta)
        )

        # v_crit of S3 for PP (most relevant — buoyant, hardest to capture)
        v_crit_s3 = results["PP"]["per_stage"][2]["v_crit"] if len(results["PP"]["per_stage"]) > 2 else 0.0

        # Write state
        self._state["eta_cascade"]  = eta_cascade
        self._state["eta_PP"]       = float(results["PP"]["eta_cascade"])
        self._state["eta_PET"]      = float(results["PET"]["eta_cascade"])
        self._state["C_out"]        = float(np.clip(C_out, 0.0, C_in))
        self._state["v_crit_s3"]    = float(v_crit_s3)

        # ── Accumulation model ────────────────────────────────────────────
        bf = float(action[2])  # bf_cmd from action vector

        # Each stage captures particles proportional to PET efficiency
        # (PET = sinking majority species; captures into channel bottom)
        for i in range(3):
            if len(results["PET"]["per_stage"]) > i:
                eta_i    = float(results["PET"]["per_stage"][i]["eta_stage"])
                captured = eta_i * float(self._state.get("C_in", 0.7)) * Q_m3s * self._dt
                self._channel_fill[i] = float(np.clip(
                    self._channel_fill[i] + captured / _CHANNEL_CAPACITY_M3,
                    0.0, 1.0,
                ))

        # bf_cmd > 0.5: drain channels into storage at FLUSH_DRAIN_RATE per step
        if bf > 0.5:
            for i in range(3):
                drained = self._channel_fill[i] * _FLUSH_DRAIN_RATE
                self._storage_fill = float(np.clip(
                    self._storage_fill
                    + drained * (_CHANNEL_CAPACITY_M3 / _STORAGE_CAPACITY_M3),
                    0.0, 1.0,
                ))
                self._channel_fill[i] = max(0.0, self._channel_fill[i] - drained)
            flush_flag = 1.0
        else:
            flush_flag = 0.0

        self._state["storage_fill"]    = self._storage_fill
        self._state["channel_fill_s1"] = self._channel_fill[0]
        self._state["channel_fill_s2"] = self._channel_fill[1]
        self._state["channel_fill_s3"] = self._channel_fill[2]
        self._state["flush_active_s1"] = flush_flag
        self._state["flush_active_s2"] = flush_flag
        self._state["flush_active_s3"] = flush_flag

        # Polarization zone diagnostics (PP — hardest to polarise quickly)
        pol_pp = self.pol_zone.state_dict(PP, Q_m3s)
        self._state.update(pol_pp)

        self._step += 1
        reward   = self._reward()
        done     = self._step >= self._max_steps
        return self._obs(), reward, done, False, self._info(results)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _voltage_scaled_stages(self, volt_norm: float) -> list[ConicalStageSpec]:
        """Return stages with voltage scaled by action (0→0V, 1→design voltage)."""
        scaled = []
        for stg in self.stages:
            new_dep = DEPConfig(
                voltage_V=stg.dep.voltage_V * float(volt_norm),
                electrode_gap_m=stg.dep.electrode_gap_m,
                tip_radius_m=stg.dep.tip_radius_m,
                eps_r_medium=stg.dep.eps_r_medium,
                mu=stg.dep.mu,
            )
            scaled.append(ConicalStageSpec(
                label=stg.label, mesh=stg.mesh, dep=new_dep,
                D_in_m=stg.D_in_m, D_tip_m=stg.D_tip_m, L_cone_m=stg.L_cone_m,
            ))
        return scaled

    def _obs(self) -> np.ndarray:
        s = self._state
        q    = float(s.get("q_processed_lmin", 10.0))
        dp   = float(s.get("delta_p_kpa", 0.0))
        ff   = (
            float(s.get("fouling_frac_s1", 0.0))
            + float(s.get("fouling_frac_s2", 0.0))
            + float(s.get("fouling_frac_s3", 0.0))
        ) / 3.0
        obs = np.array([
            np.clip(q  / self.OBS_Q_MAX,          0.0, 1.0),   # 0: q_in
            np.clip(dp / self.OBS_DP_MAX,          0.0, 1.0),   # 1: delta_p
            np.clip(ff,                            0.0, 1.0),   # 2: fouling_mean
            np.clip(s.get("eta_cascade", 0.0),     0.0, 1.0),   # 3: eta_cascade
            np.clip(s.get("C_in",  0.7),           0.0, 1.0),   # 4: C_in
            np.clip(s.get("C_out", 0.7),           0.0, 1.0),   # 5: C_out
            np.clip(s.get("voltage_norm", 0.8),    0.0, 1.0),   # 6: E_field_norm
            np.clip(s.get("v_crit_s3", 0.0)        # 7: v_crit_norm
                    / self.OBS_VCRIT_MAX,          0.0, 1.0),
            np.clip(self._step / self._max_steps,  0.0, 1.0),   # 8: step_norm
            float(s.get("bf_active", 0.0) > 0.5),               # 9: bf_active
            np.clip(s.get("eta_PP",  0.0),         0.0, 1.0),   # 10: eta_PP (buoyant)
            np.clip(s.get("eta_PET", 0.0),         0.0, 1.0),   # 11: eta_PET (dense)
        ], dtype=np.float32)
        return obs

    def _reward(self) -> float:
        """
        Simple reward: maximise capture, penalise energy and over-pressure.
        Mirrors HydrionEnv reward structure for comparability.
        """
        eta  = float(self._state.get("eta_cascade", 0.0))
        dp   = float(self._state.get("delta_p_kpa", 0.0))
        volt = float(self._state.get("voltage_norm", 0.8))
        dp_penalty   = max(0.0, dp - 80.0) / 70.0   # penalise above 80 kPa
        volt_penalty = volt * 0.05                   # small energy cost
        return float(eta - dp_penalty - volt_penalty)

    def _info(self, results: dict) -> dict:
        return {
            "eta_cascade": self._state.get("eta_cascade", 0.0),
            "eta_PP":      self._state.get("eta_PP",      0.0),
            "eta_PET":     self._state.get("eta_PET",     0.0),
            "v_crit_s3":   self._state.get("v_crit_s3",   0.0),
            "step":        self._step,
            "per_stage_PET": [
                s["eta_stage"] for s in results["PET"]["per_stage"]
            ],
        }
