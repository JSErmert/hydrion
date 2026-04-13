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
from ..physics.m5.field_models import analytical_conical_field
from ..physics.m5.particle_dynamics import (
    InputParticle, ParticleDynamicsEngine,
)
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
# Default particle set — three sizes per species to span the PSD (Phase 1)
#
# Size rationale:
#   10 µm — weak DEP (force ∝ r²), typically passes S1 → visible in S2/S3
#   25 µm — median, moderate DEP — baseline comparison
#   50 µm — strong DEP + above S1/S2 mesh openings → early capture
# ---------------------------------------------------------------------------
_DEFAULT_PARTICLES: list[InputParticle] = [
    # PP (buoyant): nDEP + buoyancy both drive toward axis → reliable apex capture
    # r0 spread 0.12–0.30 so trajectories are visually distinct and all apex-captured in S1
    InputParticle("pp-small",   "PP",  d_p_m=10e-6,  r0_norm=0.12),
    InputParticle("pp-median",  "PP",  d_p_m=25e-6,  r0_norm=0.18),
    InputParticle("pp-large",   "PP",  d_p_m=50e-6,  r0_norm=0.30),  # confirmed captured r0=0.30
    # PE (slightly buoyant): similar to PP but gravity weaker → some pass S1, captured S2
    InputParticle("pe-small",   "PE",  d_p_m=10e-6,  r0_norm=0.22),
    InputParticle("pe-median",  "PE",  d_p_m=25e-6,  r0_norm=0.35),
    InputParticle("pe-large",   "PE",  d_p_m=50e-6,  r0_norm=0.50),
    # PET (sinking): gravity opposes nDEP → starts off-axis, mostly passes cascade
    # Shows physics limit: PET requires narrow r0 or high voltage for apex capture
    InputParticle("pet-small",  "PET", d_p_m=10e-6,  r0_norm=0.55),
    InputParticle("pet-median", "PET", d_p_m=25e-6,  r0_norm=0.65),
    InputParticle("pet-large",  "PET", d_p_m=50e-6,  r0_norm=0.75),
]


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
        particles: list[InputParticle] | None = None,
        log_trajectories: bool = False,
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
        self._max_steps = int(raw_cfg.get("max_steps", 10_000))
        self._dt = float(raw_cfg.get("dt", 0.1))

        # Accumulation state — persists across steps, resets on env.reset()
        self._storage_fill: float       = 0.0
        self._channel_fill: list[float] = [0.0, 0.0, 0.0]

        # Visual deposit tracking — captured particle dots in channels and storage.
        # Updated at each 30-step animation cycle boundary; cleared on backflush/reset.
        self._channel_deposits: list[list[dict]] = [[], [], []]
        self._storage_particles: list[dict]      = []

        # Particle dynamics engine
        self._particle_engine  = ParticleDynamicsEngine()
        self._particle_set     = particles if particles is not None else list(_DEFAULT_PARTICLES)
        self._log_trajectories = log_trajectories

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
        self._channel_deposits = [[], [], []]
        self._storage_particles = []

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
        self._state["particle_streams"] = None
        self._state["voltage_norm"]    = 0.8
        self._state["storage_fill"]    = 0.0
        self._state["channel_fill_s1"] = 0.0
        self._state["channel_fill_s2"] = 0.0
        self._state["channel_fill_s3"] = 0.0
        self._state["flush_active_s1"] = 0.0
        self._state["flush_active_s2"] = 0.0
        self._state["flush_active_s3"] = 0.0
        self._state["eta_s1"]     = 0.0
        self._state["eta_s2"]     = 0.0
        self._state["eta_s3"]     = 0.0
        self._state["v_crit_s1"]  = 0.0
        self._state["v_crit_s2"]  = 0.0

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

        # ── Particle dynamics engine — cascade routing ─────────────────────
        # Each stage receives only particles that passed through previous stages.
        # Voltage-scaled stages used so field_fn reflects the applied voltage.
        bf_active = bf_cmd > 0.5

        trajs_per_stage: list[list] = [[], [], []]
        active_particles = list(self._particle_set)

        for i, stg in enumerate(stages):  # `stages` is already voltage-scaled above
            if not active_particles:
                break
            field_fn_i = analytical_conical_field(stg)
            trajs = self._particle_engine.integrate(
                particles  = active_particles,
                stage      = stg,
                stage_idx  = i,
                Q_m3s      = Q_m3s,
                field_fn   = field_fn_i,
                dt_sim     = self._dt,
                n_substeps = 100,
                backflush  = bf_active,
            )
            trajs_per_stage[i] = trajs
            # Cascade routing: only 'passed' particles enter the next stage.
            # Carry the final r_norm from this stage as r0_norm for the next —
            # a particle exiting S1 at r_norm=0.6 enters S2 at r_norm=0.6,
            # not reset to mid-radius. Resetting would erase all radial displacement
            # accumulated in the upstream stage, making S2/S3 independent of S1.
            active_particles = [
                InputParticle(
                    t.particle_id, t.species, t.d_p_m,
                    r0_norm=t.positions[-1][1] if t.positions else 0.5,
                )
                for t in trajs if t.final_status == "passed"
            ]

        # Particles still in active_particles after S3 escaped the full cascade
        escaped_device = active_particles

        # ── Cascade animation — 30-step cycle, one 10-step window per stage ──────
        # S1 animates in steps 0–9, S2 in 10–19, S3 in 20–29.
        # Each stage stream is EMPTY outside its window so the viewer sees
        # particles travel S1 → S2 → S3 in temporal sequence, not in parallel.
        _CYCLE     = 30
        _WIN       = 10
        _cycle     = self._step % _CYCLE
        _win_idx   = _cycle // _WIN             # 0=S1, 1=S2, 2=S3
        _win_phase = (_cycle % _WIN) / max(_WIN - 1, 1)   # 0.0 → 1.0

        def _stage_phase(target: int) -> "float | None":
            return _win_phase if _win_idx == target else None

        phase_s1 = _stage_phase(0)
        phase_s2 = _stage_phase(1)
        phase_s3 = _stage_phase(2)

        # At the start of each cycle: record newly captured particles into
        # the persistent channel deposit lists.  The same 9-particle set is
        # re-simulated every step; we sample captures only once per cycle to
        # avoid duplicating the same events.
        if _cycle == 0:
            for i in range(3):
                for t in trajs_per_stage[i]:
                    if t.final_status == "captured":
                        self._channel_deposits[i].append({
                            "x_norm":  t.positions[-1][0] if t.positions else 0.90,
                            "species": t.species,
                            "d_p_um":  t.d_p_m * 1e6,
                        })
                # Cap to 30 visible deposits per stage (visual budget)
                self._channel_deposits[i] = self._channel_deposits[i][-30:]

        self._state["particle_streams"] = {
            "s1": self._make_stream(trajs_per_stage[0], phase_s1),
            "s2": self._make_stream(trajs_per_stage[1], phase_s2),
            "s3": self._make_stream(trajs_per_stage[2], phase_s3),
            # Deposited particles in collection channels (persist until backflush)
            "s1Deposited": list(self._channel_deposits[0]),
            "s2Deposited": list(self._channel_deposits[1]),
            "s3Deposited": list(self._channel_deposits[2]),
            # Accumulated particles in storage chamber
            "storage": list(self._storage_particles[-20:]),
        }

        # Per-stage capture counts
        for i in range(3):
            lb = f"s{i + 1}"
            tl = trajs_per_stage[i]
            self._state[f"captured_pp_{lb}"]  = int(sum(1 for t in tl if t.species == "PP"  and t.final_status == "captured"))
            self._state[f"captured_pe_{lb}"]  = int(sum(1 for t in tl if t.species == "PE"  and t.final_status == "captured"))
            self._state[f"captured_pet_{lb}"] = int(sum(1 for t in tl if t.species == "PET" and t.final_status == "captured"))

        # Device-level escape counts
        self._state["escaped_device_pp"]  = int(sum(1 for p in escaped_device if p.species == "PP"))
        self._state["escaped_device_pet"] = int(sum(1 for p in escaped_device if p.species == "PET"))

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

        # Per-stage observables for console hierarchy rendering
        per_pet = results["PET"]["per_stage"]
        self._state["eta_s1"] = float(per_pet[0]["eta_stage"]) if len(per_pet) > 0 else 0.0
        self._state["eta_s2"] = float(per_pet[1]["eta_stage"]) if len(per_pet) > 1 else 0.0
        self._state["eta_s3"] = float(per_pet[2]["eta_stage"]) if len(per_pet) > 2 else 0.0
        self._state["v_crit_s1"] = float(per_pet[0]["v_crit"]) if len(per_pet) > 0 else 0.0
        self._state["v_crit_s2"] = float(per_pet[1]["v_crit"]) if len(per_pet) > 1 else 0.0
        # v_crit_s3 already written above

        # ── Accumulation model ────────────────────────────────────────────
        # Capture flux: each stage operates on the residual concentration
        # that passed through upstream stages (cascade attenuation).
        # Flush (bf_cmd > 0.5) drains on same step — filter still captures
        # during flush; the hydraulic pressure clears the collection channels.
        c_stage = float(self._state.get("C_in", 0.7))
        for i in range(3):
            if len(results["PET"]["per_stage"]) > i:
                eta_i    = float(results["PET"]["per_stage"][i]["eta_stage"])
                captured = eta_i * c_stage * Q_m3s * self._dt
                self._channel_fill[i] = float(np.clip(
                    self._channel_fill[i] + captured / _CHANNEL_CAPACITY_M3,
                    0.0, 1.0,
                ))
                c_stage *= (1.0 - eta_i)   # residual concentration passes to next stage

        # bf_cmd > 0.5: drain channels into storage at FLUSH_DRAIN_RATE per step
        if bf_cmd > 0.5:
            for i in range(3):
                drained = self._channel_fill[i] * _FLUSH_DRAIN_RATE
                self._storage_fill = float(np.clip(
                    self._storage_fill
                    + drained * (_CHANNEL_CAPACITY_M3 / _STORAGE_CAPACITY_M3),
                    0.0, 1.0,
                ))
                self._channel_fill[i] = max(0.0, self._channel_fill[i] - drained)

                # Move visual deposit dots from channel → storage
                n_drain = max(1, int(len(self._channel_deposits[i]) * _FLUSH_DRAIN_RATE))
                to_storage = self._channel_deposits[i][:n_drain]
                self._storage_particles.extend(to_storage)
                self._channel_deposits[i] = self._channel_deposits[i][n_drain:]

            # Cap storage particle list (visual budget only)
            self._storage_particles = self._storage_particles[-50:]
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

        # Shield-compatible normalized aliases — ShieldedEnv reads these exact keys
        # from env.truth_state. Normalization: Q_max=20 L/min, P_max=80 kPa.
        self._state["flow"]     = float(np.clip(
            self._state.get("q_processed_lmin", 0.0) / 20.0, 0.0, 1.0))
        self._state["pressure"] = float(np.clip(
            self._state.get("dp_total_pa", 0.0) / 80_000.0, 0.0, 1.0))
        ff_s1 = float(self._state.get("fouling_frac_s1", 0.0))
        ff_s2 = float(self._state.get("fouling_frac_s2", 0.0))
        ff_s3 = float(self._state.get("fouling_frac_s3", 0.0))
        self._state["clog"]     = float(np.clip((ff_s1 + ff_s2 + ff_s3) / 3.0, 0.0, 1.0))

        self._step += 1
        reward   = self._reward()
        done     = self._step >= self._max_steps
        return self._obs(), reward, done, False, self._info(results)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_stream(trajs_list: list, phase_frac: "float | None" = 1.0) -> list[dict]:
        """
        Return an animated particle position + full trajectory trail for each trajectory.

        phase_frac (0.0–1.0): which fraction along the in-cone trajectory
        to show as the current particle dot. Cycling this each env step makes
        particles appear to traverse the cone rather than always showing the
        final (past-apex) position.

        Only positions with x_norm in [0, 1] are used — positions past the apex
        are outside the visible SVG cone and would not render.

        Trail: ALL positions from inlet up to the display position, subsampled
        to at most N_TRAIL_MAX points when the trajectory is long. This produces
        a visible path showing the particle's route through the stage, including
        the radial displacement driven by nDEP and gravity.
        """
        if phase_frac is None:
            return []   # this stage is outside its animation window — show nothing
        N_TRAIL_MAX = 20   # max trail points after subsampling
        result = []
        for t in trajs_list:
            if not t.positions:
                continue

            # Restrict to in-cone positions (x_norm in [0, 1])
            inside = [(x, r) for x, r in t.positions if 0.0 <= x <= 1.0]
            if not inside:
                inside = [t.positions[0]]

            n = len(inside)
            # Map phase_frac → index into inside positions
            display_idx = min(int(phase_frac * n), n - 1)
            display_pos = inside[display_idx]

            # Full trail: all positions from inlet to (but not including) display_idx
            trail_raw = inside[:display_idx]
            if len(trail_raw) > N_TRAIL_MAX:
                # Subsample evenly to at most N_TRAIL_MAX points
                step = len(trail_raw) / N_TRAIL_MAX
                trail_raw = [trail_raw[int(i * step)] for i in range(N_TRAIL_MAX)]
            trail = [{"x_norm": pos[0], "r_norm": pos[1]} for pos in trail_raw]

            # Show "in_transit" mid-animation; reveal final_status for the last 25%
            # of frames so the apex-to-channel ejection path is visible for several
            # consecutive steps rather than only the terminal frame.
            _captured_onset = max(0, n - max(1, n // 4))
            display_status = (
                t.final_status if display_idx >= _captured_onset else "in_transit"
            )
            result.append({
                "x_norm":  display_pos[0],
                "r_norm":  display_pos[1],
                "status":  display_status,
                "species": t.species,
                "d_p_um":  t.d_p_m * 1e6,  # diameter in µm — used by frontend for size/shape scaling
                "trail":   trail,
            })
        return result

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
        dp   = float(s.get("dp_total_pa", 0.0)) / 1000.0  # Pa → kPa for normalisation
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
        Reward: maximize capture + throughput, penalise over-pressure and energy.

        Components:
            eta_cascade      — primary capture signal
            flow_bonus       — processed throughput reward (Stage 3: w_processed_flow=2.0)
            dp_penalty       — pressure above 80 kPa (CCE design limit)
            volt_penalty     — energy cost

        ShieldedEnv adds on top:
            clog_penalty     — fouling above soft threshold
            blockage_penalty — pump-on with no flow
            termination (-10) — hard pressure or clog violation
        """
        eta  = float(self._state.get("eta_cascade", 0.0))
        dp   = float(self._state.get("dp_total_pa", 0.0)) / 1000.0  # Pa → kPa
        volt = float(self._state.get("voltage_norm", 0.8))
        q    = float(self._state.get("q_processed_lmin", 0.0))

        flow_bonus   = (q / 20.0) * 0.30            # throughput term; Q_max = 20 L/min
        dp_penalty   = max(0.0, dp - 80.0) / 70.0   # penalise above 80 kPa
        volt_penalty = volt * 0.05                   # small energy cost

        return float(eta + flow_bonus - dp_penalty - volt_penalty)

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

    # ------------------------------------------------------------------
    # ScenarioRunner compatibility — mirrors HydrionEnv interface
    # ------------------------------------------------------------------

    @property
    def truth_state(self) -> dict:
        """
        Expose _state as truth_state for ScenarioRunner compatibility.
        ConicalCascadeEnv uses _state as its authoritative physics state
        (no sensor layer). Returns the live dict — mutations are reflected.
        """
        return self._state

    @property
    def sensor_state(self) -> dict:
        """
        Return empty dict — ConicalCascadeEnv has no sensor noise layer.
        Required by ScenarioRunner for ScenarioStepRecord.sensorState.
        """
        return {}

    def _update_normalized_state(self) -> None:
        """
        Sync clogging model internal state into _state.
        Called by apply_initial_state() after writing initial fouling.
        """
        self._state.update(self.clogging._state)
