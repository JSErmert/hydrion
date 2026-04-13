# hydrion/env.py
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml

from .config import HydrionConfig
from .physics.hydraulics import HydraulicsModel
from .physics.clogging import CloggingModel
from .physics.electrostatics import ElectrostaticsModel
from .physics.particles import ParticleModel
from .sensors.optical import OpticalSensorArray
from .sensors.pressure import DifferentialPressureSensor
from .sensors.flow import FlowRateSensor

# Commit 1 v1.5 imports
from .runtime.run_context import RunContext
from .runtime.seeding import set_global_seed

# Commit 2 v1.5 imports
from pathlib import Path
from .logging.writer import RunLogger

# Commit 3 v1.5 imports
from .state.init import init_truth_state, init_sensor_state
from .sensors.sensor_fusion import build_observation


class HydrionEnv(gym.Env):
    """
    HydrionEnv — Multi-Physics Digital Twin Environment

    Milestone 1 additions (v1.5+):
    - Backflush pulse state machine (3-pulse square-wave with cooldown)
    - Decomposed fouling model via CloggingModel v3
    - Area-normalized hydraulic resistance via HydraulicsModel v2
    - Bypass pressure-relief logic
    - Five-term YAML-driven Milestone 1 reward
    - maintenance_required telemetry flag (status only, not in reward)

    State separation (Commit 3):
    - truth_state: physics truth (internal)
    - sensor_state: measured outputs (may diverge from truth)
    - observation: derived from (truth_state, sensor_state) via sensor_fusion
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        render_mode=None,
        run_context: RunContext | None = None,
        version: str = "v1.5",
        seed: int | None = None,
        noise_enabled: bool | None = None,
        auto_reset: bool = True,
    ):
        super().__init__()
        self.render_mode = render_mode

        # Load YAML config
        with open(config_path, "r") as f:
            raw_cfg = yaml.safe_load(f) or {}
        self.cfg = HydrionConfig(raw_cfg)

        # ------------------------------
        # Commit 1: Identity + Determinism
        # ------------------------------
        cfg_hash       = self.cfg.config_hash()
        resolved_seed  = self.cfg.get_seed(0) if seed is None else int(seed)
        resolved_noise = self.cfg.get_noise_enabled(False) if noise_enabled is None else bool(noise_enabled)

        self.run_context = run_context or RunContext.create(
            version=version,
            seed=resolved_seed,
            noise_enabled=resolved_noise,
            config_hash=cfg_hash,
            deterministic_id=True,
        )

        # ------------------------------
        # Commit 2: Logging Skeleton
        # ------------------------------
        self._log_base_dir = Path(self.cfg.raw.get("logging", {}).get("base_dir", "runs"))
        self.logger        = RunLogger(base_dir=self._log_base_dir, enabled=True, strict=False)
        self._episode_return = 0.0

        self._active_seed: int = self.run_context.seed

        # Simulation params
        self.dt        = float(self.cfg.raw.get("sim", {}).get("dt", 0.1))
        self.max_steps = int(600.0 / self.dt)

        # Physics & sensors
        self.hydraulics    = HydraulicsModel(self.cfg)
        self.clogging      = CloggingModel(self.cfg)
        self.electrostatics = ElectrostaticsModel(self.cfg)
        self.particles     = ParticleModel(self.cfg)
        self.sensors          = OpticalSensorArray(self.cfg)
        self.pressure_sensor  = DifferentialPressureSensor(self.cfg)   # M6 Phase 1
        self.flow_sensor      = FlowRateSensor(self.cfg)               # M6 Phase 1

        # Commit 3: explicit truth/sensor state containers
        self.truth_state:  dict = {}
        self.sensor_state: dict = {}

        # ------------------------------
        # Backflush state machine config (loaded once at init)
        # ------------------------------
        bf_raw = self.cfg.raw.get("backflush", {}) or {}

        def _gf(key, default):
            try:
                return float(bf_raw.get(key, default))
            except Exception:
                return default

        self._bf_n_pulses   = int(bf_raw.get("n_pulses", 3))
        self._bf_pulse_dur  = _gf("pulse_duration_s", 0.4)
        self._bf_interpulse = _gf("interpulse_s", 0.25)
        self._bf_cooldown   = _gf("cooldown_s", 9.0)
        self._bf_trigger    = _gf("trigger_threshold", 0.30)
        self._bf_eff        = _gf("effluent_cleaning_efficiency", 0.70)

        # Total burst duration: n_pulses × pulse_dur + (n_pulses−1) × interpulse
        self._bf_burst_total = (
            self._bf_n_pulses * self._bf_pulse_dur
            + max(self._bf_n_pulses - 1, 0) * self._bf_interpulse
        )

        # ------------------------------
        # Milestone 1 reward config
        # ------------------------------
        rw_raw = self.cfg.raw.get("reward", {}) or {}

        def _gfrw(key, default):
            try:
                return float(rw_raw.get(key, default))
            except Exception:
                return default

        self._rw_processed_flow   = _gfrw("w_processed_flow",  2.0)
        self._rw_pressure_penalty = _gfrw("w_pressure_penalty", 1.0)
        self._rw_fouling_penalty  = _gfrw("w_fouling_penalty",  0.5)
        self._rw_bypass_penalty   = _gfrw("w_bypass_penalty",   0.3)
        self._rw_backflush_cost   = _gfrw("w_backflush_cost",   0.1)

        # Flow normalization reference for reward terms
        fe_raw = self.cfg.raw.get("flow_envelope", {}) or {}
        self._Q_nominal_max = float(fe_raw.get("Q_nominal_max_Lmin", 15.0))

        # Maintenance threshold (for _update_normalized_state only)
        cl_raw = self.cfg.raw.get("clogging", {}) or {}
        self._maintenance_threshold = float(cl_raw.get("maintenance_fouling_threshold", 0.70))

        # Action space: [valve, pump, backflush, node_voltage]
        self.action_space = spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space (12-D, schema obs12_v1 — immutable)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32)

        self.steps = 0
        if auto_reset:
            self.reset()

    # ---------------------------------------------------------
    # RESET
    # ---------------------------------------------------------
    def reset(self, seed=None, options=None):
        resolved_seed    = self.run_context.seed if seed is None else int(seed)
        self._active_seed = resolved_seed

        super().reset(seed=resolved_seed)
        set_global_seed(resolved_seed)

        self.steps           = 0
        self._episode_return = 0.0

        # Start logging run (Commit 2)
        run_header = {
            "run_id":       self.run_context.run_id,
            "version":      self.run_context.version,
            "seed":         self._active_seed,
            "noise_enabled": self.run_context.noise_enabled,
            "config_hash":  self.run_context.config_hash,
        }
        self.logger.start_run(
            run_id=self.run_context.run_id,
            run_header=run_header,
            config=self.cfg.raw,
        )
        self.logger.log_step({
            "event":   "reset",
            "run_id":  self.run_context.run_id,
            "timestep": 0,
            "step":    0,
            "seed":    self._active_seed,
        })

        # Commit 3: initialize truth/sensor states
        self.truth_state  = init_truth_state().data
        self.sensor_state = init_sensor_state().data

        # Reset subsystems (physics uses truth)
        self.clogging.reset(self.truth_state)
        self.hydraulics.reset()
        self.electrostatics.reset(self.truth_state)
        self.particles.reset(self.truth_state)

        # Sensor reset writes to sensor_state (and mirrors to truth for compatibility)
        self.sensors.reset(self.truth_state, sensor_state=self.sensor_state)
        self.pressure_sensor.reset(self.truth_state, sensor_state=self.sensor_state)  # M6 Phase 1
        self.flow_sensor.reset(self.truth_state, sensor_state=self.sensor_state)      # M6 Phase 1

        # Neutral kick to initialize derived values
        neutral = np.array([0.5, 0.5, 0.0, 0.5], dtype=np.float32)
        # Note: neutral has bf_cmd=0 so backflush state machine is a no-op here
        self.hydraulics.update(self.truth_state, dt=self.dt, action=neutral, clogging_model=self.clogging)
        self.electrostatics.update(self.truth_state, dt=self.dt, node_cmd=self.truth_state["node_voltage_cmd"])
        self.particles.update(
            self.truth_state,
            dt=self.dt,
            clogging_model=self.clogging,
            electrostatics_model=self.electrostatics,
        )
        self.sensors.update(self.truth_state, dt=self.dt, sensor_state=self.sensor_state)
        self.pressure_sensor.update(self.truth_state, self.sensor_state, dt=self.dt)  # M6 Phase 1
        self.flow_sensor.update(self.truth_state, self.sensor_state, dt=self.dt)      # M6 Phase 1

        self._update_normalized_state()
        return self._observe(), {}

    # ---------------------------------------------------------
    # STEP
    # ---------------------------------------------------------
    def step(self, action):
        self.steps += 1
        action = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)

        # Write actuator commands into truth before physics and state machine
        self.truth_state["valve_cmd"]        = float(action[0])
        self.truth_state["pump_cmd"]         = float(action[1])
        self.truth_state["bf_cmd"]           = float(action[2])
        self.truth_state["node_voltage_cmd"] = float(action[3])

        # ----------------------------------------------------------------
        # BACKFLUSH STATE MACHINE
        # Must run before clogging.update() so bf_active is current.
        # ----------------------------------------------------------------
        self._update_backflush_state(action)

        # ----------------------------------------------------------------
        # PHYSICS PIPELINE (immutable order)
        # Hydraulics → Clogging → Electrostatics → Particles → Sensors
        # ----------------------------------------------------------------
        self.hydraulics.update(
            state=self.truth_state,
            dt=self.dt,
            action=action,
            clogging_model=self.clogging,
        )
        self.clogging.update(self.truth_state, dt=self.dt)
        self.electrostatics.update(self.truth_state, dt=self.dt, node_cmd=self.truth_state["node_voltage_cmd"])
        self.particles.update(
            self.truth_state,
            dt=self.dt,
            clogging_model=self.clogging,
            electrostatics_model=self.electrostatics,
        )
        self.sensors.update(self.truth_state, dt=self.dt, sensor_state=self.sensor_state)
        self.pressure_sensor.update(self.truth_state, self.sensor_state, dt=self.dt)  # M6 Phase 1
        self.flow_sensor.update(self.truth_state, self.sensor_state, dt=self.dt)      # M6 Phase 1

        self._update_normalized_state()

        # ----------------------------------------------------------------
        # REWARD — Milestone 1 (five-term, YAML-driven, continuous)
        # maintenance_required is telemetry only and NOT used here.
        # ----------------------------------------------------------------
        q_proc    = float(self.truth_state.get("q_processed_lmin", 0.0))
        pressure  = float(self.truth_state.get("pressure", 0.0))
        mesh_avg  = float(self.truth_state.get("mesh_loading_avg", 0.0))
        q_bypass  = float(self.truth_state.get("q_bypass_lmin", 0.0))
        bf_active = float(self.truth_state.get("bf_active", 0.0))
        Q_norm    = max(self._Q_nominal_max, 1e-6)

        reward = (
            self._rw_processed_flow   * (q_proc / Q_norm)
            - self._rw_pressure_penalty * max(0.0, pressure - 0.50) ** 2
            - self._rw_fouling_penalty  * max(0.0, mesh_avg - 0.40)
            - self._rw_bypass_penalty   * (q_bypass / Q_norm)
            - self._rw_backflush_cost   * bf_active
        )

        terminated = False
        truncated  = self.steps >= self.max_steps

        info = {
            # Core hydraulics
            "Q_out_Lmin":        float(self.truth_state.get("Q_out_Lmin",        0.0)),
            "q_processed_lmin":  float(self.truth_state.get("q_processed_lmin",  0.0)),
            "q_bypass_lmin":     float(self.truth_state.get("q_bypass_lmin",     0.0)),
            "P_in":              float(self.truth_state.get("P_in",              0.0)),
            "dp_total_pa":       float(self.truth_state.get("dp_total_pa",       0.0)),
            "bypass_active":     float(self.truth_state.get("bypass_active",     0.0)),
            # Clogging aggregate (compatibility)
            "mesh_loading_avg":  float(self.truth_state.get("mesh_loading_avg",  0.0)),
            "capture_eff":       float(self.truth_state.get("capture_eff",       0.0)),
            # Clogging per-stage fouling fractions
            "fouling_frac_s1":   float(self.truth_state.get("fouling_frac_s1",   0.0)),
            "fouling_frac_s2":   float(self.truth_state.get("fouling_frac_s2",   0.0)),
            "fouling_frac_s3":   float(self.truth_state.get("fouling_frac_s3",   0.0)),
            # Irreversible fouling per stage
            "irreversible_s1":   float(self.truth_state.get("irreversible_s1",   0.0)),
            "irreversible_s2":   float(self.truth_state.get("irreversible_s2",   0.0)),
            "irreversible_s3":   float(self.truth_state.get("irreversible_s3",   0.0)),
            # Maintenance flag (telemetry only)
            "maintenance_required": float(self.truth_state.get("maintenance_required", 0.0)),
            # Backflush event state
            "bf_active":              float(self.truth_state.get("bf_active",              0.0)),
            "bf_pulse_idx":           float(self.truth_state.get("bf_pulse_idx",           0.0)),
            "bf_cooldown_remaining":  float(self.truth_state.get("bf_cooldown_remaining",  0.0)),
            "bf_n_bursts_completed":  float(self.truth_state.get("bf_n_bursts_completed",  0.0)),
            # Electrostatics + particles (obs12_v2: key is E_field_norm, not E_norm)
            "E_field_norm":     float(self.truth_state.get("E_field_norm",     0.0)),
            "capture_eff_part": float(self.truth_state.get("particle_capture_eff", 0.0)),
            # Sensors
            "sensor_turbidity":  float(self.sensor_state.get("sensor_turbidity",  0.0)),
            "sensor_scatter":    float(self.sensor_state.get("sensor_scatter",    0.0)),
            "dp_sensor_kPa":    float(self.sensor_state.get("dp_sensor_kPa",    0.0)),   # M6
            "flow_sensor_lmin": float(self.sensor_state.get("flow_sensor_lmin", 0.0)),   # M6
            # Run identity
            "run_id":       self.run_context.run_id,
            "version":      self.run_context.version,
            "seed":         self._active_seed,
            "noise_enabled": self.run_context.noise_enabled,
            "config_hash":  self.run_context.config_hash,
        }

        self._episode_return += float(reward)

        # Commit 2 timestep spine logging (Milestone 1 extended payload)
        self.logger.log_step({
            "event":     "step",
            "run_id":    self.run_context.run_id,
            "timestep":  int(self.steps),
            "step":      int(self.steps),
            "reward":    float(reward),
            "terminated": bool(terminated),
            "truncated":  bool(truncated),
            # Normalized channels
            "flow":     float(self.truth_state.get("flow",     0.0)),
            "pressure": float(self.truth_state.get("pressure", 0.0)),
            "clog":     float(self.truth_state.get("clog",     0.0)),
            # Hydraulics Milestone 1
            "q_processed_lmin": float(self.truth_state.get("q_processed_lmin", 0.0)),
            "q_bypass_lmin":    float(self.truth_state.get("q_bypass_lmin",    0.0)),
            "dp_total_pa":      float(self.truth_state.get("dp_total_pa",      0.0)),
            "bypass_active":    float(self.truth_state.get("bypass_active",    0.0)),
            # Clogging Milestone 1
            "fouling_frac_s1": float(self.truth_state.get("fouling_frac_s1", 0.0)),
            "fouling_frac_s2": float(self.truth_state.get("fouling_frac_s2", 0.0)),
            "fouling_frac_s3": float(self.truth_state.get("fouling_frac_s3", 0.0)),
            "irreversible_s3": float(self.truth_state.get("irreversible_s3", 0.0)),
            "maintenance_required": float(self.truth_state.get("maintenance_required", 0.0)),
            # Backflush event
            "bf_active":             float(self.truth_state.get("bf_active",             0.0)),
            "bf_burst_elapsed":      float(self.truth_state.get("bf_burst_elapsed",      0.0)),
            "bf_cooldown_remaining": float(self.truth_state.get("bf_cooldown_remaining", 0.0)),
            # Sensors
            "sensor_turbidity":  float(self.sensor_state.get("sensor_turbidity",  0.0)),
            "sensor_scatter":    float(self.sensor_state.get("sensor_scatter",    0.0)),
            "dp_sensor_kPa":    float(self.sensor_state.get("dp_sensor_kPa",    0.0)),   # M6
            "flow_sensor_lmin": float(self.sensor_state.get("flow_sensor_lmin", 0.0)),   # M6
        })

        if terminated or truncated:
            self.logger.end_run(summary={
                "episode_return": float(self._episode_return),
                "steps":          int(self.steps),
                "terminated":     bool(terminated),
                "truncated":      bool(truncated),
            })

        return self._observe(), reward, terminated, truncated, info

    # ---------------------------------------------------------
    # BACKFLUSH STATE MACHINE
    # ---------------------------------------------------------
    def _update_backflush_state(self, action: np.ndarray) -> None:
        """
        Advance the backflush pulse state machine and write results into truth_state.

        Must be called at the START of step(), before the physics pipeline,
        so that clogging.update() reads the correct bf_active value.

        Burst lifecycle:
            idle  → (bf_cmd > trigger AND cooldown == 0) → burst starts
            burst → 3 × [0.4 s pulse ON / 0.25 s interpulse OFF]    (1.7 s total)
            burst → cooldown (9 s) → idle
        """
        bf_cmd           = float(action[2])
        cooldown         = float(self.truth_state.get("bf_cooldown_remaining", 0.0))
        burst_elapsed    = float(self.truth_state.get("bf_burst_elapsed",      0.0))
        in_burst         = burst_elapsed > 0.0

        # ---- Cooldown ticks passively each step ----
        if cooldown > 0.0:
            cooldown = max(0.0, cooldown - self.dt)

        # ---- Burst start condition ----
        if bf_cmd > self._bf_trigger and cooldown <= 0.0 and not in_burst:
            # Increment burst counter before the first pulse begins
            n_bursts = float(self.truth_state.get("bf_n_bursts_completed", 0.0)) + 1.0
            self.truth_state["bf_n_bursts_completed"] = n_bursts
            self.truth_state["bf_source_efficiency"]  = self._bf_eff
            burst_elapsed = self.dt   # first step of burst counts as elapsed time
            in_burst = True

        elif in_burst:
            burst_elapsed += self.dt

        # ---- Burst completion check ----
        if in_burst and burst_elapsed >= self._bf_burst_total:
            burst_elapsed = 0.0
            cooldown      = self._bf_cooldown
            in_burst      = False

        # ---- Pulse / interpulse state within an active burst ----
        bf_active    = 0.0
        bf_pulse_idx = 0.0
        if in_burst and burst_elapsed > 0.0:
            pulse_period     = self._bf_pulse_dur + self._bf_interpulse   # 0.65 s per cycle
            phase_in_period  = burst_elapsed % pulse_period
            pulse_idx        = int(burst_elapsed / pulse_period)
            bf_pulse_idx     = float(min(pulse_idx, self._bf_n_pulses - 1))
            bf_active        = 1.0 if phase_in_period < self._bf_pulse_dur else 0.0

        # ---- Write into truth_state ----
        self.truth_state["bf_active"]             = bf_active
        self.truth_state["bf_pulse_idx"]          = bf_pulse_idx
        self.truth_state["bf_burst_elapsed"]      = burst_elapsed
        self.truth_state["bf_cooldown_remaining"] = cooldown

    # ---------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------
    def _update_normalized_state(self) -> None:
        p   = self.hydraulics.params
        Q   = float(self.truth_state.get("Q_out_Lmin", 0.0))
        P   = float(self.truth_state.get("P_in",       0.0))

        self.truth_state["flow"]     = float(np.clip(Q / max(p.Q_max_Lmin, 1e-6), 0.0, 1.0))
        self.truth_state["pressure"] = float(np.clip(P / max(p.P_max_Pa,   1e-6), 0.0, 1.0))
        self.truth_state["clog"]     = float(np.clip(self.truth_state.get("mesh_loading_avg", 0.0), 0.0, 1.0))

        # Derived maintenance flag — telemetry/status only, NOT used in reward.
        ff_s1 = float(self.truth_state.get("fouling_frac_s1", 0.0))
        ff_s2 = float(self.truth_state.get("fouling_frac_s2", 0.0))
        ff_s3 = float(self.truth_state.get("fouling_frac_s3", 0.0))
        self.truth_state["maintenance_required"] = (
            1.0 if max(ff_s1, ff_s2, ff_s3) >= self._maintenance_threshold else 0.0
        )

    def _observe(self) -> np.ndarray:
        # Commit 3: stable observation contract (obs12_v1 — immutable schema)
        return build_observation(self.truth_state, self.sensor_state)

    def render(self) -> None:
        print(
            f"Flow={self.truth_state.get('flow', 0.0):.3f}  "
            f"P={self.truth_state.get('pressure', 0.0):.3f}  "
            f"Clog={self.truth_state.get('clog', 0.0):.3f}  "
            f"FF_s3={self.truth_state.get('fouling_frac_s3', 0.0):.3f}  "
            f"BF={'ON' if self.truth_state.get('bf_active', 0.0) > 0.5 else 'off'}  "
            f"Bypass={'ON' if self.truth_state.get('bypass_active', 0.0) > 0.5 else 'off'}  "
            f"Maint={'!' if self.truth_state.get('maintenance_required', 0.0) > 0.5 else 'ok'}"
        )
