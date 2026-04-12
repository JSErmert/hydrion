# hydrion/scenarios/runner.py
"""
HydrOS Scenario Runner — first execution layer.

Responsibilities:
  1. Load a ScenarioDefinition from YAML or dict
  2. Apply initial state to a HydrionEnv
  3. Compute per-step inputs (flow profile, particle density, disturbances)
  4. Inject inputs and step Hydrion
  5. Record per-step truth/sensor state
  6. Emit event markers on state transitions

Architecture contract:
  - Runner is stateless between runs (call run() any number of times)
  - truth_state separation is respected: physics writes truth, runner reads it
  - C_in is the one controlled input channel; it is set before each step as
    the scenario's external particle concentration signal
  - Initial fouling is applied via the clogging model's internal state,
    mirroring the _force_fouling_for_testing() pattern used in validation

Usage:
    from hydrion.env import HydrionEnv
    from hydrion.scenarios.runner import ScenarioRunner, load_scenario

    env = HydrionEnv(config_path="configs/default.yaml", auto_reset=False)
    scenario = load_scenario("hydrion/scenarios/examples/baseline_nominal.yaml")
    runner = ScenarioRunner(env)
    history = runner.run(scenario)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from .disturbances import (
    apply_disturbances_to_flow,
    apply_disturbances_to_particles,
    get_active_disturbances,
)
from .markers import detect_runtime_markers, update_prev_flags
from .profiles import compute_flow_at_time, compute_particle_density_at_time
from .types import (
    BackflushEvent,
    DisturbanceEvent,
    FlowProfile,
    InitialStateConfig,
    ParticleProfile,
    ScenarioDefinition,
    ScenarioEventMarker,
    ScenarioExecutionHistory,
    ScenarioStepRecord,
)

# Hard ceiling from locked system constraints
_Q_MAX_LMIN: float = 20.0


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_scenario(path: str | Path) -> ScenarioDefinition:
    """Load a ScenarioDefinition from a YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return _parse_scenario(raw)


def _parse_scenario(raw: Dict[str, Any]) -> ScenarioDefinition:
    init_raw  = raw.get("initialState",   {}) or {}
    flow_raw  = raw.get("flowProfile",    {}) or {}
    part_raw  = raw.get("particleProfile", {}) or {}
    dist_raw  = raw.get("disturbances",   []) or []

    return ScenarioDefinition(
        id          = str(raw.get("id", "unnamed")),
        name        = str(raw.get("name", "Unnamed Scenario")),
        description = str(raw.get("description", "")),
        durationSec = float(raw.get("durationSec", 60.0)),
        dt          = float(raw.get("dt", 0.1)),
        seed        = int(raw["seed"]) if raw.get("seed") is not None else 42,
        initialState = InitialStateConfig(
            fouling_s1   = float(init_raw.get("fouling_s1",    0.0)),
            fouling_s2   = float(init_raw.get("fouling_s2",    0.0)),
            fouling_s3   = float(init_raw.get("fouling_s3",    0.0)),
            storage_fill = float(init_raw.get("storage_fill",  0.0)),
            pressure_bias= float(init_raw.get("pressure_bias", 0.0)),
            flow_bias    = float(init_raw.get("flow_bias",     0.0)),
        ),
        flowProfile = FlowProfile(
            type           = str(flow_raw.get("type", "constant")),
            baseFlowLmin   = float(flow_raw.get("baseFlowLmin",   13.5)),
            variability    = float(flow_raw.get("variability",    0.0)),
            burstAmplitude = float(flow_raw.get("burstAmplitude", 0.0)),
            burstFrequency = float(flow_raw.get("burstFrequency", 0.0)),
            transientEvents= list(flow_raw.get("transientEvents", []) or []),
        ),
        particleProfile = ParticleProfile(
            type             = str(part_raw.get("type", "fiber_dominant")),
            density          = float(part_raw.get("density",    0.30)),
            variability      = float(part_raw.get("variability", 0.0)),
            sizeDistribution = list(part_raw.get("sizeDistribution", []) or []),
        ),
        disturbances = [
            DisturbanceEvent(
                type      = str(d.get("type", "flow_spike")),
                time      = float(d.get("time", 0.0)),
                duration  = float(d.get("duration", 0.0)),
                intensity = float(d.get("intensity", 1.0)),
            )
            for d in dist_raw
        ],
        backflush_events = [
            BackflushEvent(
                time     = float(b.get("time", 0.0)),
                duration = float(b.get("duration", 2.0)),
                bf_cmd   = float(b.get("bf_cmd", 1.0)),
            )
            for b in (raw.get("backflush_events", []) or [])
        ],
    )


# ---------------------------------------------------------------------------
# Initial state application
# ---------------------------------------------------------------------------

def apply_initial_state(env: Any, initial: InitialStateConfig) -> None:
    """
    Write per-stage initial fouling into the clogging model and sync truth_state.

    Mirrors _force_fouling_for_testing() but supports per-stage values.
    Uses clogging model's configured component weights for correct
    cake/bridge/pore decomposition.

    Must be called AFTER env.reset() so that the base state is clean.
    """
    clog = env.clogging
    p    = clog.params

    stage_specs = [
        (1, initial.fouling_s1, p.cake_weight_s1, p.bridge_weight_s1, p.pore_weight_s1, p.Mc1_max),
        (2, initial.fouling_s2, p.cake_weight_s2, p.bridge_weight_s2, p.pore_weight_s2, p.Mc2_max),
        (3, initial.fouling_s3, p.cake_weight_s3, p.bridge_weight_s3, p.pore_weight_s3, p.Mc3_max),
    ]

    for i, ff, cake_w, bridge_w, pore_w, mc_max in stage_specs:
        ff = float(np.clip(ff, 0.0, 1.0))
        clog._state[f"cake_s{i}"]         = ff * cake_w
        clog._state[f"bridge_s{i}"]       = ff * bridge_w
        clog._state[f"pore_s{i}"]         = ff * pore_w
        clog._state[f"irreversible_s{i}"] = 0.0
        clog._state[f"fouling_frac_s{i}"] = ff
        clog._state[f"recoverable_s{i}"]  = ff
        clog._state[f"n{i}"]              = ff
        clog._state[f"Mc{i}"]             = ff * mc_max

    mesh_avg = (initial.fouling_s1 + initial.fouling_s2 + initial.fouling_s3) / 3.0
    clog._state["mesh_loading_avg"] = float(np.clip(mesh_avg, 0.0, 1.0))

    # Recompute capture_eff for the forced initial state
    cap = (
        p.capture_eff_baseline
        + p.capture_eff_gain * mesh_avg * (1.0 - mesh_avg) ** p.capture_eff_exponent
    )
    clog._state["capture_eff"] = float(np.clip(cap, p.capture_eff_floor, p.capture_eff_ceiling))

    # Sync clogging model's internal state into truth_state
    env.truth_state.update(clog._state)
    # Re-derive normalized channels (flow/pressure/clog/maintenance_required)
    env._update_normalized_state()


# ---------------------------------------------------------------------------
# Action builder
# ---------------------------------------------------------------------------

def _get_backflush_cmd(backflush_events: List[BackflushEvent], t: float) -> float:
    """Return bf_cmd for the first active backflush event at time t, else 0.0."""
    for ev in backflush_events:
        if ev.time <= t < ev.time + ev.duration:
            return float(ev.bf_cmd)
    return 0.0


def _build_action(flow_lmin: float, bf_cmd: float = 0.0, node_voltage: float = 0.5) -> np.ndarray:
    """
    Translate a target flow into a 4-element action vector.

    Mapping:
        valve_cmd        = 1.0  (fully open)
        pump_cmd         = flow_lmin / Q_MAX  (normalized, linear approx)
        bf_cmd           = caller-supplied (0.0 = normal, 1.0 = backflush)
        node_voltage_cmd = node_voltage
    """
    pump_cmd = float(np.clip(flow_lmin / _Q_MAX_LMIN, 0.0, 1.0))
    return np.array([1.0, pump_cmd, float(bf_cmd), node_voltage], dtype=np.float32)


# ---------------------------------------------------------------------------
# ScenarioRunner
# ---------------------------------------------------------------------------

class ScenarioRunner:
    """
    Stateless scenario execution adapter for HydrionEnv.

    Accepts a HydrionEnv at construction.  Calling run() is idempotent:
    each call resets the env to a known seed before executing.
    """

    def __init__(self, env: Any) -> None:
        self.env = env

    def run(self, scenario: ScenarioDefinition) -> ScenarioExecutionHistory:
        """
        Execute the scenario end-to-end and return a structured history.

        Execution order per step:
          1. compute flow from profile + disturbances
          2. compute particle density from profile + disturbances
          3. inject C_in into truth_state (external input channel)
          4. build action from target flow
          5. step Hydrion
          6. record step
          7. detect and emit markers
        """
        env      = self.env
        dt       = scenario.dt
        n_steps  = int(math.ceil(scenario.durationSec / dt))
        seed     = scenario.seed if scenario.seed is not None else 42

        # Seeded RNG for reproducible flow/particle variability
        rng = np.random.default_rng(seed)

        # Reset env and apply initial state
        env.reset(seed=seed)
        apply_initial_state(env, scenario.initialState)

        history = ScenarioExecutionHistory(
            scenarioId=scenario.id,
            dt=dt,
        )

        history.eventMarkers.append(ScenarioEventMarker(
            time=0.0,
            type="scenario_start",
            label=f"Scenario start: {scenario.name}",
            meta={"seed": seed, "durationSec": scenario.durationSec, "dt": dt},
        ))

        prev_flags: Dict[str, Any] = {
            "maintenance_required": 0,
            "bf_active": 0,
            "bypass_active": 0,
        }
        prev_active_disturbances: List[DisturbanceEvent] = []

        for step_idx in range(n_steps):
            t = step_idx * dt

            # ---- Compute scenario inputs ----
            base_flow    = compute_flow_at_time(scenario.flowProfile, t, rng)
            base_density = compute_particle_density_at_time(scenario.particleProfile, t, rng)

            active_dist  = get_active_disturbances(scenario.disturbances, t)
            flow         = apply_disturbances_to_flow(base_flow, active_dist)
            density      = apply_disturbances_to_particles(base_density, active_dist)

            # ---- Inject external particle concentration ----
            # C_in is the controlled input channel for scenario particle loading.
            # The particle model reads this from truth_state; we set it before step().
            env.truth_state["C_in"] = float(density)

            # ---- Step Hydrion ----
            bf_cmd = _get_backflush_cmd(scenario.backflush_events, t)
            action = _build_action(flow, bf_cmd=bf_cmd)
            obs, reward, terminated, truncated, info = env.step(action)

            # ---- Collect state ----
            truth  = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                      for k, v in env.truth_state.items()}
            sensor = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                      for k, v in env.sensor_state.items()}

            # Extract particle_streams before building truthState.
            # particle_streams is a non-numeric nested dict — kept separate to
            # preserve truthState as Record<string, number> in TypeScript.
            particle_streams = truth.pop("particle_streams", None)

            history.steps.append(ScenarioStepRecord(
                t=round(t, 6),
                stepIndex=step_idx,
                scenarioInputs={
                    "flowLmin": round(flow, 4),
                    "particleDensity": round(density, 4),
                    "activeDisturbances": [
                        {"type": d.type, "intensity": d.intensity} for d in active_dist
                    ],
                },
                truthState=truth,
                sensorState=sensor,
                reward=float(reward),
                done=bool(terminated or truncated),
                info=info,
                particle_streams=particle_streams,
            ))

            # ---- Detect markers ----
            new_markers = detect_runtime_markers(
                t=t,
                prev_flags=prev_flags,
                truth_state=env.truth_state,
                prev_active_disturbances=prev_active_disturbances,
                current_active_disturbances=active_dist,
            )
            history.eventMarkers.extend(new_markers)

            # ---- Advance flag snapshots ----
            update_prev_flags(prev_flags, env.truth_state)
            prev_active_disturbances = list(active_dist)

            if terminated or truncated:
                break

        final_t = history.steps[-1].t if history.steps else 0.0
        history.eventMarkers.append(ScenarioEventMarker(
            time=final_t,
            type="scenario_end",
            label=f"Scenario end: {scenario.name}",
            meta={"steps_executed": len(history.steps)},
        ))

        return history
