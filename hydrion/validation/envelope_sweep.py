# hydrion/validation/envelope_sweep.py
"""
HydrOS Validation Protocol v2 — Envelope Sweep.

Sweep action envelope (valve, pump, backflush, voltage) from config;
record obs/reward/info; assert observations and key metrics stay within
expected bounds. Tests stability and safety envelope.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from hydrion.env import HydrionEnv


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def run_envelope_sweep(
    config_path: str = "configs/default.yaml",
    validation_config_path: Optional[str] = None,
    scenario_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Sweep action space per validation config; check obs and truth_state bounds.

    Validation config keys (optional):
      envelope_sweep:
        seed: int
        steps_per_action: int   # steps to run at each action point
        grid:
          valve: [low, high, num] or [v1, v2, ...]
          pump: ...
          backflush: ...
          node_voltage: ...
        # or points: [[v,p,bf,nv], ...] for explicit list
        bounds:
          flow: [min, max]
          pressure: [min, max]
          clog: [min, max]
          obs_all: [min, max]
        fail_on_nan: bool
    """
    cfg_path = validation_config_path or "configs/validation/envelope_sweep.yaml"
    if not Path(cfg_path).exists():
        cfg_path = None
    val_cfg: Dict[str, Any] = load_config(cfg_path) if cfg_path else {}
    sweep_cfg = val_cfg.get("envelope_sweep") or val_cfg or {}

    scenario: Dict[str, Any] = {}
    if scenario_path and Path(scenario_path).exists():
        scenario = load_config(scenario_path)
        sweep_cfg = {**sweep_cfg, **{k: v for k, v in scenario.items() if k not in ("name", "description")}}

    seed = int(sweep_cfg.get("seed", 0))
    steps_per_action = int(sweep_cfg.get("steps_per_action", 5))
    fail_on_nan = bool(sweep_cfg.get("fail_on_nan", True))
    bounds = sweep_cfg.get("bounds") or {}
    flow_bounds = bounds.get("flow", [0.0, 1.0])
    pressure_bounds = bounds.get("pressure", [0.0, 1.0])
    clog_bounds = bounds.get("clog", [0.0, 1.0])
    obs_bounds = bounds.get("obs_all", [0.0, 1.0])

    # Build action grid
    points = sweep_cfg.get("points")
    if points is not None:
        actions_list = [np.array(p, dtype=np.float32) for p in points]
    else:
        grid = sweep_cfg.get("grid") or {}
        def expand(spec, default_lo=0.0, default_hi=1.0, default_n=3):
            if spec is None:
                return np.linspace(default_lo, default_hi, default_n)
            if isinstance(spec, list):
                if len(spec) == 3:
                    return np.linspace(float(spec[0]), float(spec[1]), int(spec[2]))
                return np.array(spec, dtype=np.float32)
            return np.linspace(default_lo, default_hi, default_n)

        v_vals = expand(grid.get("valve"))
        p_vals = expand(grid.get("pump"))
        bf_vals = expand(grid.get("backflush"), 0.0, 0.5, 2)
        nv_vals = expand(grid.get("node_voltage"))
        actions_list = []
        for v in v_vals:
            for p in p_vals:
                for bf in bf_vals:
                    for nv in nv_vals:
                        actions_list.append(np.array([v, p, bf, nv], dtype=np.float32))

    env = HydrionEnv(config_path=config_path)
    obs, _ = env.reset(seed=seed)

    results: List[Dict[str, Any]] = []
    all_passed = True

    for idx, action in enumerate(actions_list):
        env.reset(seed=seed + idx + 1)
        last_obs = None
        last_truth = None
        step_ok = True
        for _ in range(steps_per_action):
            obs, reward, term, trunc, info = env.step(action)
            last_obs = obs
            last_truth = dict(getattr(env, "truth_state", {}))
            if fail_on_nan and (not np.isfinite(obs).all() or not np.isfinite(reward)):
                step_ok = False
                all_passed = False
                break
            if term or trunc:
                break

        if last_truth is None:
            continue
        flow = float(last_truth.get("flow", 0.0))
        pressure = float(last_truth.get("pressure", 0.0))
        clog = float(last_truth.get("clog", 0.0))
        violations = []
        if flow < flow_bounds[0] or flow > flow_bounds[1]:
            violations.append(f"flow={flow:.4f}")
        if pressure < pressure_bounds[0] or pressure > pressure_bounds[1]:
            violations.append(f"pressure={pressure:.4f}")
        if clog < clog_bounds[0] or clog > clog_bounds[1]:
            violations.append(f"clog={clog:.4f}")
        if last_obs is not None and (np.any(last_obs < obs_bounds[0]) or np.any(last_obs > obs_bounds[1])):
            violations.append("obs_out_of_bounds")
        if violations:
            step_ok = False
            all_passed = False

        results.append({
            "action_index": idx,
            "action": action.tolist(),
            "steps_run": steps_per_action,
            "passed": step_ok,
            "violations": violations,
            "flow": flow,
            "pressure": pressure,
            "clog": clog,
        })

    summary = {
        "config_path": config_path,
        "validation_config_path": validation_config_path,
        "env_seed": seed,
        "all_passed": all_passed,
        "num_points": len(actions_list),
        "results": results,
    }
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description="HydrOS Envelope Sweep (Validation Protocol v2)")
    parser.add_argument("--config", default="configs/default.yaml", help="Hydrion config")
    parser.add_argument("--validation-config", default=None, help="Validation config YAML")
    parser.add_argument("--scenario", default=None, help="Scenario YAML (optional)")
    parser.add_argument("--output", default=None, help="Output results YAML")
    args = parser.parse_args()
    summary = run_envelope_sweep(
        config_path=args.config,
        validation_config_path=args.validation_config,
        scenario_path=args.scenario,
        output_path=args.output,
    )
    print("Envelope sweep all_passed:", summary["all_passed"])
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    exit(main())
