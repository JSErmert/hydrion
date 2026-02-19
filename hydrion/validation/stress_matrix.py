# hydrion/validation/stress_matrix.py
"""
HydrOS Validation Protocol v2 — Stress Matrix.

Config-driven stress testing: stability, safety envelope, and optional
coupled anomaly behavior via scenario references.
No modifications to physics, sensor fusion, or safety shield.
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


def apply_scenario_overrides(env: HydrionEnv, scenario: Dict[str, Any]) -> None:
    """
    Apply scenario overrides that only affect run conditions (e.g. seed).
    Does NOT inject into physics; only reset/seed and optional action bias.
    """
    # Seed is applied at reset; nothing else to mutate on env
    pass


def run_stress_matrix(
    config_path: str = "configs/default.yaml",
    validation_config_path: Optional[str] = None,
    scenario_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run stress matrix from validation config (and optional scenario YAML).

    Validation config keys (all optional):
      stress_matrix:
        num_episodes: int
        steps_per_episode: int | null  # null = use env max_steps
        seeds: [int] | null            # null = use config seed or 0
        action_mode: "random" | "extreme" | "sweep"
        extreme_action_fraction: float  # for extreme mode
        fail_on_nan: bool
        bounds:
          flow: [min, max]
          pressure: [min, max]
          clog: [min, max]
          obs_norm: [min, max]   # all obs in this range
    """
    cfg_path = validation_config_path or "configs/validation/stress_matrix.yaml"
    if not Path(cfg_path).exists():
        cfg_path = None
    val_cfg: Dict[str, Any] = load_config(cfg_path) if cfg_path else {}
    stress_cfg = val_cfg.get("stress_matrix") or val_cfg or {}

    scenario: Dict[str, Any] = {}
    if scenario_path and Path(scenario_path).exists():
        scenario = load_config(scenario_path)
        # Overlay scenario onto stress config (scenario takes precedence)
        stress_cfg = {**stress_cfg, **{k: v for k, v in scenario.items() if k not in ("name", "description")}}

    env = HydrionEnv(config_path=config_path)
    apply_scenario_overrides(env, scenario)

    num_episodes = int(stress_cfg.get("num_episodes", 3))
    steps_per_episode = stress_cfg.get("steps_per_episode")
    if steps_per_episode is not None:
        steps_per_episode = int(steps_per_episode)
    seeds = stress_cfg.get("seeds")
    if seeds is None:
        seeds = [env.run_context.seed] if num_episodes > 0 else []
    else:
        seeds = list(seeds)[:num_episodes]
    while len(seeds) < num_episodes:
        seeds.append(seeds[-1] + 1 if seeds else 0)

    action_mode = str(stress_cfg.get("action_mode", "random"))
    extreme_frac = float(stress_cfg.get("extreme_action_fraction", 0.3))
    fail_on_nan = bool(stress_cfg.get("fail_on_nan", True))
    bounds = stress_cfg.get("bounds") or {}
    flow_bounds = bounds.get("flow", [0.0, 1.0])
    pressure_bounds = bounds.get("pressure", [0.0, 1.0])
    clog_bounds = bounds.get("clog", [0.0, 1.0])
    obs_norm = bounds.get("obs_norm", [0.0, 1.0])

    results: List[Dict[str, Any]] = []
    all_passed = True

    for ep in range(num_episodes):
        seed = seeds[ep]
        obs, info = env.reset(seed=seed)
        max_steps = steps_per_episode if steps_per_episode is not None else env.max_steps
        steps_done = 0
        episode_ok = True
        nan_seen = False
        bound_violations: List[str] = []

        for _ in range(max_steps):
            if action_mode == "random":
                action = env.action_space.sample()
            elif action_mode == "extreme":
                u = np.random.rand(4).astype(np.float32)
                ext = np.random.rand(4) < extreme_frac
                action = np.where(ext, u, 0.5 * np.ones(4, dtype=np.float32))
            else:
                action = env.action_space.sample()

            obs, reward, term, trunc, info = env.step(action)
            steps_done += 1

            if not np.isfinite(obs).all() or not np.isfinite(reward):
                nan_seen = True
                if fail_on_nan:
                    episode_ok = False
                    break

            t = getattr(env, "truth_state", {})
            flow = float(t.get("flow", 0.0))
            pressure = float(t.get("pressure", 0.0))
            clog = float(t.get("clog", 0.0))
            if flow < flow_bounds[0] or flow > flow_bounds[1]:
                bound_violations.append(f"flow={flow:.4f}")
            if pressure < pressure_bounds[0] or pressure > pressure_bounds[1]:
                bound_violations.append(f"pressure={pressure:.4f}")
            if clog < clog_bounds[0] or clog > clog_bounds[1]:
                bound_violations.append(f"clog={clog:.4f}")
            if obs_norm and (np.any(obs < obs_norm[0]) or np.any(obs > obs_norm[1])):
                bound_violations.append("obs_out_of_norm")

            if term or trunc:
                break

        if nan_seen and fail_on_nan:
            episode_ok = False
        if bound_violations:
            episode_ok = False
        if not episode_ok:
            all_passed = False

        results.append({
            "episode": ep,
            "seed": seed,
            "steps": steps_done,
            "passed": episode_ok,
            "nan_seen": nan_seen,
            "bound_violations": bound_violations,
        })

    summary = {
        "config_path": config_path,
        "validation_config_path": validation_config_path,
        "env_seed": seeds[0] if seeds else None,
        "all_passed": all_passed,
        "num_episodes": num_episodes,
        "results": results,
    }
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description="HydrOS Stress Matrix (Validation Protocol v2)")
    parser.add_argument("--config", default="configs/default.yaml", help="Hydrion config")
    parser.add_argument("--validation-config", default=None, help="Validation config YAML")
    parser.add_argument("--scenario", default=None, help="Scenario YAML (optional)")
    parser.add_argument("--output", default=None, help="Output results YAML")
    args = parser.parse_args()
    summary = run_stress_matrix(
        config_path=args.config,
        validation_config_path=args.validation_config,
        scenario_path=args.scenario,
        output_path=args.output,
    )
    print("Stress matrix all_passed:", summary["all_passed"])
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    exit(main())
