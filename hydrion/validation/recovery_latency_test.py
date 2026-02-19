# hydrion/validation/recovery_latency_test.py
"""
HydrOS Validation Protocol v2 — Recovery Latency Test.

Apply a configurable disturbance (e.g. action step or high-load segment),
then measure steps or time until a chosen metric returns within threshold.
Config-driven: disturbance definition, target metric, threshold, max_steps.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from hydrion.env import HydrionEnv


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def run_recovery_latency_test(
    config_path: str = "configs/default.yaml",
    validation_config_path: Optional[str] = None,
    scenario_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run recovery latency test: disturb then measure steps to recovery.

    Validation config keys (optional):
      recovery_latency:
        seed: int
        disturbance:
          type: "action_hold" | "high_clog_action"
          steps: int
          action: [v, p, bf, nv]  # for action_hold
        recovery:
          metric: "flow" | "pressure" | "clog"
          target: float           # e.g. 0.5 for flow
          threshold: float        # abs difference to consider "recovered"
          max_steps: int
        post_disturbance_action: [v, p, bf, nv] | null  # null = neutral 0.5,0.5,0,0.5
    """
    cfg_path = validation_config_path or "configs/validation/recovery_latency.yaml"
    if not Path(cfg_path).exists():
        cfg_path = None
    val_cfg: Dict[str, Any] = load_config(cfg_path) if cfg_path else {}
    rl_cfg = val_cfg.get("recovery_latency") or val_cfg or {}

    seed = int(rl_cfg.get("seed", 0))
    dist = rl_cfg.get("disturbance") or {}
    dist_type = str(dist.get("type", "action_hold"))
    dist_steps = int(dist.get("steps", 20))
    dist_action = dist.get("action")
    if dist_action is None:
        dist_action = [0.2, 0.9, 0.0, 0.5]
    dist_action = np.array(dist_action, dtype=np.float32)

    rec = rl_cfg.get("recovery") or {}
    metric = str(rec.get("metric", "flow"))
    target = float(rec.get("target", 0.5))
    threshold = float(rec.get("threshold", 0.1))
    max_steps = int(rec.get("max_steps", 500))

    post_action = rl_cfg.get("post_disturbance_action")
    if post_action is None:
        post_action = np.array([0.5, 0.5, 0.0, 0.5], dtype=np.float32)
    else:
        post_action = np.array(post_action, dtype=np.float32)

    env = HydrionEnv(config_path=config_path)
    obs, _ = env.reset(seed=seed)

    # Apply disturbance
    for _ in range(dist_steps):
        obs, _, term, trunc, _ = env.step(dist_action)
        if term or trunc:
            env.reset(seed=seed + 1)

    def get_metric():
        truth = getattr(env, "truth_state", {})
        if metric == "flow":
            return float(truth.get("flow", 0.0))
        if metric == "pressure":
            return float(truth.get("pressure", 0.0))
        if metric == "clog":
            return float(truth.get("clog", 0.0))
        return 0.0

    # Recovery phase
    steps_to_recovery: Optional[int] = None
    for s in range(max_steps):
        obs, _, term, trunc, _ = env.step(post_action)
        val = get_metric()
        if abs(val - target) <= threshold:
            steps_to_recovery = s
            break
        if term or trunc:
            break

    all_passed = steps_to_recovery is not None
    summary = {
        "config_path": config_path,
        "validation_config_path": validation_config_path,
        "env_seed": seed,
        "all_passed": all_passed,
        "steps_to_recovery": steps_to_recovery,
        "max_steps": max_steps,
        "metric": metric,
        "target": target,
        "threshold": threshold,
        "disturbance_type": dist_type,
        "disturbance_steps": dist_steps,
    }
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description="HydrOS Recovery Latency Test (Validation Protocol v2)")
    parser.add_argument("--config", default="configs/default.yaml", help="Hydrion config")
    parser.add_argument("--validation-config", default=None, help="Validation config YAML")
    parser.add_argument("--scenario", default=None, help="Scenario YAML (optional)")
    parser.add_argument("--output", default=None, help="Output results YAML")
    args = parser.parse_args()
    summary = run_recovery_latency_test(
        config_path=args.config,
        validation_config_path=args.validation_config,
        scenario_path=args.scenario,
        output_path=args.output,
    )
    print("Recovery latency all_passed:", summary["all_passed"], "steps_to_recovery:", summary.get("steps_to_recovery"))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    exit(main())
