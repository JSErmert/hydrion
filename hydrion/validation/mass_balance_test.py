# hydrion/validation/mass_balance_test.py
"""
HydrOS Validation Protocol v2 — Mass Balance Test.

Validates mass-balance integrity from env outputs only (no physics changes):
- Particle/concentration: C_out <= C_in, capture_eff in [0, 1]
- Optional flow consistency checks from truth_state
Config-driven tolerances and which quantities to check.
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


def run_mass_balance_test(
    config_path: str = "configs/default.yaml",
    validation_config_path: Optional[str] = None,
    scenario_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run mass balance checks over one or more episodes; config-driven.

    Validation config keys (optional):
      mass_balance:
        seed: int
        num_steps: int
        atol: float   # absolute tolerance for float comparisons
        checks:
          c_out_le_c_in: bool
          capture_eff_bounded: bool
          flow_non_negative: bool
    """
    cfg_path = validation_config_path or "configs/validation/mass_balance.yaml"
    if not Path(cfg_path).exists():
        cfg_path = None
    val_cfg: Dict[str, Any] = load_config(cfg_path) if cfg_path else {}
    mb_cfg = val_cfg.get("mass_balance") or val_cfg or {}

    seed = int(mb_cfg.get("seed", 0))
    num_steps = int(mb_cfg.get("num_steps", 200))
    atol = float(mb_cfg.get("atol", 1e-6))
    checks = mb_cfg.get("checks") or {}
    check_c_out_le_c_in = bool(checks.get("c_out_le_c_in", True))
    check_capture_eff_bounded = bool(checks.get("capture_eff_bounded", True))
    check_flow_non_negative = bool(checks.get("flow_non_negative", True))

    env = HydrionEnv(config_path=config_path)
    obs, _ = env.reset(seed=seed)

    violations: List[str] = []
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        truth = getattr(env, "truth_state", {})

        C_in = float(truth.get("C_in", 0.0))
        C_out = float(truth.get("C_out", 0.0))
        capture_eff = float(truth.get("particle_capture_eff", 0.0))
        Q_out = float(truth.get("Q_out_Lmin", 0.0))

        if check_c_out_le_c_in and C_out > C_in + atol:
            violations.append(f"step={step} C_out={C_out:.6f} > C_in={C_in:.6f}")
        if check_capture_eff_bounded and (capture_eff < 0.0 - atol or capture_eff > 1.0 + atol):
            violations.append(f"step={step} particle_capture_eff={capture_eff:.6f} not in [0,1]")
        if check_flow_non_negative and Q_out < 0.0 - atol:
            violations.append(f"step={step} Q_out_Lmin={Q_out:.6f} < 0")

        if term or trunc:
            env.reset(seed=seed + step + 1)

    all_passed = len(violations) == 0
    summary = {
        "config_path": config_path,
        "validation_config_path": validation_config_path,
        "env_seed": seed,
        "all_passed": all_passed,
        "num_steps": num_steps,
        "violations": violations,
        "checks_applied": {
            "c_out_le_c_in": check_c_out_le_c_in,
            "capture_eff_bounded": check_capture_eff_bounded,
            "flow_non_negative": check_flow_non_negative,
        },
    }
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description="HydrOS Mass Balance Test (Validation Protocol v2)")
    parser.add_argument("--config", default="configs/default.yaml", help="Hydrion config")
    parser.add_argument("--validation-config", default=None, help="Validation config YAML")
    parser.add_argument("--scenario", default=None, help="Scenario YAML (optional)")
    parser.add_argument("--output", default=None, help="Output results YAML")
    args = parser.parse_args()
    summary = run_mass_balance_test(
        config_path=args.config,
        validation_config_path=args.validation_config,
        scenario_path=args.scenario,
        output_path=args.output,
    )
    print("Mass balance all_passed:", summary["all_passed"])
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    exit(main())
