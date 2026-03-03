# hydrion/wrappers/shielded_env.py
from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from hydrion.safety.shield import SafeRLShield, SafetyConfig


class ShieldedEnv(gym.Wrapper):
    """
    Structural safety wrapper for Hydrion (Commit 4).

    Responsibilities:
    - Apply SafeRLShield.pre_action before stepping the env
    - Apply SafeRLShield.post_step after stepping
    - Log safety events into timestep.jsonl
    - Expose safety metadata via info["safety"]

    This wrapper is intentionally thin: all domain logic lives in SafeRLShield.
    """

    def __init__(self, env: gym.Env, cfg: Optional[SafetyConfig] = None):
        super().__init__(env)
        self.shield = SafeRLShield(cfg)

        # Preserve spaces for SB3 compatibility
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    # forward any other attributes to the underlying environment to avoid
    # losing important fields such as ``cfg`` or ``dt`` when wrapped.
    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.shield.reset()
        return obs, info

    def step(self, action):
        # Ensure numpy action
        proposed = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)

        # ------------------------------
        # Pre-action safety (shield now inspects the env itself)
        # ------------------------------
        safe_action = self.shield.pre_action(proposed, self.env)
        safe_action = np.clip(np.asarray(safe_action, dtype=np.float32), 0.0, 1.0)
        projected = not np.allclose(safe_action, proposed)

        # Step underlying env
        obs, reward, terminated, truncated, info = self.env.step(safe_action)

        # ------------------------------
        # Post-step safety evaluation
        # ------------------------------
        # Post-step safety evaluation
        reward, terminated, safety_info = self.shield.post_step(
            self.env,
            obs,
            reward,
            terminated,
        )

        # record whether the shield changed the action
        safety_info["shield_intervened"] = projected

        # Attach safety info to info dict
        info = dict(info) if info is not None else {}
        info["safety"] = safety_info

        # ------------------------------
        # Commit 4: timeline logging
        # ------------------------------
        logger = getattr(self.env, "logger", None)
        run_context = getattr(self.env, "run_context", None)
        steps = getattr(self.env, "steps", None)

        if logger is not None and run_context is not None and steps is not None:
            try:
                logger.log_step({
                    "event": "safety",
                    "run_id": run_context.run_id,
                    "timestep": int(steps),
                    "projected": bool(projected),
                    "soft_pressure_violation": safety_info.get("soft_pressure_violation", False),
                    "hard_pressure_violation": safety_info.get("hard_pressure_violation", False),
                    "soft_clog_violation": safety_info.get("soft_clog_violation", False),
                    "hard_clog_violation": safety_info.get("hard_clog_violation", False),
                    "blockage_violation": safety_info.get("blockage_violation", False),
                    "penalty": float(safety_info.get("penalty", 0.0)),
                    "terminated": bool(terminated),
                })
            except Exception:
                # Never block simulation due to logging
                pass

        return obs, reward, terminated, truncated, info
