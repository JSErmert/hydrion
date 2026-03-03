from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple

import numpy as np
import gymnasium as gym


# ---------------------------------------------------------------------------
# Safety configuration
# ---------------------------------------------------------------------------

@dataclass
class SafetyConfig:
    """
    Configuration for the Safe RL Shield.

    All thresholds are expressed in *normalized* units (0–1),
    consistent with HydrionEnv's `flow`, `pressure`, and `clog` fields.
    """

    # Soft and hard pressure limits (normalized)
    max_pressure_soft: float = 0.85
    max_pressure_hard: float = 1.05  # terminate if exceeded

    # Mesh loading / clog limits (normalized)
    max_clog_soft: float = 0.80
    max_clog_hard: float = 0.98

    # Minimum flow (normalized) when pump is high – used to detect blockage
    min_flow_when_pump_high: float = 0.05
    pump_high_threshold: float = 0.75

    # Action rate limiting
    max_action_delta: float = 0.20  # max per-step change per component

    # Backflush safety (keep from saturating at 1.0 for long periods)
    max_backflush_cmd: float = 0.9

    # Penalties
    pressure_penalty_scale: float = 2.0
    clog_penalty_scale: float = 1.0
    blockage_penalty: float = 1.0
    hard_violation_penalty: float = 10.0  # added when we terminate

    # If True, terminate episode on hard violations
    terminate_on_hard_violation: bool = True


# ---------------------------------------------------------------------------
# Shield core logic
# ---------------------------------------------------------------------------

class SafeRLShield:
    """
    Safe RL Shield for HydrionEnv.

    Responsibilities:
    - Pre-process actions to keep them in a safe envelope
      (rate limiting, pump/valve sanity, backflush bounds).
    - Post-process transitions to:
        * detect safety violations
        * apply shaping penalties
        * optionally terminate the episode

    It works with *normalized* fields in `env.state`:
        flow, pressure, clog
    and the command channels:
        valve_cmd, pump_cmd, bf_cmd
    """

    def __init__(self, cfg: Optional[SafetyConfig] = None) -> None:
        self.cfg = cfg or SafetyConfig()
        self.last_action: Optional[np.ndarray] = None

    # -------------------- lifecycle -------------------- #
    def reset(self) -> None:
        self.last_action = None

    # -------------------- pre-step --------------------- #
    def pre_action(self, action: np.ndarray, env: gym.Env) -> np.ndarray:
        """
        Filter the raw action before it reaches the environment.

        The signature now accepts the full ``env`` object so that safety
        decisions may inspect ``truth_state`` or other attributes. A
        fallback is provided by the wrapper if ``env`` lacks the
        expected fields.

        Returns a "safe" action vector.
        """
        c = self.cfg

        # Always work on a copy
        a = np.array(action, dtype=np.float32).copy()
        a = np.clip(a, 0.0, 1.0)

        # 1) Rate limiting (smooth control)
        if self.last_action is not None:
            delta = a - self.last_action
            delta = np.clip(delta, -c.max_action_delta, c.max_action_delta)
            a = self.last_action + delta

        # 2) Backflush clamp
        #    Avoid extreme, sustained backflush unless the policy really wants it
        if a.shape[0] >= 3:
            a[2] = np.clip(a[2], 0.0, c.max_backflush_cmd)

        # 3) Pump vs valve sanity:
        #    If valve is nearly closed, strongly limit pump command.
        if a.shape[0] >= 2:
            valve = float(a[0])
            pump = float(a[1])

            if valve < 0.20 and pump > 0.5:
                # Linearly reduce pump as valve closes below 0.2
                scale = max(valve / 0.20, 0.1)
                a[1] = pump * scale

        self.last_action = a.copy()
        return a

    # -------------------- post-step -------------------- #
    def post_step(
        self,
        env: gym.Env,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Adjust reward and termination based on safety constraints.

        Returns:
            (new_reward, new_terminated, safety_info)
        """
        c = self.cfg
        # use truth_state if available; otherwise fall back gracefully
        s: Dict[str, Any] = getattr(env, "truth_state", getattr(env, "state", {}))

        flow = float(s.get("flow", 0.0))
        pressure = float(s.get("pressure", 0.0))
        clog = float(s.get("clog", 0.0))
        pump_cmd = float(s.get("pump_cmd", 0.0))
        bf_cmd = float(s.get("bf_cmd", 0.0))

        safety_info: Dict[str, Any] = {
            "pressure": pressure,
            "clog": clog,
            "flow": flow,
            "pump_cmd": pump_cmd,
            "bf_cmd": bf_cmd,
            "soft_pressure_violation": False,
            "hard_pressure_violation": False,
            "soft_clog_violation": False,
            "hard_clog_violation": False,
            "blockage_violation": False,
            "penalty": 0.0,
            "config": asdict(self.cfg),
        }

        penalty = 0.0
        hard_violation = False

        # --- Pressure constraints ------------------------------------ #
        if pressure > c.max_pressure_soft:
            safety_info["soft_pressure_violation"] = True
            penalty += c.pressure_penalty_scale * max(
                0.0, pressure - c.max_pressure_soft
            )

        if pressure > c.max_pressure_hard:
            safety_info["hard_pressure_violation"] = True
            hard_violation = True

        # --- Clog constraints ---------------------------------------- #
        if clog > c.max_clog_soft:
            safety_info["soft_clog_violation"] = True
            penalty += c.clog_penalty_scale * max(0.0, clog - c.max_clog_soft)

        if clog > c.max_clog_hard:
            safety_info["hard_clog_violation"] = True
            hard_violation = True

        # --- Blockage detection -------------------------------------- #
        if (
            pump_cmd > c.pump_high_threshold
            and flow < c.min_flow_when_pump_high
        ):
            safety_info["blockage_violation"] = True
            penalty += c.blockage_penalty

        # Apply penalty
        reward_after = float(reward) - penalty
        safety_info["penalty"] = float(penalty)

        # Terminate on hard violations (optional)
        if hard_violation and c.terminate_on_hard_violation and not terminated:
            terminated = True
            reward_after -= c.hard_violation_penalty

        return reward_after, terminated, safety_info


# ---------------------------------------------------------------------------
# Gymnasium wrapper for stable-baselines3
# ---------------------------------------------------------------------------

# The canonical ShieldedEnv wrapper lives in ``hydrion.wrappers``.
# We intentionally do not re-export it here to avoid circular imports.