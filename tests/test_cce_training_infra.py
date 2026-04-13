# tests/test_cce_training_infra.py
import numpy as np
import pytest
from hydrion.environments.conical_cascade_env import ConicalCascadeEnv
from hydrion.wrappers.shielded_env import ShieldedEnv
from hydrion.safety.shield import SafetyConfig


def make_env() -> ConicalCascadeEnv:
    return ConicalCascadeEnv(config_path="configs/default.yaml", seed=0)


def test_shield_alias_keys_present_after_step():
    """flow, pressure, clog must be in _state (normalized) after step.
    pressure must be > 0 when pump/valve are active (dp_total_pa > 0).
    """
    env = make_env()
    env.reset(seed=0)
    # Active pump + valve should produce nonzero pressure
    env.step(np.array([0.8, 0.8, 0.0, 0.8], dtype=np.float32))
    for key in ("flow", "pressure", "clog"):
        assert key in env._state, f"shield alias '{key}' missing from _state"
        assert 0.0 <= env._state[key] <= 1.0, f"alias '{key}' out of [0,1]: {env._state[key]}"
    # Pressure must be non-zero when the pump is on — if it's 0.0, the hydraulics key is wrong
    assert env._state["pressure"] > 0.0, (
        f"pressure alias is 0.0 — check that dp_total_pa is written by hydraulics. "
        f"Full state keys: {sorted(env._state.keys())}"
    )


def test_shielded_env_wraps_cce_without_error():
    """ShieldedEnv must wrap CCE and step without raising exceptions."""
    raw_env = make_env()
    cfg = SafetyConfig(
        max_pressure_soft=0.75,
        max_pressure_hard=1.00,
        terminate_on_hard_violation=True,
    )
    env = ShieldedEnv(raw_env, cfg=cfg)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (12,)
    action = np.array([0.5, 0.5, 0.0, 0.8], dtype=np.float32)
    obs2, reward, terminated, truncated, info = env.step(action)
    assert obs2.shape == (12,)
    assert "safety" in info
    assert isinstance(reward, float)


def test_reward_increases_with_flow():
    """Higher processed flow (same capture) must yield higher reward."""
    env = make_env()
    env.reset(seed=0)

    # Low flow: valve nearly closed
    env._state["q_processed_lmin"] = 2.0
    env._state["eta_cascade"]      = 0.80
    env._state["dp_total_pa"]      = 20_000.0
    env._state["voltage_norm"]     = 0.5
    r_low = env._reward()

    # High flow: valve open
    env._state["q_processed_lmin"] = 15.0
    env._state["eta_cascade"]      = 0.80
    env._state["dp_total_pa"]      = 20_000.0
    env._state["voltage_norm"]     = 0.5
    r_high = env._reward()

    assert r_high > r_low, (
        f"High-flow reward ({r_high:.4f}) must exceed low-flow reward ({r_low:.4f}) "
        "for equal capture efficiency"
    )
