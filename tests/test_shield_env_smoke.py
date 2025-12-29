import numpy as np

from hydrion.env import HydrionEnv
from hydrion.wrappers.shielded_env import ShieldedEnv
from hydrion.safety.shield import SafeRLShield, SafetyConfig


def test_shielded_env_runs_and_injects_info():
    env = HydrionEnv()

    # Force shield to always project by setting very low soft limits
    cfg = SafetyConfig(
        max_pressure_soft=0.0,
        max_pressure_hard=10.0,   # avoid termination
    )
    shield = SafeRLShield(cfg)

    wrapped = ShieldedEnv(env, cfg=cfg)

    obs, _ = wrapped.reset(seed=123)

    action = np.array([0.5, 0.9, 0.0, 0.5], dtype=np.float32)
    obs, reward, term, trunc, info = wrapped.step(action)

    assert "safety" in info
    assert isinstance(info["safety"], dict)
    assert "soft_pressure_violation" in info["safety"]
