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


def test_randomized_reset_produces_varied_initial_fouling():
    """Over 20 resets with randomize=True, initial fouling must vary."""
    env = ConicalCascadeEnv(
        config_path="configs/default.yaml",
        seed=42,
        randomize_on_reset=True,
    )
    fouling_values = []
    for i in range(20):
        env.reset(seed=i)
        fouling_values.append(env._state.get("fouling_frac_s1", 0.0))

    std = float(np.std(fouling_values))
    assert std > 0.01, (
        f"Fouling std across 20 resets must be > 0.01 (got {std:.4f}). "
        "Check randomize_on_reset logic."
    )
    for v in fouling_values:
        assert 0.0 <= v <= 0.30, f"Initial fouling {v:.4f} outside [0, 0.30]"


def test_deterministic_reset_when_randomize_false():
    """With randomize_on_reset=False (default), reset must always start at zero fouling."""
    env = ConicalCascadeEnv(config_path="configs/default.yaml", seed=0)
    for i in range(5):
        env.reset(seed=i)
        assert env._state.get("fouling_frac_s1", -1.0) == 0.0, \
            "fouling_frac_s1 must be 0.0 on deterministic reset"


def test_training_script_smoke_1000_steps():
    """Training stack (CCE + ShieldedEnv + VecNormalize + PPO) must run 1000 steps without error."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from hydrion.wrappers.shielded_env import ShieldedEnv
    from hydrion.safety.shield import SafetyConfig
    from hydrion.train_ppo_cce import _CCE_SAFETY_CFG, make_env

    vec_env = DummyVecEnv([make_env(seed=7)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO("MlpPolicy", vec_env, n_steps=256, batch_size=64, seed=7, verbose=0)
    model.learn(total_timesteps=1000)
    vec_env.close()
    # Reaching here means the full training stack is functional


def test_api_run_ppo_cce_falls_back_to_random_if_no_model(tmp_path, monkeypatch):
    """
    When policy_type='ppo_cce' but model files are absent,
    endpoint must not crash — it falls back to random and returns a run_id.
    """
    from fastapi.testclient import TestClient
    import hydrion.service.app as app_module
    from hydrion.service.app import app

    monkeypatch.setattr(app_module, "_PPO_CCE_MODEL_PATH",   str(tmp_path / "missing.zip"))
    monkeypatch.setattr(app_module, "_PPO_CCE_VECNORM_PATH", str(tmp_path / "missing.pkl"))
    monkeypatch.setattr(app_module, "_ppo_cce_model",    None)
    monkeypatch.setattr(app_module, "_ppo_cce_vec_norm", None)

    client = TestClient(app)
    resp = client.post("/api/run", json={
        "policy_type": "ppo_cce",
        "seed": 0,
        "config_name": "default.yaml",
        "max_steps": 5,
        "noise_enabled": False,
    })
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    assert "run_id" in resp.json()
    assert app_module._ppo_cce_model is None, "model singleton must remain unset when files are absent"


def test_api_run_ppo_cce_happy_path_with_mock_model(tmp_path, monkeypatch):
    """
    When policy_type='ppo_cce' and model files exist (mocked), endpoint must:
    - use the PPO branch (not random)
    - return 200 with run_id
    - populate the model singleton
    """
    import numpy as np
    from unittest.mock import MagicMock
    from fastapi.testclient import TestClient
    from hydrion.service.app import app
    import hydrion.service.app as app_module

    # Build a minimal mock that satisfies _load_ppo_cce's True path.
    # We monkeypatch _load_ppo_cce directly to return True and set the globals,
    # avoiding the need for real model files.
    mock_model = MagicMock()
    mock_model.predict.return_value = (np.array([[0.5, 0.5, 0.0, 0.8]]), None)

    mock_vecnorm = MagicMock()
    mock_vecnorm.normalize_obs.side_effect = lambda x: x  # identity

    def _fake_load():
        app_module._ppo_cce_model = mock_model
        app_module._ppo_cce_vec_norm = mock_vecnorm
        return True

    monkeypatch.setattr(app_module, "_load_ppo_cce", _fake_load)
    # Reset singletons so _fake_load is called
    monkeypatch.setattr(app_module, "_ppo_cce_model", None)
    monkeypatch.setattr(app_module, "_ppo_cce_vec_norm", None)

    client = TestClient(app)
    resp = client.post("/api/run", json={
        "policy_type": "ppo_cce",
        "seed": 0,
        "config_name": "default.yaml",
        "max_steps": 5,
        "noise_enabled": False,
    })
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    assert "run_id" in resp.json()
    # Confirm PPO was actually called (not random)
    assert mock_model.predict.called, "model.predict must be called when ppo_cce loads successfully"
