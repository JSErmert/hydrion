# tests/test_conical_cascade_env.py
import numpy as np
import pytest
from hydrion.environments.conical_cascade_env import ConicalCascadeEnv


def make_env() -> ConicalCascadeEnv:
    return ConicalCascadeEnv(config_path="configs/default.yaml", seed=0)


def test_accumulation_fields_in_truth_state():
    """storage_fill and channel_fill_s1/s2/s3 must appear in truth_state after step."""
    env = make_env()
    env.reset(seed=0)
    action = np.array([0.5, 0.5, 0.0, 0.8], dtype=np.float32)
    env.step(action)
    s = env._state
    assert "storage_fill"    in s, "storage_fill missing from state"
    assert "channel_fill_s1" in s, "channel_fill_s1 missing"
    assert "channel_fill_s2" in s, "channel_fill_s2 missing"
    assert "channel_fill_s3" in s, "channel_fill_s3 missing"


def test_accumulation_increases_over_steps():
    """Channel fill must grow when particles are being captured (no flush).

    With cascade attenuation, S1 dominates (highest eta). S2 and S3 receive
    residual concentration after upstream capture.
    """
    env = make_env()
    env.reset(seed=0)
    action = np.array([0.5, 0.5, 0.0, 0.8], dtype=np.float32)
    for _ in range(20):
        env.step(action)
    s = env._state
    assert s["channel_fill_s1"] > 0.0, "S1 channel fill must be > 0 (dominant capture stage)"
    total_fill = s["channel_fill_s1"] + s["channel_fill_s2"] + s["channel_fill_s3"]
    assert total_fill > 0.0, "total channel fill must be > 0 after 20 capture steps"


def test_flush_reduces_channel_fill():
    """bf_cmd > 0.5 must drain channels and increase storage_fill.

    S1 is the dominant accumulation stage (~99.65% PET capture efficiency).
    S3 receives near-zero concentration due to cascade attenuation — correct physics.
    """
    env = make_env()
    env.reset(seed=0)
    capture = np.array([0.5, 0.5, 0.0, 0.8], dtype=np.float32)
    flush   = np.array([0.5, 0.5, 1.0, 0.8], dtype=np.float32)
    for _ in range(30):
        env.step(capture)
    pre_fill    = env._state["channel_fill_s1"]   # S1 dominates — use it
    pre_storage = env._state["storage_fill"]
    for _ in range(10):
        env.step(flush)
    assert env._state["channel_fill_s1"] < pre_fill,   "flush must drain channel_fill_s1"
    assert env._state["storage_fill"]    > pre_storage, "flush must increase storage_fill"


def test_per_stage_eta_in_truth_state():
    """eta_s1, eta_s2, eta_s3 must be in state and in [0,1]."""
    env = make_env()
    env.reset(seed=0)
    action = np.array([0.5, 0.5, 0.0, 0.8], dtype=np.float32)
    for _ in range(5):
        env.step(action)
    s = env._state
    for key in ("eta_s1", "eta_s2", "eta_s3"):
        assert key in s, f"{key} missing from state"
        assert 0.0 <= s[key] <= 1.0, f"{key} out of [0,1]: {s[key]}"


def test_stage_hierarchy_s3_dominates():
    """S3 capture efficiency must exceed S1 on average at nominal flow."""
    env = make_env()
    env.reset(seed=0)
    action = np.array([0.5, 0.5, 0.0, 0.8], dtype=np.float32)
    eta_s1_samples, eta_s3_samples = [], []
    for _ in range(20):
        env.step(action)
        eta_s1_samples.append(env._state["eta_s1"])
        eta_s3_samples.append(env._state["eta_s3"])
    assert np.mean(eta_s3_samples) >= np.mean(eta_s1_samples), \
        "S3 mean efficiency must >= S1 (asymmetric stage design)"


def test_v_crit_per_stage_in_truth_state():
    """v_crit_s1, v_crit_s2, v_crit_s3 must be in state after step."""
    env = make_env()
    env.reset(seed=0)
    env.step(np.array([0.5, 0.5, 0.0, 0.8], dtype=np.float32))
    for key in ("v_crit_s1", "v_crit_s2", "v_crit_s3"):
        assert key in env._state, f"{key} missing"
