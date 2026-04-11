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
    """Channel fill must grow when particles are being captured (no flush)."""
    env = make_env()
    env.reset(seed=0)
    action = np.array([0.5, 0.5, 0.0, 0.8], dtype=np.float32)
    for _ in range(20):
        env.step(action)
    s = env._state
    total_fill = s["channel_fill_s1"] + s["channel_fill_s2"] + s["channel_fill_s3"]
    assert total_fill > 0.0, "channel fill must be > 0 after 20 capture steps"


def test_flush_reduces_channel_fill():
    """bf_cmd > 0.5 must drain channels and increase storage_fill."""
    env = make_env()
    env.reset(seed=0)
    capture = np.array([0.5, 0.5, 0.0, 0.8], dtype=np.float32)
    flush   = np.array([0.5, 0.5, 1.0, 0.8], dtype=np.float32)
    for _ in range(30):
        env.step(capture)
    pre_fill = env._state["channel_fill_s3"]
    pre_storage = env._state["storage_fill"]
    for _ in range(10):
        env.step(flush)
    assert env._state["channel_fill_s3"] < pre_fill,   "flush must drain channel_fill_s3"
    assert env._state["storage_fill"]    > pre_storage, "flush must increase storage_fill"
