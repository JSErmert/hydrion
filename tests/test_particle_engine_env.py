# tests/test_particle_engine_env.py
import json
import numpy as np
import pytest
from hydrion.environments.conical_cascade_env import ConicalCascadeEnv


@pytest.fixture
def env():
    e = ConicalCascadeEnv(config_path="configs/default.yaml", seed=0)
    e.reset(seed=0)
    return e


def test_particle_streams_in_state_after_step(env):
    """After step(), _state must contain 'particle_streams' with s1/s2/s3 keys."""
    action = np.array([0.5, 0.5, 0.0, 0.8], dtype=np.float32)
    env.step(action)
    assert "particle_streams" in env._state, "particle_streams missing from _state"
    ps = env._state["particle_streams"]
    assert "s1" in ps and "s2" in ps and "s3" in ps


def test_particle_streams_have_required_fields(env):
    """Each particle point must have x_norm, r_norm, status, species."""
    action = np.array([0.5, 0.5, 0.0, 0.8], dtype=np.float32)
    env.step(action)
    ps = env._state["particle_streams"]
    for key in ("s1", "s2", "s3"):
        for pt in ps[key]:
            assert "x_norm"  in pt, f"{key} point missing x_norm"
            assert "r_norm"  in pt, f"{key} point missing r_norm"
            assert "status"  in pt, f"{key} point missing status"
            assert "species" in pt, f"{key} point missing species"
            assert pt["status"]  in ("captured", "passed", "in_transit")
            assert pt["species"] in ("PP", "PE", "PET")


def test_particle_streams_json_serializable(env):
    """particle_streams must be JSON-serializable (required for API payload)."""
    action = np.array([0.5, 0.5, 0.0, 0.8], dtype=np.float32)
    env.step(action)
    ps = env._state["particle_streams"]
    json_str = json.dumps(ps)
    assert len(json_str) > 0


def test_truth_state_property(env):
    """truth_state property must return _state (same dict object)."""
    assert env.truth_state is env._state


def test_sensor_state_property(env):
    """sensor_state property must return an empty dict."""
    assert isinstance(env.sensor_state, dict)
    assert len(env.sensor_state) == 0


def test_cascade_routing_s2_receives_only_passed(env):
    """
    Cascade routing: s2 count <= s1 count.
    At full voltage + low flow, all S1 particles capture → s2 is empty.
    """
    # Low flow + full voltage: maximize S1 capture
    action = np.array([0.3, 0.3, 0.0, 1.0], dtype=np.float32)
    env.step(action)
    ps = env._state["particle_streams"]
    n_s1 = len(ps["s1"])
    n_s2 = len(ps["s2"])
    # Monotone count reduction invariant
    assert n_s2 <= n_s1, f"s2 ({n_s2}) cannot have more particles than s1 ({n_s1})"
    # Strong check: particles captured in s1 cannot appear in s2
    captured_in_s1 = sum(1 for pt in ps["s1"] if pt["status"] == "captured")
    if captured_in_s1 == n_s1 and n_s1 > 0:
        assert n_s2 == 0, (
            f"All {n_s1} s1 particles captured; s2 must be empty but has {n_s2}"
        )


def test_backflush_no_captures_in_streams(env):
    """During backflush, all particle_streams entries must have status='passed'.
    Verifies both that particles appear AND that none are captured."""
    action = np.array([0.5, 0.5, 1.0, 0.8], dtype=np.float32)  # bf_cmd=1.0
    env.step(action)
    ps = env._state["particle_streams"]
    total_particles = sum(len(ps[k]) for k in ("s1", "s2", "s3"))
    assert total_particles > 0, "Backflush must produce at least one particle in streams"
    for key in ("s1", "s2", "s3"):
        for pt in ps[key]:
            assert pt["status"] == "passed", (
                f"No captures during backflush. Got {pt['status']} in {key}"
            )
