# tests/test_obs8_schema.py
"""
obs8_deployment_v1 schema validation tests — M8.

Validates:
- HydrionEnv(obs_schema="obs8_deployment_v1") returns 8D observation
- Channel order: actuator_feedback (0-3) then sensor_derived (4-7)
- Channel sources trace correctly to truth_state and sensor_state
- Values are bounded (no NaN, no Inf; nominal [0, 1] with sensor noise allowance)
- obs14_v1 behavior is unchanged (HydrionEnv() still returns 14D)
- CCE still returns 12D
- obs_schema validation rejects unknown schemas

Schema version: obs8_deployment_v1 (M8 — 2026-04-14)
"""
from __future__ import annotations

import numpy as np
import pytest

from hydrion.env import HydrionEnv


# ---------------------------------------------------------------------------
# T1 — obs8_deployment_v1 shape is (8,)
# ---------------------------------------------------------------------------

def test_obs8_shape():
    env = HydrionEnv(obs_schema="obs8_deployment_v1")
    obs, _ = env.reset()
    assert obs.shape == (8,), f"Expected obs shape (8,), got {obs.shape}"
    env.close()


def test_obs8_observation_space_shape():
    env = HydrionEnv(obs_schema="obs8_deployment_v1")
    assert env.observation_space.shape == (8,), (
        f"observation_space.shape must be (8,) for obs8_deployment_v1. "
        f"Got: {env.observation_space.shape}"
    )
    env.close()


def test_obs8_observation_space_matches_actual_obs():
    env = HydrionEnv(obs_schema="obs8_deployment_v1")
    obs, _ = env.reset()
    assert env.observation_space.shape == obs.shape, (
        f"observation_space.shape={env.observation_space.shape} "
        f"does not match actual obs.shape={obs.shape}"
    )
    env.close()


# ---------------------------------------------------------------------------
# T2 — Actuator feedback channels (obs8 indices 0–3 = obs14_v1 indices 6–9)
# ---------------------------------------------------------------------------

def test_obs8_index_0_is_valve_cmd():
    env = HydrionEnv(obs_schema="obs8_deployment_v1")
    env.reset(seed=42)
    action = np.array([0.3, 0.5, 0.0, 0.4], dtype=np.float32)
    obs, _, _, _, _ = env.step(action)
    expected = float(env.truth_state.get("valve_cmd", 0.0))
    assert abs(obs[0] - expected) < 1e-5, (
        f"obs8[0] (valve_cmd) expected {expected:.6f}, got {obs[0]:.6f}"
    )
    env.close()


def test_obs8_index_1_is_pump_cmd():
    env = HydrionEnv(obs_schema="obs8_deployment_v1")
    env.reset(seed=42)
    action = np.array([0.3, 0.7, 0.0, 0.4], dtype=np.float32)
    obs, _, _, _, _ = env.step(action)
    expected = float(env.truth_state.get("pump_cmd", 0.0))
    assert abs(obs[1] - expected) < 1e-5, (
        f"obs8[1] (pump_cmd) expected {expected:.6f}, got {obs[1]:.6f}"
    )
    env.close()


def test_obs8_index_2_is_bf_cmd():
    env = HydrionEnv(obs_schema="obs8_deployment_v1")
    env.reset(seed=42)
    action = np.array([0.3, 0.5, 0.8, 0.4], dtype=np.float32)
    obs, _, _, _, _ = env.step(action)
    expected = float(env.truth_state.get("bf_cmd", 0.0))
    assert abs(obs[2] - expected) < 1e-5, (
        f"obs8[2] (bf_cmd) expected {expected:.6f}, got {obs[2]:.6f}"
    )
    env.close()


def test_obs8_index_3_is_node_voltage_cmd():
    env = HydrionEnv(obs_schema="obs8_deployment_v1")
    env.reset(seed=42)
    action = np.array([0.3, 0.5, 0.0, 0.9], dtype=np.float32)
    obs, _, _, _, _ = env.step(action)
    expected = float(env.truth_state.get("node_voltage_cmd", 0.0))
    assert abs(obs[3] - expected) < 1e-5, (
        f"obs8[3] (node_voltage_cmd) expected {expected:.6f}, got {obs[3]:.6f}"
    )
    env.close()


# ---------------------------------------------------------------------------
# T3 — Sensor derived channels (obs8 indices 4–7 = obs14_v1 indices 10–13)
# ---------------------------------------------------------------------------

def test_obs8_index_4_is_sensor_turbidity():
    env = HydrionEnv(obs_schema="obs8_deployment_v1")
    obs, _ = env.reset(seed=42)
    expected = float(env.sensor_state.get("sensor_turbidity", 0.0))
    assert abs(obs[4] - expected) < 1e-5, (
        f"obs8[4] (sensor_turbidity) expected {expected:.6f}, got {obs[4]:.6f}"
    )
    env.close()


def test_obs8_index_5_is_sensor_scatter():
    env = HydrionEnv(obs_schema="obs8_deployment_v1")
    obs, _ = env.reset(seed=42)
    expected = float(env.sensor_state.get("sensor_scatter", 0.0))
    assert abs(obs[5] - expected) < 1e-5, (
        f"obs8[5] (sensor_scatter) expected {expected:.6f}, got {obs[5]:.6f}"
    )
    env.close()


def test_obs8_index_6_is_flow_sensor_norm():
    env = HydrionEnv(obs_schema="obs8_deployment_v1")
    obs, _ = env.reset(seed=42)
    raw = env.sensor_state.get("flow_sensor_lmin", 0.0)
    expected = float(np.clip(raw / 20.0, 0.0, 1.0))
    assert abs(obs[6] - expected) < 1e-5, (
        f"obs8[6] (flow_sensor_norm) expected {expected:.6f}, got {obs[6]:.6f}"
    )
    env.close()


def test_obs8_index_7_is_dp_sensor_norm():
    env = HydrionEnv(obs_schema="obs8_deployment_v1")
    obs, _ = env.reset(seed=42)
    raw = env.sensor_state.get("dp_sensor_kPa", 0.0)
    expected = float(np.clip(raw / 80.0, 0.0, 1.0))
    assert abs(obs[7] - expected) < 1e-5, (
        f"obs8[7] (dp_sensor_norm) expected {expected:.6f}, got {obs[7]:.6f}"
    )
    env.close()


# ---------------------------------------------------------------------------
# T4 — Value validity across a rollout (no NaN, no Inf; actuator indices [0,1])
# ---------------------------------------------------------------------------

def test_obs8_values_valid_rollout():
    num_steps = 200
    env = HydrionEnv(obs_schema="obs8_deployment_v1")
    obs, _ = env.reset(seed=0)
    assert not np.any(np.isnan(obs)), f"NaN at reset: {obs}"
    assert not np.any(np.isinf(obs)), f"Inf at reset: {obs}"
    # actuator indices 0–3 must be in [0, 1] (they are raw actuator commands)
    assert np.all(obs[0:4] >= 0.0) and np.all(obs[0:4] <= 1.0), (
        f"Actuator channels out of [0,1] at reset: {obs[0:4]}"
    )
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        assert not np.any(np.isnan(obs)), f"NaN at step {step}: {obs}"
        assert not np.any(np.isinf(obs)), f"Inf at step {step}: {obs}"
        assert np.all(obs[0:4] >= 0.0) and np.all(obs[0:4] <= 1.0), (
            f"Actuator channels out of [0,1] at step {step}: {obs[0:4]}"
        )
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()


# ---------------------------------------------------------------------------
# T5 — obs14_v1 behavior unchanged (default schema still returns 14D)
# ---------------------------------------------------------------------------

def test_obs14_v1_unchanged_by_default():
    """HydrionEnv() without obs_schema must still return 14D obs."""
    env = HydrionEnv()   # default obs_schema="obs14_v1"
    obs, _ = env.reset()
    assert obs.shape == (14,), (
        f"obs14_v1 default behavior broken — expected (14,), got {obs.shape}"
    )
    assert env.observation_space.shape == (14,)
    env.close()


def test_obs14_v1_explicit_unchanged():
    env = HydrionEnv(obs_schema="obs14_v1")
    obs, _ = env.reset()
    assert obs.shape == (14,), f"Expected (14,), got {obs.shape}"
    env.close()


# ---------------------------------------------------------------------------
# T6 — CCE still returns 12D
# ---------------------------------------------------------------------------

def test_cce_obs_still_12d_after_m8():
    from hydrion.environments.conical_cascade_env import ConicalCascadeEnv
    env = ConicalCascadeEnv()
    obs, _ = env.reset()
    assert obs.shape == (12,), (
        f"CCE must remain 12D after M8. Got shape: {obs.shape}"
    )
    env.close()


# ---------------------------------------------------------------------------
# T7 — Unknown schema is rejected
# ---------------------------------------------------------------------------

def test_unknown_schema_raises():
    with pytest.raises((ValueError, KeyError)):
        env = HydrionEnv(obs_schema="obs99_unknown")
        env.close()


# ---------------------------------------------------------------------------
# T8 — obs8 and obs14 are distinct (regression: obs8 must not return 14D)
# ---------------------------------------------------------------------------

def test_obs8_and_obs14_are_different_shapes():
    env8  = HydrionEnv(obs_schema="obs8_deployment_v1")
    env14 = HydrionEnv(obs_schema="obs14_v1")
    obs8,  _ = env8.reset(seed=42)
    obs14, _ = env14.reset(seed=42)
    assert obs8.shape  == (8,),  f"obs8 shape wrong: {obs8.shape}"
    assert obs14.shape == (14,), f"obs14 shape wrong: {obs14.shape}"
    env8.close()
    env14.close()
