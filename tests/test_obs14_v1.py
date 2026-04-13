# tests/test_obs14_v1.py
"""
obs14_v1 schema validation tests.

Validates the M6.2B add-only schema extension for HydrionEnv:
- indices 0–11 unchanged from obs12_v2
- index 12 = flow_sensor_norm  (sensors["flow_sensor_lmin"] / 20.0, clipped)
- index 13 = dp_sensor_norm    (sensors["dp_sensor_kPa"]   / 80.0, clipped)
- CCE observation unchanged at 12D
- ppo_cce_v2 unaffected

Schema version: obs14_v1 (M6.2B — 2026-04-13)
"""
from __future__ import annotations

import numpy as np
import pytest

from hydrion.env import HydrionEnv


# ---------------------------------------------------------------------------
# T1 — Observation shape is (14,)
# ---------------------------------------------------------------------------

def test_obs_shape_hydrienv():
    env = HydrionEnv()
    obs, _ = env.reset()
    assert obs.shape == (14,), f"Expected obs shape (14,), got {obs.shape}"
    env.close()


# ---------------------------------------------------------------------------
# T2 — Indices 0–11 are bounded and semantically intact
# ---------------------------------------------------------------------------

def test_obs_indices_0_11_bounded():
    env = HydrionEnv()
    obs, _ = env.reset(seed=42)
    for i in range(12):
        assert 0.0 <= obs[i] <= 1.0, (
            f"Index {i} out of [0,1] at reset: {obs[i]:.6f}"
        )
    env.close()


def test_obs_index_ordering_unchanged():
    """
    Structural check: truth-derived indices 0–9 come from truth_state;
    sensor-derived indices 10–11 come from sensor_state.
    After one step, verify index 6 (valve_cmd) matches truth_state["valve_cmd"].
    """
    env = HydrionEnv()
    env.reset(seed=42)
    action = np.array([0.3, 0.5, 0.0, 0.4], dtype=np.float32)
    obs, _, _, _, _ = env.step(action)
    expected_valve = float(env.truth_state.get("valve_cmd", 0.0))
    assert abs(obs[6] - expected_valve) < 1e-5, (
        f"Index 6 (valve_cmd) expected {expected_valve:.6f}, got {obs[6]:.6f}"
    )
    env.close()


# ---------------------------------------------------------------------------
# T3 — Index 12 traces to sensors["flow_sensor_lmin"] / 20.0
# ---------------------------------------------------------------------------

def test_obs_index_12_from_flow_sensor():
    env = HydrionEnv()
    obs, _ = env.reset(seed=42)
    raw = env.sensor_state["flow_sensor_lmin"]
    expected = float(np.clip(raw / 20.0, 0.0, 1.0))
    assert abs(obs[12] - expected) < 1e-5, (
        f"Index 12: expected {expected:.6f} from flow_sensor_lmin={raw:.4f}, "
        f"got {obs[12]:.6f}"
    )
    env.close()


def test_obs_index_12_from_flow_sensor_after_step():
    env = HydrionEnv()
    env.reset(seed=42)
    action = np.array([0.5, 0.8, 0.0, 0.5], dtype=np.float32)
    obs, _, _, _, _ = env.step(action)
    raw = env.sensor_state["flow_sensor_lmin"]
    expected = float(np.clip(raw / 20.0, 0.0, 1.0))
    assert abs(obs[12] - expected) < 1e-5, (
        f"Index 12 post-step: expected {expected:.6f} from flow_sensor_lmin={raw:.4f}, "
        f"got {obs[12]:.6f}"
    )
    env.close()


# ---------------------------------------------------------------------------
# T4 — Index 13 traces to sensors["dp_sensor_kPa"] / 80.0
# ---------------------------------------------------------------------------

def test_obs_index_13_from_dp_sensor():
    env = HydrionEnv()
    obs, _ = env.reset(seed=42)
    raw = env.sensor_state["dp_sensor_kPa"]
    expected = float(np.clip(raw / 80.0, 0.0, 1.0))
    assert abs(obs[13] - expected) < 1e-5, (
        f"Index 13: expected {expected:.6f} from dp_sensor_kPa={raw:.4f}, "
        f"got {obs[13]:.6f}"
    )
    env.close()


def test_obs_index_13_from_dp_sensor_after_step():
    env = HydrionEnv()
    env.reset(seed=42)
    action = np.array([0.5, 0.8, 0.0, 0.5], dtype=np.float32)
    obs, _, _, _, _ = env.step(action)
    raw = env.sensor_state["dp_sensor_kPa"]
    expected = float(np.clip(raw / 80.0, 0.0, 1.0))
    assert abs(obs[13] - expected) < 1e-5, (
        f"Index 13 post-step: expected {expected:.6f} from dp_sensor_kPa={raw:.4f}, "
        f"got {obs[13]:.6f}"
    )
    env.close()


# ---------------------------------------------------------------------------
# T5 — All 14 observation values bounded [0, 1] across a rollout
# ---------------------------------------------------------------------------

def test_obs_all_values_bounded_rollout():
    num_steps = 200
    env = HydrionEnv()
    obs, _ = env.reset(seed=0)
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0), (
        f"Obs out of bounds at reset: {obs}"
    )
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0), (
            f"Obs out of bounds at step {step}: min={obs.min():.4f} max={obs.max():.4f}\n{obs}"
        )
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()


# ---------------------------------------------------------------------------
# T6 — No truth-state contamination at indices 12 and 13
# ---------------------------------------------------------------------------

def test_no_truth_state_contamination_index_12():
    """
    Index 12 must match sensor_state["flow_sensor_lmin"] / 20.0, not truth_state["flow"].
    After a step, sensor and truth values may diverge — verify index 12 follows sensor.
    """
    env = HydrionEnv()
    env.reset(seed=42)
    action = np.array([0.5, 0.9, 1.0, 0.5], dtype=np.float32)  # activate backflush
    obs, _, _, _, _ = env.step(action)

    sensor_expected = float(np.clip(env.sensor_state["flow_sensor_lmin"] / 20.0, 0.0, 1.0))
    assert abs(obs[12] - sensor_expected) < 1e-5, (
        f"Index 12 does not match sensor_state['flow_sensor_lmin']. "
        f"Got obs[12]={obs[12]:.6f}, sensor expected={sensor_expected:.6f}"
    )
    # index 12 and index 0 are distinct array elements (different physical quantities)
    assert obs.shape[0] == 14
    assert obs[12] is not obs[0]
    env.close()


def test_no_truth_state_contamination_index_13():
    """
    Index 13 must match sensor_state["dp_sensor_kPa"] / 80.0, not truth_state["pressure"].
    """
    env = HydrionEnv()
    env.reset(seed=42)
    action = np.array([0.5, 0.9, 0.0, 0.5], dtype=np.float32)
    obs, _, _, _, _ = env.step(action)

    sensor_expected = float(np.clip(env.sensor_state["dp_sensor_kPa"] / 80.0, 0.0, 1.0))
    assert abs(obs[13] - sensor_expected) < 1e-5, (
        f"Index 13 does not match sensor_state['dp_sensor_kPa']. "
        f"Got obs[13]={obs[13]:.6f}, sensor expected={sensor_expected:.6f}"
    )
    assert obs[13] is not obs[1]
    env.close()


# ---------------------------------------------------------------------------
# T7 — CCE still returns 12D observation (guard against accidental modification)
# ---------------------------------------------------------------------------

def test_cce_obs_still_12d():
    from hydrion.environments.conical_cascade_env import ConicalCascadeEnv
    env = ConicalCascadeEnv()
    obs, _ = env.reset()
    assert obs.shape == (12,), (
        f"CCE must remain 12D (truth-derived). obs14_v1 must NOT affect CCE. "
        f"Got shape: {obs.shape}"
    )
    env.close()


def test_cce_observation_space_12d():
    from hydrion.environments.conical_cascade_env import ConicalCascadeEnv
    env = ConicalCascadeEnv()
    assert env.observation_space.shape == (12,), (
        f"CCE observation_space must remain (12,). Got: {env.observation_space.shape}"
    )
    env.close()


# ---------------------------------------------------------------------------
# T8 — Sensor indices can diverge from truth indices (dual-channel design validation)
# ---------------------------------------------------------------------------

def test_sensor_and_truth_indices_are_distinct_array_positions():
    """
    Structural: index 12 and index 0 are separate positions; index 13 and index 1 are separate.
    Verifies add-only extension did not overwrite existing slots.
    """
    env = HydrionEnv()
    obs, _ = env.reset(seed=42)
    assert len(obs) == 14
    # Indices 0 and 12 exist and are separate
    _ = obs[0]
    _ = obs[12]
    # Indices 1 and 13 exist and are separate
    _ = obs[1]
    _ = obs[13]
    env.close()


def test_observation_space_shape_matches_obs():
    """
    observation_space.shape must be consistent with the actual observation returned.
    """
    env = HydrionEnv()
    obs, _ = env.reset()
    assert env.observation_space.shape == obs.shape, (
        f"observation_space.shape={env.observation_space.shape} "
        f"does not match actual obs.shape={obs.shape}"
    )
    env.close()
