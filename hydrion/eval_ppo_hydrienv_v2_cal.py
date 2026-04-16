"""
Hydrion — PPO Evaluation Script for M9 Calibrated Deployment Gap
-----------------------------------------------------------------
Compares ppo_hydrienv_v2_cal (obs8, M9 calibrated noise) against
ppo_hydrienv_v1 (obs14_v1) and a random baseline.

IMPORTANT: v2_cal uses configs/m9_calibrated.yaml (SP1).
           v1 uses configs/default.yaml (preserved, AC11 gate).

M9 prohibited claims (M9.3 Section 10 — BINDING):
    - "M9 demonstrates deployment readiness"
    - "The M9 calibrated gap proves hardware validity"
    - "M9 validates the obs8 schema for real-world use"
    - "Calibration has resolved the deployment gap question"
    - "The M9 result applies to any device beyond the nominated classes"
    - "M9 supersedes the M8 prohibited-claims list"

Usage:
    python -m hydrion.eval_ppo_hydrienv_v2_cal
    python -m hydrion.eval_ppo_hydrienv_v2_cal --episodes 5
"""

from __future__ import annotations

import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hydrion.env import HydrionEnv
from hydrion.wrappers.shielded_env import ShieldedEnv
from hydrion.safety.shield import SafetyConfig


_V2_CAL_MODEL_PATH   = "models/ppo_hydrienv_v2_cal.zip"
_V2_CAL_VECNORM_PATH = "models/ppo_hydrienv_v2_cal_vecnorm.pkl"
_V1_MODEL_PATH       = "models/ppo_hydrienv_v1.zip"
_V1_VECNORM_PATH     = "models/ppo_hydrienv_v1_vecnorm.pkl"

_V2_CAL_CONFIG = "configs/m9_calibrated.yaml"
_V1_CONFIG     = "configs/default.yaml"

_V2_OBS_SCHEMA = "obs8_deployment_v1"
_V2_OBS_DIM    = 8
_V1_OBS_DIM    = 14

_V1_CANONICAL_RETURN = 694.639
_V1_TOLERANCE_PCT    = 0.05

_M8_V2_REFERENCE_RETURN = 841.839

_HYDRIENV_SAFETY_CFG = SafetyConfig(
    max_pressure_soft=0.85,
    max_pressure_hard=1.05,
    max_clog_soft=0.80,
    max_clog_hard=0.98,
    terminate_on_hard_violation=True,
    hard_violation_penalty=10.0,
)

_MAX_STEPS_EVAL = 400


def make_env_v2_cal(seed: int = 0):
    def _init():
        env = HydrionEnv(
            config_path=_V2_CAL_CONFIG,
            seed=seed,
            noise_enabled=True,
            obs_schema=_V2_OBS_SCHEMA,
        )
        env.max_steps = _MAX_STEPS_EVAL
        env = ShieldedEnv(env, cfg=_HYDRIENV_SAFETY_CFG)
        env.reset(seed=seed)
        return env
    return _init


def make_env_v1(seed: int = 0):
    def _init():
        env = HydrionEnv(
            config_path=_V1_CONFIG,
            seed=seed,
            noise_enabled=True,
            obs_schema="obs14_v1",
        )
        env.max_steps = _MAX_STEPS_EVAL
        env = ShieldedEnv(env, cfg=_HYDRIENV_SAFETY_CFG)
        env.reset(seed=seed)
        return env
    return _init


def _run_v2_cal_episodes(vec_env, model, n_episodes: int) -> dict:
    returns, ep_lengths, violations = [], [], []
    act_means, act_stds = [], []
    sensor_means, sensor_stds = [], []

    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_ret, ep_steps, ep_viol = 0.0, 0, 0
        ep_act_vals, ep_sensor_vals = [], []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info_arr = vec_env.step(action)
            done = bool(done_arr[0])
            ep_ret += float(reward[0])
            ep_steps += 1
            ep_act_vals.append(obs[0][0:4].tolist())
            ep_sensor_vals.append(obs[0][4:8].tolist())
            info = info_arr[0]
            safety = info.get("safety", {})
            if any([
                safety.get("soft_pressure_violation"),
                safety.get("hard_pressure_violation"),
                safety.get("soft_clog_violation"),
                safety.get("hard_clog_violation"),
                safety.get("blockage_violation"),
            ]):
                ep_viol += 1

        returns.append(ep_ret)
        ep_lengths.append(ep_steps)
        violations.append(ep_viol)
        act_arr = np.array(ep_act_vals)
        sensor_arr = np.array(ep_sensor_vals)
        act_means.append(float(np.mean(act_arr)))
        act_stds.append(float(np.std(act_arr)))
        sensor_means.append(float(np.mean(sensor_arr)))
        sensor_stds.append(float(np.std(sensor_arr)))
        print(f"  [V2CAL] ep {ep:02d}: return={ep_ret:8.3f}  steps={ep_steps}  violations={ep_viol}")

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_ep_length": float(np.mean(ep_lengths)),
        "mean_violations": float(np.mean(violations)),
        "act_global_mean": float(np.mean(act_means)),
        "act_global_std": float(np.mean(act_stds)),
        "sensor_global_mean": float(np.mean(sensor_means)),
        "sensor_global_std": float(np.mean(sensor_stds)),
    }


def _run_v1_episodes(vec_env, model, n_episodes: int) -> dict:
    returns, ep_lengths, violations = [], [], []
    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_ret, ep_steps, ep_viol = 0.0, 0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info_arr = vec_env.step(action)
            done = bool(done_arr[0])
            ep_ret += float(reward[0])
            ep_steps += 1
            info = info_arr[0]
            safety = info.get("safety", {})
            if any([
                safety.get("soft_pressure_violation"),
                safety.get("hard_pressure_violation"),
                safety.get("soft_clog_violation"),
                safety.get("hard_clog_violation"),
                safety.get("blockage_violation"),
            ]):
                ep_viol += 1
        returns.append(ep_ret)
        ep_lengths.append(ep_steps)
        violations.append(ep_viol)
        print(f"  [V1   ] ep {ep:02d}: return={ep_ret:8.3f}  steps={ep_steps}  violations={ep_viol}")
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_ep_length": float(np.mean(ep_lengths)),
        "mean_violations": float(np.mean(violations)),
    }


def _run_random_episodes(n_episodes: int, seed: int = 300) -> dict:
    rand_vec = DummyVecEnv([make_env_v2_cal(seed=seed)])
    returns, ep_lengths = [], []
    for ep in range(n_episodes):
        obs = rand_vec.reset()
        done = False
        ep_ret, ep_steps = 0.0, 0
        while not done:
            action = rand_vec.action_space.sample()[np.newaxis, :]
            obs, reward, done_arr, _ = rand_vec.step(action)
            done = bool(done_arr[0])
            ep_ret += float(reward[0])
            ep_steps += 1
        returns.append(ep_ret)
        ep_lengths.append(ep_steps)
        print(f"  [RAND ] ep {ep:02d}: return={ep_ret:8.3f}  steps={ep_steps}")
    rand_vec.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_ep_length": float(np.mean(ep_lengths)),
    }


def evaluate(n_episodes: int = 5) -> None:
    print(f"\n{'='*72}")
    print(f"M9 Calibrated Deployment Gap Evaluation")
    print(f"v2_cal (obs8, calibrated noise) vs v1 (obs14) vs random")
    print(f"{'='*72}")
    print()
    print("M9 PROHIBITED CLAIMS (M9.3 Section 10 — binding):")
    print("  - M9 does NOT demonstrate deployment readiness")
    print("  - M9 does NOT prove hardware validity")
    print("  - M9 does NOT validate obs8 for real-world use")
    print("  - M9 does NOT resolve the deployment gap question")
    print("  - M9 does NOT generalize beyond nominated device classes")
    print("  - M9 does NOT supersede M8 prohibited claims")
    print()

    for path in [_V2_CAL_MODEL_PATH, _V2_CAL_VECNORM_PATH, _V1_MODEL_PATH, _V1_VECNORM_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required artifact not found: {path}")

    v2_cal_vec = DummyVecEnv([make_env_v2_cal(seed=100)])
    v2_cal_vec = VecNormalize.load(_V2_CAL_VECNORM_PATH, v2_cal_vec)
    v2_cal_vec.training = False
    v2_cal_vec.norm_reward = False
    v2_cal_model = PPO.load(_V2_CAL_MODEL_PATH, env=v2_cal_vec)

    _v2_shape = v2_cal_model.observation_space.shape
    assert _v2_shape == (_V2_OBS_DIM,), f"[SCHEMA LOCK FAILED] v2_cal: {_v2_shape}"
    print(f"[SCHEMA LOCK OK] v2_cal obs space: {_v2_shape} ({_V2_OBS_SCHEMA})")
    print(f"[CONFIG] v2_cal: {_V2_CAL_CONFIG}")

    v1_vec = DummyVecEnv([make_env_v1(seed=100)])
    v1_vec = VecNormalize.load(_V1_VECNORM_PATH, v1_vec)
    v1_vec.training = False
    v1_vec.norm_reward = False
    v1_model = PPO.load(_V1_MODEL_PATH, env=v1_vec)

    _v1_shape = v1_model.observation_space.shape
    assert _v1_shape == (_V1_OBS_DIM,), f"[SCHEMA LOCK FAILED] v1: {_v1_shape}"
    print(f"[SCHEMA LOCK OK] v1 obs space: {_v1_shape} (obs14_v1)")
    print(f"[CONFIG] v1: {_V1_CONFIG}")
    print()

    print(f"Evaluating ppo_hydrienv_v2_cal ({_V2_OBS_SCHEMA}, calibrated, {n_episodes} eps)...")
    v2_cal_stats = _run_v2_cal_episodes(v2_cal_vec, v2_cal_model, n_episodes)
    v2_cal_vec.close()

    print(f"\nEvaluating ppo_hydrienv_v1 (obs14_v1, {n_episodes} eps)...")
    v1_stats = _run_v1_episodes(v1_vec, v1_model, n_episodes)
    v1_vec.close()

    print(f"\nEvaluating random baseline ({_V2_OBS_SCHEMA}, calibrated env, {n_episodes} eps)...")
    rand_stats = _run_random_episodes(n_episodes, seed=300)

    delta_vs_v1_pct = (
        (v2_cal_stats["mean_return"] - v1_stats["mean_return"])
        / max(abs(v1_stats["mean_return"]), 1e-6)
    ) * 100.0
    delta_vs_random_pct = (
        (v2_cal_stats["mean_return"] - rand_stats["mean_return"])
        / max(abs(rand_stats["mean_return"]), 1e-6)
    ) * 100.0
    delta_vs_m8_v2_pct = (
        (v2_cal_stats["mean_return"] - _M8_V2_REFERENCE_RETURN)
        / max(abs(_M8_V2_REFERENCE_RETURN), 1e-6)
    ) * 100.0

    ac_m9_1 = v2_cal_stats["mean_return"] > rand_stats["mean_return"]
    ac_m9_2 = abs(v1_stats["mean_return"] - _V1_CANONICAL_RETURN) / _V1_CANONICAL_RETURN <= _V1_TOLERANCE_PCT
    ac_m9_3_act = v2_cal_stats["act_global_std"] > 0
    ac_m9_3_sensor = v2_cal_stats["sensor_global_std"] > 0

    print(f"\n{'='*72}")
    print(f"{'Metric':<38} {'v2_cal':>10} {'v1':>10} {'Random':>10}")
    print(f"{'-'*72}")
    print(f"  {'Mean return':<36} {v2_cal_stats['mean_return']:>10.3f} "
          f"{v1_stats['mean_return']:>10.3f} {rand_stats['mean_return']:>10.3f}")
    print(f"  {'Std return':<36} {v2_cal_stats['std_return']:>10.3f} "
          f"{v1_stats['std_return']:>10.3f} {rand_stats['std_return']:>10.3f}")
    print(f"  {'Mean ep length':<36} {v2_cal_stats['mean_ep_length']:>10.1f} "
          f"{v1_stats['mean_ep_length']:>10.1f} {rand_stats['mean_ep_length']:>10.1f}")
    print(f"  {'Mean violations':<36} {v2_cal_stats['mean_violations']:>10.1f} "
          f"{v1_stats['mean_violations']:>10.1f} {'N/A':>10}")
    print(f"{'-'*72}")
    print(f"  {'Calibrated gap (v2_cal - v1) %':<36} {delta_vs_v1_pct:>+10.1f}%")
    print(f"  {'v2_cal vs random %':<36} {delta_vs_random_pct:>+10.1f}%")
    print(f"  {'v2_cal vs M8 v2 (841.839) %':<36} {delta_vs_m8_v2_pct:>+10.1f}%")
    print(f"{'='*72}")

    print(f"\nAcceptance criteria (M9.3):")
    print(f"  AC-M9-1  v2_cal > random:             {'[PASS]' if ac_m9_1 else '[FAIL]'}"
          f"  ({v2_cal_stats['mean_return']:.3f} vs {rand_stats['mean_return']:.3f})")
    print(f"  AC-M9-2  AC11 v1 canonical match:     {'[PASS]' if ac_m9_2 else '[FAIL]'}"
          f"  ({v1_stats['mean_return']:.3f} vs {_V1_CANONICAL_RETURN:.3f})")
    print(f"  AC-M9-3  actuator non-constant:       {'[PASS]' if ac_m9_3_act else '[FAIL]'}"
          f"  (act_std={v2_cal_stats['act_global_std']:.4f})")
    print(f"  AC-M9-3  sensor active:               {'[PASS]' if ac_m9_3_sensor else '[FAIL]'}"
          f"  (sensor_std={v2_cal_stats['sensor_global_std']:.4f})")

    if not ac_m9_2:
        print(f"\n  [HALT] AC-M9-2 FAILED — AC11 canonical match broken.")
        print(f"  DO NOT interpret any M9 result until this is investigated.")

    print(f"\n{'-'*72}")
    print(f"Four-scenario classification (M9.3 Section 9 — binding):")
    abs_gap = abs(delta_vs_v1_pct)
    if v2_cal_stats["mean_return"] <= rand_stats["mean_return"]:
        scenario = "D: v2_cal <= random — catastrophic degradation"
        must_not = "HALT. Investigate calibrated parameter application before continuing."
    elif abs_gap > 15.0 and v2_cal_stats["mean_return"] > v1_stats["mean_return"]:
        scenario = "A: v2_cal >> v1 — positive gap persists under calibration"
        must_not = "Do NOT conclude: deployment-ready or deployment gap solved."
    elif abs_gap <= 15.0:
        scenario = "B: v2_cal ~ v1 — gap narrowed under calibration"
        must_not = "Do NOT conclude: obs8 architecture is sufficient for deployment."
    else:
        scenario = "C: v2_cal < v1 but v2_cal > random — negative gap under calibration"
        must_not = "Do NOT conclude: obs8 is inadequate (it still exceeds random)."

    print(f"  Scenario:       {scenario}")
    print(f"  {must_not}")
    print(f"{'-'*72}")
    print(f"Calibration note: sensor noise now grounded to named device classes.")
    print(f"  See M9.1R.3 + M9.1R.4 for full provenance.")
    print(f"  M8 prohibited claims remain in force for M8 artifacts.")
    print(f"{'='*72}")


def main():
    parser = argparse.ArgumentParser(
        description="M9 calibrated deployment gap evaluation: v2_cal vs v1 vs random"
    )
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.episodes)


if __name__ == "__main__":
    main()
