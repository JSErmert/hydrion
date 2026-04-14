"""
Hydrion — PPO Evaluation Script for M8 Deployment Gap Quantification
---------------------------------------------------------------------
Compares ppo_hydrienv_v2 (obs8_deployment_v1) against ppo_hydrienv_v1 (obs14_v1)
and a random-action baseline. Measures the deployment gap.

Schema information:
    ppo_hydrienv_v2: obs8_deployment_v1 (8D — actuator feedback + sensor-derived)
    ppo_hydrienv_v1: obs14_v1 (14D — full schema including physics truth channels 0–5)

M8 methodology validity conditions (M8.3 Section 6):
    MVC-1: Each policy uses its own VecNormalize pkl (never swapped)
    MVC-2: Evaluation env uses identical ShieldedEnv config for both policies
    MVC-3: Calibration confound is stated explicitly in output
    MVC-4: Comparison is labeled as cross-schema deployment-gap measurement

Result interpretation (M8.3 Section 9 — BINDING):
    v2 ~= v1:              obs8 achieves near-equivalent at 500k steps + placeholder noise.
                           NOT: "channels 0–5 unnecessary" or "deployment-ready".
    v2 < v1 (small gap):   Moderate information cost. NOT: "deployment impossible".
    v2 < v1 (large gap):   Substantial cost — investigate training budget / distillation
                           before concluding information insufficiency.
    v2 > random:           Deployment-realistic policy learns above-chance control.

Prohibited claims (M8.3 Section 10):
    - "deployment-ready"            - "channels 0–5 unnecessary"
    - "obs8 is sensor-only"         - "deployment is impossible"
    - "v2 is better than v1"        - "M8 proves deployment viability"

Usage:
    python -m hydrion.eval_ppo_hydrienv_v2
    python -m hydrion.eval_ppo_hydrienv_v2 --episodes 5
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


# ── Artifact paths — named constants (never swap these) ─────────────────────
_V2_MODEL_PATH   = "models/ppo_hydrienv_v2.zip"
_V2_VECNORM_PATH = "models/ppo_hydrienv_v2_vecnorm.pkl"   # obs8_deployment_v1 only
_V1_MODEL_PATH   = "models/ppo_hydrienv_v1.zip"
_V1_VECNORM_PATH = "models/ppo_hydrienv_v1_vecnorm.pkl"   # obs14_v1 only

# ── Schema constants ─────────────────────────────────────────────────────────
_V2_OBS_SCHEMA   = "obs8_deployment_v1"
_V2_OBS_DIM      = 8
_V1_OBS_DIM      = 14

# AC11 reference value from M7 canonical result
_V1_CANONICAL_RETURN = 694.639
_V1_TOLERANCE_PCT    = 0.05     # 5% tolerance for AC11

# MVC-2: identical ShieldedEnv config for both policy evaluations
_HYDRIENV_SAFETY_CFG = SafetyConfig(
    max_pressure_soft=0.85,
    max_pressure_hard=1.05,
    max_clog_soft=0.80,
    max_clog_hard=0.98,
    terminate_on_hard_violation=True,
    hard_violation_penalty=10.0,
)

_MAX_STEPS_EVAL = 400

# MVC-3: calibration confound statement
_CALIBRATION_CONFOUND_STATEMENT = (
    "CONFOUND (MVC-3): M8 results are confounded by calibration-pending sensor noise. "
    "The sensor channels in obs8_deployment_v1 (obs8 indices 4-7 = obs14_v1 indices 10-13) "
    "are governed by placeholder noise parameters (dp_drift_rate, dp_fouling_gain, "
    "flow_calibration_bias_std). The deployment gap measurement is internally valid "
    "under these parameters, but its predictive value for hardware deployment is bounded "
    "by the uncalibrated sensor model."
)


def make_env_v2(seed: int = 0):
    """Environment factory for ppo_hydrienv_v2 (obs8_deployment_v1)."""
    def _init():
        env = HydrionEnv(
            config_path="configs/default.yaml",
            seed=seed,
            noise_enabled=True,
            obs_schema=_V2_OBS_SCHEMA,     # obs8_deployment_v1
        )
        env.max_steps = _MAX_STEPS_EVAL
        env = ShieldedEnv(env, cfg=_HYDRIENV_SAFETY_CFG)
        env.reset(seed=seed)
        return env
    return _init


def make_env_v1(seed: int = 0):
    """Environment factory for ppo_hydrienv_v1 (obs14_v1, default schema)."""
    def _init():
        env = HydrionEnv(
            config_path="configs/default.yaml",
            seed=seed,
            noise_enabled=True,
            # obs_schema defaults to "obs14_v1" — explicit here for clarity
            obs_schema="obs14_v1",
        )
        env.max_steps = _MAX_STEPS_EVAL
        env = ShieldedEnv(env, cfg=_HYDRIENV_SAFETY_CFG)
        env.reset(seed=seed)
        return env
    return _init


def _run_ppo_v2_episodes(vec_env, model, n_episodes: int) -> dict:
    """
    Run ppo_hydrienv_v2 for n_episodes.
    obs8_deployment_v1 channel diagnostics:
        obs8 indices 0–3: actuator_feedback (obs14_v1 indices 6–9)
        obs8 indices 4–7: sensor_derived    (obs14_v1 indices 10–13)
    """
    returns, ep_lengths, violations = [], [], []
    act_means, act_stds = [], []
    sensor_means, sensor_stds = [], []

    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_ret, ep_steps, ep_viol = 0.0, 0, 0
        ep_act_vals    = []
        ep_sensor_vals = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info_arr = vec_env.step(action)
            done = bool(done_arr[0])
            ep_ret   += float(reward[0])
            ep_steps += 1

            ep_act_vals.append(obs[0][0:4].tolist())
            ep_sensor_vals.append(obs[0][4:8].tolist())

            info   = info_arr[0]
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

        act_arr    = np.array(ep_act_vals)
        sensor_arr = np.array(ep_sensor_vals)

        act_means.append(float(np.mean(act_arr)))
        act_stds.append(float(np.std(act_arr)))
        sensor_means.append(float(np.mean(sensor_arr)))
        sensor_stds.append(float(np.std(sensor_arr)))

        print(
            f"  [V2  ] ep {ep:02d}: "
            f"return={ep_ret:8.3f}  steps={ep_steps}  violations={ep_viol}  "
            f"act_mean={act_means[-1]:.4f}  sensor_mean={sensor_means[-1]:.4f}"
        )

    return {
        "mean_return":          float(np.mean(returns)),
        "std_return":           float(np.std(returns)),
        "mean_ep_length":       float(np.mean(ep_lengths)),
        "mean_violations":      float(np.mean(violations)),
        "act_global_mean":      float(np.mean(act_means)),
        "act_global_std":       float(np.mean(act_stds)),
        "sensor_global_mean":   float(np.mean(sensor_means)),
        "sensor_global_std":    float(np.mean(sensor_stds)),
    }


def _run_ppo_v1_episodes(vec_env, model, n_episodes: int) -> dict:
    """Run ppo_hydrienv_v1 for n_episodes."""
    returns, ep_lengths, violations = [], [], []

    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_ret, ep_steps, ep_viol = 0.0, 0, 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info_arr = vec_env.step(action)
            done = bool(done_arr[0])
            ep_ret   += float(reward[0])
            ep_steps += 1
            info   = info_arr[0]
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
        print(
            f"  [V1  ] ep {ep:02d}: "
            f"return={ep_ret:8.3f}  steps={ep_steps}  violations={ep_viol}"
        )

    return {
        "mean_return":     float(np.mean(returns)),
        "std_return":      float(np.std(returns)),
        "mean_ep_length":  float(np.mean(ep_lengths)),
        "mean_violations": float(np.mean(violations)),
    }


def _run_random_episodes(n_episodes: int, seed: int = 300) -> dict:
    """Random baseline under obs8_deployment_v1. No VecNormalize."""
    rand_vec = DummyVecEnv([make_env_v2(seed=seed)])
    returns, ep_lengths = [], []

    for ep in range(n_episodes):
        obs = rand_vec.reset()
        done = False
        ep_ret, ep_steps = 0.0, 0

        while not done:
            action = rand_vec.action_space.sample()[np.newaxis, :]
            obs, reward, done_arr, _ = rand_vec.step(action)
            done = bool(done_arr[0])
            ep_ret   += float(reward[0])
            ep_steps += 1

        returns.append(ep_ret)
        ep_lengths.append(ep_steps)
        print(
            f"  [RAND] ep {ep:02d}: return={ep_ret:8.3f}  steps={ep_steps}"
        )

    rand_vec.close()
    return {
        "mean_return":    float(np.mean(returns)),
        "std_return":     float(np.std(returns)),
        "mean_ep_length": float(np.mean(ep_lengths)),
    }


def evaluate(
    v2_model_path:   str = _V2_MODEL_PATH,
    v2_vecnorm_path: str = _V2_VECNORM_PATH,
    v1_model_path:   str = _V1_MODEL_PATH,
    v1_vecnorm_path: str = _V1_VECNORM_PATH,
    n_episodes:      int = 5,
) -> None:

    print(f"\n{'='*72}")
    print(f"M8 Deployment Gap Evaluation — obs8_deployment_v1 vs obs14_v1")
    print(f"MVC-4: cross-schema deployment-gap measurement (NOT policy improvement)")
    print(f"{'='*72}")

    # ── Artifact checks — fail fast ──────────────────────────────────────────
    for path in [v2_model_path, v2_vecnorm_path, v1_model_path, v1_vecnorm_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required artifact not found: {path}\n"
                f"Train ppo_hydrienv_v2 first: python -m hydrion.train_ppo_hydrienv_v2\n"
                f"Train ppo_hydrienv_v1 first: python -m hydrion.train_ppo_hydrienv_v1"
            )

    # ── Load v2 with its own VecNorm (MVC-1: NEVER use v1 vecnorm for v2) ───
    v2_vec = DummyVecEnv([make_env_v2(seed=100)])
    v2_vec = VecNormalize.load(v2_vecnorm_path, v2_vec)   # obs8 VecNorm pkl
    v2_vec.training    = False
    v2_vec.norm_reward = False
    v2_model = PPO.load(v2_model_path, env=v2_vec)

    _v2_shape = v2_model.observation_space.shape
    assert _v2_shape == (_V2_OBS_DIM,), (
        f"[SCHEMA LOCK FAILED] v2 model expects {_v2_shape}, not ({_V2_OBS_DIM},). "
        f"Wrong artifact — check {v2_model_path}."
    )
    print(f"[SCHEMA LOCK OK] v2 obs space: {_v2_shape} ({_V2_OBS_SCHEMA})")

    # ── Load v1 with its own VecNorm (MVC-1: NEVER use v2 vecnorm for v1) ───
    v1_vec = DummyVecEnv([make_env_v1(seed=100)])
    v1_vec = VecNormalize.load(v1_vecnorm_path, v1_vec)   # obs14 VecNorm pkl
    v1_vec.training    = False
    v1_vec.norm_reward = False
    v1_model = PPO.load(v1_model_path, env=v1_vec)

    _v1_shape = v1_model.observation_space.shape
    assert _v1_shape == (_V1_OBS_DIM,), (
        f"[SCHEMA LOCK FAILED] v1 model expects {_v1_shape}, not ({_V1_OBS_DIM},). "
        f"Wrong artifact — check {v1_model_path}."
    )
    print(f"[SCHEMA LOCK OK] v1 obs space: {_v1_shape} (obs14_v1)")

    print(f"\nv2 model:    {v2_model_path}  + {v2_vecnorm_path}")
    print(f"v1 model:    {v1_model_path}  + {v1_vecnorm_path}")
    print(f"Episodes:    {n_episodes}  seed=100  deterministic=True\n")

    # ── Run three evaluations ────────────────────────────────────────────────
    print(f"Evaluating ppo_hydrienv_v2 ({_V2_OBS_SCHEMA}, {n_episodes} episodes)...")
    v2_stats = _run_ppo_v2_episodes(v2_vec, v2_model, n_episodes)
    v2_vec.close()

    print(f"\nEvaluating ppo_hydrienv_v1 (obs14_v1, {n_episodes} episodes)...")
    v1_stats = _run_ppo_v1_episodes(v1_vec, v1_model, n_episodes)
    v1_vec.close()

    print(f"\nEvaluating random baseline ({_V2_OBS_SCHEMA}, {n_episodes} episodes)...")
    rand_stats = _run_random_episodes(n_episodes, seed=300)

    # ── Channel diagnostics (AC10) ───────────────────────────────────────────
    print(f"\n{'-'*72}")
    print(f"obs8_deployment_v1 channel diagnostics (VecNormalize-normalized values):")
    print(f"  Actuator feedback (obs8 indices 0-3 = obs14_v1 indices 6-9):")
    print(f"    mean={v2_stats['act_global_mean']:.4f}  std={v2_stats['act_global_std']:.4f}")
    print(f"  Sensor derived    (obs8 indices 4-7 = obs14_v1 indices 10-13):")
    print(f"    mean={v2_stats['sensor_global_mean']:.4f}  std={v2_stats['sensor_global_std']:.4f}")

    # ── Performance delta ────────────────────────────────────────────────────
    delta_pct = (
        (v2_stats["mean_return"] - v1_stats["mean_return"])
        / max(abs(v1_stats["mean_return"]), 1e-6)
    ) * 100.0
    delta_vs_random_pct = (
        (v2_stats["mean_return"] - rand_stats["mean_return"])
        / max(abs(rand_stats["mean_return"]), 1e-6)
    ) * 100.0

    # ── AC checks ─────────────────────────────────────────────────────────────
    ac9_pass  = v2_stats["mean_return"] > rand_stats["mean_return"]
    ac10_act_pass    = v2_stats["act_global_std"] > 0
    ac10_sensor_pass = v2_stats["sensor_global_std"] > 0 and abs(v2_stats["sensor_global_mean"]) > 0.001
    ac11_pass = abs(v1_stats["mean_return"] - _V1_CANONICAL_RETURN) / _V1_CANONICAL_RETURN <= _V1_TOLERANCE_PCT

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"{'Metric':<36} {'v2 (obs8)':>10} {'v1 (obs14)':>10} {'Random':>10}")
    print(f"{'-'*72}")
    print(f"  {'Mean return':<34} {v2_stats['mean_return']:>10.3f} "
          f"{v1_stats['mean_return']:>10.3f} {rand_stats['mean_return']:>10.3f}")
    print(f"  {'Std return':<34} {v2_stats['std_return']:>10.3f} "
          f"{v1_stats['std_return']:>10.3f} {rand_stats['std_return']:>10.3f}")
    print(f"  {'Mean ep length':<34} {v2_stats['mean_ep_length']:>10.1f} "
          f"{v1_stats['mean_ep_length']:>10.1f} {rand_stats['mean_ep_length']:>10.1f}")
    print(f"  {'Mean violations':<34} {v2_stats['mean_violations']:>10.1f} "
          f"{v1_stats['mean_violations']:>10.1f} {'N/A':>10}")
    print(f"{'-'*72}")
    print(f"  {'Deployment gap (v2 - v1) %':<34} {delta_pct:>+10.1f}%")
    print(f"  {'v2 vs random %':<34} {delta_vs_random_pct:>+10.1f}%")
    print(f"{'='*72}")

    print(f"\nAcceptance criteria checks (M8.3 AC9–AC11):")
    print(f"  AC9   v2 return > random:                {'[PASS]' if ac9_pass else '[FAIL]'}"
          f"  ({v2_stats['mean_return']:.3f} vs {rand_stats['mean_return']:.3f})")
    print(f"  AC10  actuator channels non-constant:    {'[PASS]' if ac10_act_pass else '[FAIL]'}"
          f"  (act_std={v2_stats['act_global_std']:.4f})")
    print(f"  AC10  sensor channels active:            {'[PASS]' if ac10_sensor_pass else '[FAIL]'}"
          f"  (sensor_mean={v2_stats['sensor_global_mean']:.4f}, std={v2_stats['sensor_global_std']:.4f})")
    print(f"  AC11  v1 return within 5% of canonical:  {'[PASS]' if ac11_pass else '[FAIL]'}"
          f"  ({v1_stats['mean_return']:.3f} vs canonical {_V1_CANONICAL_RETURN:.3f})")

    if not ac11_pass:
        print(f"\n  [WARNING] AC11 FAILED. v1 return deviates from M7 canonical by more than 5%.")
        print(f"  This indicates the evaluation environment may have changed.")
        print(f"  DO NOT interpret v2 vs v1 delta until AC11 is investigated.")

    # ── Result interpretation (M8.3 Section 9 — BINDING) ────────────────────
    print(f"\n{'-'*72}")
    print(f"Result interpretation (M8.3 Section 9 — binding):")
    abs_gap = abs(delta_pct)
    if abs_gap <= 5.0:
        scenario = "v2 approximately equals v1"
        interpretation = "obs8_deployment_v1 achieves near-equivalent performance at 500k steps."
        must_not = "Do NOT conclude: 'channels 0-5 unnecessary' or 'deployment-ready'."
    elif abs_gap <= 20.0 and v2_stats["mean_return"] < v1_stats["mean_return"]:
        scenario = "v2 < v1 (small gap)"
        interpretation = "Moderate information cost from removing physics-truth channels."
        must_not = "Do NOT conclude: 'deployment impractical' or 'sensor channels insufficient'."
    elif abs_gap > 20.0 and v2_stats["mean_return"] < v1_stats["mean_return"]:
        scenario = "v2 < v1 (large gap)"
        interpretation = (
            "Substantial information cost. Investigate before concluding insufficiency: "
            "(a) training budget (500k may not be enough — Rudin 2022), "
            "(b) distillation from ppo_hydrienv_v1 (Lee 2020 DAgger path), "
            "(c) placeholder noise quality."
        )
        must_not = "Do NOT conclude: 'deployment impossible' or 'obs8 fundamentally insufficient'."
    else:
        scenario = "v2 approximately equals or exceeds v1"
        interpretation = "obs8_deployment_v1 achieves near-equivalent or better performance."
        must_not = "Do NOT conclude: 'deployment-ready' or 'deployment gap solved'."

    print(f"  Scenario:       {scenario}")
    print(f"  Interpretation: {interpretation}")
    print(f"  {must_not}")

    # ── MVC-3: calibration confound statement ────────────────────────────────
    print(f"\n{'-'*72}")
    print(_CALIBRATION_CONFOUND_STATEMENT)
    print(f"{'-'*72}")
    print(f"MVC-4: This result is a cross-schema deployment-gap measurement.")
    print(f"       It is NOT a policy improvement comparison.")
    print(f"{'='*72}")


def main():
    parser = argparse.ArgumentParser(
        description="M8 deployment gap evaluation: v2 vs v1 vs random"
    )
    parser.add_argument("--v2-model",   default=_V2_MODEL_PATH)
    parser.add_argument("--v2-vecnorm", default=_V2_VECNORM_PATH)
    parser.add_argument("--v1-model",   default=_V1_MODEL_PATH)
    parser.add_argument("--v1-vecnorm", default=_V1_VECNORM_PATH)
    parser.add_argument("--episodes",   type=int, default=5)
    args = parser.parse_args()
    evaluate(
        args.v2_model, args.v2_vecnorm,
        args.v1_model, args.v1_vecnorm,
        args.episodes,
    )


if __name__ == "__main__":
    main()
