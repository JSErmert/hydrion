"""
Hydrion — PPO Evaluation Script for HydrionEnv (obs14_v1 baseline)
-------------------------------------------------------------------
Evaluates the trained ppo_hydrienv_v1 policy against a random-action baseline.

Schema:   obs14_v1 (14D — 10 truth-derived + 4 sensor-derived)
Artifact: models/ppo_hydrienv_v1.zip + models/ppo_hydrienv_v1_vecnorm.pkl

Sensor channel diagnostics (required per M7.4 AC8, AC9):
    obs[12] = flow_sensor_norm (VecNormalize-scaled at eval time)
    obs[13] = dp_sensor_norm   (VecNormalize-scaled at eval time; 1-step latency)
    obs[0]  = truth flow       (privileged — for dual-channel distinction check)
    obs[1]  = truth pressure   (privileged — for dual-channel distinction check)

    AC8: obs[12] and obs[13] must be non-zero (mean > 0.001) and varying (std > 0)
    AC9: max|obs[12] - obs[0]| > 0 or max|obs[13] - obs[1]| > 0 in ≥ 1 episode

    Note: values reported here are VecNormalize-normalized (centered near 0, std ~ 1).
    Non-zero mean in normalized space confirms the raw sensor channels are active
    and not uniformly zero.

Usage (from repo root):
    python -m hydrion.eval_ppo_hydrienv_v1
    python -m hydrion.eval_ppo_hydrienv_v1 --model models/ppo_hydrienv_v1.zip --episodes 5
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


_HYDRIENV_SAFETY_CFG = SafetyConfig(
    max_pressure_soft=0.85,
    max_pressure_hard=1.05,
    max_clog_soft=0.80,
    max_clog_hard=0.98,
    terminate_on_hard_violation=True,
    hard_violation_penalty=10.0,
)

_MAX_STEPS_EVAL = 400


def make_env(seed: int = 0):
    def _init():
        env = HydrionEnv(
            config_path="configs/default.yaml",
            seed=seed,
            noise_enabled=True,
        )
        env.max_steps = _MAX_STEPS_EVAL
        env = ShieldedEnv(env, cfg=_HYDRIENV_SAFETY_CFG)
        env.reset(seed=seed)
        return env
    return _init


def _run_ppo_episodes(vec_env, model, n_episodes: int) -> dict:
    """Run PPO policy for n_episodes. Collect returns, safety data, sensor diagnostics."""
    returns, ep_lengths, violations = [], [], []
    # Per-episode sensor channel accumulators
    obs12_means, obs13_means = [], []
    obs12_stds,  obs13_stds  = [], []
    obs12_minus_obs0_max = []   # dual-channel distinction
    obs13_minus_obs1_max = []   # dual-channel distinction

    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_ret, ep_steps, ep_viol = 0.0, 0, 0
        ep_obs12, ep_obs13 = [], []
        ep_obs0,  ep_obs1  = [], []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info_arr = vec_env.step(action)
            done = bool(done_arr[0])
            ep_ret   += float(reward[0])
            ep_steps += 1

            # Collect VecNormalize-scaled obs values for sensor diagnostic
            ep_obs12.append(float(obs[0][12]))
            ep_obs13.append(float(obs[0][13]))
            ep_obs0.append(float(obs[0][0]))
            ep_obs1.append(float(obs[0][1]))

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

        ep12 = np.array(ep_obs12)
        ep13 = np.array(ep_obs13)
        ep0  = np.array(ep_obs0)
        ep1  = np.array(ep_obs1)

        obs12_means.append(float(np.mean(ep12)))
        obs13_means.append(float(np.mean(ep13)))
        obs12_stds.append(float(np.std(ep12)))
        obs13_stds.append(float(np.std(ep13)))
        obs12_minus_obs0_max.append(float(np.max(np.abs(ep12 - ep0))))
        obs13_minus_obs1_max.append(float(np.max(np.abs(ep13 - ep1))))

        print(
            f"  [PPO ] ep {ep:02d}: "
            f"return={ep_ret:8.3f}  steps={ep_steps}  violations={ep_viol}  "
            f"obs[12]_mean={obs12_means[-1]:.4f}  obs[13]_mean={obs13_means[-1]:.4f}"
        )

    return {
        "mean_return":    float(np.mean(returns)),
        "std_return":     float(np.std(returns)),
        "mean_ep_length": float(np.mean(ep_lengths)),
        "mean_violations": float(np.mean(violations)),
        # Sensor diagnostics — aggregated across all episodes
        "obs12_global_mean": float(np.mean(obs12_means)),
        "obs12_global_std":  float(np.mean(obs12_stds)),
        "obs13_global_mean": float(np.mean(obs13_means)),
        "obs13_global_std":  float(np.mean(obs13_stds)),
        "obs12_minus_obs0_max_across_eps": float(np.max(obs12_minus_obs0_max)),
        "obs13_minus_obs1_max_across_eps": float(np.max(obs13_minus_obs1_max)),
    }


def _run_random_episodes(n_episodes: int, seed: int = 300) -> dict:
    """Run random-action baseline. No VecNormalize — raw env."""
    rand_vec = DummyVecEnv([make_env(seed=seed)])
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
    model_path: str = "models/ppo_hydrienv_v1.zip",
    vecnorm_path: str = "models/ppo_hydrienv_v1_vecnorm.pkl",
    n_episodes: int = 5,
) -> None:

    print(f"\n{'='*68}")
    print(f"HydrionEnv obs14_v1 Evaluation — ppo_hydrienv_v1")
    print(f"{'='*68}")

    # ── Artifact check — fail fast ─────────────────────────────────────────
    if not os.path.exists(model_path) or not os.path.exists(vecnorm_path):
        raise FileNotFoundError(
            f"ppo_hydrienv_v1 artifacts not found. Run train_ppo_hydrienv_v1 first.\n"
            f"  Expected: {model_path}\n"
            f"  Expected: {vecnorm_path}"
        )

    # ── Load model and apply schema lock ───────────────────────────────────
    ppo_vec = DummyVecEnv([make_env(seed=100)])
    ppo_vec = VecNormalize.load(vecnorm_path, ppo_vec)
    ppo_vec.training    = False
    ppo_vec.norm_reward = False
    model = PPO.load(model_path, env=ppo_vec)

    _loaded_shape = model.observation_space.shape
    assert _loaded_shape == (14,), (
        f"[SCHEMA LOCK FAILED] Loaded model expects {_loaded_shape}, not (14,). "
        f"Wrong artifact loaded — check model path."
    )
    print(f"[SCHEMA LOCK OK] Loaded model observation_space.shape = {_loaded_shape}")
    print(f"Model:    {model_path}")
    print(f"VecNorm:  {vecnorm_path}")
    print(f"Episodes: {n_episodes}  seed=100  deterministic=True\n")

    # ── PPO evaluation ─────────────────────────────────────────────────────
    print(f"Evaluating PPO policy ({n_episodes} episodes, seed=100, deterministic=True)...")
    ppo_stats = _run_ppo_episodes(ppo_vec, model, n_episodes)
    ppo_vec.close()

    # ── Random baseline ────────────────────────────────────────────────────
    print(f"\nEvaluating random-action baseline ({n_episodes} episodes, seed=300)...")
    rand_stats = _run_random_episodes(n_episodes, seed=300)

    # ── Sensor channel diagnostics ─────────────────────────────────────────
    print(f"\n{'─'*68}")
    print(f"Sensor channel diagnostics (obs values are VecNormalize-normalized):")
    print(f"  obs[12] (flow_sensor_norm):  mean={ppo_stats['obs12_global_mean']:.4f}  "
          f"std={ppo_stats['obs12_global_std']:.4f}")
    print(f"  obs[13] (dp_sensor_norm):    mean={ppo_stats['obs13_global_mean']:.4f}  "
          f"std={ppo_stats['obs13_global_std']:.4f}")
    print(f"  Dual-channel distinction (max|obs[i] - truth_analog| across all episodes):")
    print(f"    max|obs[12] - obs[0]| = {ppo_stats['obs12_minus_obs0_max_across_eps']:.4f}")
    print(f"    max|obs[13] - obs[1]| = {ppo_stats['obs13_minus_obs1_max_across_eps']:.4f}")

    # ── AC checks ─────────────────────────────────────────────────────────
    ac7_pass  = ppo_stats["mean_return"] > rand_stats["mean_return"]
    ac8_12_pass = (abs(ppo_stats["obs12_global_mean"]) > 0.001 and
                   ppo_stats["obs12_global_std"] > 0)
    ac8_13_pass = (abs(ppo_stats["obs13_global_mean"]) > 0.001 and
                   ppo_stats["obs13_global_std"] > 0)
    ac9_pass  = (ppo_stats["obs12_minus_obs0_max_across_eps"] > 0 or
                 ppo_stats["obs13_minus_obs1_max_across_eps"] > 0)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"{'Metric':<32} {'PPO':>12} {'Random':>12}")
    print(f"{'─'*68}")
    print(f"  {'Mean return':<30} {ppo_stats['mean_return']:>12.3f} "
          f"{rand_stats['mean_return']:>12.3f}")
    print(f"  {'Std return':<30} {ppo_stats['std_return']:>12.3f} "
          f"{rand_stats['std_return']:>12.3f}")
    print(f"  {'Mean ep length':<30} {ppo_stats['mean_ep_length']:>12.1f} "
          f"{rand_stats['mean_ep_length']:>12.1f}")
    print(f"  {'Mean violations':<30} {ppo_stats['mean_violations']:>12.1f} {'—':>12}")
    print(f"{'='*68}")

    print(f"\nAcceptance criteria checks:")
    print(f"  AC7  PPO return > random return:      {'[PASS]' if ac7_pass else '[FAIL]'}"
          f"  ({ppo_stats['mean_return']:.3f} vs {rand_stats['mean_return']:.3f})")
    print(f"  AC8  obs[12] non-zero and varying:    {'[PASS]' if ac8_12_pass else '[FAIL]'}"
          f"  (mean={ppo_stats['obs12_global_mean']:.4f}, std={ppo_stats['obs12_global_std']:.4f})")
    print(f"  AC8  obs[13] non-zero and varying:    {'[PASS]' if ac8_13_pass else '[FAIL]'}"
          f"  (mean={ppo_stats['obs13_global_mean']:.4f}, std={ppo_stats['obs13_global_std']:.4f})")
    print(f"  AC9  dual-channel distinction:        {'[PASS]' if ac9_pass else '[FAIL]'}"
          f"  (max diff obs12={ppo_stats['obs12_minus_obs0_max_across_eps']:.4f}, "
          f"obs13={ppo_stats['obs13_minus_obs1_max_across_eps']:.4f})")

    if not all([ac7_pass, ac8_12_pass, ac8_13_pass, ac9_pass]):
        print(f"\n  [WARNING] One or more AC checks FAILED. Review above.")
    else:
        print(f"\n  All AC7/AC8/AC9 checks passed.")

    print(f"\nNote: This baseline is NOT deployment-ready.")
    print(f"      Truth channels 0-9 are privileged inputs unavailable at hardware deployment.")
    print(f"      Strong PPO performance does NOT prove sensor-only control sufficiency.")
    print(f"      ppo_hydrienv_v1 is NOT cross-comparable to ppo_cce_v2.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ppo_hydrienv_v1 policy (obs14_v1 baseline)"
    )
    parser.add_argument("--model",    default="models/ppo_hydrienv_v1.zip")
    parser.add_argument("--vecnorm",  default="models/ppo_hydrienv_v1_vecnorm.pkl")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.model, args.vecnorm, args.episodes)


if __name__ == "__main__":
    main()
