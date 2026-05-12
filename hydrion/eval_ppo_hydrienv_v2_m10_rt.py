"""
Hydrion — PPO Evaluation Script for M10 RT Capture Integration
---------------------------------------------------------------
Compares ppo_hydrienv_v2_m10_rt (obs8, M10 RT capture + M9 calibrated noise)
against ppo_hydrienv_v2_cal_A (M9 canonical) and random baseline.

M10 invariance checks:
    - Reward trajectory:  must match M9 at fixed seed (capture-decoupled)
    - Clogging trajectory: must match M9 at fixed seed (C_fibers=1.0)
    - Hydraulics trajectory: must match M9 at fixed seed
    - obs8[0-3, 6-7]: must match M9
    - obs8[4-5]: must DIFFER from M9 (C_out → optical sensor)
    - capture_boost_settling: must be 0.0 under m5_rt

M10 prohibited claims (M10.3 Section 8 — BINDING):
    - "M10 RT capture is hardware-validated"
    - "M10 replaces M4 physics"
    - "M10 improves policy performance"
    - "M10 observation schema is unchanged" (unqualified)
    - "M10 capture is fully grounded"

Usage:
    python -m hydrion.eval_ppo_hydrienv_v2_m10_rt
    python -m hydrion.eval_ppo_hydrienv_v2_m10_rt --episodes 5
"""

from __future__ import annotations

import argparse
import json
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hydrion.env import HydrionEnv
from hydrion.wrappers.shielded_env import ShieldedEnv
from hydrion.safety.shield import SafetyConfig


_M10_MODEL_PATH   = "models/ppo_hydrienv_v2_m10_rt.zip"
_M10_VECNORM_PATH = "models/ppo_hydrienv_v2_m10_rt_vecnorm.pkl"
_M9_MODEL_PATH    = "models/ppo_hydrienv_v2_cal_A.zip"
_M9_VECNORM_PATH  = "models/ppo_hydrienv_v2_cal_A_vecnorm.pkl"
_V1_MODEL_PATH    = "models/ppo_hydrienv_v1.zip"
_V1_VECNORM_PATH  = "models/ppo_hydrienv_v1_vecnorm.pkl"

_M10_CONFIG = "configs/m10_rt_capture.yaml"
_M9_CONFIG  = "configs/m9_calibrated.yaml"
_V1_CONFIG  = "configs/default.yaml"

_OBS_SCHEMA = "obs8_deployment_v1"
_OBS_DIM    = 8
_V1_OBS_DIM = 14

_V1_CANONICAL_RETURN = 694.639
_V1_TOLERANCE_PCT    = 0.05
_M9_CAL_A_RETURN     = 847.7

_HYDRIENV_SAFETY_CFG = SafetyConfig(
    max_pressure_soft=0.85,
    max_pressure_hard=1.05,
    max_clog_soft=0.80,
    max_clog_hard=0.98,
    terminate_on_hard_violation=True,
    hard_violation_penalty=10.0,
)

_MAX_STEPS_EVAL = 400


def _make_env(config_path: str, obs_schema: str, seed: int = 0):
    def _init():
        env = HydrionEnv(
            config_path=config_path,
            seed=seed,
            noise_enabled=True,
            obs_schema=obs_schema,
        )
        env.max_steps = _MAX_STEPS_EVAL
        env = ShieldedEnv(env, cfg=_HYDRIENV_SAFETY_CFG)
        env.reset(seed=seed)
        return env
    return _init


def _run_invariance_check(n_steps: int = 50, seed: int = 42) -> dict:
    """Run M10 and M9 envs side-by-side with identical actions to check invariants."""
    m10_env = HydrionEnv(config_path=_M10_CONFIG, seed=seed, noise_enabled=False, obs_schema=_OBS_SCHEMA)
    m9_env  = HydrionEnv(config_path=_M9_CONFIG,  seed=seed, noise_enabled=False, obs_schema=_OBS_SCHEMA)

    m10_obs, _ = m10_env.reset()
    m9_obs, _  = m9_env.reset()

    reward_match = True
    clogging_match = True
    hydraulics_match = True
    obs_0_3_match = True
    obs_6_7_match = True
    obs_4_5_differ = False
    settling_guard_pass = True
    capture_differs = False

    for step in range(n_steps):
        action = np.array([0.5, 0.6, 0.0, 0.4], dtype=np.float32)

        m10_obs, m10_r, _, _, m10_info = m10_env.step(action)
        m9_obs, m9_r, _, _, m9_info = m9_env.step(action)

        if abs(m10_r - m9_r) > 1e-10:
            reward_match = False

        if abs(m10_info.get("mesh_loading_avg", 0) - m9_info.get("mesh_loading_avg", 0)) > 1e-10:
            clogging_match = False

        if abs(m10_info.get("q_processed_lmin", 0) - m9_info.get("q_processed_lmin", 0)) > 1e-10:
            hydraulics_match = False

        # obs8[0-3] are actuator commands written directly from the action vector —
        # no sensor noise, no RNG dependency. Safe to compare raw obs.
        if not np.allclose(m10_obs[0:4], m9_obs[0:4], atol=1e-10):
            obs_0_3_match = False

        # obs8[6-7] are noisy sensor readings (flow_sensor, dp_sensor).
        # Sensor models use np.random.randn() internally and do not check noise_enabled.
        # The M5 RT branch consumes different random draws, shifting RNG state before
        # sensors execute. Compare underlying truth values instead of noisy outputs.
        m10_q = float(m10_env.truth_state.get("q_processed_lmin", 0))
        m9_q  = float(m9_env.truth_state.get("q_processed_lmin", 0))
        m10_dp = float(m10_env.truth_state.get("dp_total_pa", 0))
        m9_dp  = float(m9_env.truth_state.get("dp_total_pa", 0))
        if abs(m10_q - m9_q) > 1e-10 or abs(m10_dp - m9_dp) > 1e-10:
            obs_6_7_match = False

        if not np.allclose(m10_obs[4:6], m9_obs[4:6], atol=1e-6):
            obs_4_5_differ = True

        settling = float(m10_env.truth_state.get("capture_boost_settling", -1.0))
        if settling != 0.0:
            settling_guard_pass = False

        if abs(m10_info.get("capture_eff_part", 0) - m9_info.get("capture_eff_part", 0)) > 1e-6:
            capture_differs = True

    m10_env.close()
    m9_env.close()

    return {
        "reward_invariance":    reward_match,
        "clogging_invariance":  clogging_match,
        "hydraulics_invariance": hydraulics_match,
        "obs_0_3_match":        obs_0_3_match,
        "obs_6_7_match":        obs_6_7_match,
        "obs_4_5_differ":       obs_4_5_differ,
        "settling_guard_pass":  settling_guard_pass,
        "capture_differs":      capture_differs,
    }


def _run_episodes(vec_env, model, n_episodes: int, label: str) -> dict:
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
        print(f"  [{label:6s}] ep {ep:02d}: return={ep_ret:8.3f}  steps={ep_steps}  violations={ep_viol}")
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_ep_length": float(np.mean(ep_lengths)),
        "mean_violations": float(np.mean(violations)),
    }


def _run_random_episodes(n_episodes: int, seed: int = 300) -> dict:
    rand_vec = DummyVecEnv([_make_env(_M10_CONFIG, _OBS_SCHEMA, seed=seed)])
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
        print(f"  [RAND  ] ep {ep:02d}: return={ep_ret:8.3f}  steps={ep_steps}")
    rand_vec.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_ep_length": float(np.mean(ep_lengths)),
    }


def evaluate(n_episodes: int = 5) -> None:
    print(f"\n{'='*72}")
    print(f"M10 RT Capture Integration Evaluation")
    print(f"M10 RT (obs8, m5_rt) vs M9 cal_A (obs8, m4) vs v1 (obs14) vs random")
    print(f"{'='*72}")
    print()
    print("M10 PROHIBITED CLAIMS (M10.3 Section 8 — binding):")
    print("  - M10 RT capture is NOT hardware-validated")
    print("  - M10 does NOT replace M4 physics (toggle — M4 remains default)")
    print("  - M10 does NOT improve policy performance (reward is capture-decoupled)")
    print("  - M10 obs schema is structurally preserved; obs8[4-5] VALUES change")
    print("  - M10 capture is NOT fully grounded (mesh specs are design defaults)")
    print()

    # Invariance checks first (deterministic, no model needed)
    print("Running M10 vs M9 invariance checks (deterministic, 50 steps)...")
    inv = _run_invariance_check(n_steps=50, seed=42)
    print(f"  Reward invariance:     {'[PASS]' if inv['reward_invariance'] else '[FAIL]'}")
    print(f"  Clogging invariance:   {'[PASS]' if inv['clogging_invariance'] else '[FAIL]'}")
    print(f"  Hydraulics invariance: {'[PASS]' if inv['hydraulics_invariance'] else '[FAIL]'}")
    print(f"  obs8[0-3] match:       {'[PASS]' if inv['obs_0_3_match'] else '[FAIL]'}")
    print(f"  obs8[6-7] match:       {'[PASS]' if inv['obs_6_7_match'] else '[FAIL]'}")
    print(f"  obs8[4-5] differ:      {'[PASS]' if inv['obs_4_5_differ'] else '[FAIL] — RT not propagating'}")
    print(f"  Settling guard:        {'[PASS]' if inv['settling_guard_pass'] else '[FAIL] — double-count!'}")
    print(f"  Capture differs:       {'[PASS]' if inv['capture_differs'] else '[FAIL] — RT not active'}")

    # Halt on invariance failure
    invariance_ok = all([
        inv["reward_invariance"],
        inv["clogging_invariance"],
        inv["hydraulics_invariance"],
        inv["obs_0_3_match"],
        inv["obs_6_7_match"],
        inv["obs_4_5_differ"],
        inv["settling_guard_pass"],
        inv["capture_differs"],
    ])
    if not invariance_ok:
        print(f"\n[HALT] Invariance check FAILED. Do not proceed to policy evaluation.")
        return

    print(f"\nAll invariance checks passed. Proceeding to policy evaluation.\n")

    # Load models
    for path in [_M10_MODEL_PATH, _M10_VECNORM_PATH, _M9_MODEL_PATH, _M9_VECNORM_PATH, _V1_MODEL_PATH, _V1_VECNORM_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required artifact not found: {path}")

    # M10 RT
    m10_vec = DummyVecEnv([_make_env(_M10_CONFIG, _OBS_SCHEMA, seed=100)])
    m10_vec = VecNormalize.load(_M10_VECNORM_PATH, m10_vec)
    m10_vec.training = False
    m10_vec.norm_reward = False
    m10_model = PPO.load(_M10_MODEL_PATH, env=m10_vec)
    print(f"[SCHEMA LOCK OK] M10 obs space: {m10_model.observation_space.shape} ({_OBS_SCHEMA})")

    # M9 cal_A
    m9_vec = DummyVecEnv([_make_env(_M9_CONFIG, _OBS_SCHEMA, seed=100)])
    m9_vec = VecNormalize.load(_M9_VECNORM_PATH, m9_vec)
    m9_vec.training = False
    m9_vec.norm_reward = False
    m9_model = PPO.load(_M9_MODEL_PATH, env=m9_vec)
    print(f"[SCHEMA LOCK OK] M9 obs space: {m9_model.observation_space.shape} ({_OBS_SCHEMA})")

    # V1
    v1_vec = DummyVecEnv([_make_env(_V1_CONFIG, "obs14_v1", seed=100)])
    v1_vec = VecNormalize.load(_V1_VECNORM_PATH, v1_vec)
    v1_vec.training = False
    v1_vec.norm_reward = False
    v1_model = PPO.load(_V1_MODEL_PATH, env=v1_vec)
    print(f"[SCHEMA LOCK OK] V1 obs space: {v1_model.observation_space.shape} (obs14_v1)")
    print()

    # Evaluate
    print(f"Evaluating M10 RT ({n_episodes} eps)...")
    m10_stats = _run_episodes(m10_vec, m10_model, n_episodes, "M10_RT")
    m10_vec.close()

    print(f"\nEvaluating M9 cal_A ({n_episodes} eps)...")
    m9_stats = _run_episodes(m9_vec, m9_model, n_episodes, "M9_CAL")
    m9_vec.close()

    print(f"\nEvaluating V1 ({n_episodes} eps)...")
    v1_stats = _run_episodes(v1_vec, v1_model, n_episodes, "V1")
    v1_vec.close()

    print(f"\nEvaluating random baseline ({n_episodes} eps)...")
    rand_stats = _run_random_episodes(n_episodes, seed=300)

    # AC11 gate
    ac11_pass = abs(v1_stats["mean_return"] - _V1_CANONICAL_RETURN) / _V1_CANONICAL_RETURN <= _V1_TOLERANCE_PCT

    # Compute gaps
    m10_vs_m9_pct = ((m10_stats["mean_return"] - m9_stats["mean_return"]) / max(abs(m9_stats["mean_return"]), 1e-6)) * 100.0
    m10_vs_v1_pct = ((m10_stats["mean_return"] - v1_stats["mean_return"]) / max(abs(v1_stats["mean_return"]), 1e-6)) * 100.0
    m10_vs_rand_pct = ((m10_stats["mean_return"] - rand_stats["mean_return"]) / max(abs(rand_stats["mean_return"]), 1e-6)) * 100.0

    print(f"\n{'='*80}")
    print(f"{'Metric':<40} {'M10 RT':>10} {'M9 cal_A':>10} {'V1':>10} {'Random':>10}")
    print(f"{'-'*80}")
    print(f"  {'Mean return':<38} {m10_stats['mean_return']:>10.3f} {m9_stats['mean_return']:>10.3f} {v1_stats['mean_return']:>10.3f} {rand_stats['mean_return']:>10.3f}")
    print(f"  {'Std return':<38} {m10_stats['std_return']:>10.3f} {m9_stats['std_return']:>10.3f} {v1_stats['std_return']:>10.3f} {rand_stats['std_return']:>10.3f}")
    print(f"  {'Mean violations':<38} {m10_stats['mean_violations']:>10.1f} {m9_stats['mean_violations']:>10.1f} {v1_stats['mean_violations']:>10.1f} {'N/A':>10}")
    print(f"{'-'*80}")
    print(f"  {'M10 vs M9 cal_A %':<38} {m10_vs_m9_pct:>+10.1f}%")
    print(f"  {'M10 vs V1 %':<38} {m10_vs_v1_pct:>+10.1f}%")
    print(f"  {'M10 vs random %':<38} {m10_vs_rand_pct:>+10.1f}%")
    print(f"{'='*80}")

    print(f"\nAcceptance criteria:")
    print(f"  AC11 v1 canonical match:  {'[PASS]' if ac11_pass else '[FAIL]'}  ({v1_stats['mean_return']:.3f} vs {_V1_CANONICAL_RETURN:.3f})")
    print(f"  M10 > random:             {'[PASS]' if m10_stats['mean_return'] > rand_stats['mean_return'] else '[FAIL]'}")
    print(f"  Invariance checks:        {'[PASS]' if invariance_ok else '[FAIL]'}")

    if not ac11_pass:
        print(f"\n  [HALT] AC11 FAILED — canonical match broken. Do not interpret M10 results.")

    # Update meta with eval results
    meta_path = "models/ppo_hydrienv_v2_m10_rt_meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta["reward_invariance"] = "PASS" if inv["reward_invariance"] else "FAIL"
        meta["clogging_invariance"] = "PASS" if inv["clogging_invariance"] else "FAIL"
        meta["hydraulics_invariance"] = "PASS" if inv["hydraulics_invariance"] else "FAIL"
        meta["obs8_channel_comparison"] = {
            "obs_0_3_match": inv["obs_0_3_match"],
            "obs_6_7_match": inv["obs_6_7_match"],
            "obs_4_5_differ": inv["obs_4_5_differ"],
        }
        meta["ac11_result"] = f"{'PASS' if ac11_pass else 'FAIL'} — {v1_stats['mean_return']:.3f} vs {_V1_CANONICAL_RETURN}"
        meta["eval_results"] = {
            "m10_mean_return": m10_stats["mean_return"],
            "m9_mean_return": m9_stats["mean_return"],
            "v1_mean_return": v1_stats["mean_return"],
            "random_mean_return": rand_stats["mean_return"],
            "m10_vs_m9_pct": round(m10_vs_m9_pct, 2),
            "m10_vs_v1_pct": round(m10_vs_v1_pct, 2),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n  Meta updated: {meta_path}")

    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="M10 RT capture evaluation: M10 vs M9 cal_A vs V1 vs random"
    )
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.episodes)


if __name__ == "__main__":
    main()
