"""
Hydrion — PPO Evaluation Script for ConicalCascadeEnv
------------------------------------------------------
Evaluates a trained PPO policy vs heuristic and random baselines on CCE.

Constraint 5: PPO must beat both random AND heuristic baselines to be
considered non-trivial. Heuristic: full voltage, nominal flow, BF at fouling > 0.6.

Usage (from repo root):
    python -m hydrion.eval_ppo_cce
    python -m hydrion.eval_ppo_cce --model models/ppo_cce_v1.zip --episodes 5
"""

from __future__ import annotations

import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hydrion.environments.conical_cascade_env import ConicalCascadeEnv
from hydrion.wrappers.shielded_env import ShieldedEnv
from hydrion.safety.shield import SafetyConfig


_CCE_SAFETY_CFG = SafetyConfig(
    max_pressure_soft=0.75,
    max_pressure_hard=1.00,
    max_clog_soft=0.70,
    max_clog_hard=0.95,
    terminate_on_hard_violation=True,
    hard_violation_penalty=10.0,
)

_MAX_STEPS_EVAL = 400


def make_env(seed: int):
    def _init():
        env = ConicalCascadeEnv(
            config_path="configs/default.yaml",
            seed=seed,
            randomize_on_reset=False,   # deterministic eval — always clean start
        )
        env._max_steps = _MAX_STEPS_EVAL
        env = ShieldedEnv(env, cfg=_CCE_SAFETY_CFG)
        env.reset(seed=seed)
        return env
    return _init


class HeuristicPolicy:
    """
    Rule-based baseline (Constraint 5).

    Strategy: full voltage, nominal flow, trigger backflush when fouling_mean > 0.6.
    This is the non-trivial baseline — PPO must beat it to demonstrate learning.

    obs12_v2 layout:
        [0] q_in          [1] delta_p       [2] fouling_mean  [3] eta_cascade
        [4] C_in          [5] C_out         [6] E_field_norm  [7] v_crit_norm
        [8] step_norm     [9] bf_active     [10] eta_PP       [11] eta_PET
    """
    def predict(self, obs: np.ndarray, deterministic: bool = True):
        fouling_mean = float(obs[0][2])   # obs is (1, 12) from VecEnv
        bf_cmd = 1.0 if fouling_mean > 0.6 else 0.0
        action = np.array([[0.7, 0.7, bf_cmd, 1.0]], dtype=np.float32)
        return action, None


def _run_episodes(
    vec_env,
    model,
    n_episodes: int,
    policy_label: str,
    use_vecnorm: bool = False,
) -> dict:
    """Run n_episodes and collect per-episode statistics."""
    returns, eta_list, shield_violations, bf_steps_list, ep_lengths = [], [], [], [], []

    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_ret, ep_eta_sum, ep_steps = 0.0, 0.0, 0
        ep_shield_v, ep_bf = 0, 0

        while not done:
            if use_vecnorm and hasattr(model, 'predict'):
                action, _ = model.predict(obs, deterministic=True)
            elif hasattr(model, 'predict'):
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = vec_env.action_space.sample()[np.newaxis, :]

            obs, reward, done_arr, info_arr = vec_env.step(action)
            done = bool(done_arr[0])
            ep_ret += float(reward[0])
            ep_steps += 1

            info = info_arr[0]
            ep_eta_sum += float(info.get("eta_cascade", 0.0))

            safety = info.get("safety", {})
            if any([
                safety.get("soft_pressure_violation"),
                safety.get("hard_pressure_violation"),
                safety.get("soft_clog_violation"),
                safety.get("hard_clog_violation"),
                safety.get("blockage_violation"),
            ]):
                ep_shield_v += 1

            # Count steps where backflush is active
            action_flat = action[0] if hasattr(action, '__len__') else action
            if float(action_flat[2]) > 0.5:
                ep_bf += 1

        returns.append(ep_ret)
        eta_list.append(ep_eta_sum / max(ep_steps, 1))
        shield_violations.append(ep_shield_v)
        bf_steps_list.append(ep_bf)
        ep_lengths.append(ep_steps)

        print(
            f"  [{policy_label:4s}] ep {ep:02d}: "
            f"return={ep_ret:7.3f}  "
            f"eta={ep_eta_sum / max(ep_steps, 1):.3f}  "
            f"steps={ep_steps}  "
            f"violations={ep_shield_v}  "
            f"bf_steps={ep_bf}"
        )

    return {
        "mean_return":     float(np.mean(returns)),
        "std_return":      float(np.std(returns)),
        "mean_eta":        float(np.mean(eta_list)),
        "mean_violations": float(np.mean(shield_violations)),
        "mean_bf_steps":   float(np.mean(bf_steps_list)),
        "mean_ep_length":  float(np.mean(ep_lengths)),
    }


def evaluate(
    model_path: str = "models/ppo_cce_v1.zip",
    vecnorm_path: str = "models/ppo_cce_v1_vecnorm.pkl",
    n_episodes: int = 5,
) -> None:

    # ── PPO policy ────────────────────────────────────────────────────────
    if not os.path.exists(model_path) or not os.path.exists(vecnorm_path):
        print(f"Model not found: {model_path} / {vecnorm_path}")
        print("Run `python -m hydrion.train_ppo_cce` first.")
        ppo_stats = None
    else:
        ppo_vec = DummyVecEnv([make_env(seed=100)])
        ppo_vec = VecNormalize.load(vecnorm_path, ppo_vec)
        ppo_vec.training    = False
        ppo_vec.norm_reward = False
        model = PPO.load(model_path, env=ppo_vec)
        print(f"\nEvaluating PPO policy: {model_path}")
        ppo_stats = _run_episodes(ppo_vec, model, n_episodes, "PPO", use_vecnorm=True)
        ppo_vec.close()

    # ── Heuristic baseline ────────────────────────────────────────────────
    heur_vec = DummyVecEnv([make_env(seed=200)])
    print(f"\nEvaluating heuristic baseline ({n_episodes} episodes)...")
    heur_stats = _run_episodes(heur_vec, HeuristicPolicy(), n_episodes, "HEUR")
    heur_vec.close()

    # ── Random baseline ───────────────────────────────────────────────────
    rand_vec = DummyVecEnv([make_env(seed=300)])
    print(f"\nEvaluating random baseline ({n_episodes} episodes)...")
    rand_stats = _run_episodes(rand_vec, None, n_episodes, "RAND")
    rand_vec.close()

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    if ppo_stats:
        print(f"{'Metric':<28} {'PPO':>10} {'Heuristic':>10} {'Random':>10}")
    else:
        print(f"{'Metric':<28} {'Heuristic':>10} {'Random':>10}")
    print("-" * 68)

    metrics = [
        ("Mean return",     "mean_return"),
        ("Mean eta_cascade", "mean_eta"),
        ("Mean violations",  "mean_violations"),
        ("Mean BF steps",   "mean_bf_steps"),
    ]
    for label, key in metrics:
        h = heur_stats[key]
        r = rand_stats[key]
        if ppo_stats:
            p = ppo_stats[key]
            print(f"  {label:<26} {p:>10.3f} {h:>10.3f} {r:>10.3f}")
        else:
            print(f"  {label:<26} {h:>10.3f} {r:>10.3f}")
    print("=" * 68)

    if ppo_stats:
        ppo_eta  = ppo_stats["mean_eta"]
        heur_eta = heur_stats["mean_eta"]
        rand_eta = rand_stats["mean_eta"]
        print(f"\nConvergence criteria (Constraint 5):")
        print(f"  PPO vs Random eta ratio:    {ppo_eta / max(rand_eta, 1e-6):.2f}x  (need > 3.0x)")
        print(f"  PPO vs Heuristic eta ratio: {ppo_eta / max(heur_eta, 1e-6):.2f}x  (need >= 1.0x)")
    else:
        print(f"\nHeuristic vs Random eta ratio: {heur_stats['mean_eta'] / max(rand_stats['mean_eta'], 1e-6):.2f}x")
        print("(No trained model found — heuristic/random comparison only)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO-CCE policy")
    parser.add_argument("--model",    default="models/ppo_cce_v1.zip")
    parser.add_argument("--vecnorm",  default="models/ppo_cce_v1_vecnorm.pkl")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.model, args.vecnorm, args.episodes)


if __name__ == "__main__":
    main()
