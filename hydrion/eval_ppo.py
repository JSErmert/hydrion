from __future__ import annotations

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hydrion.env import HydrionEnv
from hydrion.wrappers.shielded_env import ShieldedEnv
from hydrion.safety.shield import SafetyConfig


MODEL_PATH = "ppo_hydrion_v15_final.zip"
VECNORM_PATH = "ppo_hydrion_v15_vecnormalize.pkl"


def make_env(seed: int):
    def _init():
        env = HydrionEnv()
        env = ShieldedEnv(
            env,
            cfg=SafetyConfig(
                max_pressure_soft=0.85,
                max_pressure_hard=1.05,
                terminate_on_hard_violation=True,
            ),
        )
        env.reset(seed=seed)
        return env
    return _init


def evaluate(model_path=MODEL_PATH, vecnorm_path=VECNORM_PATH, n_episodes=3):

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(f"VecNormalize not found at: {vecnorm_path}")

    # Vectorized environment (single env, deterministic)
    vec_env = DummyVecEnv([make_env(seed=100)])

    # Load VecNormalize statistics (CRITICAL)
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    # Load PPO model
    model = PPO.load(model_path, env=vec_env)
    print(f"Loaded model from {model_path}")

    returns = []
    lengths = []
    safety_violations = []
    action_projections = []

    for ep in range(n_episodes):

        obs = vec_env.reset()
        done = False
        ep_ret = 0.0
        steps = 0
        ep_safety_violations = 0
        ep_action_projections = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info_arr = vec_env.step(action)

            done = bool(done_arr[0])
            ep_ret += float(reward[0])
            steps += 1

            info = info_arr[0]
            if "safety" in info:
                s = info["safety"]

                # TRUE safety violations (constraint breaches)
                if (
                    s.get("soft_pressure_violation", False)
                    or s.get("hard_pressure_violation", False)
                    or s.get("soft_clog_violation", False)
                    or s.get("hard_clog_violation", False)
                    or s.get("blockage_violation", False)
                ):
                    ep_safety_violations += 1

                # Action projections (control smoothing, not failure)
                if s.get("projected", False):
                    ep_action_projections += 1

        returns.append(ep_ret)
        lengths.append(steps)
        safety_violations.append(ep_safety_violations)
        action_projections.append(ep_action_projections)

        print(
            f"Episode {ep:02d}: "
            f"return={ep_ret:.3f}, "
            f"steps={steps}, "
            f"safety_violations={ep_safety_violations}, "
            f"action_projections={ep_action_projections}"
        )

    print("\n=== Evaluation Summary (Hydrion v1.5 PPO) ===")
    print("Mean return:", np.mean(returns))
    print("Std return:", np.std(returns))
    print("Mean episode length:", np.mean(lengths))
    print("Mean safety violations:", np.mean(safety_violations))
    print("Mean action projections:", np.mean(action_projections))


def main():
    evaluate(n_episodes=3)


if __name__ == "__main__":
    main()
