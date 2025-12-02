# hydrion/eval_ppo.py
from __future__ import annotations

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from hydrion.env import HydrionEnv

DEFAULT_MODEL_PATH = "ppo_hydrion_final_v3.zip"


def make_env():
    return HydrionEnv(config_path="configs/default.yaml")


def evaluate(model_path=DEFAULT_MODEL_PATH, n_episodes=3):

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    # Create vec env
    vec_env = DummyVecEnv([make_env])

    # Load PPO model
    model = PPO.load(model_path, env=vec_env)
    print(f"Loaded model from {model_path}")

    returns = []

    for ep in range(n_episodes):

        obs = vec_env.reset()
        done = False
        ep_ret = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = vec_env.step(action)

            # VecEnv returns arrays → unpack first env
            done = bool(done_arr[0])
            ep_ret += float(reward[0])
            steps += 1

        returns.append(ep_ret)
        print(f"Episode {ep:02d}: return={ep_ret:.3f}, steps={steps}")

    print("\nMean return:", np.mean(returns))
    print("Std return:", np.std(returns))


def main():
    evaluate(DEFAULT_MODEL_PATH, n_episodes=3)


if __name__ == "__main__":
    main()
