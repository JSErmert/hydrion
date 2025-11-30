# hydrion/visualize_episode.py
from __future__ import annotations

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from hydrion.env import HydrionEnv
from hydrion.utils.episode_recorder import rollout_episode
from hydrion.rendering.viz2d import plot_episode_timeseries


MODEL_PATH = "ppo_hydrion_final_12d.zip"


def make_env():
    return HydrionEnv()


def visualize_with_policy(model_path=MODEL_PATH):
    # Load env + model
    vec_env = DummyVecEnv([make_env])
    model = PPO.load(model_path, env=vec_env)

    # Rollout one episode
    history = rollout_episode(vec_env, policy=model, deterministic=True)

    # Plot results
    plot_episode_timeseries(history)


def visualize_random():
    env = HydrionEnv()
    history = rollout_episode(env, policy=None)
    plot_episode_timeseries(history)


if __name__ == "__main__":
    visualize_with_policy()
