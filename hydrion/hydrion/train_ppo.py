"""
Hydrion Digital Twin — PPO Training Script
------------------------------------------
Trains PPO on HydrionEnv with:
- Vectorized environments
- Observation normalization
- Reward scaling
- Logging via TensorBoard
- Checkpoint saving

Author: HydrionRL System
"""

import os
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
)
from stable_baselines3.common.callbacks import CheckpointCallback

# Import your environment
from hydrion.env import HydrionEnv


# --------------------------------------------------------------
# ENV FACTORY
# --------------------------------------------------------------
def make_env():
    """Factory for creating fresh HydrionEnv instances."""
    def _init():
        env = HydrionEnv()
        env = Monitor(env)
        return env
    return _init


# --------------------------------------------------------------
# MAIN TRAINING
# --------------------------------------------------------------
def main():

    # Vectorized environment (recommended for PPO)
    env = DummyVecEnv([make_env()])
    
    # Normalize observations + rewards
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    # Logging directory
    run_name = f"ppo_hydrion_{int(time.time())}"
    tensorboard_log = f"./runs/{run_name}"

    # PPO Hyperparameters
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=2048,          # rollout length
        batch_size=64,         # minibatch size
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        ent_coef=0.01,         # exploration
        learning_rate=3e-4,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )

    # Save checkpoints every X steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="./checkpoints/",
        name_prefix="ppo_hydrion",
    )

    # Train
    print("\n🚀 Starting PPO training...")
    model.learn(
        total_timesteps=500_000,     # increase for deeper training
        callback=checkpoint_callback
    )

    # Save final model + normalization stats
    model.save("ppo_hydrion_final")
    env.save("ppo_hydrion_vecnormalize.pkl")

    print("\n🎉 Training complete! Models saved.")


if __name__ == "__main__":
    main()
