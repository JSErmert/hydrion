"""
Hydrion Digital Twin — PPO Training Script (Silent Version)
-----------------------------------------------------------
Runs PPO quietly, printing only:
- Start message
- Checkpoint saves
- Final completion message
"""

import os
import time
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from hydrion.env import HydrionEnv


# --------------------------------------------------------------
# Custom silent callback (prints only essential info)
# --------------------------------------------------------------
class QuietCheckpointCallback(CheckpointCallback):
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            print(f"[Checkpoint] Saved at step {self.n_calls}")
        return True


# --------------------------------------------------------------
# ENV FACTORY
# --------------------------------------------------------------
def make_env():
    def _init():
        env = HydrionEnv()
        return Monitor(env)   # Monitor is safe, still quiet
    return _init


# --------------------------------------------------------------
# MAIN TRAINING
# --------------------------------------------------------------
def main():

    # Vectorized environment
    env = DummyVecEnv([make_env()])
    
    # Normalize observations + rewards (silent)
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        training=True
    )

    # Logging directory
    run_name = f"ppo_hydrion_{int(time.time())}"
    tensorboard_log = f"./runs/{run_name}"

    print("\n🚀 Starting PPO training (silent mode)...\n")

    # PPO — set verbose=0 to silence output
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        ent_coef=0.01,
        learning_rate=3e-4,
        clip_range=0.2,
        verbose=0,                         # 👈 QUIET MODE
        tensorboard_log=tensorboard_log,
    )

    # Silent checkpoint callback
    checkpoint_callback = QuietCheckpointCallback(
        save_freq=10_000,
        save_path="./checkpoints/",
        name_prefix="ppo_hydrion",
        verbose=0                          # 👈 suppress checkpoint spam
    )

    # TRAINING
    model.learn(
        total_timesteps=500_000,
        callback=checkpoint_callback,
        progress_bar=False                  # 👈 disable ASCII bar
    )

    # Save final model + normalization
    model.save("ppo_hydrion_final_12d")
    env.save("ppo_hydrion_vecnormalize_12d.pkl")

    print("\n🎉 Training complete! Model + normalization saved.\n")


if __name__ == "__main__":
    main()
