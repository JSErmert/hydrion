"""
Hydrion Digital Twin — PPO Training Script (v1.5, Silent)
--------------------------------------------------------
Trains PPO on Hydrion v1.5 with:
- Structural safety (ShieldedEnv)
- Truth/sensor separation
- Safety penalties & termination
- Run-level causal logging

Prints only:
- Start message
- Checkpoint saves
- Final completion message
"""

import time
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from hydrion.env import HydrionEnv
from hydrion.wrappers.shielded_env import ShieldedEnv
from hydrion.safety.shield import SafetyConfig


# --------------------------------------------------------------
# Custom silent checkpoint callback
# --------------------------------------------------------------
class QuietCheckpointCallback(CheckpointCallback):
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            print(f"[Checkpoint] Saved at step {self.n_calls}")
        return True


# --------------------------------------------------------------
# ENV FACTORY (v1.5 SAFE)
# --------------------------------------------------------------
def make_env(seed: int = 0):
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
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


# --------------------------------------------------------------
# MAIN TRAINING
# --------------------------------------------------------------
def main():

    # Vectorized environment (1 env, but future-proof)
    env = DummyVecEnv([make_env(seed=42)])

    # Normalize observations + rewards
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        training=True,
    )

    # Logging directory
    run_name = f"ppo_hydrion_v15_{int(time.time())}"
    tensorboard_log = f"./runs/{run_name}"

    print("\n🚀 Starting PPO training on Hydrion v1.5 (silent mode)...\n")

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
        verbose=0,                    # QUIET MODE
        tensorboard_log=tensorboard_log,
    )

    checkpoint_callback = QuietCheckpointCallback(
        save_freq=10_000,
        save_path="./checkpoints/",
        name_prefix="ppo_hydrion_v15",
        verbose=0,
    )

    model.learn(
        total_timesteps=500_000,
        callback=checkpoint_callback,
        progress_bar=False,
    )

    # Save final model + normalization
    model.save("ppo_hydrion_v15_final")
    env.save("ppo_hydrion_v15_vecnormalize.pkl")

    print("\n🎉 PPO v1.5 training complete! Model + normalization saved.\n")


if __name__ == "__main__":
    main()
