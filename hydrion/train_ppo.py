# hydrion/train_ppo_12d.py
"""
Quiet PPO Training on HydrionEnv (12D Observations)

Outputs:
    checkpoints/ppo_hydrion_XXXXXX_steps_12d.zip   (intermediate)
    checkpoints/ppo_hydrion_final_12d.zip          (final model)
    ppo_hydrion_vecnormalize_final_12d.pkl         (VecNormalize stats)

Silent mode: No PPO tables, no logs, no console spam.
"""

from __future__ import annotations
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback

from .env import HydrionEnv


# ============================================================
# PATHS
# ============================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS = os.path.join(ROOT, "checkpoints")
LOGS = os.path.join(ROOT, "logs")
os.makedirs(CHECKPOINTS, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

TOTAL_TIMESTEPS = 500_000
SAVE_EVERY = 100_000
SEED = 42


# ============================================================
# ENV FACTORY
# ============================================================

def make_env():
    def _init():
        return HydrionEnv()
    return _init


# ============================================================
# CALLBACK — Save every N steps
# ============================================================

class SaveEveryNSteps(BaseCallback):
    def __init__(self, save_freq: int, save_dir: str):
        super().__init__(verbose=0)
        self.save_freq = int(save_freq)
        self.save_dir = save_dir

    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0:
            tag = f"{self.num_timesteps:06d}"
            path = os.path.join(self.save_dir, f"ppo_hydrion_{tag}_steps_12d.zip")
            self.model.save(path)
        return True


# ============================================================
# MAIN TRAINING LOGIC
# ============================================================

def main():
    print("🧪 Building HydrionEnv (12D) + VecNormalize...")

    base_env = DummyVecEnv([make_env()])
    vec_env = VecNormalize(base_env, training=True, norm_obs=True, norm_reward=True)

    print("   Observation space:", vec_env.observation_space)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,            # ← QUIET MODE
        n_steps=2048,
        batch_size=64,
        gamma=0.999,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=SEED,
    )

    # Disable all internal logging
    model.set_logger(configure(folder=None, format_strings=[]))

    callback = SaveEveryNSteps(SAVE_EVERY, CHECKPOINTS)

    print(f"🚀 Training PPO for {TOTAL_TIMESTEPS:,} timesteps (silent mode)...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    final_model = os.path.join(CHECKPOINTS, "ppo_hydrion_final_12d.zip")
    vecnorm_path = os.path.join(ROOT, "ppo_hydrion_vecnormalize_final_12d.pkl")

    print("💾 Saving final PPO model:", final_model)
    model.save(final_model)

    print("💾 Saving final VecNormalize stats:", vecnorm_path)
    vec_env.save(vecnorm_path)

    print("✅ Training complete.")


if __name__ == "__main__":
    main()
