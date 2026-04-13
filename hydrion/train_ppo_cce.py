"""
Hydrion — PPO Training Script for ConicalCascadeEnv (M5 Physics)
-----------------------------------------------------------------
Trains PPO on the M5 conical cascade environment with:
    - Randomized initial fouling (episode diversity)
    - ShieldedEnv (Safe RL: pressure/clog hard limits + termination)
    - VecNormalize (observation + reward normalization)
    - TensorBoard logging
    - Deterministic seeding (Constraint 6)

Saves:
    models/ppo_cce_v1.zip          — final policy
    models/ppo_cce_v1_vecnorm.pkl  — VecNormalize statistics
    models/ppo_cce_v1_meta.json    — obs schema + reward version metadata

Checkpoints every 10k steps to checkpoints/cce/

Run from repo root:
    python -m hydrion.train_ppo_cce
"""

import json
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from hydrion.environments.conical_cascade_env import ConicalCascadeEnv
from hydrion.wrappers.shielded_env import ShieldedEnv
from hydrion.safety.shield import SafetyConfig


_TRAIN_SEED  = 42
_OBS_SCHEMA  = "obs12_v2"   # version-lock — must match ConicalCascadeEnv observation space

_CCE_SAFETY_CFG = SafetyConfig(
    max_pressure_soft=0.75,       # 60 kPa — soft warning
    max_pressure_hard=1.00,       # 80 kPa — hard limit (terminates episode)
    max_clog_soft=0.70,           # soft clog warning
    max_clog_hard=0.95,           # hard clog limit
    terminate_on_hard_violation=True,
    hard_violation_penalty=10.0,  # -10 on termination (matches Stage 3 report)
)


def _set_global_seeds(seed: int) -> None:
    """Constraint 6: training must be deterministic from a fixed seed."""
    import random as _random
    os.environ["PYTHONHASHSEED"] = str(seed)
    _random.seed(seed)
    np.random.seed(seed)
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        _torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def make_env(seed: int = 0):
    def _init():
        env = ConicalCascadeEnv(
            config_path="configs/default.yaml",
            seed=seed,
            randomize_on_reset=True,
        )
        # Override max_steps for training episodes.
        # 400 steps = 40 s simulated — long enough for fouling + backflush cycle,
        # short enough for many episodes per gradient update.
        env._max_steps = 400
        env = ShieldedEnv(env, cfg=_CCE_SAFETY_CFG)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def main():
    _set_global_seeds(_TRAIN_SEED)
    os.makedirs("models",          exist_ok=True)
    os.makedirs("checkpoints/cce", exist_ok=True)

    vec_env = DummyVecEnv([make_env(seed=_TRAIN_SEED)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        training=True,
    )

    run_name        = f"ppo_cce_{int(time.time())}"
    tensorboard_log = f"./runs/{run_name}"

    print(f"\nStarting PPO training on ConicalCascadeEnv (M5 physics)...")
    print(f"TensorBoard log: {tensorboard_log}")
    print(f"Checkpoints:     checkpoints/cce/")
    print(f"Final model:     models/ppo_cce_v1.zip\n")

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        ent_coef=0.01,
        learning_rate=3e-4,
        clip_range=0.2,
        seed=_TRAIN_SEED,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path="./checkpoints/cce/",
        name_prefix="ppo_cce",
        verbose=1,
    )

    model.learn(
        total_timesteps=500_000,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    model.save("models/ppo_cce_v1")
    vec_env.save("models/ppo_cce_v1_vecnorm.pkl")

    # Constraint 2: embed obs schema version so the serving path can verify
    # the schema before loading the model.
    with open("models/ppo_cce_v1_meta.json", "w") as f:
        json.dump({
            "obs_schema":      _OBS_SCHEMA,
            "action_schema":   "act4_v1",
            "train_seed":      _TRAIN_SEED,
            "total_timesteps": 500_000,
            "reward_version":  "phase1_v1",
        }, f, indent=2)

    print("\nTraining complete.")
    print("  Model:    models/ppo_cce_v1.zip")
    print("  VecNorm:  models/ppo_cce_v1_vecnorm.pkl")
    print("  Meta:     models/ppo_cce_v1_meta.json")


if __name__ == "__main__":
    main()
