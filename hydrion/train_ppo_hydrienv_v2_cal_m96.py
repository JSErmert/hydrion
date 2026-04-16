"""
M9.6 — Training Stability Recovery: Config A and Config B
----------------------------------------------------------
Controlled hyperparameter correction pass under calibrated sensor noise.
Uses configs/m9_calibrated.yaml (SP1 preserved).

Config A (conservative):  lr=1e-4, clip=0.1, batch=128, epochs=15, 500k steps
Config B (stability+data): lr=3e-5, clip=0.1, batch=256, epochs=20, 1M steps

Run:
    python -m hydrion.train_ppo_hydrienv_v2_cal_m96 --config A
    python -m hydrion.train_ppo_hydrienv_v2_cal_m96 --config B
"""

import json
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from hydrion.env import HydrionEnv
from hydrion.wrappers.shielded_env import ShieldedEnv
from hydrion.safety.shield import SafetyConfig


_TRAIN_SEED = 42
_OBS_SCHEMA = "obs8_deployment_v1"
_OBS_DIM = 8
_CONFIG_PATH = "configs/m9_calibrated.yaml"

_HYDRIENV_SAFETY_CFG = SafetyConfig(
    max_pressure_soft=0.85,
    max_pressure_hard=1.05,
    max_clog_soft=0.80,
    max_clog_hard=0.98,
    terminate_on_hard_violation=True,
    hard_violation_penalty=10.0,
)

CONFIGS = {
    "A": {
        "learning_rate": 1e-4,
        "clip_range": 0.1,
        "batch_size": 128,
        "n_epochs": 15,
        "total_timesteps": 500_000,
        "label": "Conservative PPO",
        "suffix": "cal_A",
    },
    "B": {
        "learning_rate": 3e-5,
        "clip_range": 0.1,
        "batch_size": 256,
        "n_epochs": 20,
        "total_timesteps": 1_000_000,
        "label": "Stability + Data",
        "suffix": "cal_B",
    },
}


def _set_global_seeds(seed: int) -> None:
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
        env = HydrionEnv(
            config_path=_CONFIG_PATH,
            seed=seed,
            noise_enabled=True,
            obs_schema=_OBS_SCHEMA,
        )
        env.max_steps = 400
        env = ShieldedEnv(env, cfg=_HYDRIENV_SAFETY_CFG)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="M9.6 Training Stability Recovery — Config A or B"
    )
    parser.add_argument(
        "--config", choices=["A", "B"], required=True,
        help="Training configuration: A (conservative) or B (stability+data)"
    )
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    suffix = cfg["suffix"]
    total_timesteps = cfg["total_timesteps"]

    _schema_check_env = HydrionEnv(config_path=_CONFIG_PATH, obs_schema=_OBS_SCHEMA)
    _actual_shape = _schema_check_env.observation_space.shape
    _schema_check_env.close()
    assert _actual_shape == (_OBS_DIM,), (
        f"[SCHEMA LOCK FAILED] {_OBS_SCHEMA} requires shape ({_OBS_DIM},). Got {_actual_shape}."
    )
    print(f"[SCHEMA LOCK OK] observation_space.shape = {_actual_shape}  ({_OBS_SCHEMA})")
    print(f"[CONFIG] {_CONFIG_PATH}  (SP1)")
    print(f"[M9.6]  Config {args.config}: {cfg['label']}")
    print(f"        lr={cfg['learning_rate']}, clip={cfg['clip_range']}, "
          f"batch={cfg['batch_size']}, epochs={cfg['n_epochs']}, steps={total_timesteps}")

    _set_global_seeds(_TRAIN_SEED)
    os.makedirs("models", exist_ok=True)
    ckpt_dir = f"checkpoints/hydrienv_v2_{suffix}"
    os.makedirs(ckpt_dir, exist_ok=True)

    vec_env = DummyVecEnv([make_env(seed=_TRAIN_SEED)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        training=True,
    )

    run_name = f"ppo_hydrienv_v2_{suffix}_{int(time.time())}"
    tensorboard_log = f"./runs/{run_name}"

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=2048,
        batch_size=cfg["batch_size"],
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=cfg["n_epochs"],
        ent_coef=0.01,
        learning_rate=cfg["learning_rate"],
        clip_range=cfg["clip_range"],
        seed=_TRAIN_SEED,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=f"./{ckpt_dir}/",
        name_prefix=f"ppo_hydrienv_v2_{suffix}",
        verbose=1,
    )

    print(f"\nStarting training — M9.6 Config {args.config}...")
    t0 = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    elapsed = time.time() - t0
    model_path = f"models/ppo_hydrienv_v2_{suffix}"
    vecnorm_path = f"models/ppo_hydrienv_v2_{suffix}_vecnorm.pkl"
    meta_path = f"models/ppo_hydrienv_v2_{suffix}_meta.json"

    model.save(model_path)
    vec_env.save(vecnorm_path)

    with open(meta_path, "w") as f:
        json.dump({
            "obs_schema": _OBS_SCHEMA,
            "obs_dim": _OBS_DIM,
            "config_path": _CONFIG_PATH,
            "train_seed": _TRAIN_SEED,
            "total_timesteps": total_timesteps,
            "m96_config": args.config,
            "m96_label": cfg["label"],
            "hyperparameters": {
                "learning_rate": cfg["learning_rate"],
                "clip_range": cfg["clip_range"],
                "batch_size": cfg["batch_size"],
                "n_epochs": cfg["n_epochs"],
                "n_steps": 2048,
                "gae_lambda": 0.95,
                "gamma": 0.99,
                "ent_coef": 0.01,
            },
            "training_time_seconds": round(elapsed, 1),
            "deployment_ready": False,
            "note": f"M9.6 training stability recovery — Config {args.config}",
        }, f, indent=2)

    print(f"\nTraining complete — Config {args.config} ({cfg['label']})")
    print(f"  Duration:  {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Model:     {model_path}.zip")
    print(f"  VecNorm:   {vecnorm_path}")
    print(f"  Meta:      {meta_path}")


if __name__ == "__main__":
    main()
