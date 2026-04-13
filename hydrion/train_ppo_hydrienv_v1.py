"""
Hydrion — PPO Training Script for HydrionEnv (obs14_v1 baseline)
-----------------------------------------------------------------
Trains PPO on HydrionEnv under the obs14_v1 observation schema:
    - obs14_v1: 14D observation (indices 0–9 truth-derived, 10–13 sensor-derived)
    - Schema lock: asserts observation_space.shape == (14,) at startup
    - ShieldedEnv (Safe RL: pressure/clog hard limits + termination)
    - VecNormalize (observation + reward normalization)
    - TensorBoard logging
    - Deterministic seeding

obs14_v1 information regime:
    Truth channels (0–9): privileged simulation-only inputs — NOT available at hardware
    deployment. The trained policy will depend on these channels. This is a documented
    limitation of the M7 baseline; deployment-mode obs design is a future phase.

    Sensor channels (10–13): noisy partial observations of the physical state.
        Index 12: flow_sensor_norm (flow_sensor_lmin / 20.0, clip [0,1])
        Index 13: dp_sensor_norm (dp_sensor_kPa / 80.0, clip [0,1]) — 1-step latency

    Calibration-pending: dp_drift_rate, dp_fouling_gain, flow_calibration_bias_std
    are placeholder values. Domain randomization of sensor params deferred.

Saves:
    models/ppo_hydrienv_v1.zip          — final policy (obs14_v1 baseline)
    models/ppo_hydrienv_v1_vecnorm.pkl  — VecNormalize statistics (obs14_v1 only)
    models/ppo_hydrienv_v1_meta.json    — obs schema + metadata + limitations record

Checkpoints every 10k steps to checkpoints/hydrienv/

Run from repo root:
    python -m hydrion.train_ppo_hydrienv_v1                  # 500k steps (default)
    python -m hydrion.train_ppo_hydrienv_v1 --steps 10000    # proof-of-concept run
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


_TRAIN_SEED  = 42
_D_P_UM      = 1.0          # submicron benchmark regime (consistent with ppo_cce_v2)
_OBS_SCHEMA  = "obs14_v1"   # version-lock — must match HydrionEnv observation space

_HYDRIENV_SAFETY_CFG = SafetyConfig(
    max_pressure_soft=0.85,
    max_pressure_hard=1.05,
    max_clog_soft=0.80,
    max_clog_hard=0.98,
    terminate_on_hard_violation=True,
    hard_violation_penalty=10.0,
)


def _set_global_seeds(seed: int) -> None:
    """Deterministic training from a fixed seed."""
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
            config_path="configs/default.yaml",
            seed=seed,
            noise_enabled=True,   # sensor noise active — required for obs14_v1 channels
        )
        # Override default episode length (6000 steps at dt=0.1) to 400 for training.
        # 400 steps = 40 s simulated — sufficient for fouling + backflush cycle.
        env.max_steps = 400
        env = ShieldedEnv(env, cfg=_HYDRIENV_SAFETY_CFG)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Train PPO on HydrionEnv (obs14_v1 baseline)"
    )
    parser.add_argument(
        "--steps", type=int, default=500_000,
        help="Total training timesteps (default: 500000)"
    )
    args = parser.parse_args()
    total_timesteps = args.steps

    # ── Schema lock check — fail fast before any training infrastructure ────
    _schema_check_env = HydrionEnv()
    _actual_shape = _schema_check_env.observation_space.shape
    _schema_check_env.close()
    assert _actual_shape == (14,), (
        f"[SCHEMA LOCK FAILED] obs14_v1 requires shape (14,). "
        f"Got {_actual_shape}. Check hydrion/sensors/sensor_fusion.py and hydrion/env.py."
    )
    print(f"[SCHEMA LOCK OK] observation_space.shape = {_actual_shape}  ({_OBS_SCHEMA})")

    _set_global_seeds(_TRAIN_SEED)
    os.makedirs("models",               exist_ok=True)
    os.makedirs("checkpoints/hydrienv", exist_ok=True)

    vec_env = DummyVecEnv([make_env(seed=_TRAIN_SEED)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        training=True,
    )

    run_name        = f"ppo_hydrienv_v1_{int(time.time())}"
    tensorboard_log = f"./runs/{run_name}"

    print(f"\nStarting PPO training on HydrionEnv (obs14_v1)...")
    print(f"Schema lock:     {_OBS_SCHEMA} — {_actual_shape} confirmed")
    print(f"Env:             HydrionEnv")
    print(f"Seed:            {_TRAIN_SEED}")
    print(f"Steps:           {total_timesteps:,}")
    print(f"TensorBoard:     {tensorboard_log}")
    print(f"Checkpoints:     checkpoints/hydrienv/")
    print(f"Final model:     models/ppo_hydrienv_v1.zip")
    print(f"")
    print(f"Note: truth channels 0-9 are privileged inputs (not available at hardware deployment).")
    print(f"      Sensor channels 10-13 carry noisy partial observations.")
    print(f"      This baseline is NOT deployment-ready. See metadata for full limitations.")

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
        save_path="./checkpoints/hydrienv/",
        name_prefix="ppo_hydrienv_v1",
        verbose=1,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    model.save("models/ppo_hydrienv_v1")
    vec_env.save("models/ppo_hydrienv_v1_vecnorm.pkl")

    with open("models/ppo_hydrienv_v1_meta.json", "w") as f:
        json.dump({
            "obs_schema":           _OBS_SCHEMA,
            "action_schema":        "act4_v1",
            "env":                  "HydrionEnv",
            "train_seed":           _TRAIN_SEED,
            "total_timesteps":      total_timesteps,
            "reward_version":       "phase1_v1",
            "d_p_um":               _D_P_UM,
            "benchmark_regime":     "submicron",
            "truth_channels":       "0-9 (privileged — not available at hardware deployment)",
            "sensor_channels":      "10-13 (noisy partial observations)",
            "noise_params_status":  "calibration-pending placeholders (dp_drift_rate, dp_fouling_gain)",
            "domain_randomization": "none — deferred to post-calibration",
            "delay_handling":       "implicit MLP (1-step dp_sensor latency)",
            "deployment_ready":     False,
            "note": (
                "First canonical HydrionEnv baseline under obs14_v1. "
                "Truth channels 0-9 are privileged training inputs not available at "
                "hardware deployment. Calibration-pending noise parameters. "
                "Not comparable to ppo_cce_v2 (different environment, physics, reward, schema)."
            ),
        }, f, indent=2)

    print("\nTraining complete.")
    print("  Model:    models/ppo_hydrienv_v1.zip")
    print("  VecNorm:  models/ppo_hydrienv_v1_vecnorm.pkl")
    print("  Meta:     models/ppo_hydrienv_v1_meta.json")


if __name__ == "__main__":
    main()
