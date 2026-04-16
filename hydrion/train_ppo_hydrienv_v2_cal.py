"""
Hydrion — PPO Training Script for HydrionEnv (obs8_deployment_v1, M9 calibrated noise)
---------------------------------------------------------------------------------------
Trains PPO on HydrionEnv under obs8_deployment_v1 with M9-calibrated sensor noise.

Config: configs/m9_calibrated.yaml (SP1 — NEVER uses configs/default.yaml)

Calibrated parameters (M9.1R.3 verified source brief):
    dp_drift_rate_kPa_per_step: 1.27e-6  (was 0.0005 — 394x reduction)
    dp_drift_max_kPa:           0.40     (was 2.0 — 5x reduction)
    dp_fouling_gain:            0.02     (was 0.5 — 25x reduction)
    turbidity_noise_std:        0.02     (was 0.01 — 2x increase)
    scatter_noise_std:          0.02     (was 0.01 — 2x increase)

Unchanged from M8 v2: PPO hyperparameters, seed, ShieldedEnv config, episode length.
Fresh VecNormalize (MVC-1 — never reuses v1 or v2 pkl).

Saves:
    models/ppo_hydrienv_v2_cal.zip          — M9 calibrated policy
    models/ppo_hydrienv_v2_cal_vecnorm.pkl  — VecNormalize for calibrated distribution
    models/ppo_hydrienv_v2_cal_meta.json    — provenance + calibration record

Run:
    python -m hydrion.train_ppo_hydrienv_v2_cal                  # 360k (default)
    python -m hydrion.train_ppo_hydrienv_v2_cal --steps 500000   # optional full-budget
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
_OBS_SCHEMA  = "obs8_deployment_v1"
_OBS_DIM     = 8
_CONFIG_PATH = "configs/m9_calibrated.yaml"

_HYDRIENV_SAFETY_CFG = SafetyConfig(
    max_pressure_soft=0.85,
    max_pressure_hard=1.05,
    max_clog_soft=0.80,
    max_clog_hard=0.98,
    terminate_on_hard_violation=True,
    hard_violation_penalty=10.0,
)

_CALIBRATION_NOTE = (
    "M9 calibrated artifact. Sensor noise parameters grounded against "
    "piezoresistive DP transmitter (Rosemount 3051 class, MEMS bound), "
    "ISO 7027 nephelometric turbidity sensor (Hanna HI88713, E+H CUS52D). "
    "See M9.1R.3 verified source brief and M9.1R.4 sources map for full provenance."
)

_CALIBRATED_PARAMS = {
    "dp_drift_rate_kPa_per_step": {"value": 1.27e-6, "old": 0.0005, "source": "S1-S4", "confidence": "HIGH"},
    "dp_drift_max_kPa":           {"value": 0.40,    "old": 2.0,    "source": "S3",     "confidence": "HIGH"},
    "dp_fouling_gain":            {"value": 0.02,    "old": 0.5,    "source": "S7-S9",  "confidence": "MEDIUM"},
    "turbidity_noise_std":        {"value": 0.02,    "old": 0.01,   "source": "S5,S6",  "confidence": "HIGH"},
    "scatter_noise_std":          {"value": 0.02,    "old": 0.01,   "source": "S5,S6,S10", "confidence": "HIGH"},
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
        description="Train PPO on HydrionEnv (obs8_deployment_v1 — M9 calibrated noise)"
    )
    parser.add_argument(
        "--steps", type=int, default=360_000,
        help="Total training timesteps (default: 360000, matching M8 v2)"
    )
    args = parser.parse_args()
    total_timesteps = args.steps

    _schema_check_env = HydrionEnv(config_path=_CONFIG_PATH, obs_schema=_OBS_SCHEMA)
    _actual_shape = _schema_check_env.observation_space.shape
    _schema_check_env.close()
    assert _actual_shape == (_OBS_DIM,), (
        f"[SCHEMA LOCK FAILED] {_OBS_SCHEMA} requires shape ({_OBS_DIM},). "
        f"Got {_actual_shape}."
    )
    print(f"[SCHEMA LOCK OK] observation_space.shape = {_actual_shape}  ({_OBS_SCHEMA})")
    print(f"[CONFIG] {_CONFIG_PATH}  (SP1 — NOT configs/default.yaml)")

    _set_global_seeds(_TRAIN_SEED)
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints/hydrienv_v2_cal", exist_ok=True)

    vec_env = DummyVecEnv([make_env(seed=_TRAIN_SEED)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        training=True,
    )

    run_name = f"ppo_hydrienv_v2_cal_{int(time.time())}"
    tensorboard_log = f"./runs/{run_name}"

    print(f"\nStarting PPO training — M9 calibrated noise")
    print(f"Config:          {_CONFIG_PATH}")
    print(f"Schema:          {_OBS_SCHEMA} — {_actual_shape}")
    print(f"Seed:            {_TRAIN_SEED}")
    print(f"Steps:           {total_timesteps:,}")
    print(f"TensorBoard:     {tensorboard_log}")
    print(f"Final model:     models/ppo_hydrienv_v2_cal.zip")
    print()
    print("Calibrated parameters:")
    for k, v in _CALIBRATED_PARAMS.items():
        print(f"  {k}: {v['old']} -> {v['value']}  [{v['confidence']}] ({v['source']})")
    print()

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
        save_path="./checkpoints/hydrienv_v2_cal/",
        name_prefix="ppo_hydrienv_v2_cal",
        verbose=1,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    model.save("models/ppo_hydrienv_v2_cal")
    vec_env.save("models/ppo_hydrienv_v2_cal_vecnorm.pkl")

    with open("models/ppo_hydrienv_v2_cal_meta.json", "w") as f:
        json.dump({
            "obs_schema":              _OBS_SCHEMA,
            "obs_dim":                 _OBS_DIM,
            "action_schema":           "act4_v1",
            "env":                     "HydrionEnv",
            "config_path":             _CONFIG_PATH,
            "train_seed":              _TRAIN_SEED,
            "total_timesteps":         total_timesteps,
            "reward_version":          "phase1_v1",
            "deployment_ready":        False,
            "calibration_note":        _CALIBRATION_NOTE,
            "calibrated_parameters":   _CALIBRATED_PARAMS,
            "device_classes": {
                "dp_sensor":  "Piezoresistive differential pressure transmitter (MEMS class, 0-80 kPa)",
                "flow_meter": "Electromagnetic (mag) flow meter (0-20 L/min, dirty-water service)",
                "turbidity":  "ISO 7027 nephelometric turbidity sensor (90deg / 860 nm)",
                "scatter":    "ISO 7027 nephelometric (secondary scatter channel)",
                "camera":     "DEFERRED — no device class committed",
            },
            "provenance_references": {
                "S1": "Rosemount 3051 Product Data Sheet (Emerson)",
                "S3": "MEMS Vision DP sensor specifications",
                "S5": "Hanna Instruments HI88713 ISO 7027 turbidimeter",
                "S6": "Endress+Hauser Turbimax CUS52D",
                "S7": "Control Global — pressure transmitter installation practice",
                "S10": "ISO 7027-1:2016",
            },
            "channel_taxonomy": {
                "0": {"obs14_v1_index": 6,  "name": "valve_cmd",         "class": "actuator_feedback", "deployment_available": True},
                "1": {"obs14_v1_index": 7,  "name": "pump_cmd",          "class": "actuator_feedback", "deployment_available": True},
                "2": {"obs14_v1_index": 8,  "name": "bf_cmd",            "class": "actuator_feedback", "deployment_available": True},
                "3": {"obs14_v1_index": 9,  "name": "node_voltage_cmd",  "class": "actuator_feedback", "deployment_available": True},
                "4": {"obs14_v1_index": 10, "name": "sensor_turbidity",  "class": "sensor_derived",    "deployment_available": True, "noise_calibrated": True},
                "5": {"obs14_v1_index": 11, "name": "sensor_scatter",    "class": "sensor_derived",    "deployment_available": True, "noise_calibrated": True},
                "6": {"obs14_v1_index": 12, "name": "flow_sensor_norm",  "class": "sensor_derived",    "deployment_available": True, "noise_calibrated": False, "note": "already grounded M6"},
                "7": {"obs14_v1_index": 13, "name": "dp_sensor_norm",    "class": "sensor_derived",    "deployment_available": True, "noise_calibrated": True},
            },
            "confounds": [
                "Single training seed (seed=42 — no cross-seed variance estimate)",
                "M4 phenomenological physics layer (Layer A) — M5 grounded physics not integrated",
                "Optical gain coefficients are placeholder design weights (NTU-to-mass mapping not grounded)",
                "Camera sensor not grounded (deferred)",
            ],
            "calibration_pending_items": [
                "NTU-to-microplastic-mass mapping (requires experimental correlation)",
                "Camera device class selection and noise grounding",
                "Multiplicative vs additive noise model architecture",
                "M5 physics integration into production loop",
            ],
        }, f, indent=2)

    print("\nTraining complete.")
    print(f"  Model:    models/ppo_hydrienv_v2_cal.zip")
    print(f"  VecNorm:  models/ppo_hydrienv_v2_cal_vecnorm.pkl")
    print(f"  Meta:     models/ppo_hydrienv_v2_cal_meta.json")
    print(f"  Config:   {_CONFIG_PATH}")
    print()
    print("Next: python -m hydrion.eval_ppo_hydrienv_v2_cal")


if __name__ == "__main__":
    main()
