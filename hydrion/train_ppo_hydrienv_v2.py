"""
Hydrion — PPO Training Script for HydrionEnv (obs8_deployment_v1)
------------------------------------------------------------------
Trains PPO on HydrionEnv under the obs8_deployment_v1 observation schema:
    - obs8_deployment_v1: 8D observation (indices 6–9 actuator feedback, 10–13 sensor-derived)
    - Schema lock: asserts observation_space.shape == (8,) at startup
    - ShieldedEnv (identical config to ppo_hydrienv_v1 — MVC-2 matched conditions)
    - VecNormalize (fresh fit — NEVER reuses ppo_hydrienv_v1_vecnorm.pkl — MVC-1)
    - TensorBoard logging
    - Deterministic seeding (seed=42 — same as v1 for matched comparison — MVC-2)

obs8_deployment_v1 information regime:
    Removed (physics_truth, 0–5): flow, pressure, clog, E_field_norm, C_out,
        particle_capture_eff — NOT available at hardware deployment.
    Retained actuator_feedback (0–3): valve_cmd, pump_cmd, bf_cmd, node_voltage_cmd.
        Controller-issued; always self-known at deployment.
    Retained sensor_derived (4–7): sensor_turbidity, sensor_scatter,
        flow_sensor_norm, dp_sensor_norm.

    Calibration-pending: dp_drift_rate, dp_fouling_gain, flow_calibration_bias_std
    are placeholder values.

M8 methodology validity conditions (M8.3 Section 6):
    MVC-1: Fresh VecNormalize — never reuse ppo_hydrienv_v1_vecnorm.pkl
    MVC-2: Matched training conditions (identical hyperparameters to v1)
    MVC-3: Calibration confound named in metadata
    MVC-4: Cross-schema deployment-gap framing in metadata

Saves:
    models/ppo_hydrienv_v2.zip          — M8 deployment-gap policy
    models/ppo_hydrienv_v2_vecnorm.pkl  — VecNormalize statistics (obs8_deployment_v1 only)
    models/ppo_hydrienv_v2_meta.json    — full channel taxonomy + limitations record

Checkpoints every 10k steps to checkpoints/hydrienv_v2/

Run from repo root:
    python -m hydrion.train_ppo_hydrienv_v2                  # 500k steps (default)
    python -m hydrion.train_ppo_hydrienv_v2 --steps 10000    # proof-of-concept run

IMPORTANT: Training requires ~15-16 minutes. Do NOT run as a background process —
the execution environment kills background processes at ~5 minutes.
Run as a foreground terminal command.
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


_TRAIN_SEED  = 42                     # MVC-2: same as ppo_hydrienv_v1
_D_P_UM      = 1.0                    # consistent with ppo_hydrienv_v1
_OBS_SCHEMA  = "obs8_deployment_v1"   # version-lock
_OBS_DIM     = 8                      # schema lock constant

# MVC-2: identical ShieldedEnv config to ppo_hydrienv_v1
_HYDRIENV_SAFETY_CFG = SafetyConfig(
    max_pressure_soft=0.85,
    max_pressure_hard=1.05,
    max_clog_soft=0.80,
    max_clog_hard=0.98,
    terminate_on_hard_violation=True,
    hard_violation_penalty=10.0,
)

# MVC-4: cross-schema comparison framing
_COMPARISON_TYPE = "cross-schema deployment-gap measurement"
_COMPARISON_TARGET = "ppo_hydrienv_v1 (obs14_v1, 14D)"

# MVC-3: calibration confound statement
_CALIBRATION_CONFOUND = (
    "calibration-pending placeholders (dp_drift_rate, dp_fouling_gain, "
    "flow_calibration_bias_std). M8 results are internally valid under these "
    "parameters; predictive value for hardware deployment is bounded by the "
    "uncalibrated sensor model."
)


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
            config_path="configs/default.yaml",
            seed=seed,
            noise_enabled=True,
            obs_schema=_OBS_SCHEMA,    # obs8_deployment_v1
        )
        env.max_steps = 400            # MVC-2: same as ppo_hydrienv_v1
        env = ShieldedEnv(env, cfg=_HYDRIENV_SAFETY_CFG)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Train PPO on HydrionEnv (obs8_deployment_v1 — M8 deployment gap)"
    )
    parser.add_argument(
        "--steps", type=int, default=500_000,
        help="Total training timesteps (default: 500000)"
    )
    args = parser.parse_args()
    total_timesteps = args.steps

    # ── Schema lock check — fail fast before any training infrastructure ────
    _schema_check_env = HydrionEnv(obs_schema=_OBS_SCHEMA)
    _actual_shape = _schema_check_env.observation_space.shape
    _schema_check_env.close()
    assert _actual_shape == (_OBS_DIM,), (
        f"[SCHEMA LOCK FAILED] {_OBS_SCHEMA} requires shape ({_OBS_DIM},). "
        f"Got {_actual_shape}. Check hydrion/sensors/sensor_fusion.py and hydrion/env.py."
    )
    print(f"[SCHEMA LOCK OK] observation_space.shape = {_actual_shape}  ({_OBS_SCHEMA})")

    _set_global_seeds(_TRAIN_SEED)
    os.makedirs("models",                exist_ok=True)
    os.makedirs("checkpoints/hydrienv_v2", exist_ok=True)

    vec_env = DummyVecEnv([make_env(seed=_TRAIN_SEED)])
    vec_env = VecNormalize(        # MVC-1: fresh VecNormalize — never reuse v1 pkl
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        training=True,
    )

    run_name        = f"ppo_hydrienv_v2_{int(time.time())}"
    tensorboard_log = f"./runs/{run_name}"

    print(f"\nStarting PPO training on HydrionEnv ({_OBS_SCHEMA})...")
    print(f"Schema lock:     {_OBS_SCHEMA} — {_actual_shape} confirmed")
    print(f"Env:             HydrionEnv")
    print(f"Seed:            {_TRAIN_SEED}")
    print(f"Steps:           {total_timesteps:,}")
    print(f"TensorBoard:     {tensorboard_log}")
    print(f"Checkpoints:     checkpoints/hydrienv_v2/")
    print(f"Final model:     models/ppo_hydrienv_v2.zip")
    print(f"")
    print(f"MVC-1: Fresh VecNormalize — ppo_hydrienv_v1_vecnorm.pkl is NOT used here.")
    print(f"MVC-2: Matched conditions — identical hyperparameters to ppo_hydrienv_v1.")
    print(f"MVC-4: This is a deployment-gap measurement, not a policy improvement.")
    print(f"")
    print(f"Removed: physics_truth channels 0-5 (unavailable at hardware deployment).")
    print(f"Retained: actuator_feedback 0-3 (obs14_v1 6-9) + sensor_derived 4-7 (obs14_v1 10-13).")

    # MVC-2: identical PPO hyperparameters to ppo_hydrienv_v1
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
        save_path="./checkpoints/hydrienv_v2/",
        name_prefix="ppo_hydrienv_v2",
        verbose=1,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    model.save("models/ppo_hydrienv_v2")
    vec_env.save("models/ppo_hydrienv_v2_vecnorm.pkl")   # MVC-1: schema-specific pkl

    # ── Full metadata per M8.3 Section 4.2 ─────────────────────────────────
    with open("models/ppo_hydrienv_v2_meta.json", "w") as f:
        json.dump({
            "obs_schema":              _OBS_SCHEMA,
            "obs_dim":                 _OBS_DIM,
            "action_schema":           "act4_v1",
            "env":                     "HydrionEnv",
            "train_seed":              _TRAIN_SEED,
            "total_timesteps":         total_timesteps,
            "reward_version":          "phase1_v1",
            "d_p_um":                  _D_P_UM,
            "benchmark_regime":        "submicron",
            "deployment_ready":        False,
            "comparison_type":         _COMPARISON_TYPE,
            "cross_schema_comparison_target": _COMPARISON_TARGET,
            "calibration_status":      _CALIBRATION_CONFOUND,
            "channel_taxonomy": {
                "0": {"obs14_v1_index": 6,  "name": "valve_cmd",         "class": "actuator_feedback", "deployment_available": True},
                "1": {"obs14_v1_index": 7,  "name": "pump_cmd",          "class": "actuator_feedback", "deployment_available": True},
                "2": {"obs14_v1_index": 8,  "name": "bf_cmd",            "class": "actuator_feedback", "deployment_available": True},
                "3": {"obs14_v1_index": 9,  "name": "node_voltage_cmd",  "class": "actuator_feedback", "deployment_available": True},
                "4": {"obs14_v1_index": 10, "name": "sensor_turbidity",  "class": "sensor_derived",    "deployment_available": True},
                "5": {"obs14_v1_index": 11, "name": "sensor_scatter",    "class": "sensor_derived",    "deployment_available": True},
                "6": {"obs14_v1_index": 12, "name": "flow_sensor_norm",  "class": "sensor_derived",    "deployment_available": True},
                "7": {"obs14_v1_index": 13, "name": "dp_sensor_norm",    "class": "sensor_derived",    "deployment_available": True},
            },
            "removed_channels": {
                "obs14_v1_0": {"name": "flow",                 "class": "physics_truth", "deployment_available": False},
                "obs14_v1_1": {"name": "pressure",             "class": "physics_truth", "deployment_available": False},
                "obs14_v1_2": {"name": "clog",                 "class": "physics_truth", "deployment_available": False},
                "obs14_v1_3": {"name": "E_field_norm",         "class": "physics_truth", "deployment_available": False},
                "obs14_v1_4": {"name": "C_out",                "class": "physics_truth", "deployment_available": False},
                "obs14_v1_5": {"name": "particle_capture_eff", "class": "physics_truth", "deployment_available": False},
            },
            "confounds": [
                "calibration-pending sensor noise (dp_drift_rate, dp_fouling_gain, flow_calibration_bias_std are placeholders)",
                "single training seed (seed=42 — no cross-seed variance estimate)",
                "500k step training budget (may be insufficient to fully exploit obs8 signals)",
            ],
            "methodology_validity_conditions": {
                "MVC-1": "Fresh VecNormalize fit — ppo_hydrienv_v1_vecnorm.pkl was NOT used",
                "MVC-2": "Matched conditions — identical PPO hyperparameters, episode length, ShieldedEnv config, seed",
                "MVC-3": "Calibration confound named in calibration_status field",
                "MVC-4": "Comparison framed as cross-schema deployment-gap measurement",
            },
            "note": (
                "M8 deployment-gap comparison artifact. Trains on deployment-realistic "
                "obs8_deployment_v1 (actuator feedback + sensor channels only). "
                "Compared against ppo_hydrienv_v1 (obs14_v1) to quantify the performance "
                "cost of removing physics-truth channels. Does NOT replace ppo_hydrienv_v1. "
                "Result interpretation: see M8.3_execution_document.md Section 9."
            ),
        }, f, indent=2)

    print("\nTraining complete.")
    print("  Model:    models/ppo_hydrienv_v2.zip")
    print("  VecNorm:  models/ppo_hydrienv_v2_vecnorm.pkl  (obs8_deployment_v1 only)")
    print("  Meta:     models/ppo_hydrienv_v2_meta.json")
    print("")
    print("MVC-3 (calibration confound):")
    print(f"  {_CALIBRATION_CONFOUND}")
    print("")
    print("Next: python -m hydrion.eval_ppo_hydrienv_v2")


if __name__ == "__main__":
    main()
