"""
Hydrion — PPO Training Script for HydrionEnv (obs8_deployment_v1, M10 RT capture)
----------------------------------------------------------------------------------
Trains PPO on HydrionEnv under obs8_deployment_v1 with M5 Rajagopalan-Tien (1976)
particle capture physics and M9-calibrated sensor noise.

Config: configs/m10_rt_capture.yaml
    - physics.capture_mode: "m5_rt" (RT 1976 single-collector capture)
    - All sensor parameters inherited from M9 calibrated config
    - Reward, clogging, hydraulics UNCHANGED from M9

M10 mandatory guards:
    - capture_boost_settling = 0.0 under m5_rt (Stokes/N_G double-count prevention)
    - psd.enabled = false, shape.enabled = false (C_fibers coupling guard)

PPO hyperparameters: unchanged from M9 Config A (lr=1e-4, clip=0.1).
Fresh VecNormalize (MVC-1 — never reuses prior pkl).

Saves:
    models/ppo_hydrienv_v2_m10_rt.zip          — M10 RT capture policy
    models/ppo_hydrienv_v2_m10_rt_vecnorm.pkl  — VecNormalize for RT distribution
    models/ppo_hydrienv_v2_m10_rt_meta.json    — provenance + M10 guard confirmation

Run:
    python -m hydrion.train_ppo_hydrienv_v2_m10_rt
    python -m hydrion.train_ppo_hydrienv_v2_m10_rt --steps 500000
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
_CONFIG_PATH = "configs/m10_rt_capture.yaml"

_HYDRIENV_SAFETY_CFG = SafetyConfig(
    max_pressure_soft=0.85,
    max_pressure_hard=1.05,
    max_clog_soft=0.80,
    max_clog_hard=0.98,
    terminate_on_hard_violation=True,
    hard_violation_penalty=10.0,
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


def _verify_settling_guard():
    """Pre-training verification: capture_boost_settling must be 0.0 under m5_rt."""
    env = HydrionEnv(
        config_path=_CONFIG_PATH,
        seed=0,
        noise_enabled=False,
        obs_schema=_OBS_SCHEMA,
    )
    env.reset()
    action = np.array([0.5, 0.5, 0.0, 0.5], dtype=np.float32)
    for _ in range(5):
        env.step(action)
    settling = float(env.truth_state.get("capture_boost_settling", -1.0))
    env.close()
    assert settling == 0.0, (
        f"[SETTLING GUARD FAILED] capture_boost_settling = {settling} under m5_rt. "
        f"Expected 0.0. RT N_G double-count detected."
    )
    print(f"[SETTLING GUARD OK] capture_boost_settling = {settling}")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Train PPO on HydrionEnv (obs8_deployment_v1 — M10 RT capture)"
    )
    parser.add_argument(
        "--steps", type=int, default=500_000,
        help="Total training timesteps (default: 500000, matching M9 Config A)"
    )
    args = parser.parse_args()
    total_timesteps = args.steps

    # Schema lock check
    _schema_check_env = HydrionEnv(config_path=_CONFIG_PATH, obs_schema=_OBS_SCHEMA)
    _actual_shape = _schema_check_env.observation_space.shape
    _capture_mode = _schema_check_env.particles._capture_mode
    _schema_check_env.close()
    assert _actual_shape == (_OBS_DIM,), (
        f"[SCHEMA LOCK FAILED] {_OBS_SCHEMA} requires shape ({_OBS_DIM},). "
        f"Got {_actual_shape}."
    )
    assert _capture_mode == "m5_rt", (
        f"[MODE CHECK FAILED] Expected capture_mode='m5_rt', got '{_capture_mode}'. "
        f"Config: {_CONFIG_PATH}"
    )
    print(f"[SCHEMA LOCK OK] observation_space.shape = {_actual_shape}  ({_OBS_SCHEMA})")
    print(f"[CAPTURE MODE OK] capture_mode = '{_capture_mode}'")
    print(f"[CONFIG] {_CONFIG_PATH}")

    # Settling guard verification
    _verify_settling_guard()

    _set_global_seeds(_TRAIN_SEED)
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints/hydrienv_v2_m10_rt", exist_ok=True)

    vec_env = DummyVecEnv([make_env(seed=_TRAIN_SEED)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        training=True,
    )

    run_name = f"ppo_hydrienv_v2_m10_rt_{int(time.time())}"
    tensorboard_log = f"./runs/{run_name}"

    print(f"\nStarting PPO training — M10 RT capture")
    print(f"Config:          {_CONFIG_PATH}")
    print(f"Schema:          {_OBS_SCHEMA} — {_actual_shape}")
    print(f"Capture mode:    m5_rt (Rajagopalan-Tien 1976)")
    print(f"Seed:            {_TRAIN_SEED}")
    print(f"Steps:           {total_timesteps:,}")
    print(f"TensorBoard:     {tensorboard_log}")
    print(f"Final model:     models/ppo_hydrienv_v2_m10_rt.zip")
    print()

    # M9 Config A hyperparameters (noise-regime-corrected)
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        ent_coef=0.01,
        learning_rate=1e-4,     # M9 Config A: lr=1e-4 (not M8's 3e-4)
        clip_range=0.1,         # M9 Config A: clip=0.1 (not M8's 0.2)
        seed=_TRAIN_SEED,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path="./checkpoints/hydrienv_v2_m10_rt/",
        name_prefix="ppo_hydrienv_v2_m10_rt",
        verbose=1,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    model.save("models/ppo_hydrienv_v2_m10_rt")
    vec_env.save("models/ppo_hydrienv_v2_m10_rt_vecnorm.pkl")

    with open("models/ppo_hydrienv_v2_m10_rt_meta.json", "w") as f:
        json.dump({
            "milestone":               "M10",
            "capture_mode":            "m5_rt",
            "capture_physics":         "Rajagopalan-Tien 1976 single-collector",
            "parent_baseline":         "ppo_hydrienv_v2_cal_A",
            "obs_schema":              _OBS_SCHEMA,
            "obs_dim":                 _OBS_DIM,
            "action_schema":           "act4_v1",
            "env":                     "HydrionEnv",
            "config_path":             _CONFIG_PATH,
            "train_seed":              _TRAIN_SEED,
            "total_timesteps":         total_timesteps,
            "learning_rate":           1e-4,
            "clip_range":              0.1,
            "reward_version":          "phase1_v1",
            "deployment_ready":        False,
            "settling_guard":          "PASS — capture_boost_settling = 0.0 verified",
            "reward_invariance":       "PENDING — verify in eval",
            "clogging_invariance":     "PENDING — verify in eval",
            "hydraulics_invariance":   "PENDING — verify in eval",
            "obs8_channel_comparison": "PENDING — verify in eval",
            "ac11_result":             "PENDING — verify in eval",
            "truth_status_note": (
                "RT 1976 capture integrated via toggle. nDEP deferred (Phase 2). "
                "Sensors inherited from M9 calibrated config. "
                "Mesh specs are [DESIGN_DEFAULT] — not hardware-calibrated. "
                "Reward is capture-decoupled. obs8[4-5] expected to differ from M9 "
                "via C_out → optical sensor pathway."
            ),
            "design_defaults": [
                "MESH_S1: d_w=125µm (opening/4 estimate)",
                "MESH_S2: d_w=50µm",
                "MESH_S3_MEMBRANE: d_w=1.5µm (microporous membrane regime)",
            ],
            "confounds": [
                "Single training seed (seed=42)",
                "M4 electrostatics still active (pDEP-flavored, not nDEP)",
                "Mesh wire diameters are design defaults, not measured",
                "No fouling coupling under RT (clean-filter formula)",
                "Optical gain coefficients are placeholder design weights",
            ],
        }, f, indent=2)

    print("\nTraining complete.")
    print(f"  Model:    models/ppo_hydrienv_v2_m10_rt.zip")
    print(f"  VecNorm:  models/ppo_hydrienv_v2_m10_rt_vecnorm.pkl")
    print(f"  Meta:     models/ppo_hydrienv_v2_m10_rt_meta.json")
    print(f"  Config:   {_CONFIG_PATH}")
    print()
    print("Next: python -m hydrion.eval_ppo_hydrienv_v2_m10_rt")


if __name__ == "__main__":
    main()
