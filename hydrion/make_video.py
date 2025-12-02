# hydrion/make_video.py
"""
Hydrion Digital Twin — Cinematic Renderer v8 (Corrected Coordinates)
Matches v5 particle flow logic BUT uses your cinematic artwork.

Fixes applied:
    - Image drawn with origin='lower'
    - Axes flipped so y=0 is TOP, y=1 is BOTTOM
    - Particle motion direction now correctly flows top → bottom
"""

from __future__ import annotations
import os
from typing import Dict, Any, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.image as mpimg

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .env import HydrionEnv


# ============================================================
# PATHS / CONSTANTS
# ============================================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")
VIDEOS_DIR = os.path.join(ROOT_DIR, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)

DEFAULT_MODEL = "ppo_hydrion_final_12d.zip"
MODEL_PATH = os.path.join(CHECKPOINTS_DIR, DEFAULT_MODEL)
VECNORM_PATH = os.path.join(ROOT_DIR, "ppo_hydrion_vecnormalize_final_12d.pkl")

BACKGROUND_PATH = os.path.join(PKG_DIR, "images", "visual_background.png")

FPS = 30
VIDEO_SECONDS = 15.0
N_FRAMES = int(FPS * VIDEO_SECONDS)
VIDEO_PATH = os.path.join(VIDEOS_DIR, "hydrion_run_v8.mp4")


# ============================================================
# ENV + MODEL LOADING
# ============================================================

def make_env():
    def _init():
        return HydrionEnv()
    return _init


def load_model_and_env():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing PPO checkpoint: {MODEL_PATH}")
    if not os.path.exists(VECNORM_PATH):
        raise FileNotFoundError(f"Missing VecNormalize stats: {VECNORM_PATH}")

    base_env = DummyVecEnv([make_env()])
    vec_env: VecNormalize = VecNormalize.load(VECNORM_PATH, base_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model: PPO = PPO.load(MODEL_PATH, env=vec_env)
    return model, vec_env


# ============================================================
# ROLLOUT EPISODE
# ============================================================

def rollout_for_animation(model: PPO, vec_env: VecNormalize, max_frames: int):
    obs = vec_env.reset()
    obs_hist = []
    rew_hist = []

    for _ in range(max_frames):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, info = vec_env.step(action)

        obs_hist.append(next_obs[0].copy())
        rew_hist.append(float(reward[0]))

        obs = next_obs
        if bool(done[0]):
            obs = vec_env.reset()

    return {
        "obs": np.stack(obs_hist, axis=0),
        "reward": np.array(rew_hist, dtype=np.float32),
    }


# ============================================================
# PARTICLE FIELD + MOTION (unchanged from your cinematic v8)
# ============================================================

REACTOR_X_MIN = 0.33
REACTOR_X_MAX = 0.67

INFLOW_Y_MIN = 0.18
INFLOW_Y_MAX = 0.27

# was 0.332, 0.459, 0.586
MESH_CENTERS_Y = np.array([0.420, 0.515, 0.610])
MESH_SLOPE_DEG = 6.5
MESH_THETA = np.deg2rad(MESH_SLOPE_DEG)
MESH_TAN = np.tan(MESH_THETA)
MESH_CENTER_X = 0.50
MESH_BAND = 0.012

PIPE_X = 0.757
PIPE_ENTRY_X = 0.73
PIPE_TOP_Y = 0.34
PIPE_BOTTOM_Y = 0.77

BASE_FALL = 0.060
FLOW_GAIN = 0.23
CURVE_STRENGTH = 0.38
SLIDE_SPEED = 0.22
PIPE_SPEED = 0.30
JITTER_SCALE = 0.002


def init_particle_positions(
    n_big=220,
    n_med=320,
    n_small=440
):
    rng = np.random.default_rng(42)

    def rand_group(n):
        xs = rng.uniform(REACTOR_X_MIN, REACTOR_X_MAX, size=n)
        ys = rng.uniform(INFLOW_Y_MIN, INFLOW_Y_MAX, size=n)
        return np.stack([xs, ys], axis=1)

    return {
        "big": rand_group(n_big),
        "med": rand_group(n_med),
        "small": rand_group(n_small),
    }


def update_particles(positions, obs_t, group, dt_vis):
    if obs_t.ndim != 1:
        obs_t = obs_t.ravel()

    flow = float(obs_t[0])
    clog = float(obs_t[2])
    e_norm = float(obs_t[3])
    capture_eff = float(obs_t[5])

    if group == "big":
        target_mesh_idx = 0
    elif group == "med":
        target_mesh_idx = 1
    else:
        target_mesh_idx = 2

    x = positions[:, 0]
    y = positions[:, 1]

    # Downward drift (y increases downward because of axis inversion)
    vy = (BASE_FALL + FLOW_GAIN * flow) * dt_vis
    y += vy

    # Rightward curvature
    depth = np.clip((y - INFLOW_Y_MIN) / (PIPE_BOTTOM_Y - INFLOW_Y_MIN), 0.0, 1.0)
    vx = CURVE_STRENGTH * np.clip(e_norm, 0.0, 1.0) * depth * dt_vis
    x += vx

    # Mesh sliding
    mesh_dir = np.array([np.cos(MESH_THETA), np.sin(MESH_THETA)])

    base_capture = 0.3 + 0.7 * np.clip(capture_eff, 0.0, 1.0)
    clog_factor = 1.0 - 0.4 * np.clip(clog, 0.0, 1.0)
    strength_global = base_capture * clog_factor

    for i, yc in enumerate(MESH_CENTERS_Y):
        y_line = yc + MESH_TAN * (x - MESH_CENTER_X)
        dist = y - y_line
        on_mesh = np.abs(dist) < MESH_BAND

        if not np.any(on_mesh):
            continue

        layer_gain = 1.0 + 0.25 * i
        strength = strength_global * layer_gain

        if i == target_mesh_idx:
            noise = np.random.normal(0.0, 0.0015, size=on_mesh.sum())
            y[on_mesh] = y_line[on_mesh] + noise

            slide = SLIDE_SPEED * (0.7 + 0.4 * e_norm + 0.3 * capture_eff) * dt_vis
            x[on_mesh] -= mesh_dir[0] * slide
            y[on_mesh] += mesh_dir[1] * slide
        else:
            slide = 0.10 * SLIDE_SPEED * dt_vis
            x[on_mesh] -= mesh_dir[0] * slide
            y[on_mesh] += mesh_dir[1] * slide

    # Enter the vertical electrostatic pipe
    entering = x >= PIPE_ENTRY_X
    if np.any(entering):
        x[entering] = PIPE_X
        y[entering] = np.maximum(y[entering], PIPE_TOP_Y)

    in_pipe = np.abs(x - PIPE_X) < 0.008
    if np.any(in_pipe):
        y[in_pipe] += PIPE_SPEED * dt_vis

    # Respawn
    gone = y > PIPE_BOTTOM_Y
    if np.any(gone):
        n = int(gone.sum())
        x[gone] = np.random.uniform(REACTOR_X_MIN, REACTOR_X_MAX, size=n)
        y[gone] = np.random.uniform(INFLOW_Y_MIN, INFLOW_Y_MAX, size=n)

    noise = np.random.normal(scale=JITTER_SCALE * dt_vis, size=positions.shape)
    x += noise[:, 0]
    y += noise[:, 1]

    x = np.clip(x, 0.25, 0.90)
    y = np.clip(y, 0.05, 0.95)

    positions[:, 0] = x
    positions[:, 1] = y
    return positions


# ============================================================
# FIGURE / ARTISTS — FIXED COORDINATES
# ============================================================

def create_vertical_reactor_figure():
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(4.0, 6.0))

    if not os.path.exists(BACKGROUND_PATH):
        raise FileNotFoundError(f"Background image not found at: {BACKGROUND_PATH}")
    img = mpimg.imread(BACKGROUND_PATH)

    # 🔥 THE FIX:
    # Draw upright image (origin='lower') but FLIP the axes so y=0 is TOP.
    ax.imshow(
        img,
        extent=(0.0, 1.0, 0.0, 1.0),
        origin="lower",   # <-- image upright
        aspect="auto",
        zorder=0,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0)  # <-- coordinate system now image-style (y=0 top)
    ax.axis("off")

    scat_big = ax.scatter([], [], s=18, color="#e8f7ff", alpha=0.95, zorder=6)
    scat_med = ax.scatter([], [], s=12, color="#bde6ff", alpha=0.90, zorder=6)
    scat_small = ax.scatter([], [], s=8, color="#8fd1ff", alpha=0.90, zorder=6)

    return fig, {
        "ax": ax,
        "scat_big": scat_big,
        "scat_med": scat_med,
        "scat_small": scat_small,
    }


# ============================================================
# ANIMATION
# ============================================================

def build_animation(history):
    obs_hist = history["obs"]

    fig, artists = create_vertical_reactor_figure()
    ax = artists["ax"]

    pos = init_particle_positions()

    scat_big = artists["scat_big"]
    scat_med = artists["scat_med"]
    scat_small = artists["scat_small"]

    def init():
        scat_big.set_offsets(pos["big"])
        scat_med.set_offsets(pos["med"])
        scat_small.set_offsets(pos["small"])
        return scat_big, scat_med, scat_small

    def animate(frame):
        obs = obs_hist[min(frame, obs_hist.shape[0] - 1)]

        pos["big"] = update_particles(pos["big"], obs, "big", dt_vis=1/FPS)
        pos["med"] = update_particles(pos["med"], obs, "med", dt_vis=1/FPS)
        pos["small"] = update_particles(pos["small"], obs, "small", dt_vis=1/FPS)

        scat_big.set_offsets(pos["big"])
        scat_med.set_offsets(pos["med"])
        scat_small.set_offsets(pos["small"])

        return scat_big, scat_med, scat_small

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=N_FRAMES,
        interval=1000.0 / FPS,
        blit=True,
    )

    return anim


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading PPO model + environment...")
    model, vec_env = load_model_and_env()

    print("Rolling out PPO episode...")
    history = rollout_for_animation(model, vec_env, N_FRAMES)

    print("Rendering cinematic animation...")
    anim = build_animation(history)

    print(f"Saving MP4 to: {VIDEO_PATH}")
    writer = animation.FFMpegWriter(
        fps=FPS,
        metadata={"artist": "HydrionRL"},
        bitrate=6000,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"],
    )
    anim.save(VIDEO_PATH, writer=writer)

    print("✅ Video saved.")


if __name__ == "__main__":
    main()
