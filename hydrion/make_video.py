# hydrion/make_video.py
"""
Hydrion Digital Twin — Cinematic Renderer v6.2
----------------------------------------------

- Uses the static artwork at hydrion/images/visual_background.png
- Overlays PPO-driven particle motion:

    * Big MPs  → top membrane
    * Med MPs  → middle membrane
    * Small MPs→ bottom membrane

- Particles:
    * Drift downward with flow
    * Curve slightly right (electrophoretic pull)
    * Snap onto their membrane and slide down-right
    * Enter the yellow vertical node line on the right
    * Drop straight down and disappear at the bottom node
    * Respawn near the inflow so the flow is continuous

Run from project root:

    python -m hydrion.make_video

Output:

    videos/hydrion_run_v6.mp4
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

# ---------------------------------------------------------------------
#  PATHS / CONSTANTS
# ---------------------------------------------------------------------

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
VIDEO_PATH = os.path.join(VIDEOS_DIR, "hydrion_run_v7.mp4")

# Image is 1024 x 1536 (w x h), so aspect ratio is 2:3.
# We'll work in normalized coords: x,y in [0,1], with y=0 at top, y=1 at bottom.

# Vessel interior region
REACTOR_X_MIN = 0.33
REACTOR_X_MAX = 0.67

# Inflow region (just below Polarize Layer)
INFLOW_Y_MIN = 0.18
INFLOW_Y_MAX = 0.27

# Slanted membranes: centers aligned to artwork (top→bottom)
MESH_CENTERS_Y = np.array([0.332, 0.459, 0.586])
MESH_SLOPE_DEG = 6.5                 # downward to the right
MESH_THETA = np.deg2rad(MESH_SLOPE_DEG)
MESH_TAN = np.tan(MESH_THETA)
MESH_CENTER_X = 0.50
MESH_BAND = 0.012                    # thickness of "capture band" around each membrane

# Yellow vertical node line on the right
PIPE_X = 0.757                       # x-position of vertical node line
PIPE_ENTRY_X = 0.73                  # where particles leave the membrane into the pipe
PIPE_TOP_Y = 0.34                    # near top node
PIPE_BOTTOM_Y = 0.77                 # bottom node

# Motion tuning
BASE_FALL = 0.060
FLOW_GAIN = 0.23
CURVE_STRENGTH = 0.38
SLIDE_SPEED = 0.22
PIPE_SPEED = 0.30

JITTER_SCALE = 0.002


# ---------------------------------------------------------------------
#  ENV + MODEL LOADING
# ---------------------------------------------------------------------

def make_env():
    def _init():
        return HydrionEnv()
    return _init


def load_model_and_env() -> Tuple[PPO, VecNormalize]:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing PPO checkpoint at: {MODEL_PATH}")
    if not os.path.exists(VECNORM_PATH):
        raise FileNotFoundError(f"Missing VecNormalize stats at: {VECNORM_PATH}")

    base_env = DummyVecEnv([make_env()])
    vec_env: VecNormalize = VecNormalize.load(VECNORM_PATH, base_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model: PPO = PPO.load(MODEL_PATH, env=vec_env)
    return model, vec_env


# ---------------------------------------------------------------------
#  ROLLOUT EPISODE (FOR VISUALIZATION)
# ---------------------------------------------------------------------

def rollout_for_animation(
    model: PPO,
    vec_env: VecNormalize,
    max_frames: int,
) -> Dict[str, np.ndarray]:
    obs = vec_env.reset()
    obs_hist: List[np.ndarray] = []
    rew_hist: List[float] = []

    for _ in range(max_frames):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, dones, infos = vec_env.step(action)

        obs_hist.append(next_obs[0].copy())
        rew_hist.append(float(reward[0]))

        obs = next_obs
        if bool(dones[0]):
            obs = vec_env.reset()

    obs_arr = np.stack(obs_hist, axis=0)
    rew_arr = np.array(rew_hist, dtype=np.float32)
    return {"obs": obs_arr, "reward": rew_arr}


# ---------------------------------------------------------------------
#  PARTICLE FIELD + MOTION
# ---------------------------------------------------------------------

def init_particle_positions(
    n_big: int = 220,
    n_med: int = 320,
    n_small: int = 440,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(42)

    def rand_group(n: int) -> np.ndarray:
        xs = rng.uniform(REACTOR_X_MIN, REACTOR_X_MAX, size=n)
        ys = rng.uniform(INFLOW_Y_MIN, INFLOW_Y_MAX, size=n)
        return np.stack([xs, ys], axis=1)

    return {
        "big": rand_group(n_big),
        "med": rand_group(n_med),
        "small": rand_group(n_small),
    }


def update_particles(
    positions: np.ndarray,
    obs_t: np.ndarray,
    group: str,
    dt_vis: float,
) -> np.ndarray:
    """
    Map v5-style motion into artwork coordinates:

    - y increases downward.
    - Downward drift from flow.
    - Rightward drift from electrophoresis.
    - Group-specific capture on one membrane, sliding along it.
    - At PIPE_ENTRY_X, particles enter the vertical node column and
      drop straight down to PIPE_BOTTOM_Y, then respawn at inflow.
    """
    if obs_t.ndim != 1:
        obs_t = obs_t.ravel()

    flow = float(obs_t[0]) if obs_t.size > 0 else 0.5
    clog = float(obs_t[2]) if obs_t.size > 2 else 0.0
    e_norm = float(obs_t[3]) if obs_t.size > 3 else 0.0
    capture_eff = float(obs_t[5]) if obs_t.size > 5 else 0.5

    if group == "big":
        target_mesh_idx = 0
    elif group == "med":
        target_mesh_idx = 1
    else:
        target_mesh_idx = 2

    x = positions[:, 0]
    y = positions[:, 1]

    # Downward advection
    vy_base = (BASE_FALL + FLOW_GAIN * flow) * dt_vis
    y += vy_base

    # Electrophoretic rightward pull (stronger deeper)
    depth = np.clip((y - INFLOW_Y_MIN) / (PIPE_BOTTOM_Y - INFLOW_Y_MIN), 0.0, 1.0)
    vx_curve = CURVE_STRENGTH * np.clip(e_norm, 0.0, 1.0) * depth * dt_vis
    x += vx_curve

    # Membrane interaction & sliding
    mesh_dir = np.array([np.cos(MESH_THETA), np.sin(MESH_THETA)])
    base_capture = 0.3 + 0.7 * np.clip(capture_eff, 0.0, 1.0)
    clog_factor = 1.0 - 0.4 * np.clip(clog, 0.0, 1.0)
    strength_global = base_capture * clog_factor

    for i, yc in enumerate(MESH_CENTERS_Y):
        # Membrane center-line equation: y = yc + m (x - x0)
        y_line = yc + MESH_TAN * (x - MESH_CENTER_X)
        dist = y - y_line
        on_mesh = np.abs(dist) < MESH_BAND

        if not np.any(on_mesh):
            continue

        layer_gain = 1.0 + 0.25 * i
        strength = strength_global * layer_gain

        if i == target_mesh_idx:
            # Clamp close to membrane and slide along it
            noise = np.random.normal(0.0, 0.0015, size=on_mesh.sum())
            y[on_mesh] = y_line[on_mesh] + noise

            slide = SLIDE_SPEED * (0.7 + 0.4 * e_norm + 0.3 * capture_eff) * dt_vis
            x[on_mesh] += mesh_dir[0] * slide
            y[on_mesh] += mesh_dir[1] * slide
        else:
            # Light deflection for non-primary meshes
            slide = 0.10 * SLIDE_SPEED * dt_vis
            x[on_mesh] += mesh_dir[0] * slide
            y[on_mesh] += mesh_dir[1] * slide

    # Enter the vertical node pipe
    entering_pipe = x >= PIPE_ENTRY_X
    if np.any(entering_pipe):
        x[entering_pipe] = PIPE_X
        # ensure they don't "enter" above the first node region
        y[entering_pipe] = np.maximum(y[entering_pipe], PIPE_TOP_Y)

    # Inside the pipe: drop straight down
    in_pipe = np.abs(x - PIPE_X) < 0.008
    if np.any(in_pipe):
        y[in_pipe] += PIPE_SPEED * dt_vis

    # Respawn after leaving bottom node
    gone = y > PIPE_BOTTOM_Y
    if np.any(gone):
        n = int(gone.sum())
        x[gone] = np.random.uniform(REACTOR_X_MIN, REACTOR_X_MAX, size=n)
        y[gone] = np.random.uniform(INFLOW_Y_MIN, INFLOW_Y_MAX, size=n)

    # Light jitter for organic feel
    noise = np.random.normal(scale=JITTER_SCALE * dt_vis, size=positions.shape)
    x += noise[:, 0]
    y += noise[:, 1]

    # Keep within canvas / vessel neighborhood
    x = np.clip(x, 0.25, 0.90)
    y = np.clip(y, 0.05, 0.95)

    positions[:, 0] = x
    positions[:, 1] = y
    return positions


# ---------------------------------------------------------------------
#  FIGURE / ARTISTS
# ---------------------------------------------------------------------

def create_vertical_reactor_figure() -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Use your static artwork as the background and overlay particles on top.
    """
    plt.style.use("default")

    # Maintain 2:3 aspect ratio
    fig, ax = plt.subplots(figsize=(4.0, 6.0))

    if not os.path.exists(BACKGROUND_PATH):
        raise FileNotFoundError(f"Background image not found at: {BACKGROUND_PATH}")
    img = mpimg.imread(BACKGROUND_PATH)

    ax.imshow(
        img,
        extent=(0.0, 1.0, 0.0, 1.0),
        aspect="auto",
        origin="upper",
        zorder=0,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0)  # y=0 top, y=1 bottom
    ax.axis("off")

    # Particles ON TOP of everything (zorder high)
    scat_big = ax.scatter([], [], s=18, color="#e8f7ff", alpha=0.95, zorder=6)
    scat_med = ax.scatter([], [], s=12, color="#bde6ff", alpha=0.90, zorder=6)
    scat_small = ax.scatter([], [], s=8, color="#8fd1ff", alpha=0.90, zorder=6)

    # HUD text (top-left of screen)
    hud_x = 0.06
    hud_y = 0.08
    dy = 0.035

    text_flow = ax.text(
        hud_x,
        hud_y,
        "",
        ha="left",
        va="top",
        fontsize=7,
        color="#e8f7ff",
        zorder=7,
    )
    text_pressure = ax.text(
        hud_x,
        hud_y + dy,
        "",
        ha="left",
        va="top",
        fontsize=7,
        color="#e8f7ff",
        zorder=7,
    )
    text_clog = ax.text(
        hud_x,
        hud_y + 2 * dy,
        "",
        ha="left",
        va="top",
        fontsize=7,
        color="#e8f7ff",
        zorder=7,
    )
    text_e = ax.text(
        hud_x,
        hud_y + 3 * dy,
        "",
        ha="left",
        va="top",
        fontsize=7,
        color="#e8f7ff",
        zorder=7,
    )
    text_turb = ax.text(
        hud_x,
        hud_y + 4 * dy,
        "",
        ha="left",
        va="top",
        fontsize=7,
        color="#e8f7ff",
        zorder=7,
    )
    text_step = ax.text(
        hud_x,
        hud_y + 5 * dy,
        "",
        ha="left",
        va="top",
        fontsize=7,
        color="#e8f7ff",
        zorder=7,
    )

    artists: Dict[str, Any] = {
        "ax": ax,
        "scat_big": scat_big,
        "scat_med": scat_med,
        "scat_small": scat_small,
        "text_flow": text_flow,
        "text_pressure": text_pressure,
        "text_clog": text_clog,
        "text_e": text_e,
        "text_turb": text_turb,
        "text_step": text_step,
    }
    return fig, artists


# ---------------------------------------------------------------------
#  ANIMATION
# ---------------------------------------------------------------------

def build_animation(history: Dict[str, np.ndarray]) -> animation.FuncAnimation:
    obs_hist = history["obs"]
    rewards = history["reward"]

    fig, artists = create_vertical_reactor_figure()
    ax = artists["ax"]

    particle_pos = init_particle_positions()
    pos_big = particle_pos["big"]
    pos_med = particle_pos["med"]
    pos_small = particle_pos["small"]

    scat_big = artists["scat_big"]
    scat_med = artists["scat_med"]
    scat_small = artists["scat_small"]

    text_flow = artists["text_flow"]
    text_pressure = artists["text_pressure"]
    text_clog = artists["text_clog"]
    text_e = artists["text_e"]
    text_turb = artists["text_turb"]
    text_step = artists["text_step"]

    def init():
        scat_big.set_offsets(pos_big)
        scat_med.set_offsets(pos_med)
        scat_small.set_offsets(pos_small)
        return (
            scat_big,
            scat_med,
            scat_small,
            text_flow,
            text_pressure,
            text_clog,
            text_e,
            text_turb,
            text_step,
        )

    def animate(frame: int):
        nonlocal pos_big, pos_med, pos_small

        idx = min(frame, obs_hist.shape[0] - 1)
        obs_t = obs_hist[idx]

        flow = float(obs_t[0]) if obs_t.size > 0 else 0.5
        pressure = float(obs_t[1]) if obs_t.size > 1 else 0.5
        clog = float(obs_t[2]) if obs_t.size > 2 else 0.0
        e_norm = float(obs_t[3]) if obs_t.size > 3 else 0.0
        turbidity = float(obs_t[10]) if obs_t.size > 10 else 0.0

        # Update particle positions
        pos_big = update_particles(pos_big, obs_t, "big", dt_vis=1.0 / FPS)
        pos_med = update_particles(pos_med, obs_t, "med", dt_vis=1.0 / FPS)
        pos_small = update_particles(pos_small, obs_t, "small", dt_vis=1.0 / FPS)

        scat_big.set_offsets(pos_big)
        scat_med.set_offsets(pos_med)
        scat_small.set_offsets(pos_small)

        # HUD
        text_flow.set_text(f"Flow:     {flow:5.2f}")
        text_pressure.set_text(f"Pressure: {pressure:5.2f}")
        text_clog.set_text(f"Clog:     {clog:5.2f}")
        text_e.set_text(f"E-field:  {e_norm:5.2f}")
        text_turb.set_text(f"Turbid.:  {turbidity:5.2f}")
        text_step.set_text(f"Frame:    {frame:4d}")

        return (
            scat_big,
            scat_med,
            scat_small,
            text_flow,
            text_pressure,
            text_clog,
            text_e,
            text_turb,
            text_step,
        )

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=N_FRAMES,
        interval=1000.0 / FPS,
        blit=True,
    )
    return anim


# ---------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------

def main():
    print("Loading PPO model + environment (12D)...")
    model, vec_env = load_model_and_env()

    print("Rolling out PPO-controlled episode...")
    history = rollout_for_animation(model, vec_env, max_frames=N_FRAMES)

    print("Building cinematic animation on background artwork...")
    anim = build_animation(history)

    print(f"Saving MP4 to {VIDEO_PATH} ...")
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
