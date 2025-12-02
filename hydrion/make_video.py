# hydrion/make_video.py
"""
Hydrion Digital Twin — Cinematic Renderer v10 (Scientific Edition)

- Preserves v8's scientifically faithful dynamics and coordinates
- Uses your cinematic artwork as the background
- 30 fps, 30-second research-realistic render
- Visual enhancements are *driven by physics* (e_norm, capture_eff, clog),
  not arbitrary cosmetics:
    * Particle core + glow layers (depth-like effect)
    * Glow intensity tied to electrostatic field + capture efficiency
    * Pipe field-bar illumination tied to e_norm
    * Subtle lab-like aesthetic while maintaining full realism

Run from project root:

    python -m hydrion.make_video

Output:

    videos/hydrion_run_v10.mp4

Requires: ffmpeg installed and on PATH.
"""

from __future__ import annotations
import os
from typing import Dict, Any, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

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
VIDEO_SECONDS = 30.0
N_FRAMES = int(FPS * VIDEO_SECONDS)
DT_VIS = 1.0 / FPS

VIDEO_PATH = os.path.join(VIDEOS_DIR, "hydrion_run_v10.mp4")


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
    obs_hist: List[np.ndarray] = []
    rew_hist: List[float] = []

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
# PARTICLE FIELD + MOTION (v8 physics, with your alignment)
# ============================================================

REACTOR_X_MIN = 0.33
REACTOR_X_MAX = 0.67

INFLOW_Y_MIN = 0.18
INFLOW_Y_MAX = 0.27

# Your aligned meshes:
# - Top assembly pushed down the most
# - Middle medium
# - Bottom a little
# (remember: larger y = lower on the image)
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

# Slightly refined jitter for research-realistic but not messy motion
JITTER_SCALE = 0.0018


def init_particle_positions(
    n_big: int = 220,
    n_med: int = 320,
    n_small: int = 440,
):
    rng = np.random.default_rng(42)

    def rand_group(n: int):
        xs = rng.uniform(REACTOR_X_MIN, REACTOR_X_MAX, size=n)
        ys = rng.uniform(INFLOW_Y_MIN, INFLOW_Y_MAX, size=n)
        return np.stack([xs, ys], axis=1)

    return {
        "big": rand_group(n_big),
        "med": rand_group(n_med),
        "small": rand_group(n_small),
    }


def update_particles(positions: np.ndarray, obs_t: np.ndarray, group: str, dt_vis: float):
    # Physics identical to v8: do not touch core logic
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

    # Rightward curvature into the pipe region
    depth = np.clip((y - INFLOW_Y_MIN) / (PIPE_BOTTOM_Y - INFLOW_Y_MIN), 0.0, 1.0)
    vx = CURVE_STRENGTH * np.clip(e_norm, 0.0, 1.0) * depth * dt_vis
    x += vx

    # Mesh sliding
    mesh_dir = np.array([np.cos(MESH_THETA), np.sin(MESH_THETA)])

    base_capture = 0.3 + 0.7 * np.clip(capture_eff, 0.0, 1.0)
    clog_factor = 1.0 - 0.4 * np.clip(clog, 0.0, 1.0)
    strength_global = base_capture * clog_factor  # (unused but kept for clarity)

    for i, yc in enumerate(MESH_CENTERS_Y):
        y_line = yc + MESH_TAN * (x - MESH_CENTER_X)
        dist = y - y_line
        on_mesh = np.abs(dist) < MESH_BAND

        if not np.any(on_mesh):
            continue

        layer_gain = 1.0 + 0.25 * i
        strength = strength_global * layer_gain  # (kept for future use)

        if i == target_mesh_idx:
            # Snap toward the mesh line with microscopic jitter
            noise = np.random.normal(0.0, 0.0015, size=on_mesh.sum())
            y[on_mesh] = y_line[on_mesh] + noise

            # Slide along the mesh in its direction
            slide = SLIDE_SPEED * (0.7 + 0.4 * e_norm + 0.3 * capture_eff) * dt_vis
            x[on_mesh] -= mesh_dir[0] * slide
            y[on_mesh] += mesh_dir[1] * slide
        else:
            # Light cross-layer influence
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

    # Respawn at inflow after passing the bottom of the pipe
    gone = y > PIPE_BOTTOM_Y
    if np.any(gone):
        n = int(gone.sum())
        x[gone] = np.random.uniform(REACTOR_X_MIN, REACTOR_X_MAX, size=n)
        y[gone] = np.random.uniform(INFLOW_Y_MIN, INFLOW_Y_MAX, size=n)

    # Micro-scale jitter to keep motion alive
    noise = np.random.normal(scale=JITTER_SCALE * dt_vis, size=positions.shape)
    x += noise[:, 0]
    y += noise[:, 1]

    x = np.clip(x, 0.25, 0.90)
    y = np.clip(y, 0.05, 0.95)

    positions[:, 0] = x
    positions[:, 1] = y
    return positions


# ============================================================
# COLOR / GLOW HELPERS
# ============================================================

def make_rgba(hex_color: str, alpha: float) -> Tuple[float, float, float, float]:
    r, g, b, _ = mcolors.to_rgba(hex_color)
    return (r, g, b, alpha)


def modulate_color_brightness(rgba: Tuple[float, float, float, float], factor: float):
    r, g, b, a = rgba
    factor = float(np.clip(factor, 0.0, 2.0))
    return (
        np.clip(r * factor, 0.0, 1.0),
        np.clip(g * factor, 0.0, 1.0),
        np.clip(b * factor, 0.0, 1.0),
        a,
    )


# Base colors (scientific cool palette)
BIG_CORE_BASE = "#e8f7ff"
MED_CORE_BASE = "#bde6ff"
SMALL_CORE_BASE = "#8fd1ff"

BIG_GLOW_BASE = "#e8f7ff"
MED_GLOW_BASE = "#bde6ff"
SMALL_GLOW_BASE = "#8fd1ff"


# ============================================================
# FIGURE / ARTISTS — BACKGROUND + PIPE FIELD BAR + GLOW
# ============================================================

def create_vertical_reactor_figure():
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(4.0, 6.0))

    if not os.path.exists(BACKGROUND_PATH):
        raise FileNotFoundError(f"Background image not found at: {BACKGROUND_PATH}")
    img = mpimg.imread(BACKGROUND_PATH)

    # Draw upright image (origin='lower') but flip axes so y=0 is top
    ax.imshow(
        img,
        extent=(0.0, 1.0, 0.0, 1.0),
        origin="lower",
        aspect="auto",
        zorder=0,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0)  # y=0 top, y=1 bottom
    ax.axis("off")

    # Field visualization around the electrostatic pipe (subtle blue column)
    field_height = PIPE_BOTTOM_Y - PIPE_TOP_Y
    pipe_field_bar = Rectangle(
        (PIPE_X - 0.02, PIPE_TOP_Y),
        0.04,
        field_height,
        linewidth=0.0,
        facecolor="#66d9ff",
        alpha=0.0,  # dynamically modulated by e_norm
        zorder=2,
    )
    ax.add_patch(pipe_field_bar)

    # Core particles (foreground)
    scat_big_core = ax.scatter([], [], s=20, color=BIG_CORE_BASE, alpha=0.90, zorder=6)
    scat_med_core = ax.scatter([], [], s=13, color=MED_CORE_BASE, alpha=0.88, zorder=6)
    scat_small_core = ax.scatter([], [], s=8, color=SMALL_CORE_BASE, alpha=0.86, zorder=6)

    # Glow particles (larger, softer, depth-like effect)
    scat_big_glow = ax.scatter([], [], s=60, color=BIG_GLOW_BASE, alpha=0.25, zorder=5)
    scat_med_glow = ax.scatter([], [], s=45, color=MED_GLOW_BASE, alpha=0.22, zorder=5)
    scat_small_glow = ax.scatter([], [], s=32, color=SMALL_GLOW_BASE, alpha=0.20, zorder=5)

    return fig, {
        "ax": ax,
        "pipe_field_bar": pipe_field_bar,
        "scat_big_core": scat_big_core,
        "scat_med_core": scat_med_core,
        "scat_small_core": scat_small_core,
        "scat_big_glow": scat_big_glow,
        "scat_med_glow": scat_med_glow,
        "scat_small_glow": scat_small_glow,
    }


# ============================================================
# ANIMATION
# ============================================================

def build_animation(history: Dict[str, np.ndarray]):
    obs_hist = history["obs"]

    fig, artists = create_vertical_reactor_figure()
    ax = artists["ax"]

    pipe_field_bar = artists["pipe_field_bar"]

    pos = init_particle_positions()

    scat_big_core = artists["scat_big_core"]
    scat_med_core = artists["scat_med_core"]
    scat_small_core = artists["scat_small_core"]

    scat_big_glow = artists["scat_big_glow"]
    scat_med_glow = artists["scat_med_glow"]
    scat_small_glow = artists["scat_small_glow"]

    # Precompute RGBA base colors
    big_core_rgba = make_rgba(BIG_CORE_BASE, alpha=0.90)
    med_core_rgba = make_rgba(MED_CORE_BASE, alpha=0.88)
    small_core_rgba = make_rgba(SMALL_CORE_BASE, alpha=0.86)

    big_glow_rgba = make_rgba(BIG_GLOW_BASE, alpha=0.25)
    med_glow_rgba = make_rgba(MED_GLOW_BASE, alpha=0.22)
    small_glow_rgba = make_rgba(SMALL_GLOW_BASE, alpha=0.20)

    def init():
        # Initial positions
        scat_big_core.set_offsets(pos["big"])
        scat_med_core.set_offsets(pos["med"])
        scat_small_core.set_offsets(pos["small"])

        scat_big_glow.set_offsets(pos["big"])
        scat_med_glow.set_offsets(pos["med"])
        scat_small_glow.set_offsets(pos["small"])

        # Initial optical state
        scat_big_core.set_facecolor(big_core_rgba)
        scat_med_core.set_facecolor(med_core_rgba)
        scat_small_core.set_facecolor(small_core_rgba)

        scat_big_glow.set_facecolor(big_glow_rgba)
        scat_med_glow.set_facecolor(med_glow_rgba)
        scat_small_glow.set_facecolor(small_glow_rgba)

        pipe_field_bar.set_alpha(0.0)

        return (
            scat_big_core,
            scat_med_core,
            scat_small_core,
            scat_big_glow,
            scat_med_glow,
            scat_small_glow,
            pipe_field_bar,
        )

    def animate(frame: int):
        obs = obs_hist[min(frame, obs_hist.shape[0] - 1)]

        # Extract key physical signals for visual encoding
        flow = float(obs[0])
        clog = float(obs[2])
        e_norm = float(obs[3])
        capture_eff = float(obs[5])

        e_level = float(np.clip(e_norm, 0.0, 1.0))
        capture_level = float(np.clip(capture_eff, 0.0, 1.0))
        clog_level = float(np.clip(clog, 0.0, 1.0))

        # Physics update (unchanged from v8)
        pos["big"] = update_particles(pos["big"], obs, "big", dt_vis=DT_VIS)
        pos["med"] = update_particles(pos["med"], obs, "med", dt_vis=DT_VIS)
        pos["small"] = update_particles(pos["small"], obs, "small", dt_vis=DT_VIS)

        # Update positions
        scat_big_core.set_offsets(pos["big"])
        scat_med_core.set_offsets(pos["med"])
        scat_small_core.set_offsets(pos["small"])

        scat_big_glow.set_offsets(pos["big"])
        scat_med_glow.set_offsets(pos["med"])
        scat_small_glow.set_offsets(pos["small"])

        # --- Scientific glow & color modulation ---

        # Core brightness slightly increases with field & capture
        core_intensity = 0.85 + 0.25 * (0.6 * e_level + 0.4 * capture_level)
        # Clogging gently dims the scene
        dim_factor = 1.0 - 0.3 * clog_level
        core_factor = core_intensity * dim_factor

        big_core_col = modulate_color_brightness(big_core_rgba, core_factor)
        med_core_col = modulate_color_brightness(med_core_rgba, core_factor)
        small_core_col = modulate_color_brightness(small_core_rgba, core_factor)

        scat_big_core.set_facecolor(big_core_col)
        scat_med_core.set_facecolor(med_core_col)
        scat_small_core.set_facecolor(small_core_col)

        # Glow intensity tracks capture efficiency more strongly
        glow_intensity = 0.7 + 0.5 * capture_level
        glow_dim = 1.0 - 0.5 * clog_level
        glow_factor = glow_intensity * glow_dim

        big_glow_col = modulate_color_brightness(big_glow_rgba, glow_factor)
        med_glow_col = modulate_color_brightness(med_glow_rgba, glow_factor)
        small_glow_col = modulate_color_brightness(small_glow_rgba, glow_factor)

        scat_big_glow.set_facecolor(big_glow_col)
        scat_med_glow.set_facecolor(med_glow_col)
        scat_small_glow.set_facecolor(small_glow_col)

        # Pipe field-bar alpha reflects electrostatic field strength
        base_alpha = 0.03
        field_alpha = base_alpha + 0.20 * e_level
        pipe_field_bar.set_alpha(float(np.clip(field_alpha, 0.0, 0.35)))

        return (
            scat_big_core,
            scat_med_core,
            scat_small_core,
            scat_big_glow,
            scat_med_glow,
            scat_small_glow,
            pipe_field_bar,
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


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading PPO model + environment...")
    model, vec_env = load_model_and_env()

    print("Rolling out PPO episode...")
    history = rollout_for_animation(model, vec_env, N_FRAMES)

    print("Rendering scientific-cinematic animation (v10)...")
    anim = build_animation(history)

    print(f"Saving MP4 to: {VIDEO_PATH}")
    writer = animation.FFMpegWriter(
        fps=FPS,
        metadata={"artist": "HydrionRL"},
        bitrate=9000,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"],
    )
    anim.save(VIDEO_PATH, writer=writer)

    print("✅ v10 scientific cinematic video saved.")


if __name__ == "__main__":
    main()
