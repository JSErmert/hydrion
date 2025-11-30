# hydrion/make_video.py
"""
Hydrion Digital Twin — Vertical Reactor Video Generator (MP4)

- Loads trained PPO model + VecNormalize stats
- Runs one evaluation episode in HydrionEnv
- Builds a vertical “reactor column” visualization matching the
  Comprehensive Design:
    * Water inflow at top
    * Polarize layer
    * Tri-Layer Electric Node Extraction
    * Sensor live feedback
    * Storage chamber + up-cycling drain
    * Water outflow at bottom
- Animates 3 particle-size groups as colored dots.

Run from project root:

    python -m hydrion.make_video

Output:

    videos/hydrion_run.mp4

Requires: ffmpeg installed and on PATH.
"""

from __future__ import annotations

import os
from typing import Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .env import HydrionEnv


# ---------------------------------------------------------------------
#  CONFIG
# ---------------------------------------------------------------------

MODEL_PATH = "ppo_hydrion_final_12d.zip"
VECNORM_PATH = "ppo_hydrion_vecnormalize_12d.pkl"

VIDEO_DIR = "videos"
VIDEO_PATH = os.path.join(VIDEO_DIR, "hydrion_run.mp4")

MAX_STEPS = 6000            # max episode steps
TARGET_FRAMES = 800         # 600–1000 as requested

# Particle groups (different sizes & colors)
N_BIG = 200     # ~500 µm
N_MED = 300     # ~100 µm
N_SMALL = 500   # ~5 µm
NUM_PARTICLES = N_BIG + N_MED + N_SMALL


# ---------------------------------------------------------------------
#  ENV + MODEL LOADING
# ---------------------------------------------------------------------


def make_env():
    """Factory for a fresh HydrionEnv (used by DummyVecEnv)."""
    def _init():
        return HydrionEnv()
    return _init


def load_model_and_env() -> Tuple[PPO, VecNormalize]:
    """Load PPO + VecNormalize with the correct HydrionEnv backend."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing trained model: {MODEL_PATH}")
    if not os.path.exists(VECNORM_PATH):
        raise FileNotFoundError(f"Missing VecNormalize stats: {VECNORM_PATH}")

    base_env = DummyVecEnv([make_env()])
    vec_env: VecNormalize = VecNormalize.load(VECNORM_PATH, base_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model: PPO = PPO.load(MODEL_PATH, env=vec_env)
    return model, vec_env


# ---------------------------------------------------------------------
#  ROLLOUT EPISODE
# ---------------------------------------------------------------------


def rollout_episode(
    vec_env: VecNormalize,
    model: PPO,
    max_steps: int = MAX_STEPS,
) -> Dict[str, np.ndarray]:
    """
    Roll out a single episode with the trained policy and record:
    - obs[t, 12]
    - actions[t, 4]
    - rewards[t]
    """
    obs = vec_env.reset()

    obs_hist = []
    act_hist = []
    rew_hist = []

    for t in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, dones, infos = vec_env.step(action)

        obs_hist.append(next_obs[0].copy())
        act_hist.append(action[0].copy())
        rew_hist.append(float(reward[0]))

        obs = next_obs
        if dones[0]:
            break

    obs_arr = np.asarray(obs_hist, dtype=np.float32)
    act_arr = np.asarray(act_hist, dtype=np.float32)
    rew_arr = np.asarray(rew_hist, dtype=np.float32)

    return {"obs": obs_arr, "actions": act_arr, "rewards": rew_arr}


# ---------------------------------------------------------------------
#  PARTICLE FIELD (VISUAL ONLY)
# ---------------------------------------------------------------------


def init_particles() -> Dict[str, np.ndarray]:
    """
    Initialize particle positions within cylinder coordinates.

    We use normalized coordinates:
        x in [0, 1] (horizontal)
        y in [0, 1] (vertical, 1 = top, 0 = bottom)
    The reactor body will be drawn within x ~ [0.2, 0.8].
    """
    # Slightly random distribution near top half
    x_all = np.random.uniform(0.3, 0.7, size=NUM_PARTICLES)
    y_all = np.random.uniform(0.6, 1.0, size=NUM_PARTICLES)

    # Split into size groups
    big_idx = np.arange(0, N_BIG)
    med_idx = np.arange(N_BIG, N_BIG + N_MED)
    small_idx = np.arange(N_BIG + N_MED, NUM_PARTICLES)

    return {
        "pos": np.stack([x_all, y_all], axis=-1),
        "big_idx": big_idx,
        "med_idx": med_idx,
        "small_idx": small_idx,
    }


def update_particles(
    positions: np.ndarray,
    obs_t: np.ndarray,
    dt_vis: float = 1.0 / 30.0,
) -> np.ndarray:
    """
    Update particle positions based on current observation.

    Uses:
        obs_t[0] -> flow (0–1)
        obs_t[3] -> E_norm (0–1)
    """

    flow = float(obs_t[0])
    e_norm = float(obs_t[3])

    # Downward advection speed (y decreases)
    vy = (0.05 + 0.25 * flow) * dt_vis

    # Weak radial drift toward electric node region (x ~ 0.5)
    center_x = 0.5
    dx = center_x - positions[:, 0]
    vx = 0.03 * e_norm * dx * dt_vis

    # Random jitter
    noise = np.random.normal(scale=0.01 * dt_vis, size=positions.shape)

    positions[:, 0] += vx + noise[:, 0]
    positions[:, 1] -= vy + noise[:, 1]  # downward

    # Keep within reactor cylinder horizontally
    positions[:, 0] = np.clip(positions[:, 0], 0.28, 0.72)

    # Wrap-around: particles reaching bottom re-enter near top
    bottom_mask = positions[:, 1] < 0.05
    positions[bottom_mask, 1] = np.random.uniform(0.8, 1.0, size=bottom_mask.sum())
    positions[bottom_mask, 0] = np.random.uniform(0.3, 0.7, size=bottom_mask.sum())

    # Also clamp very top
    positions[:, 1] = np.clip(positions[:, 1], 0.05, 0.98)

    return positions


# ---------------------------------------------------------------------
#  FIGURE / ARTISTS (VERTICAL REACTOR)
# ---------------------------------------------------------------------


def create_vertical_reactor_figure() -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Build the Matplotlib figure + artists for a vertical reactor layout
    matching the Comprehensive Design aesthetic.
    """
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(4.5, 8))

    # Dark background theme
    fig.patch.set_facecolor("#021521")
    ax.set_facecolor("#021521")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Cylindrical reactor body (rounded top & bottom using FancyBboxPatch)
    reactor = FancyBboxPatch(
        (0.25, 0.08),
        0.5,
        0.74,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=2.0,
        edgecolor="#43b3ff",
        facecolor="#021b2b",
    )
    ax.add_patch(reactor)

    # Polarize layer (top horizontal band)
    polarize = Rectangle(
        (0.27, 0.78),
        0.46,
        0.03,
        linewidth=1.5,
        edgecolor="#ffcc4d",
        facecolor="#ffcc4d",
        alpha=0.8,
    )
    ax.add_patch(polarize)

    # Tri-layer electric node extraction zone (rectangle with cross-hatch)
    node_zone = Rectangle(
        (0.30, 0.35),
        0.40,
        0.26,
        linewidth=1.5,
        edgecolor="#ffb347",
        facecolor="#102235",
        hatch="xxx",
        alpha=1.0,
    )
    ax.add_patch(node_zone)

    # Three electrode rings along side of node zone
    electrode_nodes = []
    node_ys = [0.56, 0.48, 0.40]
    for y in node_ys:
        c = Circle((0.73, y), 0.02, edgecolor="#ffcc4d",
                   facecolor="#ffcc4d", alpha=0.8)
        ax.add_patch(c)
        electrode_nodes.append(c)

    # Storage chamber below node zone
    storage = Rectangle(
        (0.30, 0.14),
        0.40,
        0.18,
        linewidth=1.5,
        edgecolor="#ffb347",
        facecolor="#052438",
        alpha=1.0,
    )
    ax.add_patch(storage)

    # Up-cycling drain (side box + pipe)
    drain_box = Rectangle(
        (0.70, 0.20),
        0.16,
        0.10,
        linewidth=1.0,
        edgecolor="#ffb347",
        facecolor="#052438",
    )
    ax.add_patch(drain_box)

    drain_pipe = Rectangle(
        (0.66, 0.23),
        0.04,
        0.04,
        linewidth=0.5,
        edgecolor="#ffb347",
        facecolor="#ffb347",
        alpha=0.8,
    )
    ax.add_patch(drain_pipe)

    # Sensor feedback icon above node zone (red)
    sensor_circle = Circle(
        (0.72, 0.73),
        0.03,
        edgecolor="#ff5959",
        facecolor="#ff5959",
        alpha=0.9,
    )
    ax.add_patch(sensor_circle)

    # Concentric halos for sensor
    sensor_halos = []
    for r in [0.045, 0.065]:
        halo = Circle(
            (0.72, 0.73),
            r,
            edgecolor="#ff5959",
            facecolor="none",
            linewidth=1.0,
            alpha=0.5,
        )
        ax.add_patch(halo)
        sensor_halos.append(halo)

    # Water inflow (droplet at top)
    inflow_drop = Circle(
        (0.50, 0.92),
        0.018,
        edgecolor="#4dd0ff",
        facecolor="#4dd0ff",
    )
    ax.add_patch(inflow_drop)

    # Water outflow droplet at bottom
    outflow_drop = Circle(
        (0.50, 0.04),
        0.018,
        edgecolor="#4dd0ff",
        facecolor="#4dd0ff",
    )
    ax.add_patch(outflow_drop)

    # Particle scatter (all groups; colored by size later)
    particle_scatter_big = ax.scatter([], [], s=18, color="#ffdd55", alpha=0.8)
    particle_scatter_med = ax.scatter([], [], s=10, color="#ff9b42", alpha=0.8)
    particle_scatter_small = ax.scatter([], [], s=6, color="#ffc4a3", alpha=0.9)

    # Text labels (yellow/white)
    title = ax.text(
        0.50,
        0.97,
        "Hydrion Comprehensive Design — Digital Twin",
        ha="center",
        va="top",
        color="#ffffff",
        fontsize=11,
    )
    label_polarize = ax.text(
        0.50,
        0.82,
        "Polarize Layer",
        ha="center",
        va="bottom",
        color="#ffcc4d",
        fontsize=8,
    )
    label_node = ax.text(
        0.50,
        0.63,
        "Tri-Layer Electric Node Extraction",
        ha="center",
        va="bottom",
        color="#ffb347",
        fontsize=8,
    )
    label_storage = ax.text(
        0.50,
        0.14,
        "Storage Chamber",
        ha="center",
        va="top",
        color="#ffb347",
        fontsize=8,
    )
    label_upcycle = ax.text(
        0.78,
        0.31,
        "Up-cycling\nDrain",
        ha="center",
        va="top",
        color="#ffb347",
        fontsize=7,
    )
    label_sensor = ax.text(
        0.72,
        0.77,
        "Sensor\nLive Feedback",
        ha="center",
        va="bottom",
        color="#ff5959",
        fontsize=7,
    )
    label_inflow = ax.text(
        0.50,
        0.93,
        "Water Inflow",
        ha="center",
        va="bottom",
        color="#4dd0ff",
        fontsize=8,
    )
    label_outflow = ax.text(
        0.50,
        0.00,
        "Water Outflow",
        ha="center",
        va="bottom",
        color="#4dd0ff",
        fontsize=8,
    )

    # HUD metrics on left/right
    text_flow = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        fontsize=8,
        color="#ffcc4d",
    )
    text_pressure = ax.text(
        0.02,
        0.92,
        "",
        transform=ax.transAxes,
        fontsize=8,
        color="#ffcc4d",
    )
    text_clog = ax.text(
        0.02,
        0.88,
        "",
        transform=ax.transAxes,
        fontsize=8,
        color="#ffcc4d",
    )
    text_e = ax.text(
        0.02,
        0.84,
        "",
        transform=ax.transAxes,
        fontsize=8,
        color="#ffb347",
    )
    text_turb = ax.text(
        0.70,
        0.96,
        "",
        transform=ax.transAxes,
        fontsize=8,
        color="#ff5959",
    )
    text_scatter = ax.text(
        0.70,
        0.92,
        "",
        transform=ax.transAxes,
        fontsize=8,
        color="#ff5959",
    )
    text_step = ax.text(
        0.70,
        0.88,
        "",
        transform=ax.transAxes,
        fontsize=8,
        color="#ffffff",
    )

    artists = {
        "ax": ax,
        "reactor": reactor,
        "polarize": polarize,
        "node_zone": node_zone,
        "electrode_nodes": electrode_nodes,
        "storage": storage,
        "drain_box": drain_box,
        "drain_pipe": drain_pipe,
        "sensor_circle": sensor_circle,
        "sensor_halos": sensor_halos,
        "inflow_drop": inflow_drop,
        "outflow_drop": outflow_drop,
        "particle_scatter_big": particle_scatter_big,
        "particle_scatter_med": particle_scatter_med,
        "particle_scatter_small": particle_scatter_small,
        "text_flow": text_flow,
        "text_pressure": text_pressure,
        "text_clog": text_clog,
        "text_e": text_e,
        "text_turb": text_turb,
        "text_scatter": text_scatter,
        "text_step": text_step,
    }

    return fig, artists


# ---------------------------------------------------------------------
#  ANIMATION CONSTRUCTION
# ---------------------------------------------------------------------


def build_animation(history: Dict[str, np.ndarray]) -> animation.FuncAnimation:
    """
    Given rollout history, construct the vertical reactor animation.
    """
    obs = history["obs"]        # [T, 12]
    rewards = history["rewards"]
    T = obs.shape[0]

    stride = max(1, T // TARGET_FRAMES)
    frame_indices = np.arange(0, T, stride)
    n_frames = len(frame_indices)

    fig, artists = create_vertical_reactor_figure()

    particles = init_particles()
    pos = particles["pos"]
    big_idx = particles["big_idx"]
    med_idx = particles["med_idx"]
    small_idx = particles["small_idx"]

    scat_big = artists["particle_scatter_big"]
    scat_med = artists["particle_scatter_med"]
    scat_small = artists["particle_scatter_small"]

    storage = artists["storage"]
    sensor_circle = artists["sensor_circle"]
    sensor_halos = artists["sensor_halos"]
    electrode_nodes = artists["electrode_nodes"]

    text_flow = artists["text_flow"]
    text_pressure = artists["text_pressure"]
    text_clog = artists["text_clog"]
    text_e = artists["text_e"]
    text_turb = artists["text_turb"]
    text_scatter = artists["text_scatter"]
    text_step = artists["text_step"]

    # Running estimate of captured fraction for storage fill
    captured_fill = 0.0

    def init():
        scat_big.set_offsets(np.empty((0, 2)))
        scat_med.set_offsets(np.empty((0, 2)))
        scat_small.set_offsets(np.empty((0, 2)))
        return [scat_big, scat_med, scat_small]

    def animate(frame_idx: int):
        nonlocal pos, captured_fill

        step = int(frame_indices[frame_idx])
        o = obs[step]

        # Hydraulics/clogging
        flow = float(o[0])
        pressure = float(o[1])
        clog = float(o[2])

        # Electrostatics
        e_norm = float(o[3])

        # Particles
        c_out = float(o[4])
        p_eff = float(o[5])

        # Sensors
        turbidity = float(o[10])
        scatter_sig = float(o[11])

        # Update particle field
        pos = update_particles(pos, o)
        scat_big.set_offsets(pos[big_idx])
        scat_med.set_offsets(pos[med_idx])
        scat_small.set_offsets(pos[small_idx])

        # Simulate captured fraction slowly increasing with p_eff
        captured_fill = min(1.0, captured_fill + 0.01 * p_eff)
        # Storage chamber color shifts with captured fraction
        base_color = np.array([0.02, 0.15, 0.28])
        capture_color = np.array([1.0, 0.72, 0.28])
        mix = base_color * (1 - captured_fill) + capture_color * captured_fill
        mix = np.clip(mix, 0.0, 1.0)
        storage.set_facecolor(mix)

        # Electrode glow intensity from e_norm
        glow_alpha = 0.3 + 0.6 * e_norm
        glow_alpha = float(np.clip(glow_alpha, 0.0, 1.0))
        glow_color = (1.0, 0.75, 0.35)
        for node in electrode_nodes:
            node.set_alpha(glow_alpha)
            node.set_facecolor(glow_color)

        # Sensor brightness from turbidity & scatter
        sensor_intensity = float(np.clip(0.3 + 0.7 * (turbidity + 0.5 * scatter_sig), 0.0, 1.0))
        sensor_color = (sensor_intensity, 0.35 * sensor_intensity, 0.35 * sensor_intensity)
        sensor_circle.set_facecolor(sensor_color)
        for halo in sensor_halos:
            halo.set_alpha(0.2 + 0.4 * sensor_intensity)

        # HUD text
        text_flow.set_text(f"Flow: {flow:.3f}")
        text_pressure.set_text(f"P_in: {pressure:.3f}")
        text_clog.set_text(f"Clog: {clog:.3f}")
        text_e.set_text(f"E_norm: {e_norm:.3f}")
        text_turb.set_text(f"Turbidity: {turbidity:.3f}")
        text_scatter.set_text(f"Scatter: {scatter_sig:.3f}")
        text_step.set_text(f"Step {step}/{T-1}  R={rewards[step]:.1f}")

        return [
            scat_big,
            scat_med,
            scat_small,
            storage,
            sensor_circle,
            *sensor_halos,
            *electrode_nodes,
            text_flow,
            text_pressure,
            text_clog,
            text_e,
            text_turb,
            text_scatter,
            text_step,
        ]

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=n_frames,
        interval=33,   # ~30 FPS
        blit=True,
    )
    return anim


# ---------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------


def main():
    os.makedirs(VIDEO_DIR, exist_ok=True)

    print("Loading PPO model + environment...")
    model, vec_env = load_model_and_env()

    print("Rolling out a PPO-controlled episode...")
    history = rollout_episode(vec_env, model, max_steps=MAX_STEPS)
    print(f"Episode length: {history['obs'].shape[0]} steps")

    print("Building animation (vertical reactor)...")
    anim = build_animation(history)

    print(f"Saving MP4 to {VIDEO_PATH} ...")
    writer = animation.FFMpegWriter(
        fps=30,
        metadata={"artist": "HydrionRL"},
        bitrate=2400,
    )
    anim.save(VIDEO_PATH, writer=writer)
    print("✅ Video saved.")


if __name__ == "__main__":
    main()
