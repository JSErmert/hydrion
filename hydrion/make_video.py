"""
Hydrion Digital Twin — 2D Video Generator (MP4)

- Loads trained PPO model + VecNormalize stats
- Runs one evaluation episode in HydrionEnv
- Builds a high-level 2D visualization (pipe, meshes, particles, field, sensor)
- Exports an MP4 animation.

Run from project root:

    python -m hydrion.make_video

Output:

    videos/hydrion_run.mp4

Note: Requires ffmpeg to be installed and on your PATH for MP4 export.
"""

from __future__ import annotations

import os
from typing import Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

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

MAX_STEPS = 6000          # max episode steps
TARGET_FRAMES = 800       # 600–1000 as requested
NUM_PARTICLES = 1000      # dense cloud


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

    # Base vec env
    base_env = DummyVecEnv([make_env()])

    # Load normalization stats and attach to env
    vec_env: VecNormalize = VecNormalize.load(VECNORM_PATH, base_env)
    vec_env.training = False
    vec_env.norm_reward = False

    # Load PPO tied to this vec env
    model: PPO = PPO.load(MODEL_PATH, env=vec_env)

    return model, vec_env


# ---------------------------------------------------------------------
#  ROLLOUT EPISODE (USING TRAINED POLICY)
# ---------------------------------------------------------------------


def rollout_episode(vec_env: VecNormalize, model: PPO, max_steps: int = MAX_STEPS) -> Dict[str, np.ndarray]:
    """
    Roll out a single episode with the trained policy and record:
    - obs[t, 12]
    - actions[t, 4]
    - rewards[t]
    """

    obs = vec_env.reset()   # VecEnv API: returns obs only

    obs_hist = []
    act_hist = []
    rew_hist = []

    for t in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, dones, infos = vec_env.step(action)

        # VecEnv wraps batch dimension -> index 0
        obs_hist.append(next_obs[0].copy())
        act_hist.append(action[0].copy())
        rew_hist.append(float(reward[0]))

        obs = next_obs

        if dones[0]:
            break

    obs_arr = np.asarray(obs_hist, dtype=np.float32)
    act_arr = np.asarray(act_hist, dtype=np.float32)
    rew_arr = np.asarray(rew_hist, dtype=np.float32)

    history = {
        "obs": obs_arr,
        "actions": act_arr,
        "rewards": rew_arr,
    }
    return history


# ---------------------------------------------------------------------
#  PARTICLE FIELD (PURELY VISUAL)
# ---------------------------------------------------------------------


def init_particles(num: int = NUM_PARTICLES) -> np.ndarray:
    """
    Initialize particles in normalized pipe coordinates [0,1]x[0,1].
    """
    # Slightly inset from pipe walls
    x = np.random.uniform(0.0, 0.2, size=num)
    y = np.random.uniform(0.1, 0.9, size=num)
    return np.stack([x, y], axis=-1)


def update_particles(positions: np.ndarray, obs_t: np.ndarray) -> np.ndarray:
    """
    Update particle positions based on current observation.
    Uses:
        obs_t[0] -> flow (0–1)
        obs_t[3] -> E_norm (0–1)
    """
    flow = float(obs_t[0])
    e_norm = float(obs_t[3])

    # Base drift speed along x (pipe direction)
    vx = 0.002 + 0.008 * flow

    # Electrostatic vertical bias (toward meshes/electrodes)
    vy_center_bias = (e_norm - 0.5) * 0.004

    # Add noise for turbulent jitter
    noise = np.random.normal(scale=0.0015, size=positions.shape)

    positions[:, 0] += vx + noise[:, 0]
    positions[:, 1] += vy_center_bias + noise[:, 1]

    # Keep y inside pipe walls
    positions[:, 1] = np.clip(positions[:, 1], 0.05, 0.95)

    # Wrap-around at outlet -> inlet
    positions[:, 0] = np.mod(positions[:, 0], 1.0)

    return positions


# ---------------------------------------------------------------------
#  VIDEO ANIMATION
# ---------------------------------------------------------------------


def create_digital_twin_figure() -> Tuple[plt.Figure, Any, Dict[str, Any]]:
    """
    Build the Matplotlib figure + artists for a 2D digital twin look.
    Returns:
        fig, ax_main, artists dict
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle("Hydrion Digital Twin — PPO-Controlled Episode", fontsize=14)

    ax.set_xlim(0, 1.2)   # extra space on right for camera
    ax.set_ylim(0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # PIPE walls
    pipe = Rectangle((0.0, 0.05), 1.0, 0.9,
                     linewidth=1.5, edgecolor="black", facecolor="none")
    ax.add_patch(pipe)

    # Mesh positions along pipe
    mesh_xs = [0.3, 0.5, 0.7]
    mesh_patches = []
    for mx in mesh_xs:
        patch = Rectangle((mx, 0.05), 0.01, 0.9,
                          linewidth=0.5,
                          edgecolor="black",
                          facecolor="#dddddd",
                          alpha=0.7)
        ax.add_patch(patch)
        mesh_patches.append(patch)

    # Electrodes (thin strips near meshes)
    electrode_patches = []
    for mx in mesh_xs:
        patch = Rectangle((mx - 0.015, 0.05), 0.005, 0.9,
                          linewidth=0.0,
                          facecolor="#88bbff",
                          alpha=0.0)   # visibility tied to E field
        ax.add_patch(patch)
        electrode_patches.append(patch)

    # Optical camera window on right
    camera_body = Rectangle((1.02, 0.3), 0.16, 0.4,
                            linewidth=1.0,
                            edgecolor="black",
                            facecolor="#222222")
    ax.add_patch(camera_body)

    sensor_rect = Rectangle((1.04, 0.32), 0.12, 0.36,
                            linewidth=0.0,
                            facecolor="#444444")
    ax.add_patch(sensor_rect)

    # Beam from pipe outlet to camera
    beam = ax.plot([0.95, 1.02], [0.5, 0.5],
                   linestyle="-",
                   linewidth=2.0,
                   color="#44e0ff",
                   alpha=0.4)[0]

    # Particle scatter
    particle_scatter = ax.scatter([], [], s=4, alpha=0.6, color="#ffaa44")

    # Text overlays for key signals
    text_flow = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=8)
    text_pressure = ax.text(0.02, 0.90, "", transform=ax.transAxes, fontsize=8)
    text_clog = ax.text(0.02, 0.84, "", transform=ax.transAxes, fontsize=8)
    text_e = ax.text(0.35, 0.96, "", transform=ax.transAxes, fontsize=8)
    text_turb = ax.text(0.35, 0.90, "", transform=ax.transAxes, fontsize=8)
    text_scatter = ax.text(0.35, 0.84, "", transform=ax.transAxes, fontsize=8)
    text_step = ax.text(0.70, 0.96, "", transform=ax.transAxes, fontsize=8)

    artists = {
        "mesh_patches": mesh_patches,
        "electrode_patches": electrode_patches,
        "sensor_rect": sensor_rect,
        "beam": beam,
        "particle_scatter": particle_scatter,
        "text_flow": text_flow,
        "text_pressure": text_pressure,
        "text_clog": text_clog,
        "text_e": text_e,
        "text_turb": text_turb,
        "text_scatter": text_scatter,
        "text_step": text_step,
    }

    return fig, ax, artists


def build_animation(history: Dict[str, np.ndarray]) -> animation.FuncAnimation:
    """
    Given rollout history, construct a high-fidelity 2D animation.
    """
    obs = history["obs"]  # shape [T, 12]
    rewards = history["rewards"]
    T = obs.shape[0]

    # Downsample to ~TARGET_FRAMES
    stride = max(1, T // TARGET_FRAMES)
    frame_indices = np.arange(0, T, stride)
    n_frames = len(frame_indices)

    # Figure + artists
    fig, ax, artists = create_digital_twin_figure()

    # Initialize particle positions
    particle_pos = init_particles(NUM_PARTICLES)

    # Convenience handles
    scatter = artists["particle_scatter"]
    sensor_rect = artists["sensor_rect"]
    electrode_patches = artists["electrode_patches"]

    text_flow = artists["text_flow"]
    text_pressure = artists["text_pressure"]
    text_clog = artists["text_clog"]
    text_e = artists["text_e"]
    text_turb = artists["text_turb"]
    text_scatter = artists["text_scatter"]
    text_step = artists["text_step"]

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        return [scatter]

    def animate(frame_idx: int):
        step = int(frame_indices[frame_idx])
        o = obs[step]

        # Update particle field
        nonlocal particle_pos
        particle_pos = update_particles(particle_pos, o)
        scatter.set_offsets(particle_pos)

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

        # Electrode visibility/color tied to E_norm
        alpha_e = 0.1 + 0.6 * e_norm
        color_e = (0.3, 0.6, 1.0)
        for ep in electrode_patches:
            ep.set_alpha(alpha_e)
            ep.set_facecolor(color_e)

        # Sensor brightness tied to turbidity & scatter signal
        # Map to pleasant teal/orange mix
        base = 0.15 + 0.5 * turbidity
        sensor_color = (base * (1 - scatter_sig * 0.5),
                        base * (0.9 + scatter_sig * 0.3),
                        base * (1.0))
        sensor_rect.set_facecolor(sensor_color)

        # Text overlays
        text_flow.set_text(f"Flow: {flow:.3f}")
        text_pressure.set_text(f"P_in: {pressure:.3f}")
        text_clog.set_text(f"Clog: {clog:.3f}")
        text_e.set_text(f"E_norm: {e_norm:.3f}")
        text_turb.set_text(f"Turbidity: {turbidity:.3f}")
        text_scatter.set_text(f"Scatter: {scatter_sig:.3f}")
        text_step.set_text(f"Step: {step} / {T-1}  Reward: {rewards[step]:.2f}")

        return [
            scatter,
            sensor_rect,
            *electrode_patches,
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
        interval=33,          # ~30 FPS
        blit=True,
    )

    return anim


# ---------------------------------------------------------------------
#  MAIN ENTRYPOINT
# ---------------------------------------------------------------------


def main():
    os.makedirs(VIDEO_DIR, exist_ok=True)

    print("Loading PPO model + environment...")
    model, vec_env = load_model_and_env()

    print("Rolling out a PPO-controlled episode...")
    history = rollout_episode(vec_env, model, max_steps=MAX_STEPS)
    print(f"Episode length: {history['obs'].shape[0]} steps")

    print("Building animation...")
    anim = build_animation(history)

    print(f"Saving MP4 to {VIDEO_PATH} ...")
    # Use FFMpegWriter for MP4
    writer = animation.FFMpegWriter(
        fps=30,
        metadata={"artist": "HydrionRL"},
        bitrate=2400,
    )
    anim.save(VIDEO_PATH, writer=writer)
    print("✅ Video saved.")


if __name__ == "__main__":
    main()
