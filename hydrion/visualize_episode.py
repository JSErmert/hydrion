"""
hydrion/visualize_episode.py

Truthful spatial episode visualization (Commit 5).
Pure observer: no mechanics, no reward, no PPO, no viz2d.
"""

from hydrion.visual_sampling.particle_sampler import ParticleSampler
from hydrion.rendering.static_geometry import draw_static_context


def visualize_episode(
    env,
    renderer,
    seed: int,
    max_steps: int = 1000,
):
    # ----------------------------
    # Reset environment
    # ----------------------------
    obs, info = env.reset(seed=seed)

    # ----------------------------
    # Initialize particle sampler
    # ----------------------------
    sampler = ParticleSampler(
        max_particles=500,
        inlet_x_range=(-0.4, 0.4),
        inlet_y=1.0,
        flow_speed=0.5,
        sink_x=0.8,
    )

    # IMPORTANT: use env.truth_state (Commit 3 contract)
    sampler.reset(
        state=env.truth_state,
        seed=seed,
    )

    # ----------------------------
    # Episode loop
    # ----------------------------
    done = False
    step_count = 0

    while not done and step_count < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # IMPORTANT: use env.dt (runtime truth)
        particles = sampler.step(dt=env.dt)

        renderer.begin_frame()
        draw_static_context(renderer)
        renderer.draw_particles(particles)
        renderer.end_frame()

        step_count += 1

    renderer.close()
