"""
hydrion/visualize_episode.py

Truthful episode visualization with static spatial context.
"""

from hydrion.visual_sampling.particle_sampler import ParticleSampler
from hydrion.rendering.static_geometry import draw_static_context


def visualize_episode(
    env,
    renderer,
    seed: int,
    max_steps: int = 1000,
):
    """
    Run a single episode with deterministic visual inspection.

    Visualization is a pure observer:
    - mechanics unchanged
    - particles sampled deterministically
    - geometry is static annotation only
    """

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

    sampler.reset(
        state=env.get_state(),
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

        particles = sampler.step(dt=env.cfg.dt)

        renderer.begin_frame()

        # --- Static spatial context (Commit 5.3)
        draw_static_context(renderer)

        # --- Dynamic witnesses (Commits 5.1 / 5.2)
        renderer.draw_particles(particles)

        renderer.end_frame()

        step_count += 1

    renderer.close()
