"""
hydrion/visualize_episode.py

Truthful episode visualization.
Pure observer: no mechanics, no reward, no mutation.
"""

from hydrion.visual_sampling.particle_sampler import ParticleSampler


def visualize_episode(
    env,
    renderer,
    seed: int,
    max_steps: int = 1000,
):
    """
    Run a single episode with deterministic visual inspection.

    Visualization is a pure witness:
    - Reads mechanical state
    - Samples visual particles
    - Renders frames
    - Never influences the environment
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
        # --- Advance environment (policy or noop action)
        action = env.action_space.sample()  # or policy(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Advance visual witnesses
        particles = sampler.step(dt=env.cfg.dt)

        # --- Render (pure observer)
        renderer.begin_frame()
        renderer.draw_particles(particles)
        renderer.end_frame()

        step_count += 1

    # ----------------------------
    # Finalize visualization
    # ----------------------------
    renderer.close()
