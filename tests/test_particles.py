# tests/test_particles.py
import numpy as np

from hydrion.physics.particles import ParticleModel


def test_particles_basic():
    part = ParticleModel(cfg=None)
    state = {
        "mesh_loading_avg": 0.2,
        "capture_eff": 0.8,
        "E_norm": 0.0,
    }

    part.reset(state)

    print("\n--- ParticleModel Test ---")
    dt = 0.1

    for i in range(10):
        # slowly increase clogging & E_norm
        state["mesh_loading_avg"] = min(1.0, state["mesh_loading_avg"] + 0.05)
        state["E_norm"] = min(1.0, state["E_norm"] + 0.05)

        part.update(state, dt=dt, clogging_model=None, electrostatics_model=None)
        print(
            f"Step {i:02d}: C_in={state['C_in']:.3f}, "
            f"C_out={state['C_out']:.3f}, "
            f"capture_eff={state['particle_capture_eff']:.3f}"
        )


if __name__ == "__main__":
    test_particles_basic()
