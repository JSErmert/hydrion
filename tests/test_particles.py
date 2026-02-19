# tests/test_particles.py
import numpy as np

from hydrion.physics.particles import ParticleModel, _compute_bin_weights


def test_psd_disabled_regression():
    """PSD disabled: outputs shape and keys match previous behavior."""
    part = ParticleModel(cfg=None)
    state = {
        "mesh_loading_avg": 0.2,
        "capture_eff": 0.8,
        "E_norm": 0.0,
    }
    part.reset(state)
    part.update(state, dt=0.1, clogging_model=None, electrostatics_model=None)

    # Core keys unchanged
    assert "C_in" in state
    assert "C_out" in state
    assert "particle_capture_eff" in state
    assert "C_fibers" in state
    # No PSD-specific keys when disabled
    assert "C_in_bin_0" not in state
    assert "C_out_bin_0" not in state
    assert "C_L" not in state

    # Mass balance
    assert state["C_out"] <= state["C_in"] + 1e-9
    assert 0.0 <= state["particle_capture_eff"] <= 1.0 + 1e-9


def test_psd_enabled_bins_sum_to_one():
    """PSD enabled: bin weights sum to 1."""
    # bins mode with explicit weights
    weights = _compute_bin_weights(
        mode="bins",
        parametric={},
        bins=[
            {"d_min_um": 0.1, "d_max_um": 1.0, "w_in": 0.2},
            {"d_min_um": 1.0, "d_max_um": 10.0, "w_in": 0.5},
            {"d_min_um": 10.0, "d_max_um": 100.0, "w_in": 0.3},
        ],
        bin_edges_um=[],
    )
    assert len(weights) == 3
    assert abs(sum(weights) - 1.0) < 1e-9

    # parametric mode
    weights2 = _compute_bin_weights(
        mode="parametric",
        parametric={"distribution": "lognormal", "mean_um": 5.0, "std_um": 2.0},
        bins=[],
        bin_edges_um=[0.1, 1.0, 10.0, 100.0],
    )
    assert len(weights2) == 3
    assert abs(sum(weights2) - 1.0) < 1e-9


def test_psd_enabled_mass_balance():
    """PSD enabled: C_out <= C_in, capture_eff bounded."""
    cfg_raw = {
        "particles": {
            "C_in_base": 0.7,
            "psd": {"enabled": True, "mode": "bins", "bins": [
                {"d_min_um": 0.1, "d_max_um": 1.0, "w_in": 0.33},
                {"d_min_um": 1.0, "d_max_um": 10.0, "w_in": 0.34},
                {"d_min_um": 10.0, "d_max_um": 100.0, "w_in": 0.33},
            ]},
            "shape": {"enabled": True, "fiber_fraction": 0.5},
        }
    }
    from hydrion.config import HydrionConfig
    cfg = HydrionConfig(raw=cfg_raw)
    part = ParticleModel(cfg=cfg)

    state = {"mesh_loading_avg": 0.3, "capture_eff": 0.8, "E_norm": 0.2}
    part.reset(state)
    part.update(state, dt=0.1, clogging_model=None, electrostatics_model=None)

    assert state["C_out"] <= state["C_in"] + 1e-9
    assert 0.0 <= state["particle_capture_eff"] <= 1.0 + 1e-9
    assert "C_fibers" in state
    assert "C_in_bin_0" in state
    assert "C_out_bin_0" in state
    # Per-bin mass balance
    for i in range(3):
        assert state[f"C_out_bin_{i}"] <= state[f"C_in_bin_{i}"] + 1e-9


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
