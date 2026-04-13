# tests/test_particles.py
# M4 — density classification, Stokes settling, per-stage capture, eta_nominal
import numpy as np

from hydrion.physics.particles import (
    ParticleModel,
    _compute_bin_weights,
    stokes_velocity_ms,
    _capture_eff_s1,
    _capture_eff_s2,
    _capture_eff_s3,
    _eta_system,
)


# ---------------------------------------------------------------------------
# Regression: existing API contract preserved
# ---------------------------------------------------------------------------

def test_psd_disabled_regression():
    """PSD disabled: C_in, C_out, particle_capture_eff, C_fibers all present."""
    part = ParticleModel(cfg=None)
    state = {"mesh_loading_avg": 0.2, "capture_eff": 0.8}
    part.reset(state)
    part.update(state, dt=0.1, clogging_model=None, electrostatics_model=None)

    assert "C_in" in state
    assert "C_out" in state
    assert "particle_capture_eff" in state
    assert "C_fibers" in state
    assert "C_in_bin_0" not in state
    assert "C_L" not in state

    assert state["C_out"] <= state["C_in"] + 1e-9
    assert 0.0 <= state["particle_capture_eff"] <= 1.0 + 1e-9


def test_psd_enabled_bins_sum_to_one():
    """PSD enabled: bin weights sum to 1."""
    weights = _compute_bin_weights(
        mode="bins",
        parametric={},
        bins=[
            {"d_min_um": 0.1,  "d_max_um": 1.0,   "w_in": 0.2},
            {"d_min_um": 1.0,  "d_max_um": 10.0,  "w_in": 0.5},
            {"d_min_um": 10.0, "d_max_um": 100.0, "w_in": 0.3},
        ],
        bin_edges_um=[],
    )
    assert len(weights) == 3
    assert abs(sum(weights) - 1.0) < 1e-9

    weights2 = _compute_bin_weights(
        mode="parametric",
        parametric={"distribution": "lognormal", "mean_um": 5.0, "std_um": 2.0},
        bins=[],
        bin_edges_um=[0.1, 1.0, 10.0, 100.0],
    )
    assert len(weights2) == 3
    assert abs(sum(weights2) - 1.0) < 1e-9


def test_psd_enabled_mass_balance():
    """PSD enabled: per-bin C_out <= C_in."""
    cfg_raw = {
        "particles": {
            "C_in_base": 0.7,
            "psd": {"enabled": True, "mode": "bins", "bins": [
                {"d_min_um": 0.1,  "d_max_um": 1.0,   "w_in": 0.33},
                {"d_min_um": 1.0,  "d_max_um": 10.0,  "w_in": 0.34},
                {"d_min_um": 10.0, "d_max_um": 100.0, "w_in": 0.33},
            ]},
            "shape": {"enabled": True, "fiber_fraction": 0.5},
        }
    }
    from hydrion.config import HydrionConfig
    cfg = HydrionConfig(raw=cfg_raw)
    part = ParticleModel(cfg=cfg)
    state = {"mesh_loading_avg": 0.3, "capture_eff": 0.8}
    part.reset(state)
    part.update(state, dt=0.1, clogging_model=None, electrostatics_model=None)

    assert state["C_out"] <= state["C_in"] + 1e-9
    assert 0.0 <= state["particle_capture_eff"] <= 1.0 + 1e-9
    assert "C_fibers" in state
    for i in range(3):
        assert state[f"C_out_bin_{i}"] <= state[f"C_in_bin_{i}"] + 1e-9


# ---------------------------------------------------------------------------
# M4 Gate 1: Stokes settling physics
# ---------------------------------------------------------------------------

def test_stokes_dense_sinks():
    """Dense particles (PET, rho=1380) have positive settling velocity."""
    v = stokes_velocity_ms(1380.0, 10e-6)
    assert v > 0, f"PET must sink: v_s={v}"


def test_stokes_buoyant_rises():
    """Buoyant particles (PP, rho=910) have negative settling velocity."""
    v = stokes_velocity_ms(910.0, 10e-6)
    assert v < 0, f"PP must rise: v_s={v}"


def test_stokes_neutral_near_zero():
    """Neutral particles (rho=1000) have near-zero settling velocity."""
    v = stokes_velocity_ms(1000.0, 10e-6)
    assert abs(v) < 1e-12, f"Neutral must have ~0 settling: v_s={v}"


# ---------------------------------------------------------------------------
# M4 Gate 2: Per-stage capture physics
# ---------------------------------------------------------------------------

def test_s1_passes_fine_particles():
    """Stage 1 (500 um mesh) captures < 5% of 10 um particles at clean state."""
    eta = _capture_eff_s1(10.0, 0.0, 0.15)
    assert eta < 0.05, f"S1 should pass 10um particles: eta_s1={eta}"


def test_s2_moderate_capture():
    """Stage 2 (100 um mesh) captures < 20% of 10 um particles at clean state."""
    eta = _capture_eff_s2(10.0, 0.0, 0.20)
    assert eta < 0.20, f"S2 at 10um: {eta}"


def test_s3_primary_capture():
    """Stage 3 (5 um mesh) captures > 50% of 10 um at low flow, clean state."""
    eta = _capture_eff_s3(10.0, 0.0, 0.10, 5.0, 0.04, 10.0)
    assert eta > 0.50, f"S3 at Q=5, clean, 10um: {eta}"


def test_s3_flow_degradation():
    """Stage 3 capture efficiency decreases at Q=20 vs Q=5."""
    eta_low  = _capture_eff_s3(10.0, 0.0, 0.10, 5.0,  0.04, 10.0)
    eta_high = _capture_eff_s3(10.0, 0.0, 0.10, 20.0, 0.04, 10.0)
    assert eta_low > eta_high, f"Flow degradation absent: Q5={eta_low}, Q20={eta_high}"


def test_s3_fouling_improves_capture():
    """Fouling slightly improves S3 capture (pore restriction effect)."""
    eta_clean  = _capture_eff_s3(10.0, 0.0, 0.10, 13.5, 0.04, 10.0)
    eta_fouled = _capture_eff_s3(10.0, 0.5, 0.10, 13.5, 0.04, 10.0)
    assert eta_fouled >= eta_clean, \
        f"Fouling should improve S3: clean={eta_clean}, fouled={eta_fouled}"


def test_eta_system_compound():
    """Compound efficiency exceeds single-stage maximum."""
    eta_s1 = _capture_eff_s1(10.0, 0.0, 0.15)
    eta_s2 = _capture_eff_s2(10.0, 0.0, 0.20)
    eta_s3 = _capture_eff_s3(10.0, 0.0, 0.10, 13.5, 0.04, 10.0)
    eta_sys = _eta_system(eta_s1, eta_s2, eta_s3)
    assert eta_sys > max(eta_s1, eta_s2, eta_s3), \
        f"Compound must exceed single-stage: sys={eta_sys}, max={max(eta_s1, eta_s2, eta_s3)}"


# ---------------------------------------------------------------------------
# M4 Gate 3: Density classification in full update()
# ---------------------------------------------------------------------------

def test_density_fractions_sum_to_one():
    """C_in_dense + C_in_neutral + C_in_buoyant = C_in."""
    part = ParticleModel(cfg=None)
    state: dict = {}
    part.reset(state)
    part.update(state, dt=0.1)
    total = state["C_in_dense"] + state["C_in_neutral"] + state["C_in_buoyant"]
    assert abs(total - state["C_in"]) < 1e-9, \
        f"Density fractions do not sum to C_in: {total} vs {state['C_in']}"


def test_buoyant_pass_through():
    """C_in_buoyant exits system uncaptured — C_out >= C_in_buoyant."""
    part = ParticleModel(cfg=None)
    state: dict = {"C_in": 0.7, "q_processed_lmin": 13.5}
    part.reset(state)
    part.update(state, dt=0.1)
    assert state["C_out"] >= state["C_in_buoyant"] - 1e-9, \
        f"Buoyant pass-through missing: C_out={state['C_out']}, C_in_buoyant={state['C_in_buoyant']}"


def test_m4_truth_state_keys_present():
    """All M4 truth_state keys written after update."""
    part = ParticleModel(cfg=None)
    state: dict = {}
    part.reset(state)
    part.update(state, dt=0.1)

    required = [
        "C_in_dense", "C_in_neutral", "C_in_buoyant", "buoyant_fraction",
        "capture_eff_s1", "capture_eff_s2", "capture_eff_s3",
        "capture_boost_settling", "eta_system", "eta_nominal",
    ]
    for key in required:
        assert key in state, f"Missing M4 truth_state key: {key}"


# ---------------------------------------------------------------------------
# M4 Gate 4: eta_nominal determinism and physical range
# ---------------------------------------------------------------------------

def test_eta_nominal_deterministic():
    """eta_nominal is identical across two separate update() calls."""
    part = ParticleModel(cfg=None)
    state1: dict = {}
    part.reset(state1)
    part.update(state1, dt=0.1)

    state2: dict = {}
    part.reset(state2)
    part.update(state2, dt=0.1)

    assert state1["eta_nominal"] == state2["eta_nominal"], \
        f"eta_nominal not deterministic: {state1['eta_nominal']} vs {state2['eta_nominal']}"


def test_eta_nominal_range():
    """eta_nominal is in [0.4, 0.98] — physically plausible for 10 um at 13.5 L/min."""
    part = ParticleModel(cfg=None)
    state: dict = {}
    part.reset(state)
    part.update(state, dt=0.1)
    assert 0.4 <= state["eta_nominal"] <= 0.98, \
        f"eta_nominal out of expected range: {state['eta_nominal']}"


def test_eta_nominal_independent_of_fouling():
    """eta_nominal uses clean-filter reference — runtime fouling does not affect it."""
    part = ParticleModel(cfg=None)

    state_clean: dict = {"fouling_frac_s1": 0.0, "fouling_frac_s2": 0.0, "fouling_frac_s3": 0.0}
    part.reset(state_clean)
    part.update(state_clean, dt=0.1)

    state_fouled: dict = {"fouling_frac_s1": 0.5, "fouling_frac_s2": 0.5, "fouling_frac_s3": 0.5}
    part.reset(state_fouled)
    part.update(state_fouled, dt=0.1)

    assert abs(state_clean["eta_nominal"] - state_fouled["eta_nominal"]) < 1e-9, \
        f"eta_nominal should not change with fouling: clean={state_clean['eta_nominal']}, fouled={state_fouled['eta_nominal']}"


# ---------------------------------------------------------------------------
# M4 Gate 5: mass balance
# ---------------------------------------------------------------------------

def test_mass_balance_total():
    """C_out <= C_in at all operating conditions."""
    part = ParticleModel(cfg=None)
    for Q in [5.0, 13.5, 20.0]:
        for ff in [0.0, 0.3, 0.7]:
            state: dict = {
                "q_processed_lmin": Q,
                "fouling_frac_s1": ff,
                "fouling_frac_s2": ff,
                "fouling_frac_s3": ff,
            }
            part.reset(state)
            part.update(state, dt=0.1)
            assert state["C_out"] <= state["C_in"] + 1e-9, \
                f"Mass balance violated at Q={Q}, ff={ff}: C_out={state['C_out']:.4f} > C_in={state['C_in']:.4f}"


if __name__ == "__main__":
    test_m4_truth_state_keys_present()
    test_eta_nominal_range()
    print("M4 particle tests: OK")
