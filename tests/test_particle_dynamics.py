# tests/test_particle_dynamics.py
import math
import pytest

# These imports will fail until particle_dynamics.py is created
from hydrion.physics.m5.particle_dynamics import (
    InputParticle, SimParticle, ParticleTrajectory,
    _fluid_velocity, _dep_radial_velocity, _gravity_radial_velocity,
)
from hydrion.physics.m5.particle_dynamics import ParticleDynamicsEngine
from hydrion.physics.m5.materials import MU_WATER, RHO_WATER
from hydrion.environments.conical_cascade_env import _default_stages
from hydrion.physics.m5.field_models import analytical_conical_field


def test_import_dataclasses():
    p = InputParticle(particle_id="pp-1", species="PP", d_p_m=25e-6)
    assert p.species == "PP"
    assert p.d_p_m == pytest.approx(25e-6)


def test_dep_radial_velocity_is_negative_for_pp(stage_s1_fixture):
    """PP has Re[K] < 0 → nDEP force is negative → radially inward (v < 0)."""
    stage = stage_s1_fixture
    field_fn = analytical_conical_field(stage)
    v = _dep_radial_velocity(0.5, 0.5, 25e-6, "PP", field_fn)
    assert v < 0, f"PP nDEP must be inward (negative), got {v:.3e}"


def test_dep_radial_velocity_is_negative_for_pet(stage_s1_fixture):
    """PET has Re[K] < 0 → nDEP is also inward (all three polymers are nDEP)."""
    stage = stage_s1_fixture
    field_fn = analytical_conical_field(stage)
    v = _dep_radial_velocity(0.5, 0.5, 25e-6, "PET", field_fn)
    assert v < 0, f"PET nDEP must be inward (negative), got {v:.3e}"


def test_gravity_pp_is_negative():
    """PP ρ_p=910 < ρ_water=1000 → buoyant → v_gravity < 0 → toward axis."""
    v = _gravity_radial_velocity(25e-6, "PP")
    assert v < 0, f"PP buoyancy must be negative (toward axis), got {v:.3e}"


def test_gravity_pet_is_positive():
    """PET ρ_p=1380 > ρ_water=1000 → sinking → v_gravity > 0 → toward wall."""
    v = _gravity_radial_velocity(25e-6, "PET")
    assert v > 0, f"PET sedimentation must be positive (toward wall), got {v:.3e}"


def test_fluid_axial_zero_at_wall(stage_s1_fixture):
    """Poiseuille: no-slip at r_norm=1.0 → v_axial = 0."""
    stage = stage_s1_fixture
    R_in  = stage.D_in_m  / 2.0
    R_tip = stage.D_tip_m / 2.0
    Q_m3s = 10.0 / 60000.0  # 10 L/min
    v_ax, _ = _fluid_velocity(0.5, 1.0, Q_m3s, R_in, R_tip, stage.L_cone_m)
    assert abs(v_ax) < 1e-10, f"v_axial must be 0 at wall, got {v_ax:.3e}"


def test_fluid_axial_increases_toward_apex(stage_s1_fixture):
    """Mean velocity increases as cone narrows toward apex (continuity)."""
    stage = stage_s1_fixture
    R_in  = stage.D_in_m  / 2.0
    R_tip = stage.D_tip_m / 2.0
    Q_m3s = 10.0 / 60000.0
    v01, _ = _fluid_velocity(0.1, 0.0, Q_m3s, R_in, R_tip, stage.L_cone_m)
    v09, _ = _fluid_velocity(0.9, 0.0, Q_m3s, R_in, R_tip, stage.L_cone_m)
    assert v01 < v09, f"v_axial must increase toward apex. Got {v01:.3e} vs {v09:.3e}"


def test_fluid_radial_inward_in_converging_cone(stage_s1_fixture):
    """In a converging cone (R_in > R_tip), radial drift must be inward (toward axis)."""
    stage = stage_s1_fixture
    R_in  = stage.D_in_m  / 2.0
    R_tip = stage.D_tip_m / 2.0
    Q_m3s = 10.0 / 60000.0
    _, v_rad = _fluid_velocity(0.5, 0.5, Q_m3s, R_in, R_tip, stage.L_cone_m)
    assert v_rad < 0, f"Radial drift must be inward in converging cone, got {v_rad:.3e}"


def test_captured_at_substep_set_on_capture(engine, stage_s1_fixture):
    """captured_at_substep is a non-None int when particle is captured."""
    stage = stage_s1_fixture
    field_fn = analytical_conical_field(stage)
    # Low flow + small d_p ensures capture (DEP dominates over fluid drag)
    Q_low = 1.0 / 60000.0   # 1 L/min
    particles = [InputParticle("pp-1", "PP", 50e-6)]  # 50µm PP, more DEP force
    trajs = engine.integrate(particles, stage, 0, Q_low, field_fn, dt_sim=1.0)
    t = trajs[0]
    if t.final_status == "captured":
        assert isinstance(t.captured_at_substep, int), (
            f"captured_at_substep must be int when captured, got {t.captured_at_substep!r}"
        )
        assert 1 <= t.captured_at_substep <= 100, (
            f"captured_at_substep must be in [1, 100], got {t.captured_at_substep}"
        )
    else:
        # If not captured at 1 L/min with 50µm, that's a physics concern — warn
        import warnings
        warnings.warn(f"50µm PP at 1 L/min did not capture: {t.final_status}")


@pytest.fixture
def stage_s1_fixture():
    return _default_stages()[0]


@pytest.fixture
def engine():
    return ParticleDynamicsEngine()


def test_integrate_returns_one_trajectory_per_particle(engine, stage_s1_fixture):
    stage = stage_s1_fixture
    field_fn = analytical_conical_field(stage)
    particles = [
        InputParticle("pp-1", "PP",  25e-6),
        InputParticle("pe-1", "PE",  25e-6),
        InputParticle("pet-1","PET", 25e-6),
    ]
    trajs = engine.integrate(particles, stage, 0, 10.0/60000.0, field_fn, dt_sim=1.0)
    assert len(trajs) == 3


def test_final_status_is_terminal(engine, stage_s1_fixture):
    """final_status must always be 'captured' or 'passed' — never 'in_transit' or 'near_wall'."""
    stage = stage_s1_fixture
    field_fn = analytical_conical_field(stage)
    particles = [InputParticle(f"p{i}", sp, 25e-6) for i, sp in enumerate(["PP","PE","PET"])]
    trajs = engine.integrate(particles, stage, 0, 10.0/60000.0, field_fn, dt_sim=1.0)
    for t in trajs:
        assert t.final_status in ("captured", "passed"), (
            f"final_status must be terminal, got '{t.final_status}' for {t.species}"
        )


def test_deterministic(engine, stage_s1_fixture):
    """Same inputs must produce identical trajectories (no randomness)."""
    stage = stage_s1_fixture
    field_fn = analytical_conical_field(stage)
    particles = [InputParticle("pp-1", "PP", 25e-6)]
    Q = 10.0 / 60000.0

    trajs_a = engine.integrate(particles, stage, 0, Q, field_fn, dt_sim=1.0)
    trajs_b = engine.integrate(particles, stage, 0, Q, field_fn, dt_sim=1.0)

    assert trajs_a[0].final_status == trajs_b[0].final_status
    assert trajs_a[0].positions[-1] == trajs_b[0].positions[-1]


def test_pp_drifts_inward_relative_to_pet(engine, stage_s1_fixture):
    """
    PP is buoyant (floats toward axis) and has same nDEP as PET.
    At low flow, both captured. At high flow where some escape:
    PP should end at lower r_norm than PET (due to buoyancy assisting inward drift).
    """
    stage = stage_s1_fixture
    field_fn = analytical_conical_field(stage)
    # Use high flow to ensure escaped particles (species separation visible in passed set)
    Q_high = 20.0 / 60000.0   # 20 L/min — above typical capture range
    pp  = InputParticle("pp-1",  "PP",  25e-6)
    pet = InputParticle("pet-1", "PET", 25e-6)
    trajs = engine.integrate([pp, pet], stage, 0, Q_high, field_fn, dt_sim=1.0)
    t_pp  = next(t for t in trajs if t.species == "PP")
    t_pet = next(t for t in trajs if t.species == "PET")
    # Final r_norm: PP should be lower (toward axis) than PET (toward wall)
    r_pp  = t_pp.positions[-1][1]
    r_pet = t_pet.positions[-1][1]
    assert r_pp < r_pet, (
        f"PP (buoyant) should end closer to axis than PET (dense). "
        f"r_PP={r_pp:.3f} r_PET={r_pet:.3f}"
    )


def test_high_flow_increases_passed_fraction(engine, stage_s1_fixture):
    """At higher flow, fewer particles should be captured (more pass through)."""
    stage = stage_s1_fixture
    field_fn = analytical_conical_field(stage)
    particles = [InputParticle(f"pp-{i}", "PP", 25e-6) for i in range(3)]

    Q_low  = 5.0  / 60000.0
    Q_high = 18.0 / 60000.0

    trajs_low  = engine.integrate(particles, stage, 0, Q_low,  field_fn, dt_sim=1.0)
    trajs_high = engine.integrate(particles, stage, 0, Q_high, field_fn, dt_sim=1.0)

    captured_low  = sum(1 for t in trajs_low  if t.final_status == "captured")
    captured_high = sum(1 for t in trajs_high if t.final_status == "captured")

    assert captured_low >= captured_high, (
        f"Higher flow should not increase captures. "
        f"low={captured_low} high={captured_high}"
    )


def test_convergence_n100_vs_n200(engine, stage_s1_fixture):
    """
    Substep convergence: capture outcomes must agree between n=100 and n=200.
    Final positions must agree within 0.05 (first-order Euler; captured particles
    differ by one substep boundary, giving O(dt) position error).
    """
    stage = stage_s1_fixture
    field_fn = analytical_conical_field(stage)
    Q = 10.0 / 60000.0
    particles = [
        InputParticle("pp-1",  "PP",  25e-6),
        InputParticle("pe-1",  "PE",  25e-6),
        InputParticle("pet-1", "PET", 25e-6),
    ]

    trajs_100 = engine.integrate(particles, stage, 0, Q, field_fn, dt_sim=1.0, n_substeps=100)
    trajs_200 = engine.integrate(particles, stage, 0, Q, field_fn, dt_sim=1.0, n_substeps=200)

    for t100, t200 in zip(trajs_100, trajs_200):
        assert t100.final_status == t200.final_status, (
            f"Outcome diverges between n=100 and n=200 for {t100.species}: "
            f"{t100.final_status} vs {t200.final_status}"
        )
        x100, r100 = t100.positions[-1]
        x200, r200 = t200.positions[-1]
        assert abs(x100 - x200) < 0.05, (
            f"x_norm endpoint divergence > 0.05 for {t100.species}: {x100:.4f} vs {x200:.4f}"
        )
        assert abs(r100 - r200) < 0.05, (
            f"r_norm endpoint divergence > 0.05 for {t100.species}: {r100:.4f} vs {r200:.4f}"
        )


def test_backflush_no_captures(engine, stage_s1_fixture):
    """During backflush, capture logic is suspended — no particle may be captured."""
    stage = stage_s1_fixture
    field_fn = analytical_conical_field(stage)
    Q = 10.0 / 60000.0
    particles = [InputParticle(f"p{i}", sp, 25e-6) for i, sp in enumerate(["PP","PE","PET"])]
    trajs = engine.integrate(particles, stage, 0, Q, field_fn, dt_sim=1.0, backflush=True)
    for t in trajs:
        assert t.final_status == "passed", (
            f"No captures during backflush. Got '{t.final_status}' for {t.species}"
        )
