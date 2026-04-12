# tests/test_particle_dynamics.py
import math
import pytest

# These imports will fail until particle_dynamics.py is created
from hydrion.physics.m5.particle_dynamics import (
    InputParticle, SimParticle, ParticleTrajectory,
    _fluid_velocity, _dep_radial_velocity, _gravity_radial_velocity,
)
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


@pytest.fixture
def stage_s1_fixture():
    return _default_stages()[0]
