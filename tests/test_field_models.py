"""
Test suite for M5 field models (analytical_conical_field, fem_field_from_table).

These tests validate the callable interface and physics-grounded field computation.
"""
import pytest
from hydrion.environments.conical_cascade_env import _default_stages
from hydrion.physics.m5.field_models import analytical_conical_field


@pytest.fixture
def stage_s1():
    return _default_stages()[0]   # D_in=80mm, D_tip=20mm, L=120mm


def test_import():
    """Step 1: Verify analytical_conical_field can be imported."""
    assert callable(analytical_conical_field)


def test_apex_on_axis_equals_grad_E2_apex(stage_s1):
    """
    At x_norm=1.0 (apex), r_norm=0.0 (axis):
        R(1.0) = R_tip  → concentration = (R_tip/R_tip)^4 = 1.0
        wall_enh(0.0)   = 1 + beta_r*0 = 1.0
        → grad_E2 = grad_E2_apex exactly.
    """
    field_fn = analytical_conical_field(stage_s1)
    result = field_fn(1.0, 0.0)
    assert result == pytest.approx(stage_s1.dep.grad_E2, rel=1e-6)


def test_increases_toward_apex(stage_s1):
    """Concentration factor must increase monotonically toward apex at r_norm=0."""
    field_fn = analytical_conical_field(stage_s1)
    v01 = field_fn(0.1, 0.0)
    v05 = field_fn(0.5, 0.0)
    v09 = field_fn(0.9, 0.0)
    assert v01 < v05 < v09, (
        f"grad_E2 must increase toward apex. Got {v01:.3e} < {v05:.3e} < {v09:.3e}"
    )


def test_increases_toward_wall(stage_s1):
    """Wall enhancement must increase monotonically toward wall at fixed x_norm."""
    field_fn = analytical_conical_field(stage_s1)
    v02 = field_fn(0.5, 0.2)
    v05 = field_fn(0.5, 0.5)
    v09 = field_fn(0.5, 0.9)
    assert v02 < v05 < v09, (
        f"grad_E2 must increase toward wall. Got {v02:.3e} < {v05:.3e} < {v09:.3e}"
    )


def test_n_field_conc_parameter(stage_s1):
    """Higher n_field_conc reduces concentration away from apex (concentration scales as (R_tip/R)^n)."""
    fn4 = analytical_conical_field(stage_s1, n_field_conc=4)
    fn6 = analytical_conical_field(stage_s1, n_field_conc=6)
    # At x=0.5, R_tip/R(0.5) = 0.4 < 1, so (0.4)^6 < (0.4)^4
    assert fn6(0.5, 0.0) < fn4(0.5, 0.0)


def test_beta_r_parameter(stage_s1):
    """Higher beta_r increases wall enhancement."""
    fn_low  = analytical_conical_field(stage_s1, beta_r=0.5)
    fn_high = analytical_conical_field(stage_s1, beta_r=3.0)
    assert fn_high(0.5, 0.8) > fn_low(0.5, 0.8)


def test_returns_positive_values(stage_s1):
    """grad_E2 must be positive everywhere."""
    field_fn = analytical_conical_field(stage_s1)
    for x in [0.0, 0.3, 0.7, 1.0]:
        for r in [0.0, 0.3, 0.7, 1.0]:
            assert field_fn(x, r) > 0, f"grad_E2 must be positive at ({x}, {r})"


def test_fem_field_from_table_interface():
    """fem_field_from_table returns a callable with the same interface.
    Requires scipy — skipped if not available."""
    scipy = pytest.importorskip("scipy")
    import numpy as np
    from hydrion.physics.m5.field_models import fem_field_from_table
    x_edges = np.array([0.0, 0.5, 1.0])
    r_edges = np.array([0.0, 0.5, 1.0])
    table = np.ones((3, 3)) * 1e12   # uniform field
    field_fn = fem_field_from_table(table, x_edges, r_edges)
    result = field_fn(0.25, 0.25)
    assert isinstance(result, float)
    assert result == pytest.approx(1e12, rel=1e-3)
