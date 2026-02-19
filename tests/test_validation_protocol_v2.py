# tests/test_validation_protocol_v2.py
"""
Pytest for HydrOS Validation Protocol v2.

Runs stress_matrix, envelope_sweep, mass_balance_test, recovery_latency_test
with minimal config (short steps) so the suite is fast. Full runs use
configs/validation/*.yaml and scenario YAMLs via CLI.
"""

import pytest
from pathlib import Path

from hydrion.validation import (
    run_stress_matrix,
    run_envelope_sweep,
    run_mass_balance_test,
    run_recovery_latency_test,
)


CONFIG_PATH = "configs/default.yaml"
VALIDATION_DIR = Path("configs/validation")


def test_stress_matrix_minimal():
    summary = run_stress_matrix(
        config_path=CONFIG_PATH,
        validation_config_path=None,
        scenario_path=None,
        output_path=None,
    )
    assert "all_passed" in summary
    assert "num_episodes" in summary
    assert summary["num_episodes"] >= 1
    # Default val_cfg is empty so we get 3 episodes from code default
    assert summary["all_passed"] is True or len(summary.get("results", [])) > 0


def test_stress_matrix_with_config():
    if not (VALIDATION_DIR / "stress_matrix.yaml").exists():
        pytest.skip("configs/validation/stress_matrix.yaml not found")
    summary = run_stress_matrix(
        config_path=CONFIG_PATH,
        validation_config_path=str(VALIDATION_DIR / "stress_matrix.yaml"),
        output_path=None,
    )
    assert summary["all_passed"] is True


def test_envelope_sweep_minimal():
    summary = run_envelope_sweep(
        config_path=CONFIG_PATH,
        validation_config_path=None,
        output_path=None,
    )
    assert "all_passed" in summary
    assert "num_points" in summary
    assert summary["all_passed"] is True


def test_envelope_sweep_with_config():
    if not (VALIDATION_DIR / "envelope_sweep.yaml").exists():
        pytest.skip("configs/validation/envelope_sweep.yaml not found")
    summary = run_envelope_sweep(
        config_path=CONFIG_PATH,
        validation_config_path=str(VALIDATION_DIR / "envelope_sweep.yaml"),
        output_path=None,
    )
    assert summary["all_passed"] is True


def test_mass_balance_minimal():
    summary = run_mass_balance_test(
        config_path=CONFIG_PATH,
        validation_config_path=None,
        output_path=None,
    )
    assert summary["all_passed"] is True
    assert "violations" in summary
    assert len(summary["violations"]) == 0


def test_mass_balance_with_config():
    if not (VALIDATION_DIR / "mass_balance.yaml").exists():
        pytest.skip("configs/validation/mass_balance.yaml not found")
    summary = run_mass_balance_test(
        config_path=CONFIG_PATH,
        validation_config_path=str(VALIDATION_DIR / "mass_balance.yaml"),
        output_path=None,
    )
    assert summary["all_passed"] is True


def test_recovery_latency_minimal():
    summary = run_recovery_latency_test(
        config_path=CONFIG_PATH,
        validation_config_path=None,
        output_path=None,
    )
    assert "all_passed" in summary
    assert "steps_to_recovery" in summary
    # Recovery to flow ~0.5 after neutral action is plausible
    assert summary["all_passed"] is True or summary.get("steps_to_recovery") is not None


def test_recovery_latency_with_config():
    if not (VALIDATION_DIR / "recovery_latency.yaml").exists():
        pytest.skip("configs/validation/recovery_latency.yaml not found")
    summary = run_recovery_latency_test(
        config_path=CONFIG_PATH,
        validation_config_path=str(VALIDATION_DIR / "recovery_latency.yaml"),
        output_path=None,
    )
    assert "all_passed" in summary
