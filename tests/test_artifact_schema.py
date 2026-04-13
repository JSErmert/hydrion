import json
import os
import tempfile
from pathlib import Path

import pytest

from hydrion.logging import artifacts


def test_manifest_includes_artifact_schema(tmp_path):
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    mani = {"run_id": "foo", "seed": 1}
    artifacts.write_manifest(run_dir, mani)
    data = json.loads((run_dir / "manifest.json").read_text())
    assert data["run_id"] == "foo"
    assert data["seed"] == 1
    assert data.get("artifact_schema") == "run_artifact_v1"


def test_metrics_stability_flag(tmp_path):
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()

    # create spine with two stable steps (pressure <=0.8, flow>=0.4, clog<=0.85, shield_intervened False)
    stable_step = {
        "step_idx": 0,
        "sim_time_s": 0.1,
        "truth": {
            "flow_norm": 0.5,
            "pressure_norm": 0.5,
            "clog_norm": 0.1,
            "E_field_norm": 0.0,   # obs12_v2 key (was E_norm in v1)
            "C_out": 0.0,
            "particle_capture_eff": 0.0,
        },
        "sensors": {"turbidity": 0.0, "scatter": 0.0},
        "actions": {"valve_cmd": 0, "pump_cmd": 0, "bf_cmd": 0, "node_voltage_cmd": 0},
        "reward": 0,
        "done": False,
        "truncated": False,
        "events": {"anomaly_active": False},
        "safety": {"shield_intervened": False, "violation": False, "violation_kind": None},
    }
    # write 50 stable steps to satisfy window
    for i in range(50):
        artifacts.append_spine_step(run_dir, {**stable_step, "step_idx": i})

    metrics = artifacts.compute_metrics(run_dir)
    assert metrics["stability_reached"] is True
    assert metrics["time_to_stability_steps"] == 0

    # invalid not reached
    run_dir2 = tmp_path / "run_test2"
    run_dir2.mkdir()
    # a single unstable step
    bad = dict(stable_step)
    bad["truth"]["pressure_norm"] = 1.0
    artifacts.append_spine_step(run_dir2, bad)
    metrics2 = artifacts.compute_metrics(run_dir2)
    assert metrics2["stability_reached"] is False
    assert metrics2["time_to_stability_steps"] is None


def test_append_spine_step_validates_keys(tmp_path):
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    payload = {"step_idx": 0}  # missing many keys
    with pytest.raises(ValueError) as exc:
        artifacts.append_spine_step(run_dir, payload)
    assert "missing required key" in str(exc.value)

    # test nested missing
    payload = {
        "step_idx": 0,
        "sim_time_s": 0,
        "truth": {},
        "sensors": {"turbidity": 0, "scatter": 0},
        "actions": {"valve_cmd": 0, "pump_cmd": 0, "bf_cmd": 0, "node_voltage_cmd": 0},
        "reward": 0,
        "done": False,
        "truncated": False,
        "events": {},
        "safety": {"shield_intervened": False, "violation": False, "violation_kind": None},
    }
    with pytest.raises(ValueError):
        artifacts.append_spine_step(run_dir, payload)
