# tests/test_logging_smoke.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hydrion.env import HydrionEnv


def test_run_folder_and_files_created(tmp_path: Path):
    # Use temp directory so repo doesn't get polluted in test
    env = HydrionEnv()
    # Override logger base dir to tmp
    env._log_base_dir = tmp_path
    env.logger.base_dir = tmp_path

    obs, _ = env.reset(seed=42)
    action = np.array([0.6, 0.7, 0.0, 0.5], dtype=np.float32)

    # Step a few times
    for _ in range(5):
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            break

    run_dir = tmp_path / env.run_context.run_id
    assert run_dir.exists() and run_dir.is_dir()

    sim_run = run_dir / "simulation_run.json"
    sim_cfg = run_dir / "sim_config.json"
    tlog = run_dir / "timestep.jsonl"

    assert sim_run.exists()
    assert sim_cfg.exists()
    assert tlog.exists()

    # Validate JSON headers parse
    run_header = json.loads(sim_run.read_text(encoding="utf-8"))
    assert run_header["run_id"] == env.run_context.run_id

    cfg = json.loads(sim_cfg.read_text(encoding="utf-8"))
    assert isinstance(cfg, dict)

    # Ensure timestep log has at least reset + steps
    lines = tlog.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
