# hydrion/logging/paths.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    """
    Standardized filesystem layout for a single run.
    """
    root: Path
    simulation_run_json: Path
    sim_config_json: Path
    timestep_jsonl: Path

    @staticmethod
    def for_run(base_dir: str | Path, run_id: str) -> "RunPaths":
        base = Path(base_dir)
        root = base / run_id
        return RunPaths(
            root=root,
            simulation_run_json=root / "simulation_run.json",
            sim_config_json=root / "sim_config.json",
            timestep_jsonl=root / "timestep.jsonl",
        )
