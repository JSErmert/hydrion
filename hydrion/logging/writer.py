# hydrion/logging/writer.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .paths import RunPaths


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json_dump(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":"), ensure_ascii=True))
        f.write("\n")


@dataclass
class RunLogger:
    """
    Minimal research-grade logger (Commit 2):
    - creates per-run folder
    - writes simulation_run.json (identity + timestamps)
    - writes sim_config.json (full config snapshot)
    - appends timestep.jsonl per step (spine)

    No prints. Never throws during training unless strict=True.
    """
    base_dir: Path
    enabled: bool = True
    strict: bool = False

    _paths: Optional[RunPaths] = None
    _run_id: Optional[str] = None
    _is_open: bool = False

    def start_run(self, *, run_id: str, run_header: Dict[str, Any], config: Dict[str, Any]) -> RunPaths:
        if not self.enabled:
            self._run_id = run_id
            self._paths = RunPaths.for_run(self.base_dir, run_id)
            self._is_open = True
            return self._paths

        try:
            self._run_id = run_id
            self._paths = RunPaths.for_run(self.base_dir, run_id)
            self._paths.root.mkdir(parents=True, exist_ok=True)

            header = dict(run_header)
            header.setdefault("run_id", run_id)
            header.setdefault("created_utc", _utc_iso())
            _safe_json_dump(self._paths.simulation_run_json, header)

            cfg = dict(config)
            cfg.setdefault("_logged_utc", _utc_iso())
            _safe_json_dump(self._paths.sim_config_json, cfg)

            # Ensure timestep file exists (touch)
            self._paths.timestep_jsonl.touch(exist_ok=True)

            self._is_open = True
            return self._paths
        except Exception as e:
            if self.strict:
                raise
            # disabled-on-error fallback
            self.enabled = False
            self._is_open = True
            return RunPaths.for_run(self.base_dir, run_id)

    def log_step(self, row: Dict[str, Any]) -> None:
        if not self._is_open or self._paths is None:
            return
        if not self.enabled:
            return

        try:
            _append_jsonl(self._paths.timestep_jsonl, row)
        except Exception:
            if self.strict:
                raise
            self.enabled = False  # fail closed (stop logging) but keep sim running

    def end_run(self, summary: Optional[Dict[str, Any]] = None) -> None:
        """
        Commit 2: optional end marker. We simply append a final row with event='run_end'.
        """
        if not self._is_open or self._paths is None or self._run_id is None:
            return
        if not self.enabled:
            self._is_open = False
            return

        try:
            row = {
                "event": "run_end",
                "run_id": self._run_id,
                "t_utc": _utc_iso(),
            }
            if summary:
                row["summary"] = summary
            _append_jsonl(self._paths.timestep_jsonl, row)
        except Exception:
            if self.strict:
                raise
        finally:
            self._is_open = False
