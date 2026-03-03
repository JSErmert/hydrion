from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_run_directory(seed: int) -> Path:
    """Create and return a fresh run directory.

    The name follows the frozen pattern ``run_<timestamp>_<seed>`` where
    ``timestamp`` is the current UTC unix epoch (integer). The directory is
    created under the workspace ``runs/`` folder.
    """
    timestamp = int(datetime.now(timezone.utc).timestamp())
    run_id = f"run_{timestamp}_{seed}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_manifest(run_dir: Path, manifest: Dict[str, Any]) -> None:
    """Write ``manifest.json`` to the given run directory.

    The caller is responsible for supplying a dictionary that conforms to the
    frozen schema described in the project specification. This helper will
    also *ensure* that the mandatory ``artifact_schema`` field is present so
    that every manifest produced going forward is versioned. Existing keys
    are preserved and the structure is otherwise untouched.
    """
    # enforce artifact schema version
    manifest = dict(manifest)
    manifest.setdefault("artifact_schema", "run_artifact_v1")

    path = run_dir / "manifest.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def append_spine_step(run_dir: Path, step_payload: Dict[str, Any]) -> None:
    """Append one JSON object as a line to ``spine.jsonl``.

    The file is created if it does not already exist. Objects are written
    in compact form to keep filesize reasonable.

    This function performs a strict, runtime validation of the payload
    against the canonical spine schema. Missing required keys will raise
    ``ValueError`` so that callers fail fast instead of producing corrupt
    artifacts.
    """
    # --- validation -------------------------------------------------------
    required = [
        "step_idx",
        "sim_time_s",
        "truth",
        "sensors",
        "actions",
        "reward",
        "done",
        "truncated",
        "events",
        "safety",
    ]
    for key in required:
        if key not in step_payload:
            raise ValueError(f"spine payload missing required key '{key}'")

    # nested checks
    truth_req = [
        "flow_norm",
        "pressure_norm",
        "clog_norm",
        "E_norm",
        "C_out",
        "particle_capture_eff",
    ]
    for tkey in truth_req:
        if tkey not in step_payload["truth"]:
            raise ValueError(f"spine.truth missing required key '{tkey}'")

    sensors_req = ["turbidity", "scatter"]
    for skey in sensors_req:
        if skey not in step_payload["sensors"]:
            raise ValueError(f"spine.sensors missing required key '{skey}'")

    actions_req = [
        "valve_cmd",
        "pump_cmd",
        "bf_cmd",
        "node_voltage_cmd",
    ]
    for akey in actions_req:
        if akey not in step_payload["actions"]:
            raise ValueError(f"spine.actions missing required key '{akey}'")

    safety_req = ["shield_intervened", "violation", "violation_kind"]
    for skey in safety_req:
        if skey not in step_payload["safety"]:
            raise ValueError(f"spine.safety missing required key '{skey}'")

    # --- write to file ----------------------------------------------------
    path = run_dir / "spine.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(step_payload, separators=(",", ":"), ensure_ascii=True))
        f.write("\n")


def compute_metrics(run_dir: Path, stability_window: int = 50) -> Dict[str, Any]:
    """Scan ``spine.jsonl`` and compute a fixed set of run metrics.

    The resulting dictionary is written to ``metrics.json`` and also
    returned to the caller.

    ``stability_window`` configures the number of consecutive steps required
    to consider the system "stable". The default value of 50 matches the
    frozen project definition.
    """
    spine_path = run_dir / "spine.jsonl"
    metrics: Dict[str, Any] = {}
    steps: list[Dict[str, Any]] = []

    if spine_path.exists():
        with spine_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    steps.append(json.loads(line))
                except json.JSONDecodeError:
                    # skip malformed lines
                    continue

    # initialise counters
    metrics["actuation_count_total"] = 0
    metrics["actuation_count_nontrivial"] = 0
    metrics["shield_interventions"] = 0
    metrics["violation_count"] = 0
    metrics["energy_proxy"] = 0.0

    prev_actions: Optional[Dict[str, Any]] = None
    capture_vals: list[float] = []

    stable_steps = 0
    time_to_stability_steps: Optional[int] = None
    time_to_stability_s: Optional[float] = None

    for idx, step in enumerate(steps):
        actions = step.get("actions", {})
        if actions:
            # energy proxy is simple sum of all command magnitudes
            for v in actions.values():
                metrics["energy_proxy"] += float(v)

            # nontrivial if any command is non-zero
            if any(float(v) != 0.0 for v in actions.values()):
                metrics["actuation_count_nontrivial"] += 1

            if prev_actions is not None:
                if any(
                    abs(float(actions.get(k, 0)) - float(prev_actions.get(k, 0))) > 1e-6
                    for k in actions
                ):
                    metrics["actuation_count_total"] += 1
            else:
                # first action counts as an actuation
                metrics["actuation_count_total"] += 1
            prev_actions = actions.copy()

        if step.get("safety", {}).get("shield_intervened", False):
            metrics["shield_interventions"] += 1

        if step.get("safety", {}).get("violation", False):
            metrics["violation_count"] += 1

        ce = step.get("truth", {}).get("particle_capture_eff")
        if ce is not None:
            try:
                capture_vals.append(float(ce))
            except Exception:
                pass

        # check stability window
        stable = (
            float(step.get("truth", {}).get("pressure_norm", 0.0)) <= 0.80
            and float(step.get("truth", {}).get("flow_norm", 0.0)) >= 0.40
            and float(step.get("truth", {}).get("clog_norm", 0.0)) <= 0.85
            and not step.get("safety", {}).get("shield_intervened", False)
        )

        if stable:
            stable_steps += 1
        else:
            stable_steps = 0

        if time_to_stability_steps is None and stable_steps >= stability_window:
            time_to_stability_steps = idx - stability_window + 1
            time_to_stability_s = float(step.get("sim_time_s", 0.0))

    metrics["time_to_stability_steps"] = time_to_stability_steps
    metrics["time_to_stability_s"] = time_to_stability_s
    # explicit binary flag for whether the stability window was ever reached
    metrics["stability_reached"] = time_to_stability_steps is not None
    metrics["capture_efficiency_final"] = capture_vals[-1] if capture_vals else None
    metrics["capture_efficiency_mean"] = (
        sum(capture_vals) / len(capture_vals) if capture_vals else None
    )

    # persist
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    return metrics
