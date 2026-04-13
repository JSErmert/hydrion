from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
import yaml
import numpy as _np

from hydrion.env import HydrionEnv
from hydrion.environments.conical_cascade_env import ConicalCascadeEnv
from hydrion.wrappers.shielded_env import ShieldedEnv
from hydrion.logging import artifacts
from hydrion.scenarios import ScenarioRunner, load_scenario

# ---------------------------------------------------------------------------
# PPO-CCE model — lazy-loaded singleton (Constraint 3: cache, not per-request)
# ---------------------------------------------------------------------------
_PPO_CCE_MODEL_PATH   = "models/ppo_cce_v1.zip"
_PPO_CCE_VECNORM_PATH = "models/ppo_cce_v1_vecnorm.pkl"
_ppo_cce_model    = None   # PPO instance, populated on first ppo_cce request
_ppo_cce_vec_norm = None   # VecNormalize instance, populated on first ppo_cce request


def _load_ppo_cce() -> bool:
    """
    Load PPO-CCE model and VecNormalize on first call. Cache for process lifetime.
    Returns True if loaded successfully, False if files are absent.
    """
    global _ppo_cce_model, _ppo_cce_vec_norm
    if _ppo_cce_model is not None:
        return True
    if not Path(_PPO_CCE_MODEL_PATH).exists() or not Path(_PPO_CCE_VECNORM_PATH).exists():
        return False
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from hydrion.environments.conical_cascade_env import ConicalCascadeEnv as _CCE
        from hydrion.safety.shield import SafetyConfig as _SC

        _safety = _SC(
            max_pressure_soft=0.75,
            max_pressure_hard=1.00,
            max_clog_soft=0.70,
            max_clog_hard=0.95,
            terminate_on_hard_violation=True,
            hard_violation_penalty=10.0,
        )

        def _make():
            env = _CCE(config_path="configs/default.yaml")
            env._max_steps = 400
            env = ShieldedEnv(env, cfg=_safety)
            return env

        vec = DummyVecEnv([_make])
        vec = VecNormalize.load(_PPO_CCE_VECNORM_PATH, vec)
        vec.training    = False
        vec.norm_reward = False
        model = PPO.load(_PPO_CCE_MODEL_PATH, env=vec)
        # Only assign globals after both succeed (atomic)
        _ppo_cce_vec_norm = vec
        _ppo_cce_model    = model
        return True
    except Exception as exc:
        print(f"[warn] PPO-CCE model load failed: {exc}")
        return False


class RunRequest(BaseModel):
    policy_type: str
    seed: int
    config_name: str
    max_steps: int
    noise_enabled: bool


class ScenarioRunRequest(BaseModel):
    scenario_id: str


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve built React frontend — must be mounted AFTER all /api routes
_DIST = Path(__file__).parent.parent.parent / "apps" / "hydros-console" / "dist"


@app.post("/api/run")
def start_run(req: RunRequest) -> Dict[str, Any]:
    """Kick off a new simulation run and persist canonical artifacts."""
    # instantiate environment with requested parameters
    config_path = f"configs/{req.config_name}"
    if not Path(config_path).exists():
        raise HTTPException(status_code=400, detail="config not found")

    env = HydrionEnv(config_path=config_path, seed=req.seed, noise_enabled=req.noise_enabled)
    env = ShieldedEnv(env)  # canonical wrapper

    run_dir = artifacts.create_run_directory(req.seed)
    manifest: Dict[str, Any] = {
        "run_id": run_dir.name,
        "artifact_schema": "run_artifact_v1",
        "engine_version": "hydrion_v1.5",
        "obs_schema": "obs12_v1",
        "act_schema": "act4_v1",
        "config_hash": env.cfg.config_hash(),
        "seed": req.seed,
        "policy_type": req.policy_type,
        "noise_enabled": req.noise_enabled,
        "dt": env.dt,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    artifacts.write_manifest(run_dir, manifest)

    # Determine policy and choose environment for run loop
    if req.policy_type == "ppo_cce" and _load_ppo_cce():
        _use_ppo_cce = True
        manifest["obs_schema"] = "obs12_v2"
        from hydrion.environments.conical_cascade_env import ConicalCascadeEnv
        from hydrion.safety.shield import SafetyConfig
        from hydrion.wrappers.shielded_env import ShieldedEnv as _ShieldedEnv
        _safety_cfg = SafetyConfig(
            max_pressure_soft=0.75, max_pressure_hard=1.00,
            max_clog_soft=0.70,     max_clog_hard=0.95,
            terminate_on_hard_violation=True,
            hard_violation_penalty=10.0,
        )
        run_env = _ShieldedEnv(
            ConicalCascadeEnv(config_path=f"configs/{req.config_name}"),
            cfg=_safety_cfg,
        )
        run_env.env._max_steps = req.max_steps
    else:
        _use_ppo_cce = False
        run_env = env  # existing HydrionEnv

    # run loop
    obs, info = run_env.reset(seed=req.seed)
    for step_idx in range(req.max_steps):
        if _use_ppo_cce:
            obs_vec  = _np.array([obs])
            obs_norm = _ppo_cce_vec_norm.normalize_obs(obs_vec)
            action, _ = _ppo_cce_model.predict(obs_norm, deterministic=True)
            action = action[0]
        else:
            action = run_env.action_space.sample()

        obs, reward, terminated, truncated, info = run_env.step(action)

        step_payload: Dict[str, Any] = {
            "step_idx": step_idx,
            "sim_time_s": float(env.steps * env.dt),
            "truth": {
                "flow_norm": float(env.truth_state.get("flow", 0.0)),
                "pressure_norm": float(env.truth_state.get("pressure", 0.0)),
                "clog_norm": float(env.truth_state.get("clog", 0.0)),
                "E_norm": float(env.truth_state.get("E_norm", 0.0)),
                "C_out": float(env.truth_state.get("C_out", 0.0)),
                "particle_capture_eff": float(env.truth_state.get("particle_capture_eff", 0.0)),
            },
            "sensors": {
                "turbidity": float(env.sensor_state.get("sensor_turbidity", 0.0)),
                "scatter": float(env.sensor_state.get("sensor_scatter", 0.0)),
            },
            "actions": {
                "valve_cmd": float(env.truth_state.get("valve_cmd", 0.0)),
                "pump_cmd": float(env.truth_state.get("pump_cmd", 0.0)),
                "bf_cmd": float(env.truth_state.get("bf_cmd", 0.0)),
                "node_voltage_cmd": float(env.truth_state.get("node_voltage_cmd", 0.0)),
            },
            "reward": float(reward),
            "done": bool(terminated),
            "truncated": bool(truncated),
            "events": {"anomaly_active": False},
            "safety": {
                "shield_intervened": info.get("safety", {}).get("shield_intervened", False),
                "violation": bool(
                    info.get("safety", {}).get("soft_pressure_violation", False)
                    or info.get("safety", {}).get("hard_pressure_violation", False)
                    or info.get("safety", {}).get("soft_clog_violation", False)
                    or info.get("safety", {}).get("hard_clog_violation", False)
                    or info.get("safety", {}).get("blockage_violation", False)
                ),
                "violation_kind": None,
            },
        }
        artifacts.append_spine_step(run_dir, step_payload)

        if terminated or truncated:
            break

    artifacts.compute_metrics(run_dir)
    return {"run_id": run_dir.name}


@app.get("/api/runs")
def list_runs():
    root = Path("runs")
    if not root.exists():
        return []
    return [p.name for p in sorted(root.iterdir()) if p.is_dir()]


@app.get("/api/runs/{run_id}/manifest")
def get_manifest(run_id: str):
    path = Path("runs") / run_id / "manifest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="manifest not found")
    return json_load(path)


@app.get("/api/runs/{run_id}/spine")
def get_spine(run_id: str):
    path = Path("runs") / run_id / "spine.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail="spine not found")
    # return as list of objects
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


@app.get("/api/runs/{run_id}/metrics")
def get_metrics(run_id: str):
    path = Path("runs") / run_id / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="metrics not found")
    return json_load(path)


@app.get("/api/scenarios")
def list_scenarios():
    """Return metadata for all available scenario YAML files."""
    examples_dir = Path("hydrion/scenarios/examples")
    if not examples_dir.exists():
        return []
    result = []
    for yaml_file in sorted(examples_dir.glob("*.yaml")):
        try:
            with open(yaml_file, "r") as f:
                raw = yaml.safe_load(f) or {}
            result.append({
                "id": str(raw.get("id", yaml_file.stem)),
                "name": str(raw.get("name", yaml_file.stem)),
                "description": str(raw.get("description", "")),
            })
        except Exception:
            pass
    return result


@app.post("/api/scenarios/run")
def run_scenario(req: ScenarioRunRequest) -> Dict[str, Any]:
    """Execute a named scenario and return the full ScenarioExecutionHistory."""
    examples_dir = Path("hydrion/scenarios/examples")
    yaml_path = examples_dir / f"{req.scenario_id}.yaml"
    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail=f"scenario not found: {req.scenario_id}")

    scenario = load_scenario(yaml_path)
    # ConicalCascadeEnv used directly (no ShieldedEnv) — intentional for scenario visualization.
    # Scenario runner is not a policy training path; safety rail not required here.
    env = ConicalCascadeEnv(config_path="configs/default.yaml")
    runner = ScenarioRunner(env)
    history = runner.run(scenario)
    return history.to_dict()


# small helper

def json_load(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


# Serve static assets (JS/CSS/etc.) and SPA index fallback
if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str = ""):
        """Return index.html for all non-API routes (React SPA routing)."""
        index = _DIST / "index.html"
        if index.exists():
            return FileResponse(str(index))
        raise HTTPException(status_code=404, detail="Frontend not built")
