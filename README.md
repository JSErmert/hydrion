# HydrOS

**Research-grade digital twin and reinforcement-learning control platform for a handheld microplastic-extraction device.**

Built solo by Joshua Ermert (BS Management Information Systems + double minor Computer Science & Interdisciplinary Studies, SDSU Weber Honors College, May 2026).

---

## What HydrOS Is

HydrOS simulates a multi-stage physical extraction system end-to-end — hydraulics, clogging dynamics, electrostatic capture, particle physics, sensor realism — then trains and evaluates safe-RL policies (PPO) against it. The objective is a control system that operates reliably across the full envelope of physical conditions a real handheld device would encounter, not just on the cases its training data covered.

It is a **digital twin**, a **control-system simulator**, a **research platform**, and a **hardware-forward architecture** intended to converge toward a physical deployment.

This repository is not a class assignment or a tutorial reimplementation. It is a multi-milestone solo build that has carried a calibrated PPO policy through nine major milestones (M1–M9.9), most recently closing M9 with a verified **+22.0% calibrated capture-rate gap** under Scenario A.

---

## What's Inside

### Simulation engine — `hydrion/`

A modular physics pipeline with strict pipeline discipline (hydraulics → clogging → backflush → electrostatics → particles → sensors → RL). The system enforces a 12D observation vector and 4D action space across all evaluations to keep policy comparison meaningful across milestones.

- **`env.py`** — Gymnasium-compatible environment wrapping the full physics pipeline
- **`physics/`** — modular physics layers (hydraulics, clogging, electrostatic field, particle dynamics)
- **`sensors/`** — sensor-realism layer with calibrated noise models
- **`safety/`** — action filtering, constraint enforcement, failure detection (Safe RL)
- **`state/`** — truth-state vs. sensor-state separation (never mixed)
- **`scenarios/`** — calibrated evaluation scenarios (A, B, C, D) with fixed seeds
- **`runtime/`**, **`service/`** — orchestration and service interfaces

### Rendering subsystem — `hydrion/rendering/`

**Side-effect-free observability and visualization pipeline** designed to make internal dynamics legible, anomalies obvious, and RL behavior interpretable. All visualization is read-only relative to physics state.

- **`observatory.py`** — main dashboard class integrating all visualization components
- **`renderer_mpl.py`** — matplotlib renderer for time-series + composite views
- **`episode_history.py`** — episode data recorder (truth state, sensor state, actions, rewards, info)
- **`anomaly_detector.py`** — runtime anomaly detection across NaNs, bounds violations, shield events, termination causes
- **`time_series.py`** — labeled plot composition with units, scales, and legends
- **`static_geometry.py`** — scene-composition primitives

### Animation + video pipeline — `hydrion/make_video.py`

Loads a trained PPO model + VecNormalize stats, runs one evaluation episode, and composes an animated **vertical reactor-column visualization** matching the system's physical design:

- water inflow → polarize layer → tri-layer electric node extraction → sensor feedback → storage chamber → outflow
- three particle-size groups (~500 µm, ~100 µm, ~5 µm) animated as colored point clouds
- targets 600–1000 frames per episode using `matplotlib.animation` with `FancyBboxPatch`, `Circle`, `Rectangle` scene primitives
- exports MP4 via **FFMPEG**

Output: `videos/hydrion_run.mp4`

### Research observability console — `apps/hydros-console/`

**React 18 + TypeScript + Vite** front-end for real-time research observability — run configuration, scenario selection, panel-based visualization of system internals. Designed to make headless RL runs introspectable without changing the simulation.

### Training + evaluation — `hydrion/train_ppo_*.py`, `hydrion/eval_ppo_*.py`

Multiple training/evaluation scripts versioned through milestone progression (v1, v2, calibrated, M10 real-time). Stable Baselines3 PPO backbone with `VecNormalize` and `DummyVecEnv`.

### Validation discipline — `tests/`, `docs/`

Stress testing, envelope sweeps, mass-balance checks, recovery-dynamics validation. Every milestone closes with a written execution report, validation results, and a sealed milestone document under `docs/orchestration/`.

---

## Engineering Philosophy

HydrOS is built under a small number of non-negotiable constraints:

1. **Pipeline discipline** — physics layers compose in a fixed order. No skipping. No bypassing.
2. **State separation** — `truth_state` is authoritative physical reality; `sensor_state` is measured reality. They are never mixed.
3. **Stable interfaces** — 12D observation, 4D action. Locked across milestones to keep policy comparison valid.
4. **Validation first** — every change ships with expected behavior, validation method, and failure conditions.
5. **Side-effect-free observability** — visualization may read everything; it may modify nothing.
6. **Hardware-forward thinking** — every architectural decision must work toward eventual physical deployment, not just simulation polish.

---

## Documentation

Full system context lives under `docs/`. Start with:

- `docs/MASTER_ARCHITECTURE.md`
- `docs/context/01_SYSTEM_IDENTITY.md`
- `docs/context/02_ARCHITECTURE_CONSTRAINTS.md`
- `docs/context/04_CURRENT_ENGINE_STATUS.md`
- `docs/context/06_LOCKED_SYSTEM_CONSTRAINTS.md`
- `docs/context/09_REALISM_ROADMAP.md`
- `docs/system_spec_overview_rl_pipeline.md`

Milestone closeouts (M1 through M9.9) live under `docs/orchestration/`.

---

## Quick Start

```bash
pip install -e .
# Train (PPO, calibrated environment)
python -m hydrion.train_ppo_hydrienv_v2_cal
# Evaluate
python -m hydrion.eval_ppo_hydrienv_v2_cal
# Generate animated video (requires FFMPEG on PATH)
python -m hydrion.make_video
# Run research console
cd apps/hydros-console && npm install && npm run dev
```

---

## Status

**Phase 1.5 — Research Console + Realism Backbone.** Most recent milestone: **M9.9 closeout** — sensor layer L2→L3/L4, +22.0% calibrated capture-rate gap confirmed under Scenario A.

---

## Author

Joshua Ermert — [linkedin.com/in/josh-ermert-79496b176](https://www.linkedin.com/in/josh-ermert-79496b176/)
