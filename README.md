# HydrOS

**Research-grade digital twin and reinforcement-learning control platform for a handheld microplastic-extraction device.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TypeScript](https://img.shields.io/badge/typescript-5-blue.svg)](https://www.typescriptlang.org/)
[![Status: Phase 1.5](https://img.shields.io/badge/status-Phase%201.5-orange.svg)](#status)

Built solo by [Joshua Ermert](https://www.linkedin.com/in/josh-ermert-79496b176/) (B.S. Management Information Systems + double minor in Computer Science and Interdisciplinary Studies, SDSU Weber Honors College, May 2026).

---

## What HydrOS is

HydrOS simulates a multi-stage physical extraction system end-to-end — hydraulics, clogging dynamics, electrostatic capture, particle physics, sensor realism — and trains safe-RL policies (PPO) against it. The objective is a control system that operates reliably across the full envelope of physical conditions a real handheld device would encounter, not just the conditions its training data covered.

It is a **digital twin**, a **control-system simulator**, a **research platform**, and a **hardware-forward architecture** intended to converge toward a physical deployment.

This repository is a multi-milestone solo build that has carried a calibrated PPO policy through nine major milestones (M1–M9.9), most recently closing M9 with a verified **+22.0% calibrated capture-rate gap** under Scenario A.

For the system layout and engineering constraints, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Quick start

### Install

```bash
pip install -e .
```

### Train / evaluate / render to video

```bash
# Train PPO (calibrated environment)
python -m hydrion.train_ppo_hydrienv_v2_cal

# Evaluate
python -m hydrion.eval_ppo_hydrienv_v2_cal

# Generate the animated reactor-column MP4 (requires FFMPEG on PATH)
python -m hydrion.make_video
```

### Run the realtime research console

The console is a React + Vite frontend backed by a FastAPI service. A `2D / 3D` toggle in the corner switches between a diagnostic SVG view that renders backend particle tracers directly and an in-development WebGL view that uses the same `HydrosDisplayState` schema. Both surface live simulation state.

**Prerequisites:** Python 3.10+, Node.js 18+, optional FFMPEG.

**Terminal 1 — backend (FastAPI on port 8000):**

```bash
uvicorn hydrion.service.app:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — frontend (Vite on port 5173):**

```bash
cd apps/hydros-console
npm install   # one-time
npm run dev
```

Vite proxies `/api/*` to the backend, so one localhost URL serves both — no CORS configuration needed. Open <http://127.0.0.1:5173>:

1. The reactor housing renders; idle state with no particles yet.
2. Pick a scenario (e.g. `baseline_nominal`, `backflush_recovery_demo`) from the dropdown.
3. Click **RUN** — the backend computes the scenario's step-by-step history.
4. Click ▶ play — the simulation plays in real time, with backend state flowing into the visualization every frame.

For `backflush_recovery_demo`, watch the fouling crust accumulate on the cone walls; at t=30s the flush event triggers and on-mesh particles visibly shed into the channels.

---

## What's inside

For the canonical pipeline order, state separation rules, and the immutable 12D / 4D contract, see [`ARCHITECTURE.md`](ARCHITECTURE.md). Short overview:

### Simulation engine — `hydrion/`

A modular Python physics pipeline with strict ordering: **hydraulics → clogging → backflush → electrostatics → particles → sensors → RL**. Stable Baselines3 PPO with `VecNormalize`, Safe-RL action filtering, constraint enforcement, failure detection.

### Visualization + video — `hydrion/rendering/`, `hydrion/make_video.py`

Side-effect-free observability subsystem: episode recorder, anomaly detector across NaN / bounds-violation / shield-event / termination categories, time-series composition. The video pipeline composes a vertical reactor-column scene with three particle-size groups (500 / 100 / 5 µm) animated across 600–1000 frames per episode, exported as MP4 via FFMPEG.

### Research console — `apps/hydros-console/`

React 18 + TypeScript + Vite real-time observability surface with a 2D SVG diagnostic view (`ConicalCascadeView`) and an in-development WebGL 3D view (`WebGLCascadeView`, built on Three.js / React Three Fiber). The WebGL surface is recent and actively evolving — early graphics-programming work as preparation for deeper rendering practice.

### Validation discipline — `tests/`, `docs/`

Every milestone closes with a written execution report, validation results, and a sealed milestone document.

---

## Engineering constraints

1. **Pipeline discipline** — physics layers compose in fixed order, no skipping, no bypassing
2. **State separation** — `truth_state` is physical reality, `sensor_state` is measured reality, they are never silently mixed
3. **Stable interfaces** — 12D observation, 4D action, locked across milestones to preserve policy comparison
4. **Validation first** — every change ships with expected behavior, validation method, and failure conditions
5. **Side-effect-free observability** — visualization may read everything, it may modify nothing
6. **Hardware-forward thinking** — every architectural decision must work toward eventual physical deployment, not just simulation polish

---

## Status

**Phase 1.5 — Research Console + Realism Backbone.** Most recent milestone: **M9.9 closeout** — sensor layer L2 → L3/L4, +22.0% calibrated capture-rate gap confirmed under Scenario A. WebGL 3D console view in active early development.

---

## Repository

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system layers, pipeline, state separation, repo layout
- [`SECURITY.md`](SECURITY.md) — responsible-disclosure policy
- [`LICENSE`](LICENSE) — MIT

---

## Author

Joshua Ermert — [linkedin.com/in/josh-ermert-79496b176](https://www.linkedin.com/in/josh-ermert-79496b176/) · [github.com/JSErmert](https://github.com/JSErmert)
