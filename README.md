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

### Real-time WebGL 3D renderer — `apps/hydros-console/src/components/WebGLCascadeView.tsx`

**GPU-rendered 3D visualization of the reactor cascade**, built with Three.js via React Three Fiber, exposed in the console as a 2D / 3D toggle alongside the existing SVG `ConicalCascadeView`.

- **3D conical cascade geometry** — three filtration stages (coarse → medium → fine) rendered as physically-based cone meshes with `MeshPhysicalMaterial`; transmissive housing, PBR metalness/roughness, emissive intensity driven by per-stage electrostatic field strength.
- **GPU-instanced particle field** — up to 1500 particles rendered in a single `InstancedMesh` draw call; per-instance position, size (mapped from particle diameter in µm), color (mapped from polymer species PP / PE / PET), and capture-status all uploaded to the GPU as `InstancedBufferAttribute` streams updated every frame from the same `HydrosDisplayState` the SVG view consumes.
- **Custom GLSL shaders** — vertex shader composes instanced sphere positions with per-instance scale; fragment shader implements Lambertian-style key lighting, rim lighting for edge highlights, and per-instance emissive glow for captured particles.
- **Pulsing emissive field rendering** — animated `emissiveIntensity` on each stage cone driven by `useFrame` clock + per-stage `mult` (S1=0.4, S2=0.7, S3=1.0), giving a visible electrostatic-field pulse synchronized with stage activation.
- **Auto-rotate `OrbitControls`** with clamped polar angle and zoom range; HUD overlay shows live particle count and mean field strength.

Same `displayState` schema as the existing SVG view — switching between 2D and 3D is one button click in the console.

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

### Training / evaluation / video pipeline

```bash
pip install -e .
# Train (PPO, calibrated environment)
python -m hydrion.train_ppo_hydrienv_v2_cal
# Evaluate
python -m hydrion.eval_ppo_hydrienv_v2_cal
# Generate animated video (requires FFMPEG on PATH)
python -m hydrion.make_video
```

### Viewing the realtime WebGL 3D renderer

The HydrOS reactor is observable through a 60fps WebGL 3D view in `apps/hydros-console/`. The 3D view is the default surface; a diagnostic 2D SVG view is available via the in-app toggle.

**Prerequisites**

- Python 3.10+ with `pip` (backend)
- Node.js 18+ with `npm` (frontend)

**1. Start the FastAPI backend on port 8000** (terminal 1):

```bash
pip install -e .
uvicorn hydrion.service.app:app --host 127.0.0.1 --port 8000
```

The backend exposes `/api/scenarios` (list available scenarios) and `/api/scenarios/run/{id}` (run a scenario and return its step-by-step history).

**2. Start the Vite dev server on port 5173** (terminal 2):

```bash
cd apps/hydros-console
npm install   # one-time
npm run dev   # starts Vite at http://127.0.0.1:5173
```

Vite proxies `/api/*` to the backend, so a single localhost URL serves both. No CORS configuration is needed.

**3. Use the 3D view**

1. Open <http://127.0.0.1:5173> — the reactor housing renders but no particles flow yet (idle state).
2. Pick a scenario from the **SCENARIO** dropdown at the bottom — e.g. `baseline_nominal` or `backflush_recovery_demo`.
3. Click **RUN** — the backend computes the scenario step-by-step history.
4. Click the **▶ play** button — the reactor visibly fills from the inflow over ~12 seconds. The 1500-particle GPU-instanced field is driven by live backend state (capture efficiency, fouling fraction, backflush flag per stage); per-frame `InstancedBufferAttribute` updates flow into custom GLSL shaders.
5. For `backflush_recovery_demo`, watch the fouling crust accumulate on the cone walls; at t=30s the flush event fires and on-mesh particles visibly shed into the channels.

**Toggling to the diagnostic 2D view**

A `2D / 3D` toggle is in the top-right corner of the machine view. 2D mode renders backend particle tracers directly via SVG (useful for verifying backend simulation correctness against the rendered surface); 3D mode runs the full state-driven synthetic field (1500 particles, 60fps, render-loop decoupled from the 10Hz simulation ticks via wall-clock interpolation).

---

## Status

**Phase 1.5 — Research Console + Realism Backbone.** Most recent milestone: **M9.9 closeout** — sensor layer L2→L3/L4, +22.0% calibrated capture-rate gap confirmed under Scenario A.

---

## Author

Joshua Ermert — [linkedin.com/in/josh-ermert-79496b176](https://www.linkedin.com/in/josh-ermert-79496b176/)
