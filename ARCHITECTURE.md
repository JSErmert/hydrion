# HydrOS — Architecture

A top-level reference for how HydrOS is structured, what each layer is responsible for, and how the pieces compose. For a high-level overview, see [`README.md`](README.md). For deeper module-by-module documentation, see [`docs/`](docs/).

---

## 1. What HydrOS is

HydrOS is a **research-grade digital twin and reinforcement-learning control platform** for a handheld microplastic-extraction device for laundry outflow. It simulates the full physical pipeline (hydraulics → clogging → backflush → electrostatic capture → particle dynamics → sensor realism) and trains safe-RL control policies against that simulation, then exposes the running system through a real-time observability console.

The objective: build a controllable, validated, hardware-forward platform that can be evaluated end-to-end before any physical device is built.

---

## 2. System layers

### 2.1 Physics engine — `hydrion/`

A modular Python simulation pipeline with strict pipeline discipline. Modules compose in a fixed order and never bypass each other.

- `hydrion/physics/` — hydraulics, clogging, electrostatic field, particle dynamics
- `hydrion/sensors/` — sensor-realism layer with calibrated noise models
- `hydrion/safety/` — action filtering, constraint enforcement, failure detection (Safe RL)
- `hydrion/state/` — `truth_state` vs `sensor_state` separation, never silently mixed
- `hydrion/scenarios/` — calibrated evaluation scenarios (A, B, C, D) with fixed seeds
- `hydrion/runtime/`, `hydrion/service/` — orchestration and FastAPI service layer
- `hydrion/env.py` — Gymnasium-compatible environment wrapping the full pipeline

### 2.2 Stable interface contracts

- **12D observation vector** — locked across all milestones (M1–M9.9) so policy comparisons remain valid
- **4D action vector** — same lock; policies trained at one milestone can be evaluated at any later milestone

### 2.3 Validation and reporting — `tests/`, `hydrion/validation/`, `docs/`

Every milestone closes with a written execution report, validation results, and a sealed milestone document. Standard checks: stress matrices, envelope sweeps, mass-balance, recovery-dynamics validation.

### 2.4 Training and evaluation — `hydrion/train_ppo_*.py`, `hydrion/eval_ppo_*.py`

Stable Baselines3 PPO backbone with `VecNormalize` and `DummyVecEnv`. Multiple training/evaluation scripts versioned across milestones.

### 2.5 Visualization and video pipeline — `hydrion/rendering/`, `hydrion/make_video.py`

- `hydrion/rendering/observatory.py` — main dashboard class integrating all visualization components
- `hydrion/rendering/renderer_mpl.py` — matplotlib renderer for time-series and composite views
- `hydrion/rendering/episode_history.py` — episode recorder (truth state, sensor state, actions, rewards, info)
- `hydrion/rendering/anomaly_detector.py` — runtime anomaly detection across NaN / bounds-violation / shield-event / termination-cause categories
- `hydrion/make_video.py` — loads a trained PPO model + VecNormalize stats, runs one evaluation episode, composes the vertical reactor-column visualization across 600–1000 frames per episode, exports MP4 via FFMPEG

### 2.6 Research console — `apps/hydros-console/`

A React 18 + TypeScript + Vite real-time research-observability surface. Provides:

- Scenario selection and execution against the FastAPI service layer
- A 2D SVG diagnostic view (`ConicalCascadeView`) that renders backend particle tracers directly
- An in-development WebGL 3D view (`WebGLCascadeView`) built on Three.js / React Three Fiber, providing a state-driven synthetic field for visual study — both views consume the same `HydrosDisplayState` schema

The console is observation-only relative to the physics engine — visualization may read everything, it may modify nothing.

---

## 3. Canonical simulation pipeline

The physics pipeline executes in a fixed order each timestep:

```
Hydraulics → Clogging → Electrostatics → Particles → Sensors → Observation → Safety (wrapper)
```

This ordering is **immutable**. It defines system semantics, data dependencies, and policy behavior across all milestones.

---

## 4. State separation

| State | Owned by | Updated by | Read by |
| ----- | -------- | ---------- | ------- |
| `truth_state` | physics modules | physics modules only | sensors, observation, telemetry |
| `sensor_state` | sensors | sensors only | observation, telemetry |
| `observation` (12D) | wrapper | derived from truth + sensor | policy |
| `action` (4D) | policy / controller | policy each step | wrapper → physics |
| `reward` | env step | env step | policy |
| `telemetry` | env step | physics + sensors | console |

`truth_state` and `sensor_state` are never silently mixed in any output — every consumer reads from exactly one side of the boundary.

---

## 5. Repository layout

```
hydrOS/
├── hydrion/                  # Python physics engine + RL training/eval
│   ├── physics/              # hydraulics, clogging, electrostatics, particles
│   ├── sensors/              # sensor-realism layer
│   ├── safety/               # Safe-RL action filtering, constraint enforcement
│   ├── state/                # truth_state / sensor_state separation
│   ├── scenarios/            # calibrated evaluation scenarios
│   ├── service/              # FastAPI service exposing /api/scenarios
│   ├── rendering/            # CPU-side visualization + episode recording
│   ├── env.py                # Gymnasium environment
│   ├── train_ppo_*.py        # PPO training scripts (per milestone)
│   ├── eval_ppo_*.py         # PPO evaluation scripts (per milestone)
│   └── make_video.py         # MP4 export pipeline (FFMPEG)
│
├── apps/hydros-console/      # React 18 + TypeScript + Vite console
│   └── src/components/
│       ├── ConicalCascadeView.tsx     # 2D SVG diagnostic view
│       └── WebGLCascadeView.tsx       # In-development WebGL 3D view (Three.js / R3F)
│
├── tests/                    # validation tests, stress matrices
├── configs/                  # scenario and run configuration
├── docs/                     # public-facing documentation
└── models/                   # model metadata (weights gitignored)
```

---

## 6. Engineering constraints

HydrOS is built under a small number of non-negotiable rules:

1. **Pipeline discipline** — physics layers compose in fixed order, no skipping, no bypassing
2. **State separation** — `truth_state` is physical reality, `sensor_state` is measured reality, they are never mixed
3. **Stable interfaces** — 12D observation, 4D action, locked across milestones for cross-milestone comparability
4. **Validation first** — every change ships with expected behavior, validation method, and failure conditions
5. **Side-effect-free observability** — visualization may read everything, it may modify nothing
6. **Hardware-forward thinking** — every architectural decision must work toward eventual physical deployment, not just simulation polish

---

## 7. Status

**Phase 1.5 — Research Console + Realism Backbone.** Most recent milestone: **M9.9 closeout** — sensor layer L2 → L3/L4, +22.0% calibrated capture-rate gap confirmed under Scenario A. WebGL 3D console preview added in active early development.

---

## 8. Further reading

- [`README.md`](README.md) — what HydrOS is, quick start, viewing guide
- [`docs/system_spec_overview_rl_pipeline.md`](docs/system_spec_overview_rl_pipeline.md) — RL pipeline reference
- [`docs/context/`](docs/context/) — deeper system context documents
- [`SECURITY.md`](SECURITY.md) — responsible disclosure policy
- [`LICENSE`](LICENSE) — MIT
