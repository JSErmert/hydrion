# System Spec Overview — RL Pipeline

**Date:** 2026-04-12
**Branch:** HydrOS-x-Claude-Code
**Status:** Pre-implementation — infrastructure does not yet exist

---

## 1. What the Console Is Right Now

The HydrOS console is a **physics visualization and scenario playback system**, not a learning system.

### What is running:
- `ConicalCascadeEnv` — M5 conical cascade environment (Gymnasium-compatible)
- `ScenarioRunner` — executes pre-scripted YAML scenarios against the environment
- Console playback — steps through the resulting `ScenarioExecutionHistory` frame by frame

### What is NOT running:
- No RL agent is being trained
- No policy is learning or improving between runs
- `policy_type` field in `/api/run` is accepted by the schema but **ignored** — `action_space.sample()` is always called (random actions only)
- There is no saved policy checkpoint anywhere in the repo

### Current step count in scenario runs:
The scenarios (`baseline_nominal`, `backflush_recovery_demo`) default to **300 steps**. Bumping this to 3,000 would replay more scripted physics — it would not produce learning. Step count in scenario playback is not the same as training steps in RL.

---

## 2. Physics Grounding Status

The console telemetry is grounded in peer-reviewed publications for its **formulas and physical constants**. Device geometry and operating parameters are design defaults until a physical device is characterized.

### Confirmed peer-reviewed

| Component | Source |
|---|---|
| Mesh capture (RT single-collector efficiency) | Rajagopalan & Tien, AIChE J. 22(3):523 (1976) |
| nDEP force direction and magnitude | Pethig et al., J. Phys. D 25:881 (1992); Gascoyne & Vykoukal, Electrophoresis 23:1973 (2002) |
| Clausius-Mossotti factors (PP/PE/PET) | Brandrup & Immergut, Polymer Handbook 4th ed. (1999); Neagu et al., J. Appl. Phys. 92:6365 (2002) |
| nDEP confirmed on actual microplastic fragments | RSC Advances/PMC (2025) — PP, PE, PET, 25–50µm |
| Stokes drag, Poiseuille velocity profile | Standard fluid mechanics |
| Polymer densities for buoyancy/gravity split | Published polymer databases (peer-reviewed sources only) |
| Hamaker constant H = 0.5×10⁻²⁰ J | Visser (1972); Gregory (1981) |
| Water permittivity ε_r = 80.2 | Fernández et al., J. Phys. Chem. Ref. Data 24:33 (1995) |

### Design defaults (not yet measured from physical device)

| Parameter | Tag | Note |
|---|---|---|
| Cone dimensions (D_in, D_tip, L_cone) per stage | `[DESIGN_DEFAULT]` | Physically plausible, not measured |
| Electrode voltage (500V), gap, tip radius | `[DESIGN_DEFAULT]` | Must be characterized per device build |
| Mesh opening sizes (500µm / 100µm / 5µm) | `[DESIGN_DEFAULT]` | Wire specs (d_w, pitch) still open |
| Polymer mixture fractions (PP 8%, PE 7%, PET 70%) | `[DESIGN_DEFAULT]` | Representative laundry outflow estimate |

**Summary:** The model computes the right physics with the right equations. The numbers driving those equations are placeholders. The console accurately visualizes the physics model as implemented; it does not yet claim to accurately represent a specific built device.

---

## 3. The Environment: ConicalCascadeEnv

### Observation space — obs12_v2 (12-dimensional, all [0,1])

| Index | Key | Description |
|---|---|---|
| 0 | `q_in` | Processed flow rate (normalised to 25 L/min) |
| 1 | `delta_p` | Differential pressure (normalised to 150 kPa) |
| 2 | `fouling_mean` | Mean fouling fraction across S1/S2/S3 |
| 3 | `eta_cascade` | Weighted compound capture efficiency |
| 4 | `C_in` | Inlet particle concentration [0,1] |
| 5 | `C_out` | Outlet particle concentration [0,1] |
| 6 | `E_field_norm` | Applied voltage normalised to design voltage |
| 7 | `v_crit_norm` | DEP critical velocity in S3 (normalised to 0.10 m/s) |
| 8 | `step_norm` | Time progress within episode [0,1] |
| 9 | `bf_active` | Backflush active flag {0,1} |
| 10 | `eta_PP` | PP-specific capture efficiency (buoyant species) |
| 11 | `eta_PET` | PET-specific capture efficiency (dense species) |

### Action space — act4_v1 (4-dimensional, all [0,1])

| Index | Key | Effect |
|---|---|---|
| 0 | `valve_cmd` | Flow valve position |
| 1 | `pump_cmd` | Pump power |
| 2 | `bf_cmd` | Backflush trigger (> 0.5 activates) |
| 3 | `voltage_norm` | DEP electrode voltage (scales 0→design voltage) |

### Reward function

```
r = eta_cascade
    - max(0, delta_p_kpa - 80) / 70    ← pressure penalty above 80 kPa
    - voltage_norm × 0.05               ← energy cost
```

Objective: maximize capture efficiency while staying within pressure limits and minimizing energy use.

### Gymnasium compatibility
`ConicalCascadeEnv` is fully Gymnasium-compatible (`gym.Env`). It can be used directly with any Stable-Baselines3 or CleanRL algorithm without modification.

---

## 4. What Is Needed: RL Training Pipeline

### 4.1 Training script

A standalone script (not the web server) that:

1. Instantiates `ConicalCascadeEnv`
2. Wraps it if needed (e.g., `VecEnv` for parallel rollouts)
3. Trains a PPO agent using Stable-Baselines3
4. Saves checkpoints at intervals
5. Logs training metrics (mean reward, capture efficiency, pressure violations)

Approximate structure:
```python
from stable_baselines3 import PPO
from hydrion.environments.conical_cascade_env import ConicalCascadeEnv

env = ConicalCascadeEnv(config_path="configs/default.yaml")
model = PPO("MlpPolicy", env, verbose=1,
            n_steps=2048, batch_size=64, n_epochs=10,
            learning_rate=3e-4, tensorboard_log="runs/ppo_tb/")
model.learn(total_timesteps=500_000, callback=checkpoint_callback)
model.save("models/ppo_cce_v1")
```

### 4.2 Step count requirements

| Phase | Steps | Purpose |
|---|---|---|
| Proof-of-concept | 100k | Confirm agent beats random baseline |
| Initial convergence | 500k | Policy stabilises on nominal conditions |
| Robustness | 1–2M | Policy handles disturbances (clogging, high load) |
| Fine-tuning | 3–5M | Energy efficiency and soft constraint optimization |

3,000 steps is not enough for any phase. The environment has continuous actions, a 12-dimensional observation, and multiple competing objectives. PPO needs at minimum **100k steps to show meaningful learning**, and **500k+ to produce a usable policy**.

### 4.3 Checkpoint and loading infrastructure

The `/api/run` endpoint currently ignores `policy_type`. To actually run a trained policy:

1. Save trained model to `models/` directory
2. Load in `app.py` when `policy_type == "ppo"`
3. Replace `env.action_space.sample()` with `model.predict(obs)`

```python
# In app.py — what this should look like
if req.policy_type == "ppo":
    from stable_baselines3 import PPO
    model = PPO.load("models/ppo_cce_v1")
    action, _ = model.predict(obs, deterministic=True)
else:
    action = env.action_space.sample()
```

### 4.4 Evaluation scenarios

After training, the policy should be evaluated on:

| Scenario | What it tests |
|---|---|
| `baseline_nominal` | Does it maintain high capture at nominal load? |
| `backflush_recovery_demo` | Does it trigger backflush at the right time? |
| High-fouling scenario (to be built) | Does it respond to rapid clogging? |
| High-flow stress scenario (to be built) | Does it manage pressure under high throughput? |

### 4.5 What a trained policy should discover

Given the reward function, an optimal policy should learn to:
- Hold voltage at 0.8–1.0 during active filtration (high capture, energy cost acceptable)
- Reduce voltage when flow is low (energy saving, capture still acceptable)
- Trigger backflush when `fouling_mean > 0.6–0.7` (not too early, not too late)
- Modulate valve/pump to keep pressure below 80 kPa while maximising throughput
- Not trigger unnecessary backflush (backflush has a cooldown and temporarily disrupts capture)

---

## 5. Current Gap Summary

| Capability | Status |
|---|---|
| Physics environment (ConicalCascadeEnv) | Done — Gymnasium-compatible |
| Reward function | Done — eta_cascade - dp_penalty - volt_penalty |
| Observation space (obs12_v2) | Done |
| Action space (act4_v1) | Done |
| Console visualization | Done — M5 Baseline 1 |
| Scenario playback | Done |
| RL training script | **Not built** |
| PPO model checkpoint | **Does not exist** |
| Trained policy loading in API | **Not built** — policy_type ignored |
| Training evaluation scenarios | **Partially built** (2 scenarios exist) |
| TensorBoard or training metrics logging | **Not built** |
| Long-episode curriculum (disturbances, load variation) | **Not built** |

---

## 6. Recommended Next Step

Before training: define the **episode structure**.

Currently `max_steps` in the config is 10,000 but scenarios run 300 steps. A training episode needs:
- A defined start condition (clean filter vs. pre-fouled)
- A defined termination condition (step limit, catastrophic fouling, or time-based)
- A randomized initial condition range so the policy generalizes across operating points, not just one fixed start

Without curriculum design, the agent will overfit to the specific initial conditions it always starts from and fail in deployment when conditions differ.

---

*This document reflects the state of the system as of 2026-04-12. All physics constants are sourced from peer-reviewed literature. All device geometry values are [DESIGN_DEFAULT] pending physical device characterization.*
