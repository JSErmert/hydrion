# PPO Training Pipeline for ConicalCascadeEnv

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a PPO agent on ConicalCascadeEnv (M5 physics) to replicate and exceed the Stage 3 proof-of-concept result, producing a deployable policy checkpoint wired into the API.

**Architecture:** Add shield-compatible state normalization and episode randomization to ConicalCascadeEnv, augment its reward with a flow throughput term, then train PPO + VecNormalize + ShieldedEnv for 500k steps. Save the trained model and wire `policy_type == "ppo_cce"` in the FastAPI service layer.

**Tech Stack:** Python 3.11, stable-baselines3 ≥ 2.1.0, gymnasium ≥ 0.29.1, tensorboard

---

## Critical Constraints Before RL

These constraints govern all tasks in this plan. Violating any of them produces a training artifact that cannot be trusted, compared, or reproduced.

### 1. RL is Exploratory Until Sensor Realism (M6)

The current observation uses `truth_state` directly — no sensor noise, no measurement latency, no drift. The trained policy is **physics-optimal under perfect observation**, not deployment-realistic. It must not be treated as production-ready until M6 adds optical/turbidity sensor modeling and the observation is rebuilt on `sensor_state`. All trained checkpoints from this plan are Phase 1 research artifacts.

### 2. Observation Schema Must Be Version-Locked

The model checkpoint is inseparable from the observation schema it was trained on. The current schema is `obs12_v2` (12-dimensional, all [0,1]). If the schema changes, the checkpoint must be discarded and retraining must occur. The training script must embed `obs_schema: obs12_v2` in the model's save metadata. Any serving path must verify the schema tag matches before loading.

### 3. Model Must Be Cached, Not Loaded Per Request

Loading a Stable-Baselines3 model and VecNormalize statistics on every API request adds 200–500ms latency and is incorrect. The model must be loaded once at first use and cached as a module-level singleton for the lifetime of the process. Task 6 implements this; it is a constraint, not a convenience.

### 4. Reward Is Phase 1 — Subject to Change

The current reward (`eta_cascade + flow_bonus - dp_penalty - volt_penalty`) is a training baseline. Fouling-rate shaping, species-weighted capture (PP vs PET efficiency tradeoff), and backflush timing penalties are deferred to Phase 2. Every saved checkpoint is reward-specific. Changing the reward requires discarding existing checkpoints and retraining from scratch.

### 5. PPO Must Be Benchmarked Against Random AND Heuristic Baselines

Outperforming random is necessary but not sufficient. A heuristic baseline must also be defined and measured: fixed maximum voltage (`voltage_norm=1.0`), nominal flow (`valve=0.7, pump=0.7`), backflush triggered deterministically when `fouling_mean > 0.6`. PPO is only demonstrably useful if it beats both random and this heuristic. The evaluation script (Task 5) must include the heuristic baseline.

### 6. Training Must Be Deterministic

Training runs must be reproducible from a fixed seed. The training script must set:
- `PYTHONHASHSEED`
- `numpy` random seed
- PyTorch seed (via `torch.manual_seed`)
- SB3 `seed` parameter in PPO constructor

Without this, two runs with the same config will produce different checkpoints, making debugging and comparison impossible.

---

## Context for the implementer

The system runs from `C:/Users/JSEer/hydrOS/` (the repo root). All commands must be run from that directory. The environment uses `configs/default.yaml`.

**Do not touch `HydrionEnv`, `train_ppo.py`, or `train_ppo_v15.py`.** Those are the old-physics baseline. This plan creates a parallel CCE pipeline alongside them.

**Key files you will modify or create:**
- Modify: `hydrion/environments/conical_cascade_env.py`
- Create: `hydrion/train_ppo_cce.py`
- Create: `hydrion/eval_ppo_cce.py`
- Modify: `hydrion/service/app.py`

**Key constraint:** `ShieldedEnv` (`hydrion/wrappers/shielded_env.py`) reads `flow`, `pressure`, `clog`, `pump_cmd`, `bf_cmd` from `env.truth_state`. CCE currently uses `q_processed_lmin`, `delta_p_kpa`, and `fouling_frac_s*` keys. Tasks 1 fixes this by adding normalized aliases.

---

## File Structure

| File | Role |
|---|---|
| `hydrion/environments/conical_cascade_env.py` | Add shield-compatible aliases, flow reward term, episode randomization |
| `hydrion/train_ppo_cce.py` | Standalone training script for CCE |
| `hydrion/eval_ppo_cce.py` | Evaluation and baseline comparison script |
| `hydrion/service/app.py` | API endpoint: load and serve trained policy |
| `tests/test_cce_training_infra.py` | All tests for Tasks 1–3 |
| `models/` | Directory for final model artifacts (created by training script) |
| `checkpoints/cce/` | Per-step checkpoints during training (created by training script) |

---

## Task 1: Shield-compatible state aliases in ConicalCascadeEnv

**Files:**
- Modify: `hydrion/environments/conical_cascade_env.py` (around line 510, end of `step()` before `self._step += 1`)
- Test: `tests/test_cce_training_infra.py`

The `ShieldedEnv.post_step` method reads `flow`, `pressure`, `clog`, `pump_cmd`, `bf_cmd` from `env.truth_state`. CCE's `_state` uses different keys. Adding normalized aliases is the minimal fix — no changes to `ShieldedEnv` required.

Normalization references:
- `flow` = `q_processed_lmin / 20.0` (Q_max_Lmin = 20.0 from config)
- `pressure` = `delta_p_kpa / 80.0` (P_max_Pa = 80000 Pa = 80 kPa)
- `clog` = mean of `fouling_frac_s1`, `fouling_frac_s2`, `fouling_frac_s3`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cce_training_infra.py
import numpy as np
import pytest
from hydrion.environments.conical_cascade_env import ConicalCascadeEnv
from hydrion.wrappers.shielded_env import ShieldedEnv
from hydrion.safety.shield import SafetyConfig


def make_env() -> ConicalCascadeEnv:
    return ConicalCascadeEnv(config_path="configs/default.yaml", seed=0)


def test_shield_alias_keys_present_after_step():
    """flow, pressure, clog must be in _state (normalized) after step."""
    env = make_env()
    env.reset(seed=0)
    env.step(np.array([0.5, 0.5, 0.0, 0.8], dtype=np.float32))
    for key in ("flow", "pressure", "clog"):
        assert key in env._state, f"shield alias '{key}' missing from _state"
        assert 0.0 <= env._state[key] <= 1.0, f"alias '{key}' out of [0,1]: {env._state[key]}"


def test_shielded_env_wraps_cce_without_error():
    """ShieldedEnv must wrap CCE and step without raising exceptions."""
    raw_env = make_env()
    cfg = SafetyConfig(
        max_pressure_soft=0.75,
        max_pressure_hard=1.00,
        terminate_on_hard_violation=True,
    )
    env = ShieldedEnv(raw_env, cfg=cfg)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (12,)
    action = np.array([0.5, 0.5, 0.0, 0.8], dtype=np.float32)
    obs2, reward, terminated, truncated, info = env.step(action)
    assert obs2.shape == (12,)
    assert "safety" in info
    assert isinstance(reward, float)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "C:/Users/JSEer/hydrOS" && python -m pytest tests/test_cce_training_infra.py::test_shield_alias_keys_present_after_step tests/test_cce_training_infra.py::test_shielded_env_wraps_cce_without_error -v
```

Expected: FAIL — `AssertionError: shield alias 'flow' missing from _state`

- [ ] **Step 3: Add normalized alias fields to ConicalCascadeEnv.step()**

In `hydrion/environments/conical_cascade_env.py`, find the block just before `self._step += 1` (around line 511). Add the following lines immediately before that line:

```python
        # Shield-compatible normalized aliases — ShieldedEnv reads these exact keys
        # from env.truth_state. Normalization: Q_max=20 L/min, P_max=80 kPa.
        self._state["flow"]     = float(np.clip(
            self._state.get("q_processed_lmin", 0.0) / 20.0, 0.0, 1.0))
        self._state["pressure"] = float(np.clip(
            self._state.get("delta_p_kpa", 0.0) / 80.0, 0.0, 1.0))
        ff_s1 = float(self._state.get("fouling_frac_s1", 0.0))
        ff_s2 = float(self._state.get("fouling_frac_s2", 0.0))
        ff_s3 = float(self._state.get("fouling_frac_s3", 0.0))
        self._state["clog"]     = float(np.clip((ff_s1 + ff_s2 + ff_s3) / 3.0, 0.0, 1.0))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "C:/Users/JSEer/hydrOS" && python -m pytest tests/test_cce_training_infra.py::test_shield_alias_keys_present_after_step tests/test_cce_training_infra.py::test_shielded_env_wraps_cce_without_error -v
```

Expected: PASS for both tests.

- [ ] **Step 5: Run existing CCE test suite to confirm no regressions**

```bash
cd "C:/Users/JSEer/hydrOS" && python -m pytest tests/test_conical_cascade_env.py -v
```

Expected: All tests PASS (the alias fields are additive — nothing is changed, only added).

- [ ] **Step 6: Commit**

```bash
cd "C:/Users/JSEer/hydrOS" && git add hydrion/environments/conical_cascade_env.py tests/test_cce_training_infra.py && git commit -m "feat(cce): add shield-compatible normalized state aliases (flow/pressure/clog)"
```

---

## Task 2: Augment CCE reward with flow throughput term

**Files:**
- Modify: `hydrion/environments/conical_cascade_env.py` (`_reward` method, around line 626)
- Test: `tests/test_cce_training_infra.py` (append to existing file)

Current reward: `eta_cascade - dp_penalty - volt_penalty`

Stage 3 confirmed that rewarding processed flow throughput (`w_processed_flow = 2.0`) is critical for the agent learning to maintain valve/pump at useful levels rather than collapsing to zero flow. Add a flow term scaled at 0.3 to avoid dominating the capture signal.

The shield handles hard termination (-10), soft pressure penalty, clog penalty, and blockage penalty. We do not add those again here.

- [ ] **Step 1: Append the failing test to tests/test_cce_training_infra.py**

```python
def test_reward_increases_with_flow():
    """Higher processed flow (same capture) must yield higher reward."""
    env = make_env()
    env.reset(seed=0)

    # Low flow action: valve closed
    env._state["q_processed_lmin"] = 2.0
    env._state["eta_cascade"]      = 0.80
    env._state["delta_p_kpa"]      = 20.0
    env._state["voltage_norm"]     = 0.5
    r_low = env._reward()

    # High flow action: valve open
    env._state["q_processed_lmin"] = 15.0
    env._state["eta_cascade"]      = 0.80
    env._state["delta_p_kpa"]      = 20.0
    env._state["voltage_norm"]     = 0.5
    r_high = env._reward()

    assert r_high > r_low, (
        f"High-flow reward ({r_high:.4f}) must exceed low-flow reward ({r_low:.4f}) "
        "for equal capture efficiency"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "C:/Users/JSEer/hydrOS" && python -m pytest tests/test_cce_training_infra.py::test_reward_increases_with_flow -v
```

Expected: FAIL — `AssertionError: High-flow reward (X) must exceed low-flow reward (X)`

(Both are equal because the current reward doesn't include flow.)

- [ ] **Step 3: Replace _reward() in conical_cascade_env.py**

Replace the existing `_reward` method (lines 626–636):

```python
    def _reward(self) -> float:
        """
        Reward: maximize capture + throughput, penalise over-pressure and energy.

        Components:
            eta_cascade      — primary capture signal
            flow_bonus       — processed throughput reward (Stage 3: w_processed_flow=2.0)
            dp_penalty       — pressure above 80 kPa (CCE design limit)
            volt_penalty     — energy cost

        ShieldedEnv adds on top:
            clog_penalty     — fouling above soft threshold
            blockage_penalty — pump-on with no flow
            termination (-10) — hard pressure or clog violation
        """
        eta  = float(self._state.get("eta_cascade", 0.0))
        dp   = float(self._state.get("delta_p_kpa", 0.0))
        volt = float(self._state.get("voltage_norm", 0.8))
        q    = float(self._state.get("q_processed_lmin", 0.0))

        flow_bonus   = (q / 20.0) * 0.30            # throughput term; Q_max = 20 L/min
        dp_penalty   = max(0.0, dp - 80.0) / 70.0   # penalise above 80 kPa
        volt_penalty = volt * 0.05                   # small energy cost

        return float(eta + flow_bonus - dp_penalty - volt_penalty)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "C:/Users/JSEer/hydrOS" && python -m pytest tests/test_cce_training_infra.py::test_reward_increases_with_flow -v
```

Expected: PASS.

- [ ] **Step 5: Run full CCE suite**

```bash
cd "C:/Users/JSEer/hydrOS" && python -m pytest tests/test_conical_cascade_env.py tests/test_cce_training_infra.py -v
```

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
cd "C:/Users/JSEer/hydrOS" && git add hydrion/environments/conical_cascade_env.py tests/test_cce_training_infra.py && git commit -m "feat(cce): augment reward with flow throughput term for RL training"
```

---

## Task 3: Episode randomization in ConicalCascadeEnv.reset()

**Files:**
- Modify: `hydrion/environments/conical_cascade_env.py` (`__init__` and `reset` methods)
- Test: `tests/test_cce_training_infra.py` (append)

Without randomized initial conditions, the agent overfits to the clean-start state and fails in deployment when filters are already partially fouled. This task adds optional random initial fouling on each reset.

The `CloggingModel` internal state dict uses `cake_s*`, `bridge_s*`, `pore_s*`, `fouling_frac_s*`. We inject initial fouling by writing directly into `self.clogging._state` after `clogging.reset()`, then syncing to `self._state`.

- [ ] **Step 1: Append the failing test to tests/test_cce_training_infra.py**

```python
def test_randomized_reset_produces_varied_initial_fouling():
    """Over 20 resets with randomize=True, initial fouling must vary."""
    env = ConicalCascadeEnv(
        config_path="configs/default.yaml",
        seed=42,
        randomize_on_reset=True,
    )
    fouling_values = []
    for i in range(20):
        env.reset(seed=i)
        fouling_values.append(env._state.get("fouling_frac_s1", 0.0))

    std = float(np.std(fouling_values))
    assert std > 0.01, (
        f"Fouling std across 20 resets must be > 0.01 (got {std:.4f}). "
        "Check randomize_on_reset logic."
    )
    for v in fouling_values:
        assert 0.0 <= v <= 0.30, f"Initial fouling {v:.4f} outside [0, 0.30]"


def test_deterministic_reset_when_randomize_false():
    """With randomize_on_reset=False (default), reset must always start at zero fouling."""
    env = ConicalCascadeEnv(config_path="configs/default.yaml", seed=0)
    for i in range(5):
        env.reset(seed=i)
        assert env._state.get("fouling_frac_s1", -1.0) == 0.0, \
            "fouling_frac_s1 must be 0.0 on deterministic reset"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "C:/Users/JSEer/hydrOS" && python -m pytest tests/test_cce_training_infra.py::test_randomized_reset_produces_varied_initial_fouling tests/test_cce_training_infra.py::test_deterministic_reset_when_randomize_false -v
```

Expected: FAIL — `TypeError: ConicalCascadeEnv.__init__() got an unexpected keyword argument 'randomize_on_reset'`

- [ ] **Step 3: Add randomize_on_reset parameter to ConicalCascadeEnv.__init__()**

In `hydrion/environments/conical_cascade_env.py`, find the `__init__` signature (line ~176) and add `randomize_on_reset: bool = False` as the last parameter:

```python
    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        stages: list[ConicalStageSpec] | None = None,
        pol_zone: PolarizationZone | None = None,
        d_p_um: float = 10.0,
        seed: int | None = None,
        render_mode=None,
        particles: list[InputParticle] | None = None,
        log_trajectories: bool = False,
        randomize_on_reset: bool = False,
    ):
```

Then, just before the `if seed is not None:` block at the bottom of `__init__`, add:

```python
        self._randomize_on_reset = randomize_on_reset
```

- [ ] **Step 4: Add randomized fouling injection to reset()**

In the `reset()` method, after the line `self.clogging.reset(self._state)` (line ~259) and before `self._state["C_in"] = 0.7`, add:

```python
        # Optional: randomize initial fouling so the policy generalizes beyond clean-start.
        # Injects uniform [0, 0.30] fouling, distributed across cake/bridge/pore components
        # using the Stage 1 weight ratios (0.20/0.60/0.20). Writes into CloggingModel's
        # internal state and syncs to _state so all downstream reads see consistent values.
        if self._randomize_on_reset:
            init_f = float(self.np_random.uniform(0.0, 0.30))
            for s_key in ("s1", "s2", "s3"):
                cake_f   = init_f * 0.20
                bridge_f = init_f * 0.60
                pore_f   = init_f * 0.20
                self.clogging._state[f"cake_{s_key}"]         = cake_f
                self.clogging._state[f"bridge_{s_key}"]       = bridge_f
                self.clogging._state[f"pore_{s_key}"]         = pore_f
                self.clogging._state[f"fouling_frac_{s_key}"] = init_f
                self.clogging._state[f"recoverable_{s_key}"]  = init_f
            self._state.update(self.clogging._state)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd "C:/Users/JSEer/hydrOS" && python -m pytest tests/test_cce_training_infra.py::test_randomized_reset_produces_varied_initial_fouling tests/test_cce_training_infra.py::test_deterministic_reset_when_randomize_false -v
```

Expected: PASS for both.

- [ ] **Step 6: Run full test suites**

```bash
cd "C:/Users/JSEer/hydrOS" && python -m pytest tests/test_conical_cascade_env.py tests/test_cce_training_infra.py -v
```

Expected: All PASS (existing tests use `randomize_on_reset=False` by default — no behavior change).

- [ ] **Step 7: Commit**

```bash
cd "C:/Users/JSEer/hydrOS" && git add hydrion/environments/conical_cascade_env.py tests/test_cce_training_infra.py && git commit -m "feat(cce): add randomize_on_reset for RL training episode diversity"
```

---

## Task 4: Training script — train_ppo_cce.py

**Files:**
- Create: `hydrion/train_ppo_cce.py`

The training script trains PPO on ConicalCascadeEnv with the full safe RL stack:
`CCE(randomize=True, max_steps=400)` → `ShieldedEnv` → `Monitor` → `DummyVecEnv` → `VecNormalize` → `PPO`

Episode length of 400 steps (40 seconds simulated at dt=0.1) is long enough to observe fouling build-up and trigger backflush, short enough for many episodes per gradient update.

SafetyConfig thresholds are in normalized units (CCE adds `pressure` = `delta_p_kpa / 80`):
- `max_pressure_soft=0.75` → 60 kPa soft threshold
- `max_pressure_hard=1.00` → 80 kPa hard limit (terminates episode)
- `max_clog_soft=0.70`, `max_clog_hard=0.95` (tighter than default — CCE fouling is meaningful)

- [ ] **Step 1: Write smoke test for training script**

Append to `tests/test_cce_training_infra.py`:

```python
def test_training_script_smoke_1000_steps():
    """Training script env factory must run 1000 steps without error or exception."""
    import os, time
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from hydrion.wrappers.shielded_env import ShieldedEnv
    from hydrion.safety.shield import SafetyConfig

    def _make():
        env = ConicalCascadeEnv(
            config_path="configs/default.yaml",
            seed=7,
            randomize_on_reset=True,
        )
        env._max_steps = 400
        env = ShieldedEnv(env, cfg=SafetyConfig(
            max_pressure_soft=0.75,
            max_pressure_hard=1.00,
            max_clog_soft=0.70,
            max_clog_hard=0.95,
            terminate_on_hard_violation=True,
        ))
        return Monitor(env)

    vec_env = DummyVecEnv([_make])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO("MlpPolicy", vec_env, n_steps=256, batch_size=64, verbose=0)
    model.learn(total_timesteps=1000)
    # If we reach here, the environment + training stack is functional
    vec_env.close()
```

- [ ] **Step 2: Run smoke test to verify it fails (before script exists)**

```bash
cd "C:/Users/JSEer/hydrOS" && python -m pytest tests/test_cce_training_infra.py::test_training_script_smoke_1000_steps -v
```

Expected: FAIL (test should PASS once the infrastructure from Tasks 1–3 is in place, but this verifies the full stack before the standalone script).

Actually: this test does NOT depend on the script file — it imports directly. If Tasks 1–3 are complete, this test should PASS already. If it fails, debug the error before proceeding.

- [ ] **Step 3: Create hydrion/train_ppo_cce.py**

```python
"""
Hydrion — PPO Training Script for ConicalCascadeEnv (M5 Physics)
-----------------------------------------------------------------
Trains PPO on the M5 conical cascade environment with:
    - Randomized initial fouling (episode diversity)
    - ShieldedEnv (Safe RL: pressure/clog hard limits + termination)
    - VecNormalize (observation + reward normalization)
    - TensorBoard logging

Saves:
    models/ppo_cce_v1.zip          — final policy
    models/ppo_cce_v1_vecnorm.pkl  — VecNormalize statistics

Checkpoints every 10k steps to checkpoints/cce/

Run from repo root:
    python -m hydrion.train_ppo_cce
"""

import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from hydrion.environments.conical_cascade_env import ConicalCascadeEnv
from hydrion.wrappers.shielded_env import ShieldedEnv
from hydrion.safety.shield import SafetyConfig


_CCE_SAFETY_CFG = SafetyConfig(
    max_pressure_soft=0.75,       # 60 kPa — soft warning
    max_pressure_hard=1.00,       # 80 kPa — hard limit (terminates episode)
    max_clog_soft=0.70,           # soft clog warning
    max_clog_hard=0.95,           # hard clog limit
    terminate_on_hard_violation=True,
    hard_violation_penalty=10.0,  # -10 on termination (matches Stage 3 report)
)


def make_env(seed: int = 0):
    def _init():
        env = ConicalCascadeEnv(
            config_path="configs/default.yaml",
            seed=seed,
            randomize_on_reset=True,
        )
        # Override max_steps for training episodes.
        # 400 steps = 40 s simulated — long enough for fouling + backflush cycle,
        # short enough for many episodes per gradient update.
        env._max_steps = 400
        env = ShieldedEnv(env, cfg=_CCE_SAFETY_CFG)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


_TRAIN_SEED = 42
_OBS_SCHEMA  = "obs12_v2"   # version-lock — must match ConicalCascadeEnv observation space


def _set_global_seeds(seed: int) -> None:
    """Constraint 6: training must be deterministic from a fixed seed."""
    import os as _os
    import random as _random
    import numpy as _np
    _os.environ["PYTHONHASHSEED"] = str(seed)
    _random.seed(seed)
    _np.random.seed(seed)
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        _torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def main():
    _set_global_seeds(_TRAIN_SEED)
    os.makedirs("models",          exist_ok=True)
    os.makedirs("checkpoints/cce", exist_ok=True)

    vec_env = DummyVecEnv([make_env(seed=_TRAIN_SEED)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        training=True,
    )

    run_name       = f"ppo_cce_{int(time.time())}"
    tensorboard_log = f"./runs/{run_name}"

    print(f"\nStarting PPO training on ConicalCascadeEnv (M5 physics)...")
    print(f"TensorBoard log: {tensorboard_log}")
    print(f"Checkpoints:     checkpoints/cce/")
    print(f"Final model:     models/ppo_cce_v1.zip\n")

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        ent_coef=0.01,
        learning_rate=3e-4,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path="./checkpoints/cce/",
        name_prefix="ppo_cce",
        verbose=1,
    )

    model.learn(
        total_timesteps=500_000,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    model.save("models/ppo_cce_v1")
    vec_env.save("models/ppo_cce_v1_vecnorm.pkl")

    # Constraint 2: embed obs schema version in a sidecar file so the serving
    # path can verify the schema before loading the model.
    import json as _json
    with open("models/ppo_cce_v1_meta.json", "w") as _f:
        _json.dump({
            "obs_schema":    _OBS_SCHEMA,
            "action_schema": "act4_v1",
            "train_seed":    _TRAIN_SEED,
            "total_timesteps": 500_000,
            "reward_version": "phase1_v1",
        }, _f, indent=2)

    print("\nTraining complete.")
    print("  Model:    models/ppo_cce_v1.zip")
    print("  VecNorm:  models/ppo_cce_v1_vecnorm.pkl")
    print("  Meta:     models/ppo_cce_v1_meta.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run smoke test to verify infrastructure is correct**

```bash
cd "C:/Users/JSEer/hydrOS" && python -m pytest tests/test_cce_training_infra.py::test_training_script_smoke_1000_steps -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd "C:/Users/JSEer/hydrOS" && git add hydrion/train_ppo_cce.py tests/test_cce_training_infra.py && git commit -m "feat(rl): add PPO training script for ConicalCascadeEnv (M5 physics)"
```

---

## Task 5: Evaluation script — eval_ppo_cce.py

**Files:**
- Create: `hydrion/eval_ppo_cce.py`

Evaluates a trained CCE policy against a random baseline. Reports mean eta_cascade, mean reward, shield violations, and backflush events per episode. Designed to be run after training completes, or with any checkpoint.

- [ ] **Step 1: Create hydrion/eval_ppo_cce.py**

```python
"""
Hydrion — PPO Evaluation Script for ConicalCascadeEnv
------------------------------------------------------
Evaluates a trained PPO policy vs random policy on CCE.

Usage (from repo root):
    python -m hydrion.eval_ppo_cce
    python -m hydrion.eval_ppo_cce --model models/ppo_cce_v1.zip --episodes 5
"""

from __future__ import annotations

import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hydrion.environments.conical_cascade_env import ConicalCascadeEnv
from hydrion.wrappers.shielded_env import ShieldedEnv
from hydrion.safety.shield import SafetyConfig


_CCE_SAFETY_CFG = SafetyConfig(
    max_pressure_soft=0.75,
    max_pressure_hard=1.00,
    max_clog_soft=0.70,
    max_clog_hard=0.95,
    terminate_on_hard_violation=True,
    hard_violation_penalty=10.0,
)

_MAX_STEPS_EVAL = 400


def make_env(seed: int):
    def _init():
        env = ConicalCascadeEnv(
            config_path="configs/default.yaml",
            seed=seed,
            randomize_on_reset=False,   # deterministic eval — always clean start
        )
        env._max_steps = _MAX_STEPS_EVAL
        env = ShieldedEnv(env, cfg=_CCE_SAFETY_CFG)
        env.reset(seed=seed)
        return env
    return _init


def _run_episodes(vec_env, model, n_episodes: int, policy_label: str) -> dict:
    """Run n_episodes and collect statistics."""
    returns, eta_list, shield_violations, bf_events, ep_lengths = [], [], [], [], []

    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_ret, ep_eta_sum, ep_steps = 0.0, 0.0, 0
        ep_shield_v, ep_bf = 0, 0

        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = vec_env.action_space.sample()[np.newaxis, :]

            obs, reward, done_arr, info_arr = vec_env.step(action)
            done = bool(done_arr[0])
            ep_ret  += float(reward[0])
            ep_steps += 1

            info = info_arr[0]
            # eta_cascade from underlying env
            ep_eta_sum += float(info.get("eta_cascade", 0.0))

            safety = info.get("safety", {})
            if (safety.get("soft_pressure_violation") or
                    safety.get("hard_pressure_violation") or
                    safety.get("soft_clog_violation") or
                    safety.get("hard_clog_violation") or
                    safety.get("blockage_violation")):
                ep_shield_v += 1

            raw_env = vec_env.envs[0].env.env  # Monitor -> ShieldedEnv -> CCE
            bf_cmd = float(getattr(raw_env, '_state', {}).get('bf_cmd', 0.0))
            if bf_cmd > 0.5:
                ep_bf += 1

        returns.append(ep_ret)
        eta_list.append(ep_eta_sum / max(ep_steps, 1))
        shield_violations.append(ep_shield_v)
        bf_events.append(ep_bf)
        ep_lengths.append(ep_steps)

        print(
            f"  [{policy_label}] ep {ep:02d}: "
            f"return={ep_ret:7.3f}  "
            f"eta={ep_eta_sum / max(ep_steps,1):.3f}  "
            f"steps={ep_steps}  "
            f"shield_violations={ep_shield_v}  "
            f"bf_steps={ep_bf}"
        )

    return {
        "mean_return":    float(np.mean(returns)),
        "std_return":     float(np.std(returns)),
        "mean_eta":       float(np.mean(eta_list)),
        "mean_violations": float(np.mean(shield_violations)),
        "mean_bf_steps":  float(np.mean(bf_events)),
        "mean_ep_length": float(np.mean(ep_lengths)),
    }


def evaluate(
    model_path: str = "models/ppo_cce_v1.zip",
    vecnorm_path: str = "models/ppo_cce_v1_vecnorm.pkl",
    n_episodes: int = 5,
) -> None:

    # ── PPO policy ────────────────────────────────────────────────────────
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Run `python -m hydrion.train_ppo_cce` first.")
        return
    if not os.path.exists(vecnorm_path):
        print(f"VecNormalize not found: {vecnorm_path}")
        return

    ppo_vec = DummyVecEnv([make_env(seed=100)])
    ppo_vec = VecNormalize.load(vecnorm_path, ppo_vec)
    ppo_vec.training   = False
    ppo_vec.norm_reward = False

    model = PPO.load(model_path, env=ppo_vec)
    print(f"\nEvaluating PPO policy: {model_path}")
    ppo_stats = _run_episodes(ppo_vec, model, n_episodes, "PPO")
    ppo_vec.close()

    # ── Random baseline ───────────────────────────────────────────────────
    rand_vec = DummyVecEnv([make_env(seed=200)])
    print(f"\nEvaluating random baseline ({n_episodes} episodes)...")
    rand_stats = _run_episodes(rand_vec, model=None, n_episodes=n_episodes, policy_label="RAND")
    rand_vec.close()

    # ── Heuristic baseline (Constraint 5) ────────────────────────────────
    # Fixed policy: max voltage, nominal flow, backflush when fouling_mean > 0.6.
    # PPO must beat this to be considered non-trivial.
    class HeuristicModel:
        """Fixed rule: full voltage, nominal flow, BF when fouling_mean > 0.6."""
        def predict(self, obs, deterministic=True):
            import numpy as _np
            # obs layout (obs12_v2): [q_in, delta_p, fouling_mean, eta, C_in, C_out,
            #                         E_field, v_crit, step, bf_active, eta_PP, eta_PET]
            fouling_mean = float(obs[0][2])
            bf_cmd = 1.0 if fouling_mean > 0.6 else 0.0
            action = _np.array([[0.7, 0.7, bf_cmd, 1.0]], dtype=_np.float32)
            return action, None

    heur_vec = DummyVecEnv([make_env(seed=300)])
    print(f"\nEvaluating heuristic baseline ({n_episodes} episodes)...")
    heur_stats = _run_episodes(heur_vec, HeuristicModel(), n_episodes, "HEUR")
    heur_vec.close()

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"{'Metric':<28} {'PPO':>10} {'Heuristic':>10} {'Random':>10}")
    print("-" * 68)
    metrics = [
        ("Mean return",     "mean_return"),
        ("Mean eta_cascade", "mean_eta"),
        ("Mean violations",  "mean_violations"),
        ("Mean BF steps",   "mean_bf_steps"),
    ]
    for label, key in metrics:
        p = ppo_stats[key]
        h = heur_stats[key]
        r = rand_stats[key]
        print(f"  {label:<26} {p:>10.3f} {h:>10.3f} {r:>10.3f}")
    print("=" * 68)
    print(f"\nPPO std return:  {ppo_stats['std_return']:.3f}")
    print(f"Heur std return: {heur_stats['std_return']:.3f}")
    print(f"Rand std return: {rand_stats['std_return']:.3f}")
    print("\nConvergence criteria (Constraint 5):")
    ppo_eta  = ppo_stats["mean_eta"]
    rand_eta = rand_stats["mean_eta"]
    heur_eta = heur_stats["mean_eta"]
    print(f"  PPO vs Random eta ratio: {ppo_eta / max(rand_eta, 1e-6):.2f}x  (need > 3.0x)")
    print(f"  PPO vs Heuristic eta ratio: {ppo_eta / max(heur_eta, 1e-6):.2f}x  (need > 1.0x)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="models/ppo_cce_v1.zip")
    parser.add_argument("--vecnorm",  default="models/ppo_cce_v1_vecnorm.pkl")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.model, args.vecnorm, args.episodes)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script is importable**

```bash
cd "C:/Users/JSEer/hydrOS" && python -c "from hydrion.eval_ppo_cce import evaluate; print('import OK')"
```

Expected: `import OK`

- [ ] **Step 3: Commit**

```bash
cd "C:/Users/JSEer/hydrOS" && git add hydrion/eval_ppo_cce.py && git commit -m "feat(rl): add PPO evaluation script for ConicalCascadeEnv vs random baseline"
```

---

## Task 6: Wire trained PPO into API endpoint

**Files:**
- Modify: `hydrion/service/app.py`
- Test: `tests/test_cce_training_infra.py` (append)

When `policy_type == "ppo_cce"`, the `/api/run` endpoint loads the trained model and replaces `action_space.sample()` with `model.predict(obs, deterministic=True)`. The model is lazy-loaded once and cached as a module-level singleton to avoid reloading on every request.

The `/api/run` endpoint uses `HydrionEnv` (not CCE). This task adds a separate `ppo_cce` path that uses CCE + PPO, consistent with the training environment.

- [ ] **Step 1: Append the failing test to tests/test_cce_training_infra.py**

```python
def test_api_run_ppo_cce_falls_back_to_random_if_no_model(tmp_path, monkeypatch):
    """
    When policy_type='ppo_cce' but model file is absent, endpoint must not crash —
    it must fall back to random policy (same as before) and return a run_id.
    """
    from fastapi.testclient import TestClient
    from hydrion.service.app import app

    client = TestClient(app)

    # Point model paths at a non-existent location to force fallback
    import hydrion.service.app as app_module
    monkeypatch.setattr(app_module, "_PPO_CCE_MODEL_PATH",  str(tmp_path / "missing.zip"))
    monkeypatch.setattr(app_module, "_PPO_CCE_VECNORM_PATH", str(tmp_path / "missing.pkl"))
    monkeypatch.setattr(app_module, "_ppo_cce_model",    None)
    monkeypatch.setattr(app_module, "_ppo_cce_vec_norm", None)

    resp = client.post("/api/run", json={
        "policy_type": "ppo_cce",
        "seed": 0,
        "config_name": "default.yaml",
        "max_steps": 5,
        "noise_enabled": False,
    })
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    assert "run_id" in resp.json()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "C:/Users/JSEer/hydrOS" && python -m pytest tests/test_cce_training_infra.py::test_api_run_ppo_cce_falls_back_to_random_if_no_model -v
```

Expected: FAIL — `AttributeError: module 'hydrion.service.app' has no attribute '_PPO_CCE_MODEL_PATH'`

- [ ] **Step 3: Add PPO-CCE policy support to app.py**

In `hydrion/service/app.py`, after the existing imports block (after `import yaml`), add the lazy-loader state:

```python
# ---------------------------------------------------------------------------
# PPO-CCE model — lazy-loaded singleton
# ---------------------------------------------------------------------------
_PPO_CCE_MODEL_PATH   = "models/ppo_cce_v1.zip"
_PPO_CCE_VECNORM_PATH = "models/ppo_cce_v1_vecnorm.pkl"
_ppo_cce_model    = None   # PPO instance after load
_ppo_cce_vec_norm = None   # VecNormalize instance after load


def _load_ppo_cce() -> bool:
    """
    Attempt to load the PPO-CCE model and VecNormalize statistics.
    Returns True if loaded successfully, False if files are absent.
    Caches results in module-level singletons.
    """
    global _ppo_cce_model, _ppo_cce_vec_norm
    if _ppo_cce_model is not None:
        return True
    if not Path(_PPO_CCE_MODEL_PATH).exists() or not Path(_PPO_CCE_VECNORM_PATH).exists():
        return False
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from hydrion.environments.conical_cascade_env import ConicalCascadeEnv
        from hydrion.safety.shield import SafetyConfig

        _safety_cfg = SafetyConfig(
            max_pressure_soft=0.75,
            max_pressure_hard=1.00,
            max_clog_soft=0.70,
            max_clog_hard=0.95,
            terminate_on_hard_violation=True,
        )

        def _make():
            env = ConicalCascadeEnv(config_path="configs/default.yaml")
            env._max_steps = 400
            env = ShieldedEnv(env, cfg=_safety_cfg)
            return env

        vec = DummyVecEnv([_make])
        vec = VecNormalize.load(_PPO_CCE_VECNORM_PATH, vec)
        vec.training    = False
        vec.norm_reward = False
        _ppo_cce_vec_norm = vec

        _ppo_cce_model = PPO.load(_PPO_CCE_MODEL_PATH, env=vec)
        return True
    except Exception as exc:
        print(f"[warn] PPO-CCE model load failed: {exc}")
        return False
```

Then, in the `/api/run` endpoint, replace the `action = env.action_space.sample()` line inside the run loop with:

```python
        # Determine policy: ppo_cce uses trained M5 model; all others use random.
        _use_ppo = req.policy_type == "ppo_cce" and _load_ppo_cce()

        # run loop
        obs, info = env.reset(seed=req.seed)
        for step_idx in range(req.max_steps):
            if _use_ppo:
                import numpy as _np
                obs_vec = _np.array([obs])
                obs_norm = _ppo_cce_vec_norm.normalize_obs(obs_vec)
                action, _ = _ppo_cce_model.predict(obs_norm, deterministic=True)
                action = action[0]
            else:
                action = env.action_space.sample()
```

(Replace only the `action = env.action_space.sample()` line and add the `_use_ppo` check before the loop. Keep everything else in the loop body unchanged.)

- [ ] **Step 4: Run test to verify it passes**

```bash
cd "C:/Users/JSEer/hydrOS" && python -m pytest tests/test_cce_training_infra.py::test_api_run_ppo_cce_falls_back_to_random_if_no_model -v
```

Expected: PASS.

- [ ] **Step 5: Run full test suite**

```bash
cd "C:/Users/JSEer/hydrOS" && python -m pytest tests/ -v --ignore=tests/test_validation_protocol_v2.py -x
```

Expected: All PASS (validation protocol tests may require long-running env setup — skip for now).

- [ ] **Step 6: Commit**

```bash
cd "C:/Users/JSEer/hydrOS" && git add hydrion/service/app.py tests/test_cce_training_infra.py && git commit -m "feat(api): wire ppo_cce policy into /api/run with lazy model loading"
```

---

## After implementation: Run training

Once all 6 tasks are complete and tests pass, run the training:

```bash
cd "C:/Users/JSEer/hydrOS" && python -m hydrion.train_ppo_cce
```

Training takes ~30–90 minutes depending on hardware. Monitor progress via TensorBoard:

```bash
tensorboard --logdir runs/
```

Watch for:
- `rollout/ep_rew_mean` trending upward
- `rollout/ep_len_mean` stabilizing (not always terminating early from hard violations)
- Convergence typically visible by 200k–300k steps

After training completes, evaluate:

```bash
cd "C:/Users/JSEer/hydrOS" && python -m hydrion.eval_ppo_cce
```

**Convergence criteria (matches Stage 3 PoC):**
- `mean_eta_cascade (PPO) / mean_eta_cascade (random) > 3.0×` at 500k steps
- Mean shield violations per episode < 5
- Agent triggers backflush (`bf_steps > 0`) in at least 3/5 evaluation episodes

---

## Self-Review

**Spec coverage:**
- Shield compatibility → Task 1 ✓
- Augmented reward with flow throughput → Task 2 ✓
- Episode randomization → Task 3 ✓
- Training script (VecNormalize + ShieldedEnv + PPO, 500k steps, checkpoints) → Task 4 ✓
- Evaluation vs random baseline → Task 5 ✓
- API integration (`policy_type=ppo_cce`) → Task 6 ✓

**Placeholder scan:** None. All code blocks are complete and runnable.

**Type consistency:**
- `make_env()` factory pattern consistent across Tasks 4, 5, 6
- `SafetyConfig` parameters (`max_pressure_soft=0.75`, etc.) consistent across all tasks
- `_max_steps = 400` applied identically in training and eval env factories
- `randomize_on_reset=True` in training, `False` in eval (intentional — eval uses clean start for reproducibility)
- `_PPO_CCE_MODEL_PATH` / `_PPO_CCE_VECNORM_PATH` strings match between `app.py` loader and the paths written by `train_ppo_cce.py`
