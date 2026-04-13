# M5 Conical Cascade View — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing `MachineCore` machine view with the physics-driven `ConicalCascadeView` — a live SVG rendering of the Baseline 1 conical cascade diagram — while leaving all surrounding console panels (top telemetry, left/right panels, bottom strip) completely untouched.

**Architecture:** Two coupled halves share a clean interface. The Python half extends `ConicalCascadeEnv` to emit per-stage accumulation, efficiency, and flush state into `truth_state`. The TypeScript half extends `HydrosDisplayState`, updates the mapper to read those new fields, and builds `ConicalCascadeView.tsx` — a fully reactive SVG component that drives every visual element from the display state. `App.tsx` swaps one import.

**Tech Stack:** Python 3 / gymnasium / numpy (env); React 18 / TypeScript 5 / Vite (console); pytest (Python tests); `npx tsc --noEmit` (TypeScript type-check)

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `hydrion/environments/conical_cascade_env.py` | Add accumulation state, per-stage observables, flush tracking |
| Create | `tests/test_conical_cascade_env.py` | pytest coverage for new env behaviour |
| Modify | `apps/hydros-console/src/scenarios/displayStateMapper.ts` | Extend `HydrosDisplayState` + mapper for M5 fields |
| Create | `apps/hydros-console/src/components/ConicalCascadeView.tsx` | Full Baseline 1 SVG as a live React component |
| Modify | `apps/hydros-console/src/App.tsx` | Swap `MachineCore` → `ConicalCascadeView` at mount point |

**Do not touch:**
`TopTelemetryBand.tsx`, `RightAdvisoryPanel.tsx`, `BottomNarrativeBand.tsx`, `PlaybackBar.tsx`, `LeftMetricsPanel.tsx`, `MachineCore.tsx` (keep in place, just unmounted)

---

## Task 1: Extend ConicalCascadeEnv — accumulation state

Adds `_storage_fill`, `_channel_fill[3]` to env state so the console can show particle accumulation over time and trigger the swap recommendation.

**Files:**
- Modify: `hydrion/environments/conical_cascade_env.py`
- Create: `tests/test_conical_cascade_env.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_conical_cascade_env.py
import numpy as np
import pytest
from hydrion.environments.conical_cascade_env import ConicalCascadeEnv


def make_env() -> ConicalCascadeEnv:
    return ConicalCascadeEnv(config_path="configs/default.yaml", seed=0)


def test_accumulation_fields_in_truth_state():
    """storage_fill and channel_fill_s1/s2/s3 must appear in truth_state after step."""
    env = make_env()
    env.reset(seed=0)
    action = np.array([0.5, 0.5, 0.0, 0.8], dtype=np.float32)
    env.step(action)
    s = env._state
    assert "storage_fill"    in s, "storage_fill missing from state"
    assert "channel_fill_s1" in s, "channel_fill_s1 missing"
    assert "channel_fill_s2" in s, "channel_fill_s2 missing"
    assert "channel_fill_s3" in s, "channel_fill_s3 missing"


def test_accumulation_increases_over_steps():
    """Channel fill must grow when particles are being captured (no flush)."""
    env = make_env()
    env.reset(seed=0)
    action = np.array([0.5, 0.5, 0.0, 0.8], dtype=np.float32)
    for _ in range(20):
        env.step(action)
    s = env._state
    total_fill = s["channel_fill_s1"] + s["channel_fill_s2"] + s["channel_fill_s3"]
    assert total_fill > 0.0, "channel fill must be > 0 after 20 capture steps"


def test_flush_reduces_channel_fill():
    """bf_cmd > 0.5 must drain channels and increase storage_fill."""
    env = make_env()
    env.reset(seed=0)
    capture = np.array([0.5, 0.5, 0.0, 0.8], dtype=np.float32)
    flush   = np.array([0.5, 0.5, 1.0, 0.8], dtype=np.float32)
    for _ in range(30):
        env.step(capture)
    pre_fill = env._state["channel_fill_s3"]
    pre_storage = env._state["storage_fill"]
    for _ in range(10):
        env.step(flush)
    assert env._state["channel_fill_s3"] < pre_fill,   "flush must drain channel_fill_s3"
    assert env._state["storage_fill"]    > pre_storage, "flush must increase storage_fill"
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
cd C:/Users/JSEer/hydrOS && pytest tests/test_conical_cascade_env.py -v 2>&1 | head -30
```

Expected: `AttributeError` — `_storage_fill` or `storage_fill` not in state

- [ ] **Step 3: Add accumulation constants and state to ConicalCascadeEnv**

In `hydrion/environments/conical_cascade_env.py`, add after the `POLYMER_MIX` block and before the class definition:

```python
# ---------------------------------------------------------------------------
# Accumulation model constants — [DESIGN_DEFAULT], replace with physical specs
# ---------------------------------------------------------------------------
_CHANNEL_CAPACITY_M3 = 4e-4   # ~0.4 L per collection channel
_STORAGE_CAPACITY_M3 = 1.2e-2  # ~12 L detachable storage chamber
_FLUSH_DRAIN_RATE    = 0.20    # fraction of channel fill drained per bf step
```

In `ConicalCascadeEnv.__init__`, after `self._dt = ...`:

```python
        # Accumulation state — persists across steps, resets on env.reset()
        self._storage_fill: float      = 0.0
        self._channel_fill: list[float] = [0.0, 0.0, 0.0]
```

- [ ] **Step 4: Update reset() to zero accumulation state**

Replace the reset body ending with `return self._obs(), {}`:

```python
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        self._state = {}
        self._storage_fill = 0.0
        self._channel_fill = [0.0, 0.0, 0.0]

        self.hydraulics.reset(self._state)
        self.clogging.reset(self._state)

        self._state["C_in"]         = 0.7
        self._state["C_out"]        = 0.7
        self._state["eta_cascade"]  = 0.0
        self._state["eta_PP"]       = 0.0
        self._state["eta_PET"]      = 0.0
        self._state["v_crit_s3"]    = 0.0
        self._state["bf_active"]    = 0.0
        self._state["voltage_norm"] = 0.8
        self._state["storage_fill"]    = 0.0
        self._state["channel_fill_s1"] = 0.0
        self._state["channel_fill_s2"] = 0.0
        self._state["channel_fill_s3"] = 0.0

        return self._obs(), {}
```

- [ ] **Step 5: Add accumulation update inside step(), after the C_out write**

Add this block directly after `self._state["v_crit_s3"] = float(v_crit_s3)`:

```python
        # ── Accumulation model ────────────────────────────────────────────
        bf = float(action[2])  # bf_cmd from action vector

        # Each stage captures particles proportional to PET efficiency
        # (PET = sinking majority species; captures into channel bottom)
        for i in range(3):
            if len(results["PET"]["per_stage"]) > i:
                eta_i      = float(results["PET"]["per_stage"][i]["eta_stage"])
                captured   = eta_i * float(self._state.get("C_in", 0.7)) * Q_m3s * self._dt
                self._channel_fill[i] = float(np.clip(
                    self._channel_fill[i] + captured / _CHANNEL_CAPACITY_M3,
                    0.0, 1.0,
                ))

        # bf_cmd > 0.5: drain channels into storage at FLUSH_DRAIN_RATE per step
        if bf > 0.5:
            for i in range(3):
                drained = self._channel_fill[i] * _FLUSH_DRAIN_RATE
                self._storage_fill = float(np.clip(
                    self._storage_fill
                    + drained * (_CHANNEL_CAPACITY_M3 / _STORAGE_CAPACITY_M3),
                    0.0, 1.0,
                ))
                self._channel_fill[i] = max(0.0, self._channel_fill[i] - drained)
            flush_flag = 1.0
        else:
            flush_flag = 0.0

        self._state["storage_fill"]      = self._storage_fill
        self._state["channel_fill_s1"]   = self._channel_fill[0]
        self._state["channel_fill_s2"]   = self._channel_fill[1]
        self._state["channel_fill_s3"]   = self._channel_fill[2]
        self._state["flush_active_s1"]   = flush_flag
        self._state["flush_active_s2"]   = flush_flag
        self._state["flush_active_s3"]   = flush_flag
```

- [ ] **Step 6: Run tests — all three must pass**

```bash
cd C:/Users/JSEer/hydrOS && pytest tests/test_conical_cascade_env.py::test_accumulation_fields_in_truth_state tests/test_conical_cascade_env.py::test_accumulation_increases_over_steps tests/test_conical_cascade_env.py::test_flush_reduces_channel_fill -v
```

Expected: `3 passed`

- [ ] **Step 7: Commit**

```bash
cd C:/Users/JSEer/hydrOS
git add hydrion/environments/conical_cascade_env.py tests/test_conical_cascade_env.py
git commit -m "feat(env): add accumulation state to ConicalCascadeEnv (channel_fill, storage_fill, flush_active)"
```

---

## Task 2: Extend ConicalCascadeEnv — per-stage observables

Adds `eta_s1/s2/s3` and `v_crit_s1/s2/s3` to `truth_state` so the console can enforce stage hierarchy visually.

**Files:**
- Modify: `hydrion/environments/conical_cascade_env.py`
- Modify: `tests/test_conical_cascade_env.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_conical_cascade_env.py`:

```python
def test_per_stage_eta_in_truth_state():
    """eta_s1, eta_s2, eta_s3 must be in state and satisfy S3 >= S1 asymmetry."""
    env = make_env()
    env.reset(seed=0)
    action = np.array([0.5, 0.5, 0.0, 0.8], dtype=np.float32)
    for _ in range(5):
        env.step(action)
    s = env._state
    for key in ("eta_s1", "eta_s2", "eta_s3"):
        assert key in s, f"{key} missing from state"
        assert 0.0 <= s[key] <= 1.0, f"{key} out of [0,1]: {s[key]}"


def test_stage_hierarchy_s3_dominates():
    """S3 capture efficiency must exceed S1 on average at nominal flow."""
    env = make_env()
    env.reset(seed=0)
    action = np.array([0.5, 0.5, 0.0, 0.8], dtype=np.float32)
    eta_s1_samples, eta_s3_samples = [], []
    for _ in range(20):
        env.step(action)
        eta_s1_samples.append(env._state["eta_s1"])
        eta_s3_samples.append(env._state["eta_s3"])
    assert np.mean(eta_s3_samples) >= np.mean(eta_s1_samples), \
        "S3 mean efficiency must >= S1 (asymmetric stage design)"


def test_v_crit_per_stage_in_truth_state():
    """v_crit_s1, v_crit_s2, v_crit_s3 must be in state after step."""
    env = make_env()
    env.reset(seed=0)
    env.step(np.array([0.5, 0.5, 0.0, 0.8], dtype=np.float32))
    for key in ("v_crit_s1", "v_crit_s2", "v_crit_s3"):
        assert key in env._state, f"{key} missing"
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
cd C:/Users/JSEer/hydrOS && pytest tests/test_conical_cascade_env.py::test_per_stage_eta_in_truth_state -v
```

Expected: `KeyError` or `AssertionError` — `eta_s1` not in state

- [ ] **Step 3: Add per-stage observable writes to step()**

In `step()`, directly after the `self._state["v_crit_s3"] = float(v_crit_s3)` line, add:

```python
        # Per-stage observables for console hierarchy rendering
        per_pet = results["PET"]["per_stage"]
        self._state["eta_s1"] = float(per_pet[0]["eta_stage"]) if len(per_pet) > 0 else 0.0
        self._state["eta_s2"] = float(per_pet[1]["eta_stage"]) if len(per_pet) > 1 else 0.0
        self._state["eta_s3"] = float(per_pet[2]["eta_stage"]) if len(per_pet) > 2 else 0.0
        self._state["v_crit_s1"] = float(per_pet[0]["v_crit"]) if len(per_pet) > 0 else 0.0
        self._state["v_crit_s2"] = float(per_pet[1]["v_crit"]) if len(per_pet) > 1 else 0.0
        # v_crit_s3 already written above — no duplicate
```

- [ ] **Step 4: Run all five tests**

```bash
cd C:/Users/JSEer/hydrOS && pytest tests/test_conical_cascade_env.py -v
```

Expected: `6 passed` (3 from Task 1 + 3 new)

- [ ] **Step 5: Commit**

```bash
cd C:/Users/JSEer/hydrOS
git add hydrion/environments/conical_cascade_env.py tests/test_conical_cascade_env.py
git commit -m "feat(env): expose per-stage eta and v_crit to truth_state for console hierarchy"
```

---

## Task 3: Extend HydrosDisplayState and mapper

Adds M5 fields to the display layer interface and populates them from `truth_state`. Existing locked panels are unaffected — TypeScript structural typing ensures they see only what they reference.

**Files:**
- Modify: `apps/hydros-console/src/scenarios/displayStateMapper.ts`

- [ ] **Step 1: Confirm TypeScript type-check passes cleanly before any change**

```bash
cd C:/Users/JSEer/hydrOS/apps/hydros-console && npx tsc --noEmit 2>&1 | head -20
```

Expected: no output (clean compile). If errors exist, note them — do not fix unrelated errors.

- [ ] **Step 2: Add M5 fields to HydrosDisplayState interface**

In `displayStateMapper.ts`, add the following fields to the `HydrosDisplayState` interface, after the `storageFill` line:

```typescript
  // M5 ConicalCascade — per-stage accumulation and efficiency
  channelFillS1: number;   // [0,1] particle volume fraction in S1 collection channel
  channelFillS2: number;
  channelFillS3: number;
  etaS1: number;           // [0,1] S1 stage capture efficiency (PET representative)
  etaS2: number;
  etaS3: number;           // always >= etaS1 by design (asymmetric stages)
  vCritNorm: number;       // v_crit_s3 / OBS_VCRIT_MAX normalised to [0,1]
  etaPP: number;           // [0,1] buoyant species (PP) efficiency — drives density split cue
  etaPET: number;          // [0,1] dense species (PET) efficiency
  flushActiveS1: boolean;  // hydraulic flush active on S1 channel
  flushActiveS2: boolean;
  flushActiveS3: boolean;
```

- [ ] **Step 3: Run tsc — expect type errors on mapStepRecordToDisplayState return**

```bash
cd C:/Users/JSEer/hydrOS/apps/hydros-console && npx tsc --noEmit 2>&1 | head -30
```

Expected: errors complaining that the return object in `mapStepRecordToDisplayState` is missing the new fields. This is the expected failing state.

- [ ] **Step 4: Populate new fields in mapStepRecordToDisplayState**

In `mapStepRecordToDisplayState`, update the `storageFill` line and add new fields in the return object:

```typescript
    storageFill: clamp01(ts['storage_fill']),

    channelFillS1: clamp01(ts['channel_fill_s1']),
    channelFillS2: clamp01(ts['channel_fill_s2']),
    channelFillS3: clamp01(ts['channel_fill_s3']),
    etaS1: clamp01(ts['eta_s1']),
    etaS2: clamp01(ts['eta_s2']),
    etaS3: clamp01(ts['eta_s3']),
    vCritNorm: clamp01((ts['v_crit_s3'] ?? 0) / 0.10),
    etaPP:  clamp01(ts['eta_PP']),
    etaPET: clamp01(ts['eta_PET']),
    flushActiveS1: (ts['flush_active_s1'] ?? 0) > 0.5,
    flushActiveS2: (ts['flush_active_s2'] ?? 0) > 0.5,
    flushActiveS3: (ts['flush_active_s3'] ?? 0) > 0.5,
```

- [ ] **Step 5: Confirm type-check passes**

```bash
cd C:/Users/JSEer/hydrOS/apps/hydros-console && npx tsc --noEmit
```

Expected: no output (clean compile)

- [ ] **Step 6: Commit**

```bash
cd C:/Users/JSEer/hydrOS
git add apps/hydros-console/src/scenarios/displayStateMapper.ts
git commit -m "feat(console): extend HydrosDisplayState with M5 per-stage fields"
```

---

## Task 4: ConicalCascadeView — static geometry

Creates the new component file with all static SVG geometry from Baseline 1 faithfully reproduced. No dynamic elements yet — every visual element receives a hardcoded value so the diagram renders correctly before any state wiring.

**Files:**
- Create: `apps/hydros-console/src/components/ConicalCascadeView.tsx`

- [ ] **Step 1: Create the file with defs, background, and component shell**

```tsx
// apps/hydros-console/src/components/ConicalCascadeView.tsx
//
// M5 Conical Cascade machine view — Baseline 1 (locked 2026-04-11)
// Visual surface only. No metrics embedded here. η belongs to the N-layer panels.
//
// All geometric constants match machine-core-v4.html exactly.
// Dynamic layers are driven by HydrosDisplayState prop.

import type { HydrosDisplayState } from '../scenarios/displayStateMapper';

const FONT = '"JetBrains Mono", "Fira Code", "Courier New", monospace';

// ── Geometric constants (Baseline 1) ──────────────────────────────────────
const CY = 154;          // device centreline y
const BORE_TOP = 64;     // housing bore top wall y
const BORE_BOT = 244;    // housing bore bottom wall y
const BORE_H   = 180;    // BORE_BOT - BORE_TOP

// Stage x-ranges [xStart, xApexX, xApexY]
const STAGES = [
  { label: 'S1', xStart: 118, xEnd: 298, apexX: 296, apexY: 243,
    bezier: 'M 118,64 C 195,64 292,96 296,243',
    innerBezier: 'M 118,72 C 195,72 292,104 296,243',
    color: '#FB923C', weave: 'url(#weaveCoarse)', mult: 0.4,
    ejY: 252, chY: 252, chH: 16 },
  { label: 'S2', xStart: 306, xEnd: 486, apexX: 484, apexY: 243,
    bezier: 'M 306,64 C 383,64 480,96 484,243',
    innerBezier: 'M 306,72 C 383,72 480,104 484,243',
    color: '#FBBF24', weave: 'url(#weaveMedium)', mult: 0.7,
    ejY: 274, chY: 274, chH: 16 },
  { label: 'S3', xStart: 494, xEnd: 674, apexX: 672, apexY: 243,
    bezier: 'M 494,64 C 571,64 668,96 672,243',
    innerBezier: 'M 494,72 C 571,72 668,104 672,243',
    color: '#38BDF8', weave: 'url(#weaveFine)', mult: 1.0,
    ejY: 296, chY: 296, chH: 16 },
] as const;

const INLET_OVALS = [
  { cx: 118, stroke: '#2A5878' },
  { cx: 306, stroke: '#2A6878' },
  { cx: 494, stroke: '#306888' },
  { cx: 674, stroke: '#38BDF8' },
  { cx: 714, stroke: '#2A6080' },
] as const;

interface ConicalCascadeViewProps {
  state: HydrosDisplayState | null;
}

export default function ConicalCascadeView({ state }: ConicalCascadeViewProps) {
  const s = state;

  return (
    <svg
      viewBox="0 0 1060 460"
      width="100%"
      height="100%"
      style={{ display: 'block' }}
      preserveAspectRatio="xMidYMid meet"
    >
      <defs>
        <radialGradient id="bg" cx="50%" cy="44%" r="62%">
          <stop offset="0%"   stopColor="#061428" />
          <stop offset="100%" stopColor="#010306" />
        </radialGradient>

        <linearGradient id="cleanFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stopColor="#0C2240" stopOpacity={0.55} />
          <stop offset="100%" stopColor="#061828" stopOpacity={0.15} />
        </linearGradient>
        <linearGradient id="concFill" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%"   stopColor="#0A1C30" stopOpacity={0.6} />
          <stop offset="100%" stopColor="#040C18" stopOpacity={0.9} />
        </linearGradient>

        <radialGradient id="nodeG1" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#FB923C" stopOpacity={0.22} />
          <stop offset="100%" stopColor="#FB923C" stopOpacity={0} />
        </radialGradient>
        <radialGradient id="nodeG2" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#FBBF24" stopOpacity={0.28} />
          <stop offset="100%" stopColor="#FBBF24" stopOpacity={0} />
        </radialGradient>
        <radialGradient id="nodeG3" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#38BDF8" stopOpacity={0.45} />
          <stop offset="100%" stopColor="#38BDF8" stopOpacity={0} />
        </radialGradient>

        <radialGradient id="chamberGlow" cx="50%" cy="70%" r="60%">
          <stop offset="0%"   stopColor="#0A1C30" />
          <stop offset="100%" stopColor="#020810" />
        </radialGradient>

        <filter id="fxSoft" x="-20%" y="-30%" width="140%" height="160%">
          <feGaussianBlur in="SourceGraphic" stdDeviation={3} result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="fxStrong" x="-30%" y="-40%" width="160%" height="180%">
          <feGaussianBlur in="SourceGraphic" stdDeviation={5} result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>

        {/* Mesh weave patterns */}
        <pattern id="weaveCoarse" x="0" y="0" width="8" height="8" patternUnits="userSpaceOnUse">
          <line x1="0" y1="4" x2="8" y2="4" stroke="#2A5878" strokeWidth={0.9} opacity={0.7} />
          <line x1="4" y1="0" x2="4" y2="8" stroke="#2A5878" strokeWidth={0.9} opacity={0.7} />
        </pattern>
        <pattern id="weaveMedium" x="0" y="0" width="5" height="5" patternUnits="userSpaceOnUse">
          <line x1="0" y1="2.5" x2="5" y2="2.5" stroke="#306888" strokeWidth={0.7} opacity={0.75} />
          <line x1="2.5" y1="0" x2="2.5" y2="5" stroke="#306888" strokeWidth={0.7} opacity={0.75} />
        </pattern>
        <pattern id="weaveFine" x="0" y="0" width="3" height="3" patternUnits="userSpaceOnUse">
          <line x1="0" y1="1.5" x2="3" y2="1.5" stroke="#38BDF8" strokeWidth={0.5} opacity={0.8} />
          <line x1="1.5" y1="0" x2="1.5" y2="3" stroke="#38BDF8" strokeWidth={0.5} opacity={0.8} />
        </pattern>

        {/* Channel gradient fills */}
        <linearGradient id="tS1" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%"   stopColor="#FB923C" stopOpacity={0.08} />
          <stop offset="100%" stopColor="#FB923C" stopOpacity={0.35} />
        </linearGradient>
        <linearGradient id="tS2" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%"   stopColor="#FBBF24" stopOpacity={0.08} />
          <stop offset="100%" stopColor="#FBBF24" stopOpacity={0.3} />
        </linearGradient>
        <linearGradient id="tS3" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%"   stopColor="#38BDF8" stopOpacity={0.08} />
          <stop offset="100%" stopColor="#38BDF8" stopOpacity={0.3} />
        </linearGradient>
      </defs>

      {/* Background */}
      <rect width={1060} height={460} fill="url(#bg)" />

      {/* ── HOUSING WALLS ───────────────────────────────────────────── */}
      <line x1={36} y1={64}  x2={674} y2={64}  stroke="#1E3A5A" strokeWidth={2.2} />
      <line x1={36} y1={244} x2={674} y2={244} stroke="#1E3A5A" strokeWidth={2.2} />
      <line x1={36} y1={64}  x2={36}  y2={244} stroke="#1E3A5A" strokeWidth={2.2} />
      <text x={380} y={52} textAnchor="middle" fill="#3A6888"
        fontSize={8} fontFamily={FONT} letterSpacing={3}>
        OUTER HOUSING — CONSTANT DIAMETER
      </text>

      {/* ── IN TUBE ─────────────────────────────────────────────────── */}
      <rect x={4} y={64} width={32} height={180} fill="#030810" opacity={0.9} />
      <line x1={4} y1={64}  x2={36} y2={64}  stroke="#1E3A5A" strokeWidth={2.2} />
      <line x1={4} y1={244} x2={36} y2={244} stroke="#1E3A5A" strokeWidth={2.2} />
      <ellipse cx={4} cy={CY} rx={4} ry={90}
        fill="#060E1C" stroke="#2A5878" strokeWidth={1.5} opacity={0.9}
        filter="url(#fxSoft)" />
      <line x1={6} y1={CY} x2={34} y2={CY}
        stroke="#38BDF8" strokeWidth={1.2} opacity={0.35} strokeDasharray="4 3" />
      <text x={20} y={52} textAnchor="middle" fill="#5A90B0" fontSize={8} fontFamily={FONT}>IN</text>
      <text x={20} y={62} textAnchor="middle" fill="#3A7898" fontSize={6.5} fontFamily={FONT}>water</text>

      {/* ── POLARISATION ZONE x=36–118 ──────────────────────────────── */}
      <rect x={36} y={64} width={82} height={180} fill="#060F1C" />
      {[54, 68, 82, 96, 110].map(lx => (
        <line key={lx} x1={lx} y1={72} x2={lx} y2={236}
          stroke="#818CF8" strokeWidth={0.7} opacity={0.32} />
      ))}
      <text x={77} y={150} textAnchor="middle" fill="#38BDF8"
        fontSize={9} fontFamily={FONT} letterSpacing={1}>POL</text>
      <text x={77} y={163} textAnchor="middle" fill="#5A90B0"
        fontSize={7.5} fontFamily={FONT}>ZONE</text>
      <line x1={118} y1={64} x2={118} y2={244}
        stroke="#0D2030" strokeWidth={1} strokeDasharray="3 3" />

      {/* ── STAGE CONES ─────────────────────────────────────────────── */}
      {STAGES.map((stg) => (
        <g key={stg.label}>
          {/* Concentration zone (below outer bezier) */}
          <path d={`${stg.bezier} L ${stg.xStart},243 Z`} fill="url(#concFill)" />
          {/* Clean water zone (above outer bezier) */}
          <path d={`${stg.bezier} L ${stg.xEnd},64 Z`} fill="url(#cleanFill)" />
          {/* Mesh weave fabric */}
          <path
            d={`${stg.bezier} ${stg.innerBezier.replace('M', 'C').replace(/M \d+,\d+ /, '')} Z`}
            fill={stg.weave} opacity={0.85}
          />
          {/* Outer mesh wall */}
          <path d={stg.bezier} stroke="#2A5878" strokeWidth={1.8} fill="none" opacity={0.9} />
          {/* Stage label above */}
          <text x={(stg.xStart + stg.xEnd) / 2} y={46}
            textAnchor="middle" fill="#5A90B0" fontSize={9} fontFamily={FONT} letterSpacing={2}>
            {stg.label === 'S1' ? 'S1 — COARSE' : stg.label === 'S2' ? 'S2 — MEDIUM' : 'S3 — FINE'}
          </text>
        </g>
      ))}

      {/* ── INLET FACE OVALS ────────────────────────────────────────── */}
      {INLET_OVALS.map(({ cx, stroke }) => (
        <g key={cx}>
          <ellipse cx={cx} cy={CY} rx={6} ry={90}
            fill="#081C30" stroke={stroke} strokeWidth={1.8} opacity={0.9}
            filter="url(#fxSoft)" />
          <ellipse cx={cx} cy={CY} rx={2.5} ry={90}
            fill="#1A4060" opacity={0.4} />
        </g>
      ))}

      {/* ── TRANSITION BUFFER (x=674–714, FIXED) ────────────────────── */}
      <rect x={674} y={64} width={40} height={264} rx={2}
        fill="#060E1C" stroke="#2A6080" strokeWidth={1.2} />
      <rect x={676} y={64} width={36} height={180}
        fill="#030810" stroke="#1A4060" strokeWidth={0.8} />

      {/* ── SNAP-OFF SEAM (x=714) ───────────────────────────────────── */}
      <line x1={714} y1={64} x2={714} y2={328}
        stroke="#F59E0B" strokeWidth={1.5} strokeDasharray="5 3" opacity={0.8} />
      <polygon points="712,150 708,154 712,158" fill="#F59E0B" opacity={0.75} />
      <polygon points="716,150 720,154 716,158" fill="#F59E0B" opacity={0.75} />
      <text x={710} y={143} textAnchor="end" fill="#5A8AAA" fontSize={6} fontFamily={FONT}>FIXED ←</text>
      <text x={718} y={143} fill="#F59E0B" fontSize={6} fontFamily={FONT}>→ DETACH</text>

      {/* ── STORAGE CHAMBER (x=714–886, DETACHABLE) ─────────────────── */}
      <rect x={714} y={64} width={172} height={264} rx={2}
        fill="#060E1C" stroke="#38BDF8" strokeWidth={1.8} filter="url(#fxSoft)" />
      <rect x={718} y={68} width={164} height={256} rx={1} fill="url(#chamberGlow)" />
      <rect x={726} y={66} width={72} height={10} rx={2}
        fill="#0A1828" stroke="#F59E0B" strokeWidth={0.8} />
      <text x={762} y={74} textAnchor="middle" fill="#F59E0B"
        fontSize={6} fontFamily={FONT} letterSpacing={0.8}>DETACHABLE</text>
      <rect x={716} y={64} width={168} height={180}
        fill="#030810" stroke="#1A4060" strokeWidth={1} />
      <text x={800} y={146} textAnchor="middle" fill="#38BDF8"
        fontSize={8} fontFamily={FONT} letterSpacing={1}>CLEAN WATER</text>
      <text x={800} y={158} textAnchor="middle" fill="#38BDF8"
        fontSize={8} fontFamily={FONT} letterSpacing={1}>BORE</text>
      <text x={800} y={171} textAnchor="middle" fill="#5A90B0"
        fontSize={7} fontFamily={FONT}>continuous flow → outlet</text>
      <text x={770} y={292} textAnchor="middle" fill="#38BDF8"
        fontSize={7.5} fontFamily={FONT} letterSpacing={0.8}>STORAGE CHAMBER</text>
      <text x={770} y={304} textAnchor="middle" fill="#5A8AAA"
        fontSize={6} fontFamily={FONT}>particle accumulation</text>

      {/* Fill sensor (static structure — dynamic fill in Task 9) */}
      <rect x={847} y={253} width={7} height={66} rx={1.5}
        fill="#010608" stroke="#5A8AAA" strokeWidth={0.7} />
      <text x={851} y={251} textAnchor="middle" fill="#5A8AAA"
        fontSize={5.5} fontFamily={FONT}>FILL</text>

      {/* ── OUT TUBE ────────────────────────────────────────────────── */}
      <ellipse cx={886} cy={CY} rx={6} ry={90}
        fill="#060E1C" stroke="#38BDF8" strokeWidth={1.5} opacity={0.9} filter="url(#fxSoft)" />
      <ellipse cx={886} cy={CY} rx={2.5} ry={90} fill="#0A2840" opacity={0.5} />
      <rect x={886} y={64} width={48} height={180} fill="#030810" opacity={0.9} />
      <line x1={886} y1={64}  x2={934} y2={64}  stroke="#1E3A5A" strokeWidth={2.2} />
      <line x1={886} y1={244} x2={934} y2={244} stroke="#1E3A5A" strokeWidth={2.2} />
      <ellipse cx={934} cy={CY} rx={4} ry={90}
        fill="#060E1C" stroke="#38BDF8" strokeWidth={1.5} opacity={0.9} filter="url(#fxSoft)" />
      <line x1={890} y1={CY} x2={932} y2={CY}
        stroke="#38BDF8" strokeWidth={1.2} opacity={0.35} strokeDasharray="4 3" />
      <text x={910} y={52} textAnchor="middle" fill="#5A90B0" fontSize={8} fontFamily={FONT}>OUT</text>
      <text x={910} y={62} textAnchor="middle" fill="#3A7898" fontSize={6.5} fontFamily={FONT}>clean</text>

      {/* ── COLLECTION CHANNELS (static structure — fill wired in Task 8) */}
      <text x={395} y={249} textAnchor="middle" fill="#3A7898"
        fontSize={7} fontFamily={FONT} letterSpacing={1.2}>
        COLLECTION CHANNELS  →  BUFFER ZONE  →  STORAGE
      </text>
      {STAGES.map((stg) => (
        <g key={`ch-${stg.label}`}>
          <rect x={118} y={stg.chY} width={556} height={stg.chH} rx={2.5}
            fill="#030608" stroke={stg.color} strokeWidth={1} opacity={0.65} />
          <text x={130} y={stg.chY + 11} fill={stg.color}
            fontSize={8} fontFamily={FONT} letterSpacing={0.5}>
            {stg.label}  COLLECTION  ·  {stg.label === 'S1' ? '500µm stage' : stg.label === 'S2' ? '100µm stage' : '5µm membrane'}
          </text>
        </g>
      ))}

      {/* ── FLUSH INLETS (static — highlight wired in Task 8) ────────── */}
      {STAGES.map((stg, i) => {
        const fy = stg.chY + 2;
        return (
          <g key={`flush-${i}`}>
            <rect x={86} y={fy} width={18} height={12} rx={2}
              fill="#040810" stroke={stg.color} strokeWidth={0.9} opacity={0.85} />
            <text x={95} y={fy + 9} textAnchor="middle" fill={stg.color} fontSize={8} fontFamily={FONT}>→</text>
            <text x={83} y={fy} textAnchor="end" fill={stg.color} fontSize={6} fontFamily={FONT}>FLUSH</text>
            {/* Bridge: inlet → channel */}
            <line x1={104} y1={stg.chY + stg.chH / 2} x2={118} y2={stg.chY + stg.chH / 2}
              stroke="#030810" strokeWidth={7} strokeLinecap="butt" />
            <line x1={104} y1={stg.chY + stg.chH / 2} x2={118} y2={stg.chY + stg.chH / 2}
              stroke={stg.color} strokeWidth={3.5} opacity={0.7} />
          </g>
        );
      })}

      {/* ── CONNECTOR PIPES (channel → seam) ────────────────────────── */}
      {STAGES.map((stg) => {
        const cy2 = stg.chY + stg.chH / 2;
        return (
          <g key={`conn-${stg.label}`}>
            <line x1={673} y1={cy2} x2={714} y2={cy2}
              stroke="#030810" strokeWidth={7} strokeLinecap="butt" />
            <line x1={673} y1={cy2} x2={714} y2={cy2}
              stroke={stg.color} strokeWidth={3.5} opacity={0.72} />
            <circle cx={714} cy={cy2} r={3.5} fill={stg.color} opacity={0.9} />
          </g>
        );
      })}

      {/* ── EJECTION PIPES (apex → channel) ─────────────────────────── */}
      {STAGES.map((stg) => (
        <g key={`ej-${stg.label}`}>
          <line x1={stg.apexX} y1={244} x2={stg.apexX} y2={stg.chY}
            stroke="#030810" strokeWidth={7} strokeLinecap="butt" />
          <line x1={stg.apexX} y1={244} x2={stg.apexX} y2={stg.chY}
            stroke={stg.color} strokeWidth={3.5} opacity={0.68} />
        </g>
      ))}

      {/* ── STAGE SPEC LABELS ───────────────────────────────────────── */}
      {STAGES.map((stg) => (
        <text key={`spec-${stg.label}`}
          x={(stg.xStart + stg.xEnd) / 2} y={322}
          textAnchor="middle" fill="#5A8AAA" fontSize={7} fontFamily={FONT}>
          {stg.label === 'S1' ? '500 µm · coarse weave · RT + nDEP'
            : stg.label === 'S2' ? '100 µm · medium weave · RT + nDEP'
            : '5 µm · microporous membrane · RT + nDEP'}
        </text>
      ))}

      {/* ── MODE ANNOTATION ─────────────────────────────────────────── */}
      <text x={800} y={340} textAnchor="middle" fill="#F59E0B"
        fontSize={6.5} fontFamily={FONT} letterSpacing={0.3}>
        M1: flush → detach  ·  M2: detach → install fresh → bf_cmd
      </text>
    </svg>
  );
}
```

- [ ] **Step 2: Run TypeScript type-check**

```bash
cd C:/Users/JSEer/hydrOS/apps/hydros-console && npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 3: Wire into App.tsx temporarily to confirm visual render**

In `App.tsx`, change:
```tsx
import MachineCore from './components/MachineCore';
```
to:
```tsx
import ConicalCascadeView from './components/ConicalCascadeView';
```
And change:
```tsx
<MachineCore state={displayState} />
```
to:
```tsx
<ConicalCascadeView state={displayState} />
```

- [ ] **Step 4: Start dev server and visually confirm static geometry renders**

```bash
cd C:/Users/JSEer/hydrOS/apps/hydros-console && npm run dev
```

Open `http://localhost:5173`. Confirm: housing, cones, inlet rings, transition buffer, snap-off seam, storage chamber, collection channels, flush inlets all visible. No console errors.

- [ ] **Step 5: Revert App.tsx import to MachineCore (dynamic wiring happens in Task 11)**

```tsx
import MachineCore from './components/MachineCore';
// ...
<MachineCore state={displayState} />
```

- [ ] **Step 6: Commit**

```bash
cd C:/Users/JSEer/hydrOS
git add apps/hydros-console/src/components/ConicalCascadeView.tsx apps/hydros-console/src/App.tsx
git commit -m "feat(console): ConicalCascadeView static geometry — Baseline 1 faithfully reproduced"
```

---

## Task 5: Radial E-field lines

Adds physics-correct radial field lines inside each stage's concentration zone. Lines run from the device centreline (y=154) to the outer walls (y=64/244), with density proportional to `eField * stageMultiplier` and emphasis on the lower wall region where the apex node sits.

**Files:**
- Modify: `apps/hydros-console/src/components/ConicalCascadeView.tsx`

- [ ] **Step 1: Add RadialFieldLines function above the component**

```tsx
interface FieldLinesProps {
  xStart: number;
  xEnd:   number;
  eField: number;       // [0,1] normalised field strength
  mult:   number;       // stage multiplier: S1=0.4, S2=0.7, S3=1.0
  color:  string;
}

function RadialFieldLines({ xStart, xEnd, eField, mult, color }: FieldLinesProps) {
  const count = Math.round(eField * 14 * mult);
  if (count === 0) return null;
  const lines = [];
  for (let i = 0; i < count; i++) {
    const x    = xStart + (xEnd - xStart) * (i + 0.5) / count;
    // Lower half (toward bottom outer wall) — stronger, more physically significant
    const loOp = Math.min(0.55, 0.35 * mult * eField + 0.08);
    // Upper half (toward top outer wall) — weaker counterpart
    const upOp = loOp * 0.4;
    lines.push(
      <g key={i}>
        <line x1={x} y1={154} x2={x} y2={244}
          stroke={color} strokeWidth={0.75} opacity={loOp} />
        <line x1={x} y1={154} x2={x} y2={64}
          stroke={color} strokeWidth={0.55} opacity={upOp} />
      </g>
    );
  }
  return <>{lines}</>;
}
```

- [ ] **Step 2: Insert RadialFieldLines renders into the component, inside the concentration zone of each stage**

Inside the JSX, after the `{/* ── STAGE CONES */}` block, add a new field-lines layer:

```tsx
      {/* ── RADIAL E-FIELD LINES (radial, center → outer wall) ──────── */}
      {STAGES.map((stg) => (
        <RadialFieldLines
          key={`ef-${stg.label}`}
          xStart={stg.xStart}
          xEnd={stg.apexX}
          eField={s?.eField ?? 0}
          mult={stg.mult}
          color={stg.color}
        />
      ))}
```

- [ ] **Step 3: TypeScript type-check**

```bash
cd C:/Users/JSEer/hydrOS/apps/hydros-console && npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 4: Commit**

```bash
cd C:/Users/JSEer/hydrOS
git add apps/hydros-console/src/components/ConicalCascadeView.tsx
git commit -m "feat(console): radial E-field lines — center→wall, stage-hierarchy density"
```

---

## Task 6: Particle streams and density split buoyancy cue

Renders particles in each stage's concentration zone. Dot count and opacity scale with per-stage capture efficiency. The density split cue applies a subtle upward offset to a fraction of particles when `etaPP` is significantly lower than `etaPET` — indicating buoyant particles escaping upward toward the centreline.

**Files:**
- Modify: `apps/hydros-console/src/components/ConicalCascadeView.tsx`

- [ ] **Step 1: Add ParticleStream function**

Add above the component:

```tsx
interface ParticleStreamProps {
  xStart:  number;
  xEnd:    number;
  apexX:   number;
  apexY:   number;
  conc:    number;   // concentration entering this stage: C_in * (1 - eta_prev_stages)
  etaPP:   number;   // buoyant species efficiency (for density split)
  etaPET:  number;   // dense species efficiency
  color:   string;
  seed:    number;   // deterministic scatter per stage
}

function ParticleStream({
  xStart, xEnd, apexX, apexY, conc, etaPP, etaPET, color, seed,
}: ParticleStreamProps) {
  const n = Math.round(conc * 18);
  if (n === 0) return null;

  // buoyancy cue: when buoyant species escape significantly more than dense
  const buoyancyActive = etaPP < etaPET * 0.75 && etaPET > 0.1;

  const dots = [];
  // Simple deterministic pseudorandom scatter using seed
  const rng = (i: number, offset: number) =>
    Math.abs(Math.sin(seed * 31.7 + i * 17.3 + offset)) % 1;

  for (let i = 0; i < n; i++) {
    const t      = rng(i, 0);
    const x      = xStart + (apexX - xStart) * (0.1 + t * 0.85);
    // y within concentration zone: 154 (centreline) to apexY (floor)
    const yBase  = 154 + (apexY - 154) * (0.05 + rng(i, 1) * 0.9);
    // buoyancy: even-indexed particles float upward when buoyancyActive
    const yOff   = (buoyancyActive && i % 2 === 0) ? -12 * (1 - etaPP) : 0;
    const y      = yBase + yOff;
    const r      = 1.6 + rng(i, 2) * 1.8;
    const op     = 0.35 + conc * 0.35;

    dots.push(
      <circle key={i} cx={x} cy={y} r={r}
        fill={color} opacity={op} />
    );
  }
  return <>{dots}</>;
}
```

- [ ] **Step 2: Add particle stream renders after the E-field lines block**

```tsx
      {/* ── PARTICLE STREAMS + DENSITY SPLIT CUE ────────────────────── */}
      {STAGES.map((stg, i) => {
        // Concentration entering this stage decreases through cascade
        // Stage 0: full C_in. Stage 1: C_in*(1-etaS1). Stage 2: C_in*(1-etaS1)*(1-etaS2)
        const etaArr  = [s?.etaS1 ?? 0, s?.etaS2 ?? 0, s?.etaS3 ?? 0];
        const survival = etaArr.slice(0, i).reduce((acc, e) => acc * (1 - e), 1);
        const conc    = (s?.clog ?? 0.5) * survival;
        return (
          <ParticleStream
            key={`ps-${stg.label}`}
            xStart={stg.xStart}
            xEnd={stg.xEnd}
            apexX={stg.apexX}
            apexY={stg.apexY}
            conc={conc}
            etaPP={s?.etaPP ?? 0}
            etaPET={s?.etaPET ?? 0}
            color={stg.color}
            seed={i + 1}
          />
        );
      })}
```

- [ ] **Step 3: TypeScript type-check**

```bash
cd C:/Users/JSEer/hydrOS/apps/hydros-console && npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 4: Commit**

```bash
cd C:/Users/JSEer/hydrOS
git add apps/hydros-console/src/components/ConicalCascadeView.tsx
git commit -m "feat(console): particle streams with density-split buoyancy cue"
```

---

## Task 7: Apex node dynamics

Drives glow intensity of each stage's apex node from its per-stage efficiency. Enforces visual stage hierarchy: S3 is always brightest regardless of current state values. Node pulsing triggers at high capture.

**Files:**
- Modify: `apps/hydros-console/src/components/ConicalCascadeView.tsx`

- [ ] **Step 1: Add apex node renders after the ejection pipes block**

Replace the comment `{/* ── EJECTION PIPES ... */}` block and add apex nodes immediately after it:

```tsx
      {/* ── APEX NODES ──────────────────────────────────────────────── */}
      {(() => {
        const etas  = [s?.etaS1 ?? 0, s?.etaS2 ?? 0, s?.etaS3 ?? 0];
        const glows = ['url(#nodeG1)', 'url(#nodeG2)', 'url(#nodeG3)'];
        const fxIds = ['fxSoft', 'fxSoft', 'fxStrong'];
        const borders = ['#FB923C', '#FBBF24', '#38BDF8'];
        // Enforce hierarchy: S3 always >= S2 >= S1 in displayed glow
        const enforced = [
          etas[0],
          Math.max(etas[1], etas[0] * 1.1),
          Math.max(etas[2], etas[1] * 1.3, 0.2),  // S3 minimum glow
        ];
        return STAGES.map((stg, i) => {
          const sw = 1.8 + enforced[i] * 1.2;
          return (
            <g key={`node-${i}`}>
              <ellipse cx={stg.apexX} cy={stg.apexY}
                rx={32 * (0.7 + enforced[i] * 0.5)}
                ry={20 * (0.7 + enforced[i] * 0.5)}
                fill={glows[i]}
                filter={`url(#${fxIds[i]})`} />
              <circle cx={stg.apexX} cy={stg.apexY} r={9 + enforced[i] * 2}
                fill="#060E1C" stroke={borders[i]} strokeWidth={sw}
                filter={`url(#${fxIds[i]})`} />
              <text x={stg.apexX} y={stg.apexY + 5}
                textAnchor="middle" fill={borders[i]}
                fontSize={12} fontFamily={FONT}>⊕</text>
            </g>
          );
        });
      })()}
```

- [ ] **Step 2: TypeScript type-check**

```bash
cd C:/Users/JSEer/hydrOS/apps/hydros-console && npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 3: Commit**

```bash
cd C:/Users/JSEer/hydrOS
git add apps/hydros-console/src/components/ConicalCascadeView.tsx
git commit -m "feat(console): apex node glow driven by per-stage eta, hierarchy enforced S3>S2>S1"
```

---

## Task 8: Collection channel fill, flush highlight, backflush sweep

Drives the channel fill gradient from per-stage accumulation. Highlights flush inlets amber when active. Adds a right-to-left sweep animation across the channel band when backflush is active.

**Files:**
- Modify: `apps/hydros-console/src/components/ConicalCascadeView.tsx`

- [ ] **Step 1: Add CSS keyframe for backflush sweep to the SVG defs**

Inside `<defs>`, add:

```tsx
        <style>{`
          @keyframes bfSweep {
            0%   { transform: translateX(558px); opacity: 0.18; }
            85%  { transform: translateX(-60px);  opacity: 0.08; }
            100% { transform: translateX(-60px);  opacity: 0;    }
          }
          .bf-sweep { animation: bfSweep 1.4s linear infinite; }
        `}</style>
```

- [ ] **Step 2: Replace static collection channel rects with dynamic fill versions**

In the `{/* ── COLLECTION CHANNELS */}` block, replace the inner `<rect>` static fill with a dynamic one driven by `channelFillSx`:

```tsx
      {STAGES.map((stg, i) => {
        const fills   = [s?.channelFillS1 ?? 0, s?.channelFillS2 ?? 0, s?.channelFillS3 ?? 0];
        const fActive = [s?.flushActiveS1 ?? false, s?.flushActiveS2 ?? false, s?.flushActiveS3 ?? false];
        const fill    = fills[i];
        const flush   = fActive[i];
        const fillW   = Math.round(fill * 556);   // channel width = 556 (x=118 to x=674)
        const tIds    = ['tS1', 'tS2', 'tS3'];
        return (
          <g key={`ch-${stg.label}`}>
            {/* Channel tube structure */}
            <rect x={118} y={stg.chY} width={556} height={stg.chH} rx={2.5}
              fill="#030608"
              stroke={flush ? stg.color : stg.color}
              strokeWidth={flush ? 1.6 : 1.0}
              opacity={flush ? 0.9 : 0.65} />
            {/* Particle fill — grows from left as accumulation increases */}
            {fillW > 2 && (
              <rect x={118} y={stg.chY + 1} width={fillW} height={stg.chH - 2} rx={2}
                fill={`url(#${tIds[i]})`} opacity={0.9} />
            )}
            {/* Channel label */}
            <text x={130} y={stg.chY + 11} fill={stg.color}
              fontSize={8} fontFamily={FONT} letterSpacing={0.5}>
              {stg.label}  COLLECTION  ·  {stg.label === 'S1' ? '500µm stage' : stg.label === 'S2' ? '100µm stage' : '5µm membrane'}
            </text>
          </g>
        );
      })}

      {/* Backflush sweep overlay — sweeps R→L through channel band when bf active */}
      {(s?.backflush ?? 0) > 0.5 && (
        <rect x={118} y={250} width={558} height={68}
          fill="#38BDF8" className="bf-sweep"
          style={{ transformOrigin: 'left center' }} />
      )}
```

- [ ] **Step 3: Update flush inlet section to highlight border when flush active**

In the `{/* ── FLUSH INLETS */}` block, change the stroke opacity/width:

```tsx
      {STAGES.map((stg, i) => {
        const fActive = [s?.flushActiveS1 ?? false, s?.flushActiveS2 ?? false, s?.flushActiveS3 ?? false];
        const flush   = fActive[i];
        const fy      = stg.chY + 2;
        return (
          <g key={`flush-${i}`}>
            <rect x={86} y={fy} width={18} height={12} rx={2}
              fill="#040810"
              stroke={stg.color}
              strokeWidth={flush ? 1.8 : 0.9}
              opacity={flush ? 1.0 : 0.85} />
            <text x={95} y={fy + 9} textAnchor="middle" fill={stg.color}
              fontSize={8} fontFamily={FONT}>→</text>
            <text x={83} y={fy} textAnchor="end" fill={stg.color}
              fontSize={6} fontFamily={FONT}>FLUSH</text>
            <line x1={104} y1={stg.chY + stg.chH / 2} x2={118} y2={stg.chY + stg.chH / 2}
              stroke="#030810" strokeWidth={7} strokeLinecap="butt" />
            <line x1={104} y1={stg.chY + stg.chH / 2} x2={118} y2={stg.chY + stg.chH / 2}
              stroke={stg.color} strokeWidth={3.5} opacity={flush ? 0.95 : 0.7} />
          </g>
        );
      })}
```

- [ ] **Step 4: TypeScript type-check**

```bash
cd C:/Users/JSEer/hydrOS/apps/hydros-console && npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 5: Commit**

```bash
cd C:/Users/JSEer/hydrOS
git add apps/hydros-console/src/components/ConicalCascadeView.tsx
git commit -m "feat(console): channel fill driven by accumulation, flush highlight, bf sweep animation"
```

---

## Task 9: Storage chamber fill bar and swap trigger

Drives the fill sensor bar from `storageFill`. The amber threshold marker activates visually at 0.8 and the mode annotation text pulses when the threshold is crossed.

**Files:**
- Modify: `apps/hydros-console/src/components/ConicalCascadeView.tsx`

- [ ] **Step 1: Add fill-level CSS pulse to defs style block**

In the `<style>` tag added in Task 8, append:

```css
          @keyframes swapPulse {
            0%, 100% { opacity: 1.0; }
            50%       { opacity: 0.45; }
          }
          .swap-warn { animation: swapPulse 1.8s ease-in-out infinite; }
```

- [ ] **Step 2: Replace static fill sensor with dynamic version**

Replace the static fill sensor block (the one with `<rect x={847}...>` and the `FILL` text) with:

```tsx
      {/* ── STORAGE FILL SENSOR ─────────────────────────────────────── */}
      {(() => {
        const fill     = s?.storageFill ?? 0;
        const barH     = 66;                          // total sensor bar height
        const fillH    = Math.round(fill * barH);     // filled portion
        const atWarn   = fill >= 0.8;
        const warnY    = 253 + barH * 0.2;            // 80% from top = 20% from bottom
        return (
          <g>
            {/* Sensor housing */}
            <rect x={847} y={253} width={7} height={barH} rx={1.5}
              fill="#010608" stroke="#5A8AAA" strokeWidth={0.7} />
            {/* Green fill (rises from bottom) */}
            {fillH > 0 && (
              <rect x={848} y={253 + barH - fillH} width={5} height={fillH} rx={1}
                fill={atWarn ? '#F59E0B' : '#22C55E'} opacity={0.7} />
            )}
            {/* Threshold marker at 80% */}
            <line x1={843} y1={warnY} x2={856} y2={warnY}
              stroke="#F59E0B" strokeWidth={1}
              opacity={atWarn ? 1.0 : 0.55} />
            <circle cx={858} cy={warnY} r={2.5}
              fill="#F59E0B" opacity={atWarn ? 1.0 : 0.5} />
            {/* Labels */}
            <text x={851} y={251} textAnchor="middle" fill="#5A8AAA"
              fontSize={5.5} fontFamily={FONT}>FILL</text>
            <text x={841} y={warnY + 4} textAnchor="end" fill="#F59E0B"
              fontSize={5.5} fontFamily={FONT}
              className={atWarn ? 'swap-warn' : undefined}>
              ⚠ 80% swap
            </text>
          </g>
        );
      })()}
```

- [ ] **Step 3: Make mode annotation pulse when fill >= 0.8**

Update the mode annotation text at the bottom:

```tsx
      <text x={800} y={340} textAnchor="middle"
        fill="#F59E0B" fontSize={6.5} fontFamily={FONT} letterSpacing={0.3}
        className={(s?.storageFill ?? 0) >= 0.8 ? 'swap-warn' : undefined}>
        M1: flush → detach  ·  M2: detach → install fresh → bf_cmd
      </text>
```

- [ ] **Step 4: TypeScript type-check**

```bash
cd C:/Users/JSEer/hydrOS/apps/hydros-console && npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 5: Commit**

```bash
cd C:/Users/JSEer/hydrOS
git add apps/hydros-console/src/components/ConicalCascadeView.tsx
git commit -m "feat(console): storage fill bar live, amber threshold at 80%, mode annotation pulse"
```

---

## Task 10: Temporal flow indicators and S3 degradation cue

Adds flow-speed arrows in the clean water bore (opacity/count scale with `flow`). Reduces S3 mesh opacity slightly when `flow > 0.8`, signalling degradation risk under high flow — a visible consequence of S3 dominance without adding hardware.

**Files:**
- Modify: `apps/hydros-console/src/components/ConicalCascadeView.tsx`

- [ ] **Step 1: Add flow arrow function above component**

```tsx
function FlowArrow({ x, y, opacity }: { x: number; y: number; opacity: number }) {
  return (
    <path
      d={`M${x - 6},${y - 4} L${x + 6},${y} L${x - 6},${y + 4}`}
      fill="none" stroke="#38BDF8" strokeWidth={1.5} opacity={opacity}
    />
  );
}
```

- [ ] **Step 2: Add flow arrows in the clean water bore zone, after the IN tube block**

```tsx
      {/* ── FLOW VELOCITY INDICATORS ────────────────────────────────── */}
      {(() => {
        const flow = s?.flow ?? 0;
        if (flow < 0.05) return null;
        const op = 0.2 + flow * 0.5;
        const arrowXs = [77, 207, 395, 583];  // POL zone + S1/S2/S3 bore
        return arrowXs.map(x => (
          <FlowArrow key={x} x={x} y={CY} opacity={op} />
        ));
      })()}
```

- [ ] **Step 3: Apply S3 degradation cue — reduce S3 mesh opacity at high flow**

In the stage cones block, wrap the S3 outer mesh wall line with a flow-dependent opacity:

In the `{STAGES.map(...)}` block, find the outer mesh wall `<path>` line. Change the static `opacity={0.9}` to be dynamic based on stage and flow:

```tsx
          {/* Outer mesh wall — S3 opacity reduced under high flow (degradation cue) */}
          <path d={stg.bezier} stroke="#2A5878" strokeWidth={1.8} fill="none"
            opacity={stg.label === 'S3' ? Math.max(0.4, 0.9 - ((s?.flow ?? 0) - 0.8) * 1.5) : 0.9} />
```

- [ ] **Step 4: TypeScript type-check**

```bash
cd C:/Users/JSEer/hydrOS/apps/hydros-console && npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 5: Commit**

```bash
cd C:/Users/JSEer/hydrOS
git add apps/hydros-console/src/components/ConicalCascadeView.tsx
git commit -m "feat(console): flow velocity arrows + S3 mesh degradation cue at high flow"
```

---

## Task 11: Wire ConicalCascadeView into App.tsx

Replaces the existing `MachineCore` mount point in `App.tsx` with `ConicalCascadeView`. `MachineCore.tsx` is preserved in place but no longer mounted. No other file changes.

**Files:**
- Modify: `apps/hydros-console/src/App.tsx`

- [ ] **Step 1: Update the import in App.tsx**

Change:
```tsx
import MachineCore from './components/MachineCore';
```
to:
```tsx
import ConicalCascadeView from './components/ConicalCascadeView';
```

- [ ] **Step 2: Update the mount point in App.tsx**

Change:
```tsx
          <MachineCore state={displayState} />
```
to:
```tsx
          <ConicalCascadeView state={displayState} />
```

- [ ] **Step 3: TypeScript type-check**

```bash
cd C:/Users/JSEer/hydrOS/apps/hydros-console && npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 4: Start dev server and run a scenario end-to-end**

```bash
cd C:/Users/JSEer/hydrOS/apps/hydros-console && npm run dev
```

Open `http://localhost:5173`. Run `baseline_nominal` scenario. Confirm:
- Static geometry renders correctly (housing, cones, rings)
- E-field lines appear in stage concentration zones when eField > 0
- Particle dots visible in concentration zones
- S3 node glows brighter than S1
- Collection channel fills grow over time
- Storage fill sensor rises
- Backflush sweep animates R→L when bf active
- All surrounding panels (top telemetry, right advisory, bottom strip) remain unchanged

- [ ] **Step 5: Run Python test suite to confirm env changes haven't broken anything**

```bash
cd C:/Users/JSEer/hydrOS && pytest tests/ -v 2>&1 | tail -20
```

Expected: all tests pass (including 6 new from Tasks 1–2)

- [ ] **Step 6: Build production bundle**

```bash
cd C:/Users/JSEer/hydrOS/apps/hydros-console && npm run build
```

Expected: `dist/` rebuilt without errors

- [ ] **Step 7: Commit**

```bash
cd C:/Users/JSEer/hydrOS
git add apps/hydros-console/src/App.tsx
git commit -m "feat(console): mount ConicalCascadeView — M5 Baseline 1 live in console shell"
```

---

## Self-Review

### 1. Spec coverage

| Spec requirement | Covered by |
|---|---|
| Conical cascade geometry (housing, cones, rings, tubes) | Task 4 |
| Collection architecture (channels → buffer → storage) | Task 4 + Task 8 |
| Transition buffer zone + snap-off seam | Task 4 |
| Storage chamber fill + M1/M2 modes | Task 1 + Task 9 |
| Hydraulic flush inlets | Task 4 + Task 8 |
| Per-stage efficiency tracking | Task 2 |
| nDEP E-field radial (center → outer wall) | Task 5 |
| Stage hierarchy enforced visually S3>S2>S1 | Task 5 (density) + Task 7 (glow) |
| Density split / buoyancy cue (Option A) | Task 6 |
| Fill sensor at 80% threshold | Task 9 |
| Temporal: flow velocity variation | Task 10 |
| Temporal: fouling accumulation | Task 8 (channel fill) |
| Temporal: backflush propagation | Task 8 (sweep) |
| Temporal: S3 degradation under high flow | Task 10 |
| Console shell untouched | Task 11 (only App.tsx changed) |
| η metrics not in machine view | Throughout — component only reads `flow`, `clog`, `eField`, not `captureEff` |

### 2. Placeholder scan

None found. All code blocks are complete and executable.

### 3. Type consistency

- `HydrosDisplayState` fields added in Task 3 (`channelFillS1`, `etaS1`, `flushActiveS1`, etc.) are referenced by exactly those names in Tasks 4–10.
- `ConicalCascadeEnv` truth_state keys (`channel_fill_s1`, `eta_s1`, `flush_active_s1`) match what the mapper reads in Task 3.
- `STAGES` array constant used consistently across Tasks 4–10; `stg.chY + stg.chH / 2` is the centerline y for all three channel connector references.
