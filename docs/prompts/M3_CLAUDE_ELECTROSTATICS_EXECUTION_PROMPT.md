# M3 ELECTROSTATICS EXECUTION PROMPT
# HydrOS — Radial Field Model, obs12_v2

You are executing Milestone 3 of the HydrOS physics grounding operation.

This is NOT a research task.
This is NOT exploratory.

This is a **contained physics replacement** of one module — `electrostatics.py` —
and the downstream coupling updates required to make it function.

---

# Required Reading (Before Any Action)

Read these in full before touching any file:

1. `docs/reports/M3_ELECTROSTATIC_PHYSICS_CORRECTION_REPORT.md` — authoritative M3 scope
2. `docs/research/2026-04-10-physics-correction-report.md` — original Issue 1 diagnosis
3. `docs/context/06_LOCKED_SYSTEM_CONSTRAINTS.md` §D — voltage + 30/70 allocation (locked)
4. `docs/context/04_CURRENT_ENGINE_STATUS.md` §7 — current electrostatics state

---

# M1.5 Pre-flight (Verify Before Starting)

Confirm all of the following are true. If any fail, stop and fix M1.5 first.

```bash
cd C:/Users/JSEer/hydrOS
python -m pytest tests/ -v
```

Expected: **25/25 passing**

Spot-check YAML:
```bash
python -c "
import yaml
with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
e = cfg['electrostatics']
assert e['V_max_realism'] == 2500.0,  'V_max_realism wrong'
assert e['V_hard_clamp']  == 3000.0,  'V_hard_clamp wrong'
c = cfg['clogging']
assert c['dep_exponent'] == 1.0, 'dep_exponent wrong'
print('M1.5 pre-flight: OK')
"
```

---

# Execution Order

Tasks must run in sequence. Do not skip verification steps.

```
Task 1: Replace electrostatics.py         — radial field, two subsystems
Task 2: Update particles.py coupling      — E_norm → E_capture_gain
Task 3: Update sensor_fusion.py           — E_norm → E_field_norm, obs12_v2
Task 4: Update default.yaml               — new electrostatics parameters
Task 5: Validate + update docs            — all gates must pass
```

---

# Task 1 — Replace `hydrion/physics/electrostatics.py`

## What changes

- Remove `gap_m`, `E_norm_ref` from params
- Add `r_inner_m`, `r_outer_m`, `t_E_ref_s`, `V_ring_ref`, `ring_weight`, `node_weight`, `stage_volume_L`
- Field computation: `E_r(r) = V / (r × ln(r_outer/r_inner))` — radial, not axial
- Two subsystems: `InletPolarizationRing` (30%) + `OuterWallCollectorNode` (70%)
- New state outputs: `E_field_kVm`, `charge_factor`, `node_capture_gain`, `E_capture_gain`, `E_field_norm`
- Remove state outputs: `E_field` [V/m], `E_norm` [dimensionless]

## Full replacement

Write this file exactly:

```python
# hydrion/physics/electrostatics.py
"""
ElectrostaticsModel v2 — Radial field, two-subsystem architecture.

Replaces the axial scalar-gap model (v1) with a cylindrical radial field
model grounded in the concentric electrode geometry of the HydrOS device.

Architecture (06_LOCKED_SYSTEM_CONSTRAINTS.md §D):
    InletPolarizationRing  — 30% capture contribution, upstream charge conditioning
    OuterWallCollectorNode — 70% capture contribution, radial field at collection wall

Field geometry:
    E_r(r) = V / (r × ln(r_outer / r_inner))       [V/m]
    At collection wall (r = r_outer):
        E_r_wall = V / (r_outer × ln(r_outer / r_inner))
        E_field_kVm = E_r_wall / 1000.0             [kV/m]

Output:
    E_capture_gain ∈ [0, 1]  — normalized electrostatic capture boost
                               consumed by particles.py as a gain signal
    E_field_kVm              — physical field at collection wall [kV/m]
    E_field_norm             — E_field_kVm / E_field_kVm_max ∈ [0, 1]
                               replaces E_norm in obs12_v2 (index 3)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class ElectrostaticsParams:
    # Voltage bounds (locked system constraints — 06_LOCKED_SYSTEM_CONSTRAINTS.md §D)
    V_max_realism: float = 2500.0   # [V] upper operational bound
    V_hard_clamp:  float = 3000.0   # [V] absolute safety ceiling — never exceed

    # First-order voltage dynamics
    tau_charge: float = 0.5         # [s] time constant to reach command
    leak_rate:  float = 0.1         # [1/s] passive voltage decay

    # Radial geometry — concentric cylindrical capacitor
    r_inner_m: float = 0.005        # [m] counter-electrode (central rod) radius
    r_outer_m: float = 0.040        # [m] outer collection wall radius

    # Residence time model
    t_E_ref_s:      float = 2.0     # [s] residence time at which tanh → 0.76 saturation
    stage_volume_L: float = 0.25    # [L] effective stage volume (placeholder — bench calibration)

    # Ring conditioning reference
    V_ring_ref: float = 500.0       # [V] voltage at which ring tanh → 0.76 saturation

    # 30/70 functional allocation (locked — 06_LOCKED_SYSTEM_CONSTRAINTS.md §D)
    ring_weight: float = 0.30       # share of E capture from InletPolarizationRing
    node_weight: float = 0.70       # share of E capture from OuterWallCollectorNode

    eps: float = 1e-8


class ElectrostaticsModel:
    """
    ElectrostaticsModel v2

    Reads from state / actions:
        node_voltage_cmd  in [0, 1]   (action vector index 3)
        q_processed_lmin               (from hydraulics, for residence time)

    Writes to truth_state:
        V_node           [V]     actual node voltage after first-order dynamics
        E_field_kVm      [kV/m]  radial field at collection wall
        E_field_norm     []      E_field_kVm / E_field_kVm_max in [0, 1] (obs12_v2 index 3)
        charge_factor    []      InletPolarizationRing contribution in [0, ring_weight]
        node_capture_gain []     OuterWallCollectorNode contribution in [0, node_weight]
        E_capture_gain   []      total electrostatic capture boost in [0, 1]
    """

    def __init__(self, cfg: Any | None = None) -> None:
        e_raw: Dict[str, float] = {}
        if cfg is not None and hasattr(cfg, "raw"):
            e_raw = getattr(cfg, "raw", {}).get("electrostatics", {}) or {}

        self.params = ElectrostaticsParams(
            V_max_realism   = float(e_raw.get("V_max_realism",  2500.0)),
            V_hard_clamp    = float(e_raw.get("V_hard_clamp",   3000.0)),
            tau_charge      = float(e_raw.get("tau_charge",     0.5)),
            leak_rate       = float(e_raw.get("leak_rate",      0.1)),
            r_inner_m       = float(e_raw.get("r_inner_m",      0.005)),
            r_outer_m       = float(e_raw.get("r_outer_m",      0.040)),
            t_E_ref_s       = float(e_raw.get("t_E_ref_s",      2.0)),
            stage_volume_L  = float(e_raw.get("stage_volume_L", 0.25)),
            V_ring_ref      = float(e_raw.get("V_ring_ref",     500.0)),
            ring_weight     = float(e_raw.get("ring_weight",    0.30)),
            node_weight     = float(e_raw.get("node_weight",    0.70)),
            eps             = float(e_raw.get("eps",            1e-8)),
        )

        p = self.params
        # Precompute geometric constant: ln(r_outer / r_inner)
        self._ln_ratio = float(np.log(
            max(p.r_outer_m, p.eps) / max(p.r_inner_m, p.eps)
        ))
        # Precompute maximum field at V_max_realism for normalisation reference
        self._E_field_kVm_max = float(
            (p.V_max_realism / 1000.0) /
            max(p.r_outer_m * self._ln_ratio, p.eps)
        )

        self.state: Dict[str, float] = {}
        self._reset_state()

    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self.state = {
            "V_node":            0.0,
            "E_field_kVm":       0.0,
            "E_field_norm":      0.0,
            "charge_factor":     0.0,
            "node_capture_gain": 0.0,
            "E_capture_gain":    0.0,
        }

    def reset(self, state: Dict[str, float]) -> None:
        self._reset_state()
        state.update(self.state)

    def get_state(self) -> Dict[str, float]:
        return dict(self.state)

    # ------------------------------------------------------------------

    def update(
        self,
        state: Dict[str, float],
        dt: float,
        node_cmd: float,
    ) -> None:
        p = self.params
        node_cmd = float(np.clip(node_cmd, 0.0, 1.0))

        # ── Voltage dynamics (first-order approach with leakage) ──────────
        V_node = float(self.state.get("V_node", 0.0))
        V_target = float(np.clip(node_cmd * p.V_max_realism, 0.0, p.V_hard_clamp))
        dV = ((V_target - V_node) / max(p.tau_charge, p.eps) - p.leak_rate * V_node) * dt
        V_node = float(np.clip(V_node + dV, 0.0, p.V_hard_clamp))

        # ── Radial field at collection wall ───────────────────────────────
        # E_r(r_outer) = V / (r_outer × ln(r_outer / r_inner))
        E_r_wall_Vm = V_node / max(p.r_outer_m * self._ln_ratio, p.eps)
        E_field_kVm = E_r_wall_Vm / 1000.0
        E_field_norm = float(np.clip(
            E_field_kVm / max(self._E_field_kVm_max, p.eps), 0.0, 1.0
        ))

        # ── Residence time ────────────────────────────────────────────────
        Q_proc_Ls = max(float(state.get("q_processed_lmin", 0.0)) / 60.0, p.eps)
        t_residence_s = p.stage_volume_L / Q_proc_Ls
        t_sat = float(np.tanh(t_residence_s / max(p.t_E_ref_s, p.eps)))

        # ── SubSystem A: InletPolarizationRing (30%) ──────────────────────
        # Charge conditioning — scales with voltage and residence time.
        # Output in [0, ring_weight].
        V_norm = float(np.clip(V_node / max(p.V_ring_ref, p.eps), 0.0, 5.0))
        charge_factor = p.ring_weight * float(np.tanh(V_norm)) * t_sat

        # ── SubSystem B: OuterWallCollectorNode (70%) ─────────────────────
        # Primary capture via radial field.
        # Scales with normalised E-field and residence time.
        # Output in [0, node_weight].
        node_capture_gain = p.node_weight * E_field_norm * t_sat

        # ── Combined gain (normalised in [0, 1]) ──────────────────────────
        E_capture_gain = float(np.clip(charge_factor + node_capture_gain, 0.0, 1.0))

        # ── Write state ───────────────────────────────────────────────────
        self.state.update(
            V_node            = float(V_node),
            E_field_kVm       = float(E_field_kVm),
            E_field_norm      = float(E_field_norm),
            charge_factor     = float(charge_factor),
            node_capture_gain = float(node_capture_gain),
            E_capture_gain    = float(E_capture_gain),
        )
        state.update(self.state)
```

## Verify Task 1

```bash
python -c "
import sys; sys.path.insert(0, '.')
from hydrion.physics.electrostatics import ElectrostaticsModel

m = ElectrostaticsModel()
state = {}
m.reset(state)

# Test 1: zero voltage -> zero gain
for _ in range(100): m.update(state, dt=0.1, node_cmd=0.0)
s = m.get_state()
assert s['E_capture_gain'] < 0.01, f'V=0 gain not zero: {s[\"E_capture_gain\"]}'
assert s['E_field_kVm']    < 0.01, f'V=0 field not zero: {s[\"E_field_kVm\"]}'
print('Test 1 PASS: zero voltage -> zero gain')

# Test 2: full voltage -> E_field_kVm near 30.1 kV/m (analytical)
m2 = ElectrostaticsModel()
state2 = {'q_processed_lmin': 13.5}
m2.reset(state2)
for _ in range(200): m2.update(state2, dt=0.1, node_cmd=1.0)
s2 = m2.get_state()
assert 25.0 < s2['E_field_kVm'] < 36.0, f'E_field_kVm out of range: {s2[\"E_field_kVm\"]}'
assert s2['E_field_norm'] <= 1.0,         'E_field_norm > 1'
assert s2['V_node']       <= 3000.0,      'V_node exceeded V_hard_clamp'
print(f'Test 2 PASS: E_field_kVm={s2[\"E_field_kVm\"]:.1f} kV/m, E_capture_gain={s2[\"E_capture_gain\"]:.3f}')

# Test 3: higher flow -> lower capture gain (residence time effect)
m_low  = ElectrostaticsModel()
m_high = ElectrostaticsModel()
s_low  = {'q_processed_lmin': 5.0}
s_high = {'q_processed_lmin': 20.0}
m_low.reset(s_low); m_high.reset(s_high)
for _ in range(200):
    m_low.update(s_low,   dt=0.1, node_cmd=0.8)
    m_high.update(s_high, dt=0.1, node_cmd=0.8)
g_low  = m_low.get_state()['E_capture_gain']
g_high = m_high.get_state()['E_capture_gain']
assert g_low > g_high, f'Residence time effect failed: Q=5 gain={g_low:.3f} vs Q=20 gain={g_high:.3f}'
print(f'Test 3 PASS: Q=5 gain={g_low:.3f} > Q=20 gain={g_high:.3f}')

print('Task 1 verification: ALL PASS')
"
```

Expected:
```
Test 1 PASS: zero voltage -> zero gain
Test 2 PASS: E_field_kVm=30.x kV/m, E_capture_gain=...
Test 3 PASS: Q=5 gain=... > Q=20 gain=...
Task 1 verification: ALL PASS
```

---

# Task 2 — Update `hydrion/physics/particles.py`

## What changes

Lines 173–180: replace `E_norm` read with `E_capture_gain` read.

`alpha_E` in `ParticleParams` stays at 0.4 — it scales the normalised [0, 1] signal.
The electrostatics model now produces a bounded, physically meaningful signal.

## Edit

Find this block (lines 173–180 in the current file):

```python
        E_norm = 0.0
        if electrostatics_model is not None:
            E_norm = float(electrostatics_model.get_state().get("E_norm", 0.0))
        else:
            E_norm = float(state.get("E_norm", 0.0))

        # Boost capture efficiency with clogging + electrostatics
        capture_eff = capture_eff_base + p.alpha_clog * mesh_avg + p.alpha_E * E_norm
```

Replace with:

```python
        E_capture_gain = 0.0
        if electrostatics_model is not None:
            E_capture_gain = float(electrostatics_model.get_state().get("E_capture_gain", 0.0))
        else:
            E_capture_gain = float(state.get("E_capture_gain", 0.0))

        # Boost capture efficiency with clogging + electrostatics
        # E_capture_gain in [0, 1] — normalised signal from ElectrostaticsModel v2
        capture_eff = capture_eff_base + p.alpha_clog * mesh_avg + p.alpha_E * E_capture_gain
```

## Verify Task 2

```bash
python -c "
import sys; sys.path.insert(0, '.')
from hydrion.physics.particles import ParticleModel
from hydrion.physics.electrostatics import ElectrostaticsModel

pm     = ParticleModel()
em_on  = ElectrostaticsModel()
em_off = ElectrostaticsModel()

base = {'q_processed_lmin': 13.5, 'mesh_loading_avg': 0.0, 'capture_eff': 0.8, 'C_in': 0.7}
s_on  = dict(base)
s_off = dict(base)

em_on.reset(s_on);   em_off.reset(s_off)
pm.reset(s_on);      pm.reset(s_off)

for _ in range(200): em_on.update(s_on,   dt=0.1, node_cmd=1.0)
for _ in range(200): em_off.update(s_off, dt=0.1, node_cmd=0.0)

pm.update(s_on,  dt=0.1, electrostatics_model=em_on)
pm.update(s_off, dt=0.1, electrostatics_model=em_off)

eff_on  = s_on['particle_capture_eff']
eff_off = s_off['particle_capture_eff']
assert eff_on > eff_off, f'Voltage ON should increase capture: {eff_on:.3f} vs {eff_off:.3f}'
print(f'Task 2 PASS: capture_eff ON={eff_on:.3f} > OFF={eff_off:.3f}')
"
```

Expected: `Task 2 PASS: capture_eff ON=... > OFF=...`

---

# Task 3 — Update `hydrion/sensors/sensor_fusion.py`

## What changes

- Line 24: `truth.get("E_norm", 0.0)` → `truth.get("E_field_norm", 0.0)`
- Docstring: schema label `Commit 3` / `obs12_v1` → `obs12_v2` with full index map

## Full replacement

Write this file exactly:

```python
# hydrion/sensors/sensor_fusion.py
from __future__ import annotations
import numpy as np


def build_observation(truth: dict, sensors: dict) -> np.ndarray:
    """
    obs12_v2 — stable observation contract.

    Builds the 12D observation vector strictly from:
    - normalized truth_state values
    - measured sensor_state values

    Schema version: obs12_v2 (M3 — 2026-04-10)
    Change from v1: index 3 is now E_field_norm (physical kV/m normalised to [0,1])
                    replacing E_norm (dimensionless arbitrary reference in [0,2]).

    Index mapping:
        0   flow
        1   pressure
        2   clog
        3   E_field_norm        <- obs12_v2 (was E_norm in obs12_v1)
        4   C_out
        5   particle_capture_eff
        6   valve_cmd
        7   pump_cmd
        8   bf_cmd
        9   node_voltage_cmd
        10  sensor_turbidity
        11  sensor_scatter

    This function is the *single source of truth* for the RL observation.
    DO NOT change index ordering without bumping the schema version label.
    """
    return np.array(
        [
            # hydraulics (normalized)
            float(truth.get("flow", 0.0)),
            float(truth.get("pressure", 0.0)),
            float(truth.get("clog", 0.0)),

            # electrostatics (obs12_v2: E_field_norm replaces E_norm)
            float(truth.get("E_field_norm", 0.0)),

            # particle transport
            float(truth.get("C_out", 0.0)),
            float(truth.get("particle_capture_eff", 0.0)),

            # actuator commands
            float(truth.get("valve_cmd", 0.0)),
            float(truth.get("pump_cmd", 0.0)),
            float(truth.get("bf_cmd", 0.0)),
            float(truth.get("node_voltage_cmd", 0.0)),

            # optical sensors (measured)
            float(sensors.get("sensor_turbidity", 0.0)),
            float(sensors.get("sensor_scatter", 0.0)),
        ],
        dtype=np.float32,
    )
```

## Verify Task 3

```bash
python -c "
import sys; sys.path.insert(0, '.')
from hydrion.sensors.sensor_fusion import build_observation
import numpy as np

truth = {
    'flow': 0.5, 'pressure': 0.3, 'clog': 0.1,
    'E_field_norm': 0.75,
    'C_out': 0.1, 'particle_capture_eff': 0.85,
    'valve_cmd': 1.0, 'pump_cmd': 0.8, 'bf_cmd': 0.0, 'node_voltage_cmd': 0.5,
}
sensors = {'sensor_turbidity': 0.2, 'sensor_scatter': 0.1}

obs = build_observation(truth, sensors)
assert obs.shape == (12,),          f'Wrong shape: {obs.shape}'
assert abs(obs[3] - 0.75) < 1e-6,  f'Index 3 wrong: {obs[3]}'
assert 0.0 <= obs[3] <= 1.0,       f'E_field_norm out of [0,1]: {obs[3]}'
print(f'Task 3 PASS: obs[3]={obs[3]:.3f} (E_field_norm), shape={obs.shape}')
"
```

Expected: `Task 3 PASS: obs[3]=0.750 (E_field_norm), shape=(12,)`

---

# Task 4 — Update `configs/default.yaml`

## What changes

Replace the `electrostatics:` section entirely. Remove `gap_m` and `E_norm_ref`. Add radial geometry, residence time, and 30/70 allocation parameters.

## Find and replace

**Remove this block:**
```yaml
electrostatics:
  V_max_realism: 2500.0   # [V] upper operational bound (locked: 06_LOCKED_SYSTEM_CONSTRAINTS.md)
  V_hard_clamp:  3000.0   # [V] absolute safety ceiling — system must never exceed this
  tau_charge:    0.5      # [s] first-order voltage rise time constant
  leak_rate:     0.1      # [1/s] passive voltage decay
  gap_m:         0.01     # [m] characteristic electrode gap (interim — replaced by radial model in M3)
  E_norm_ref:    3.0e5    # [V/m] normalization reference for E_norm observation (~300 kV/m at V_hard_clamp)
```

**Replace with:**
```yaml
# ---------------------------------------------------------------------------
# Electrostatics — M3: radial field model (obs12_v2)
# ---------------------------------------------------------------------------
electrostatics:
  # Voltage bounds (locked — 06_LOCKED_SYSTEM_CONSTRAINTS.md §D)
  V_max_realism:  2500.0    # [V]   upper operational bound; node_cmd=1.0 maps here
  V_hard_clamp:   3000.0    # [V]   absolute safety ceiling — system must never exceed

  # First-order voltage dynamics
  tau_charge:     0.5       # [s]   time constant to reach commanded voltage
  leak_rate:      0.1       # [1/s] passive voltage decay when command drops

  # Radial geometry — concentric cylindrical capacitor (M3 grounding)
  # Physical basis: counter-electrode = central rod; collection wall = outer cylinder
  # E_r(r_outer) = V / (r_outer × ln(r_outer / r_inner))
  r_inner_m:      0.005     # [m]   counter-electrode (central rod) radius — placeholder
  r_outer_m:      0.040     # [m]   outer collection wall radius — placeholder

  # Residence time model (shared by ring + node subsystems)
  t_E_ref_s:      2.0       # [s]   residence time at which tanh -> 0.76 (saturation reference)
  stage_volume_L: 0.25      # [L]   effective stage volume — placeholder; bench calibration target

  # InletPolarizationRing — upstream charge conditioning (30% contribution)
  V_ring_ref:     500.0     # [V]   ring conditioning voltage reference (ring tanh -> 0.76)
  ring_weight:    0.30      # []    locked allocation — 06_LOCKED_SYSTEM_CONSTRAINTS.md §D

  # OuterWallCollectorNode — radial field primary capture (70% contribution)
  node_weight:    0.70      # []    locked allocation — 06_LOCKED_SYSTEM_CONSTRAINTS.md §D
```

## Verify Task 4

```bash
python -c "
import yaml
with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
e = cfg['electrostatics']
required = ['V_max_realism','V_hard_clamp','tau_charge','leak_rate',
            'r_inner_m','r_outer_m','t_E_ref_s','stage_volume_L',
            'V_ring_ref','ring_weight','node_weight']
for k in required:
    assert k in e, f'Missing key: {k}'
assert 'gap_m'      not in e, 'gap_m should be removed'
assert 'E_norm_ref' not in e, 'E_norm_ref should be removed'
assert e['ring_weight'] + e['node_weight'] == 1.0, '30/70 allocation must sum to 1.0'
print('Task 4 PASS: YAML electrostatics section correct')
"
```

Expected: `Task 4 PASS: YAML electrostatics section correct`

---

# Task 5 — Validate and Update Docs

## 5.1 Full test suite

```bash
cd C:/Users/JSEer/hydrOS
python -m pytest tests/ -v
```

Expected: **25/25 passing**. Do NOT proceed to doc updates if any test fails.

## 5.2 M3 physics validation (all 5 gates)

```bash
python -c "
import sys; sys.path.insert(0, '.')
import yaml, numpy as np
from hydrion.config import HydrionConfig
from hydrion.physics.electrostatics import ElectrostaticsModel
from hydrion.physics.particles import ParticleModel
from hydrion.sensors.sensor_fusion import build_observation

with open('configs/default.yaml') as f:
    cfg = HydrionConfig(raw=yaml.safe_load(f))

em = ElectrostaticsModel(cfg)
pm = ParticleModel(cfg)

# Gate 1: V_node never exceeds V_hard_clamp
state = {'q_processed_lmin': 13.5}
em.reset(state); pm.reset(state)
for _ in range(500): em.update(state, dt=0.1, node_cmd=1.0)
assert state['V_node'] <= 3000.0, f'FAIL Gate 1: V_node={state[\"V_node\"]}'
print(f'Gate 1 PASS: V_node={state[\"V_node\"]:.1f} V <= 3000 V')

# Gate 2: E_field_kVm within +-5% of analytical value ~30.06 kV/m
assert 28.5 < state['E_field_kVm'] < 31.5, f'FAIL Gate 2: E_field_kVm={state[\"E_field_kVm\"]:.2f}'
print(f'Gate 2 PASS: E_field_kVm={state[\"E_field_kVm\"]:.2f} kV/m (expected ~30.1)')

# Gate 3: E_field_norm in [0, 1]
assert 0.0 <= state['E_field_norm'] <= 1.0, f'FAIL Gate 3: {state[\"E_field_norm\"]}'
print(f'Gate 3 PASS: E_field_norm={state[\"E_field_norm\"]:.3f}')

# Gate 4: voltage ON vs OFF — measurable capture delta
em_off = ElectrostaticsModel(cfg)
s_off  = {'q_processed_lmin': 13.5, 'mesh_loading_avg': 0.0, 'capture_eff': 0.8, 'C_in': 0.7}
em_off.reset(s_off); pm.reset(s_off)
for _ in range(200): em_off.update(s_off, dt=0.1, node_cmd=0.0)
state.update({'mesh_loading_avg': 0.0, 'capture_eff': 0.8, 'C_in': 0.7})
pm.update(state, dt=0.1, electrostatics_model=em)
pm.update(s_off, dt=0.1, electrostatics_model=em_off)
delta = state['particle_capture_eff'] - s_off['particle_capture_eff']
assert delta > 0.05, f'FAIL Gate 4: delta={delta:.4f}'
print(f'Gate 4 PASS: ON={state[\"particle_capture_eff\"]:.3f} OFF={s_off[\"particle_capture_eff\"]:.3f} delta={delta:.3f}')

# Gate 5: obs12_v2 — index 3 is E_field_norm in [0,1]
truth = dict(state)
truth.update({'flow':0.5,'pressure':0.3,'clog':0.1,'C_out':0.1,
              'valve_cmd':1.0,'pump_cmd':0.8,'bf_cmd':0.0,'node_voltage_cmd':0.5})
sensors = {'sensor_turbidity':0.2,'sensor_scatter':0.1}
obs = build_observation(truth, sensors)
assert obs.shape == (12,)
assert 0.0 <= obs[3] <= 1.0, f'FAIL Gate 5: obs[3]={obs[3]}'
print(f'Gate 5 PASS: obs[3]=E_field_norm={obs[3]:.3f}, shape={obs.shape}')

print()
print('M3 VALIDATION: ALL 5 GATES PASSED')
"
```

All 5 gates must print PASS.

## 5.3 Update `docs/calibration/M2-2.5_CALIBRATION_PARAMETER_REGISTER.md`

Add these rows to **§5.6 Electrostatics** after the existing `V_hard_clamp` row:

```markdown
| Counter-electrode radius | Electrostatics | `electrostatics.r_inner_m` | 0.005 | m | Placeholder | Low | M3 geometry estimate — central rod radius | radial field validation | bench geometry measurement | dominant sensitivity: ln(r_outer/r_inner) |
| Collection wall radius | Electrostatics | `electrostatics.r_outer_m` | 0.040 | m | Placeholder | Low | M3 geometry estimate — stage housing outer radius | radial field validation | bench geometry measurement | primary calibration target for E_field_kVm |
| Residence time saturation reference | Electrostatics | `electrostatics.t_E_ref_s` | 2.0 | s | Placeholder | Low | engineering assumption for tanh saturation | voltage step-response test | bench electrostatic test | shared by ring and node subsystems |
| Stage effective volume | Electrostatics | `electrostatics.stage_volume_L` | 0.25 | L | Placeholder | Low | estimated from r_outer and representative stage height | residence time validation | bench geometry measurement | determines flow-rate dependence of capture gain |
| Ring conditioning voltage reference | Electrostatics | `electrostatics.V_ring_ref` | 500.0 | V | Placeholder | Low | engineering assumption; ring saturates above ~500 V | ring sensitivity test | electrostatic bench test | InletPolarizationRing saturation point |
```

## 5.4 Update `docs/context/04_CURRENT_ENGINE_STATUS.md` §7

Replace the existing Electrostatics limitations and M1.5 changes block with:

```markdown
### M3 Changes (2026-04-10)

- Axial scalar-gap model replaced with cylindrical radial field model
- Two subsystems implemented: `InletPolarizationRing` (30%) + `OuterWallCollectorNode` (70%)
- Field at collection wall: `E_r(r_outer) = V / (r_outer × ln(r_outer/r_inner))`
- `E_field_kVm` [kV/m] stored in truth_state (replaces dimensionless `E_field` [V/m])
- `E_capture_gain` in [0, 1] output to particles module (replaces `E_norm` pass-through)
- Residence time coupling: capture gain decreases at high flow rates (physically correct)
- `V_max_realism = 2500 V` (operational bound), `V_hard_clamp = 3000 V` (safety ceiling)

### Observation Schema

- **obs12_v2** active from M3
- Index 3: `E_field_norm` = `E_field_kVm / E_field_kVm_max` in [0, 1]
  (replaces `E_norm` in [0, 2] from obs12_v1)

### Limitations (post-M3)

- r_inner_m, r_outer_m, stage_volume_L are geometry placeholders — bench measurement required
- No conductivity dependence on field or capture
- No particle-size dependence on E_capture_gain
- Per-stage electrostatic parameters (separate V_node per stage) deferred to M3.5 / M4
- alpha_E in particles.py remains a scalar gain; M4 will replace with per-stage curves
```

## 5.5 Commit

```bash
git add hydrion/physics/electrostatics.py \
        hydrion/physics/particles.py \
        hydrion/sensors/sensor_fusion.py \
        configs/default.yaml \
        docs/calibration/M2-2.5_CALIBRATION_PARAMETER_REGISTER.md \
        docs/context/04_CURRENT_ENGINE_STATUS.md

git commit -m "fix(physics/M3): radial field electrostatics, two-subsystem model, obs12_v2

- ElectrostaticsModel v2: E_r(r)=V/(r*ln(ro/ri)), InletPolarizationRing + OuterWallCollectorNode
- E_field_kVm in truth_state; E_capture_gain replaces E_norm as particles.py input
- sensor_fusion: obs12_v2 — index 3 is E_field_norm in [0,1]
- default.yaml: radial geometry params, removed gap_m and E_norm_ref
- Calibration register + engine status docs updated"
```

---

# M3 Exit Checklist

Do not mark M3 complete until every item is confirmed:

- [ ] `electrostatics.py` uses `E_r(r) = V / (r × ln(r_outer/r_inner))`
- [ ] `InletPolarizationRing` and `OuterWallCollectorNode` are separate, documented subsystems
- [ ] `E_field_kVm` stored in truth_state
- [ ] `E_capture_gain` in [0, 1] stored in truth_state
- [ ] `E_norm` removed from truth_state and all code paths
- [ ] `sensor_fusion.py` index 3 = `E_field_norm`; schema label = `obs12_v2`
- [ ] `particles.py` reads `E_capture_gain`, not `E_norm`
- [ ] `default.yaml` electrostatics section complete; `gap_m` and `E_norm_ref` removed
- [ ] Gate 1: V_node <= V_hard_clamp (3000 V) at all times
- [ ] Gate 2: E_field_kVm in [28.5, 31.5] kV/m at V_max_realism
- [ ] Gate 3: E_field_norm in [0, 1]
- [ ] Gate 4: voltage ON/OFF delta_capture_eff > 0.05 at Q=13.5, clean filter
- [ ] Gate 5: obs[3] = E_field_norm, shape=(12,)
- [ ] Full test suite: 25/25 pass
- [ ] `04_CURRENT_ENGINE_STATUS.md` updated
- [ ] `CALIBRATION_PARAMETER_REGISTER.md` updated with new M3 parameters

---

# What You Must NOT Do

- Do NOT implement buoyant-phase capture — that is M4 scope
- Do NOT add per-stage separate V_node commands — one voltage command drives all stages
- Do NOT extend the observation vector beyond 12D — obs12_v2 is a replacement at index 3 only
- Do NOT change the reward function — that is M6 scope
- Do NOT alter truth_state / sensor_state separation
- Do NOT change index ordering in `sensor_fusion.py` without versioning
- Do NOT add YAML parameters without updating `CALIBRATION_PARAMETER_REGISTER.md`
- Do NOT start M4 before all M3 exit checklist items are confirmed

---

# What M3 Unlocks

After this prompt is complete:

- **M4** — per-stage size-dependent capture curves can couple into `E_capture_gain`
- **Phase 4** — schematic correction (radial fans, outer-wall nodes, downward collection tubes)
- **Console** — `eFieldKVm` telemetry display is now backed by a physical truth_state value

---

# Final Standard

Every change must answer:

> "Does this make HydrOS behave more like the real device?"

If the answer is no, do not apply it.

HydrOS is being grounded. Not optimized.

**Physics first. Code second. UI last.**
