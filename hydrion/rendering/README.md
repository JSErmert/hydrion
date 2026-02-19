# HydrOS Research Observatory

**Identity**: Research Observatory for internal dynamics and research analysis (not consumer UI)

## Overview

The Observatory provides comprehensive visualization and observability for HydrOS episodes. It is designed to:
- Make internal dynamics legible
- Make anomalies obvious  
- Make RL behavior interpretable

All visualization is **side-effect free** and does not modify simulation state.

## Architecture

```
hydrion/rendering/
├── __init__.py              # Module exports
├── episode_history.py        # Episode data recorder
├── time_series.py            # Time-series plotting utilities
├── anomaly_detector.py        # Anomaly detection and visualization
├── observatory.py            # Main dashboard class
└── README.md                 # This file
```

## Core Components

### 1. EpisodeHistory

Records episode data without side effects:
- `truth_state`: Physics truth (flow, pressure, clog, E_norm, C_out, etc.)
- `sensor_state`: Measured outputs (sensor_turbidity, sensor_scatter)
- `actions`: [valve, pump, backflush, node_voltage]
- `rewards`: Reward values
- `info`: Info dict (may contain safety, validation outputs)

**Usage:**
```python
from hydrion.rendering import EpisodeHistory

history = EpisodeHistory()
history.record_step(
    step=0,
    truth_state=env.truth_state,
    sensor_state=env.sensor_state,
    action=action,
    reward=reward,
    info=info,
    dt=env.dt,
)
```

### 2. Observatory

Main dashboard class that integrates all visualization components.

**Usage:**
```python
from hydrion.rendering import Observatory
from hydrion.utils.visualization import record_episode_with_observatory

# Create Observatory
observatory = Observatory(save_dir="outputs/episode_001", time_axis="time")

# Record episode
record_episode_with_observatory(env, observatory, policy=None, max_steps=1000)

# Generate dashboard
figures = observatory.plot_dashboard(save=True, show=False)
```

### 3. Time-Series Plots

Professional plots with labeled axes, units, and legends:

- **Core variables**: flow, pressure, clog, E_norm, C_out, particle_capture_eff
- **Actions**: valve, pump, backflush, node_voltage traces
- **Rewards**: Instantaneous and cumulative reward traces
- **PSD observability**: Per-bin concentrations (C_in_bin_i, C_out_bin_i) if PSD enabled
- **Shape observability**: fiber_fraction, C_fibers if shape enabled

**Usage:**
```python
from hydrion.rendering import plot_time_series, plot_actions

# Plot core variables
fig, axes = plot_time_series(history, variables=["flow", "pressure", "clog"])

# Plot actions
fig, ax = plot_actions(history)
```

### 4. Anomaly Detection

Detects and visualizes:
- **NaNs/Infs**: Invalid values in state variables
- **Bounds violations**: Values outside expected ranges
- **Shield events**: Safety interventions (projections, violations)
- **Termination causes**: Episode termination reasons

**Usage:**
```python
from hydrion.rendering import AnomalyDetector, plot_anomalies

detector = AnomalyDetector(history)
summary = detector.summary()  # Counts by type

# Visualize anomalies
fig, axes = plot_anomalies(history, detector=detector)
```

## Features

### ✅ Time-Series Plots
- Key truth variables (flow, pressure, clog, E_norm, C_out, capture_eff)
- Action traces synchronized with state
- Reward traces (instantaneous and cumulative)

### ✅ PSD Observability
- If PSD enabled: per-bin concentrations (C_in_bin_i, C_out_bin_i)
- PSD summary (C_L, C_M, C_S) if available
- Gracefully degrades if PSD disabled

### ✅ Shape Observability
- If fiber_fraction / C_fibers present: shows them
- Gracefully degrades if shape disabled

### ✅ Anomaly Visibility
- Highlights NaNs, bounds violations
- Shield events (if exposed in info)
- Termination causes

### ✅ Episode Playback
- Save frames: `observatory.save_frames()`
- Create video: `observatory.create_video_from_frames()` (requires ffmpeg)

## Constraints

### Non-Negotiable
- ✅ May read: truth_state, sensor_state, info, reward, validation outputs
- ❌ May not modify: physics, safety, sensors, observation logic, env.step ordering, determinism
- ✅ All visualization logic under: `hydrion/rendering/*`
- ✅ Side-effect free: no random calls without fixed seeds
- ✅ Visuals correspond to real physical variables
- ✅ Default output works without GPU and without heavy dependencies (matplotlib)
- ✅ Must not change RL contract: observation remains 12D

## Example

See `examples/observatory_example.py` for a complete example.

```python
from hydrion.env import HydrionEnv
from hydrion.rendering import Observatory
from hydrion.utils.visualization import record_episode_with_observatory

# Setup
env = HydrionEnv()
observatory = Observatory(save_dir="outputs/episode_001")

# Record episode
record_episode_with_observatory(env, observatory, max_steps=1000)

# Generate dashboard
figures = observatory.plot_dashboard(save=True, show=False)

# Check anomalies
summary = observatory.get_anomaly_summary()
print(f"Anomalies: {summary}")

# Save frames for playback
observatory.save_frames(variables=["flow", "pressure", "clog"])

# Create video (requires ffmpeg)
video_path = observatory.create_video_from_frames(fps=10)
```

## Output Quality

- **Professional, publishable plots**: Labeled axes, units where known, consistent scales, legends
- **Modular code**: Clear separation of concerns, docstrings
- **Clear entry points**: Observatory class as main interface
- **Minimal, incremental changes**: Treats changes like a PR

## Dependencies

- `numpy`: Array operations
- `matplotlib`: Plotting (default, no GPU required)
- `ffmpeg`: Optional, for video creation (must be on PATH)

## Pipeline Semantics

Visual layers respect pipeline order:
1. **Hydraulics** → Flow, pressure
2. **Clogging** → Clog state
3. **Electrostatics** → E_norm
4. **Particles** → C_out, capture_eff, PSD/shape
5. **Sensors** → sensor_turbidity, sensor_scatter
