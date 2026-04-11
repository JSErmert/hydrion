# hydrion/scenarios/profiles.py
"""
Flow and particle density profile computation for HydrOS scenarios.

Each function is pure (no side effects) and accepts an optional numpy RNG
for reproducible variability.  The caller is responsible for advancing
the RNG consistently across timesteps.

Physical bounds enforced:
  flow:    [0, Q_MAX_LMIN]
  density: [0, 1]
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from .types import FlowProfile, ParticleProfile

Q_MAX_LMIN: float = 20.0   # hard ceiling from locked system constraints


# ---------------------------------------------------------------------------
# Flow profile
# ---------------------------------------------------------------------------

def compute_flow_at_time(
    profile: FlowProfile,
    t: float,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Return the target flow in L/min at simulation time t.

    Profile types:
        constant  — baseFlowLmin
        ramp      — linear ramp from 0 to baseFlowLmin over first 30 s
        burst     — base + sinusoidal burst (burstAmplitude, burstFrequency Hz)
        realistic — base ± Gaussian noise scaled by variability, plus transient
                    event overlays if defined
    """
    base = profile.baseFlowLmin
    ptype = profile.type.lower()

    if ptype == "constant":
        flow = base

    elif ptype == "ramp":
        ramp_duration = 30.0
        flow = base * min(t / max(ramp_duration, 1e-6), 1.0)

    elif ptype == "burst":
        amp  = profile.burstAmplitude
        freq = profile.burstFrequency if profile.burstFrequency > 0 else 0.1
        flow = base + amp * math.sin(2.0 * math.pi * freq * t)

    elif ptype == "realistic":
        if rng is not None and profile.variability > 0.0:
            noise = float(rng.normal(0.0, profile.variability * base))
        else:
            noise = 0.0
        flow = base + noise
        # Apply transient event overlays
        for event in profile.transientEvents:
            e_start = float(event.get("time", 0.0))
            e_end   = e_start + float(event.get("duration", 0.0))
            if e_start <= t < e_end:
                flow = float(event.get("flow", flow))
    else:
        flow = base

    return float(np.clip(flow, 0.0, Q_MAX_LMIN))


# ---------------------------------------------------------------------------
# Particle density profile
# ---------------------------------------------------------------------------

def compute_particle_density_at_time(
    profile: ParticleProfile,
    t: float,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Return normalized particle density in [0, 1] at simulation time t.

    Profile types:
        fiber_dominant — moderate steady density, low variability (laundry default)
        mixed          — moderate density with balanced variability
        heavy_load     — high density with variability (stress scenario)
    """
    ptype = profile.type.lower()
    base  = float(np.clip(profile.density, 0.0, 1.0))
    var   = profile.variability

    if rng is not None and var > 0.0:
        noise = float(rng.normal(0.0, var * base))
    else:
        noise = 0.0

    if ptype == "fiber_dominant":
        density = base + noise

    elif ptype == "mixed":
        density = base + noise

    elif ptype == "heavy_load":
        density = base + noise

    else:
        density = base + noise

    return float(np.clip(density, 0.0, 1.0))
