# hydrion/scenarios/types.py
"""
HydrOS Scenario type definitions.

Python dataclasses mirroring the TypeScript types in:
  docs/visualization/M2-2.5_HYDROS_FIRST_SCENARIO_EXECUTION_LAYER.md

All mutable fields use field(default_factory=...) to avoid dataclass
shared-mutable-default traps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Scenario definition (static, loaded from YAML)
# ---------------------------------------------------------------------------

@dataclass
class InitialStateConfig:
    fouling_s1: float = 0.0
    fouling_s2: float = 0.0
    fouling_s3: float = 0.0
    storage_fill: float = 0.0
    pressure_bias: float = 0.0
    flow_bias: float = 0.0


@dataclass
class FlowProfile:
    type: str = "constant"          # "constant" | "ramp" | "burst" | "realistic"
    baseFlowLmin: float = 13.5
    variability: float = 0.0
    burstAmplitude: float = 0.0
    burstFrequency: float = 0.0
    transientEvents: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ParticleProfile:
    type: str = "fiber_dominant"    # "fiber_dominant" | "mixed" | "heavy_load"
    density: float = 0.30
    variability: float = 0.0
    sizeDistribution: List[float] = field(default_factory=list)


@dataclass
class DisturbanceEvent:
    type: str = "flow_spike"        # see spec for full enum
    time: float = 0.0
    duration: float = 0.0
    intensity: float = 1.0


@dataclass
class BackflushEvent:
    time: float = 0.0
    duration: float = 2.0
    bf_cmd: float = 1.0


@dataclass
class ScenarioDefinition:
    id: str
    name: str
    description: str
    initialState: InitialStateConfig
    flowProfile: FlowProfile
    particleProfile: ParticleProfile
    disturbances: List[DisturbanceEvent]
    durationSec: float
    dt: float
    seed: Optional[int] = 42
    backflush_events: List[BackflushEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Runtime / output types
# ---------------------------------------------------------------------------

@dataclass
class ScenarioEventMarker:
    time: float
    type: str   # see spec for full enum
    label: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"time": self.time, "type": self.type, "label": self.label, "meta": self.meta}


@dataclass
class ScenarioStepRecord:
    t: float
    stepIndex: int
    scenarioInputs: Dict[str, Any]
    truthState: Dict[str, Any]
    sensorState: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]
    particle_streams: Optional[Dict[str, List[Dict[str, Any]]]] = None   # from ParticleDynamicsEngine

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "stepIndex": self.stepIndex,
            "scenarioInputs": self.scenarioInputs,
            "truthState": self.truthState,
            "sensorState": self.sensorState,
            "reward": self.reward,
            "done": self.done,
            "info": self.info,
            "particleStreams": self.particle_streams,
        }


@dataclass
class ScenarioExecutionHistory:
    scenarioId: str
    dt: float
    steps: List[ScenarioStepRecord] = field(default_factory=list)
    eventMarkers: List[ScenarioEventMarker] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenarioId": self.scenarioId,
            "dt": self.dt,
            "steps": [s.to_dict() for s in self.steps],
            "eventMarkers": [m.to_dict() for m in self.eventMarkers],
        }
