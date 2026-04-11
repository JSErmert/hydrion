# hydrion/scenarios/__init__.py
"""
HydrOS Scenario Execution Layer — first working version (M2-2.5).

Public API:
    load_scenario(path)  → ScenarioDefinition
    ScenarioRunner(env)  → .run(scenario) → ScenarioExecutionHistory
"""

from .runner import ScenarioRunner, load_scenario
from .types import (
    BackflushEvent,
    DisturbanceEvent,
    FlowProfile,
    InitialStateConfig,
    ParticleProfile,
    ScenarioDefinition,
    ScenarioEventMarker,
    ScenarioExecutionHistory,
    ScenarioStepRecord,
)

__all__ = [
    "ScenarioRunner",
    "load_scenario",
    "ScenarioDefinition",
    "InitialStateConfig",
    "FlowProfile",
    "ParticleProfile",
    "BackflushEvent",
    "DisturbanceEvent",
    "ScenarioEventMarker",
    "ScenarioExecutionHistory",
    "ScenarioStepRecord",
]
