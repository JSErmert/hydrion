# tests/test_particle_scenario_runner.py
import pytest
from hydrion.scenarios.runner import ScenarioRunner, load_scenario
from hydrion.environments.conical_cascade_env import ConicalCascadeEnv


def test_scenario_runner_produces_particle_streams():
    """
    ScenarioRunner with ConicalCascadeEnv must populate particleStreams
    on each ScenarioStepRecord after the first step.
    """
    env = ConicalCascadeEnv(config_path="configs/default.yaml")
    runner = ScenarioRunner(env)
    scenario = load_scenario("hydrion/scenarios/examples/baseline_nominal.yaml")
    history = runner.run(scenario)

    # At least one step should have particle_streams
    found = False
    for step in history.steps:
        if step.particle_streams is not None:
            found = True
            ps = step.particle_streams
            assert "s1" in ps and "s2" in ps and "s3" in ps
            # At least some particles must appear
            total = len(ps["s1"]) + len(ps["s2"]) + len(ps["s3"])
            assert total > 0, "particle_streams must have at least one particle"
            break

    assert found, "No step contained particle_streams — check CCE._state['particle_streams']"


def test_scenario_step_dict_has_particleStreams():
    """ScenarioStepRecord.to_dict() must include 'particleStreams' key."""
    env = ConicalCascadeEnv(config_path="configs/default.yaml")
    runner = ScenarioRunner(env)
    scenario = load_scenario("hydrion/scenarios/examples/baseline_nominal.yaml")
    history = runner.run(scenario)
    d = history.to_dict()
    step_dict = d["steps"][1]   # step 0 may have None (before first integration)
    assert "particleStreams" in step_dict
