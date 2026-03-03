import numpy as np

from hydrion.env import HydrionEnv
from hydrion.wrappers.shielded_env import ShieldedEnv
from hydrion.safety.shield import SafetyConfig


def test_shield_integrity():
    """Ensure the shield wrapper consistently uses truth_state and logs interventions.

    We force a pressure violation by setting a zero soft limit, and we
    choose actions that violate the rate limit to trigger a projection
    (``shield_intervened``). The test asserts that the flag toggles
    and that a violation is recorded in the safety info.
    """
    env = HydrionEnv()

    # configure shield to always report a pressure violation and to be
    # aggressive about rate limiting so we can see an intervention
    cfg = SafetyConfig(
        max_pressure_soft=0.0,
        max_pressure_hard=10.0,  # avoid hard termination
        max_action_delta=0.1,
    )
    wrapped = ShieldedEnv(env, cfg=cfg)

    obs, _ = wrapped.reset(seed=0)

    # first step: use nonzero command to provoke pressure rise
    # (zero action produced zero pressure and therefore no violation).
    action1 = np.array([0.5, 0.5, 0.0, 0.5], dtype=np.float32)
    _, _, _, _, info1 = wrapped.step(action1)
    assert info1["safety"]["shield_intervened"] is False
    assert (
        info1["safety"].get("soft_pressure_violation", False)
        or info1["safety"].get("hard_pressure_violation", False)
    )

    # second step: large jump should be rate-limited by the shield
    action2 = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    _, _, _, _, info2 = wrapped.step(action2)
    assert info2["safety"]["shield_intervened"] is True

    # ensure at least one of the two steps recorded a pressure violation
    assert (
        info1["safety"].get("soft_pressure_violation", False)
        or info1["safety"].get("hard_pressure_violation", False)
        or info2["safety"].get("soft_pressure_violation", False)
        or info2["safety"].get("hard_pressure_violation", False)
    )
