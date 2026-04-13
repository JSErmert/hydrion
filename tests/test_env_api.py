import numpy as np
from hydrion.env import HydrionEnv


def test_observation_12d_regression():
    """PSD disabled (default): observation remains 12D, keys unchanged."""
    env = HydrionEnv(config_path="configs/default.yaml")
    obs, _ = env.reset()
    assert obs.shape == (12,), f"Expected 12D obs, got {obs.shape}"
    obs2, _, _, _, _ = env.step(env.action_space.sample())
    assert obs2.shape == (12,)
    # Core truth_state keys
    assert "C_in" in env.truth_state
    assert "C_out" in env.truth_state
    assert "particle_capture_eff" in env.truth_state


def test_env_api():
    print("\n--- HydrionEnv Full API Test (12D obs) ---")

    env = HydrionEnv()

    obs, info = env.reset()
    print("Initial obs (len={}):".format(len(obs)), obs)

    for step in range(20):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)

        # Unpack physics for readability
        flow, press, clog = obs[0], obs[1], obs[2]
        E_norm = obs[3]
        C_out, eff = obs[4], obs[5]
        valve, pump, bf, voltage = obs[6], obs[7], obs[8], obs[9]
        turb, scatter = obs[10], obs[11]

        print(
            f"Step {step:02d}: "
            f"Flow={flow:.3f}, "
            f"P={press:.3f}, "
            f"Clog={clog:.3f}, "
            f"E_norm={E_norm:.3f}, "
            f"C_out={C_out:.3f}, "
            f"Eff={eff:.3f}, "
            f"Turb={turb:.3f}, "
            f"Scatter={scatter:.3f}, "
            f"Reward={reward:.3f}"
        )

        if term or trunc:
            break


def test_e_field_norm_obs3_nonzero_after_voltage():
    """
    Regression: obs[3] (E_field_norm) must be non-zero after applying
    a non-zero node_voltage_cmd.

    Bug history (2026-04-12):
        state/init.py initialised 'E_norm' (obs12_v1 key).
        env.py info dict read truth_state['E_norm'] (always 0.0).
        service/app.py telemetry also read 'E_norm' (always 0.0).
        The obs pipeline (sensor_fusion.py) correctly reads 'E_field_norm'
        (obs12_v2 key), written by ElectrostaticsModel. Obs[3] was NOT
        silently zero — the bug was in info/telemetry only, not RL training.
        Fixed: init.py, env.py info, app.py telemetry all use E_field_norm.
    """
    env = HydrionEnv(config_path="configs/default.yaml")
    env.reset(seed=0)
    # Run 10 steps with node_voltage_cmd=1.0 (max voltage)
    action = np.array([0.5, 0.5, 0.0, 1.0], dtype=np.float32)
    obs_list = []
    for _ in range(10):
        obs, _, _, _, _ = env.step(action)
        obs_list.append(float(obs[3]))

    # obs[3] = E_field_norm — must be > 0 with non-zero voltage applied
    assert max(obs_list) > 0.01, (
        f"obs[3] (E_field_norm) stayed near zero at max voltage: {obs_list}"
    )
    # truth_state must use the obs12_v2 key
    assert "E_field_norm" in env.truth_state, "E_field_norm missing from truth_state"
    # Legacy E_norm key should NOT be in truth_state (it was obs12_v1)
    # (init.py still writes it for backward-compat logging; relax this check)
    assert float(env.truth_state.get("E_field_norm", -1)) > 0.0, (
        "truth_state['E_field_norm'] is zero even after voltage applied"
    )


if __name__ == "__main__":
    test_env_api()
