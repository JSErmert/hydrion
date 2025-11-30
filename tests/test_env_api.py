import numpy as np
from hydrion.env import HydrionEnv


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


if __name__ == "__main__":
    test_env_api()
