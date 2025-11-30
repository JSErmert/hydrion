import numpy as np
from hydrion.env import HydrionEnv


def test_env_api():
    print("\n--- HydrionEnv Full API Test ---")

    env = HydrionEnv()

    obs, info = env.reset()
    print("Initial obs:", obs)

    for step in range(20):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(
            f"Step {step:02d}: "
            f"Flow={obs[0]:.3f}, "
            f"P={obs[1]:.3f}, "
            f"Clog={obs[2]:.3f}, "
            f"Reward={reward:.3f}"
        )

        if term or trunc:
            break


if __name__ == "__main__":
    test_env_api()
