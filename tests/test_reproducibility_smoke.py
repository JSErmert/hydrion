# tests/test_reproducibility_smoke.py
import numpy as np
from hydrion.env import HydrionEnv


def rollout(env: HydrionEnv, seed: int, steps: int = 25):
    obs, _ = env.reset(seed=seed)
    obs_seq = [obs.copy()]
    info_seq = []

    # Fixed action to remove action randomness
    action = np.array([0.6, 0.7, 0.0, 0.5], dtype=np.float32)

    for _ in range(steps):
        obs, reward, term, trunc, info = env.step(action)
        obs_seq.append(obs.copy())
        info_seq.append((reward, term, trunc, dict(info)))
        if term or trunc:
            break

    return np.stack(obs_seq), info_seq


def test_reproducibility_same_seed_same_rollout():
    env1 = HydrionEnv()
    env2 = HydrionEnv()

    seed = 123
    obs1, info1 = rollout(env1, seed=seed, steps=25)
    obs2, info2 = rollout(env2, seed=seed, steps=25)

    # Exact equality should hold here because we reseed np.random each reset.
    assert obs1.shape == obs2.shape
    assert np.allclose(obs1, obs2, atol=0.0)

    # Check rewards match too
    rewards1 = [x[0] for x in info1]
    rewards2 = [x[0] for x in info2]
    assert len(rewards1) == len(rewards2)
    assert np.allclose(rewards1, rewards2, atol=0.0)
