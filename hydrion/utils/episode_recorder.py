# hydrion/utils/episode_recorder.py
from __future__ import annotations
from typing import List, Dict

def record_step(state: dict) -> dict:
    """Return a shallow copy of the environment state for later analysis."""
    return dict(state)

def rollout_episode(env, policy=None, deterministic=True, max_steps=6000):
    """Run one full episode and record state history."""
    history = []

    # VecEnv reset returns only obs
    obs = env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        if policy is None:
            action = env.action_space.sample()
        else:
            action, _ = policy.predict(obs, deterministic=deterministic)

        # SB3 VecEnv returns obs, reward, done, info
        obs, reward, dones, infos = env.step(action)

        # VecEnv outputs arrays → use first element
        done = bool(dones[0])

        # record state
        # Each env inside a vec_env has its own attributes at index 0
        # For custom attributes like `state`, access through env.envs[0]
        state_dict = env.envs[0].state.copy()
        history.append(state_dict)

        steps += 1

    return history
