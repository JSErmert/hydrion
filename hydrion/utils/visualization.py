"""
hydrion/utils/visualization.py

Convenience utilities for integrating Observatory with HydrionEnv.
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path
import numpy as np

from ..rendering import Observatory


def create_observatory(
    save_dir: Optional[str | Path] = None,
    time_axis: str = "time",
) -> Observatory:
    """
    Create an Observatory instance for episode visualization.
    
    Args:
        save_dir: Optional directory to save plots/frames
        time_axis: "time" or "step" for x-axis
    
    Returns:
        Observatory instance
    """
    return Observatory(save_dir=save_dir, time_axis=time_axis)


def record_episode_with_observatory(
    env,
    observatory: Observatory,
    policy=None,
    max_steps: int = 6000,
    deterministic: bool = True,
):
    """
    Run an episode and record data in Observatory.
    
    Args:
        env: HydrionEnv instance
        observatory: Observatory instance
        policy: Optional policy for action selection (defaults to random)
        max_steps: Maximum steps per episode
        deterministic: Whether to use deterministic policy
    
    Returns:
        (observations, rewards, terminated, truncated, info)
    """
    observatory.reset()
    
    # Reset environment
    obs, info = env.reset()
    done = False
    step = 0
    
    observations = [obs]
    rewards = []
    terminated = False
    truncated = False
    final_info = {}
    
    while not done and step < max_steps:
        # Get action
        if policy is None:
            action = env.action_space.sample()
        else:
            action, _ = policy.predict(obs, deterministic=deterministic)
        
        # Step environment
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        
        # Record step in observatory
        observatory.record_step(
            step=step,
            truth_state=env.truth_state,
            sensor_state=env.sensor_state,
            action=action,
            reward=reward,
            info=info,
            observation=obs,
            dt=getattr(env, "dt", None),
        )
        
        observations.append(obs)
        rewards.append(reward)
        terminated = term
        truncated = trunc
        final_info = info
        
        step += 1
    
    # Finalize episode
    observatory.finalize_episode(terminated=terminated, truncated=truncated)
    
    return observations, rewards, terminated, truncated, final_info
