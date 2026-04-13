"""
HydrOS environments package.

Environments:
    ConicalCascadeEnv  — M5 exploration: conical cascade + RT + nDEP physics
                         Runs in parallel with HydrionEnv for comparison.
"""
from .conical_cascade_env import ConicalCascadeEnv

__all__ = ["ConicalCascadeEnv"]
