# hydrion/runtime/seeding.py
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """
    Seed all relevant RNG sources used by Hydrion.

    Commit 1 goal: deterministic rollouts (especially sensor noise using np.random).

    Notes:
    - This seeds numpy's *global* RNG (np.random.*).
    - Later (v1.5+), we can migrate subsystems to use local Generator objects.
    """
    seed = int(seed)

    # Python
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # Some libs read this
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Torch (optional)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            # Deterministic mode (can reduce performance, but improves reproducibility)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch not installed or not desired — that's fine.
        pass
