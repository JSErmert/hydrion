from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class HydrionConfig:
    """
    Lightweight configuration wrapper.
    Allows access via cfg.raw[...] and attribute-style access.
    """
    raw: Dict[str, Any]

    def __getattr__(self, item):
        try:
            return self.raw[item]
        except KeyError:
            raise AttributeError(item)
