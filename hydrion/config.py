# hydrion/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import hashlib
import json


def stable_config_hash(raw: Dict[str, Any]) -> str:
    """
    Stable fingerprint for a config dict.
    Uses JSON canonicalization: sorted keys, no whitespace variance.
    """
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class HydrionConfig:
    """
    Lightweight configuration wrapper.

    Commit 1 additions:
    - stable config hashing (reproducibility anchor)
    """
    raw: Dict[str, Any]

    def __getattr__(self, item):
        try:
            return self.raw[item]
        except KeyError:
            raise AttributeError(item)

    def config_hash(self) -> str:
        return stable_config_hash(self.raw)

    def get_seed(self, default: int = 0) -> int:
        """
        Optional place to store seed in config if desired:
        sim:
          seed: 123
        """
        sim = self.raw.get("sim", {}) or {}
        return int(sim.get("seed", default))

    def get_noise_enabled(self, default: bool = False) -> bool:
        """
        Optional place to store noise_enabled in config if desired:
        sim:
          noise_enabled: true
        """
        sim = self.raw.get("sim", {}) or {}
        return bool(sim.get("noise_enabled", default))
