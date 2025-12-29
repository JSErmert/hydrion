# hydrion/state/types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TruthState:
    """
    Physics truth. This is the internal ground truth state updated by physics modules.
    """
    data: Dict[str, Any]


@dataclass
class SensorState:
    """
    What sensors measure (may diverge from truth once noise/anomalies are added).
    """
    data: Dict[str, Any]
