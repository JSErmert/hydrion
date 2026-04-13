# hydrion/sensors/flow.py
"""
FlowRateSensor — M6 Phase 1 sensor realism.

Models an electromagnetic (magnetic) flow meter measuring processed filter flow rate.

Source truth:   truth_state["q_processed_lmin"]  (L/min, written by HydraulicsModel)
Sensor output:  sensor_state["flow_sensor_lmin"]

Model:
    sensor_q = max(0, truth_q × (1 + N(0, σ_q)) + calibration_bias)

    calibration_bias is sampled once at episode reset from N(0, σ_bias²) and held
    fixed for the episode, simulating factory calibration offset.

Sensor type rationale (M6 Research Brief, 2026-04-13):
    Electromagnetic (mag) meter chosen for dirty/particle-laden laundry wastewater service.
    Faraday-principle measurement is drift-free and unaffected by particle loading.
    No latency buffer needed: electronic latency < 1 ms (negligible vs. 100 ms step).

Physics/architecture constraints:
- Reads ONLY from truth_state (no sensor_state reads).
- Writes ONLY to sensor_state["flow_sensor_lmin"].
- Does NOT write to truth_state.

Parameter defaults (source: M6 Sensor Realism Research Brief, 2026-04-13):
    flow_noise_frac:    0.01   Multiplicative noise fraction (1% reading; mag meter spec ±0.5–1%)
    flow_bias_std_lmin: 0.2    Std dev of factory calibration offset [L/min] (±0.1–0.5 L/min range)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class FlowRateSensor:
    """
    Electromagnetic flow meter model for M6 sensor realism.

    READS from truth_state:  q_processed_lmin
    WRITES to sensor_state:  flow_sensor_lmin
    NEVER writes to truth_state.
    """

    def __init__(self, cfg: Any | None = None) -> None:
        s_raw: Dict[str, Any] = {}
        if cfg is not None and hasattr(cfg, "raw"):
            s_raw = getattr(cfg, "raw", {}).get("sensors", {}).get("flow", {}) or {}

        self._noise_frac: float = float(s_raw.get("flow_noise_frac",    0.01))
        self._bias_std:   float = float(s_raw.get("flow_bias_std_lmin", 0.2))

        # Per-episode calibration bias (sampled at reset)
        self._calibration_bias: float = 0.0

    # ------------------------------------------------------------------
    # Public API  (mirrors OpticalSensorArray interface)
    # ------------------------------------------------------------------

    def reset(
        self,
        truth_state: Dict[str, float],
        sensor_state: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Sample a new calibration offset for this episode; zero sensor_state key.

        The bias is fixed for the duration of the episode, simulating factory
        calibration error that persists until the unit is re-calibrated.
        """
        self._calibration_bias = float(np.random.randn()) * self._bias_std
        if sensor_state is not None:
            sensor_state["flow_sensor_lmin"] = 0.0

    def update(
        self,
        truth_state: Dict[str, float],
        sensor_state: Dict[str, float],
        dt: float = 0.1,
    ) -> None:
        """
        Compute noisy flow reading and write to sensor_state only.

        Must be called after HydraulicsModel.update() so q_processed_lmin is current.
        """
        # 1. Read truth flow
        truth_q = float(truth_state.get("q_processed_lmin", 0.0))

        # 2. Multiplicative noise (signal-dependent, as with mag meters)
        noise_frac = float(np.random.randn()) * self._noise_frac
        sensor_q = truth_q * (1.0 + noise_frac) + self._calibration_bias

        # 3. Write to sensor_state ONLY — never touch truth_state
        sensor_state["flow_sensor_lmin"] = float(max(0.0, sensor_q))
