# hydrion/sensors/pressure.py
"""
DifferentialPressureSensor — M6 Phase 1 sensor realism.

Models a low-cost MEMS differential pressure sensor measuring total filter ΔP.

Source truth:   truth_state["dp_total_pa"]  (Pa, written by HydraulicsModel)
Sensor output:  sensor_state["dp_sensor_kPa"]

Model:
    sensor_dp = max(0, truth_dp_kPa + N(0, σ_dp) + drift_t + fouling_gain × mesh_loading_avg)
    drift_t   = clip(drift_{t-1} + N(0, σ_drift_per_step), −drift_max, +drift_max)

Latency is implemented via a circular deque: the sensor reports the measurement
from `dp_latency_steps` steps ago (1–2 steps ≈ 10–200 ms at 10 Hz control loop).

Physics/architecture constraints:
- Reads ONLY from truth_state (no sensor_state reads).
- Writes ONLY to sensor_state["dp_sensor_kPa"].
- Does NOT write to truth_state.

Parameter defaults (source: M6 Sensor Realism Research Brief, 2026-04-13):
    dp_noise_kPa:              0.25   MEMS DP noise floor (~0.5% FS, 0–50 kPa)
    dp_drift_rate_kPa_per_step:0.0005 Random walk drift per step (calibration-pending field value)
    dp_drift_max_kPa:          2.0    Drift accumulation cap (conservative)
    dp_fouling_gain:           0.5    Sensor port fouling bias proportional to mesh_loading_avg
                                      (mechanism confirmed; gain calibration-pending)
    dp_latency_steps:          1      1-step observation lag (physically realistic for MEMS)
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional

import numpy as np


class DifferentialPressureSensor:
    """
    MEMS differential pressure sensor model for M6 sensor realism.

    READS from truth_state:  dp_total_pa, mesh_loading_avg
    WRITES to sensor_state:  dp_sensor_kPa
    NEVER writes to truth_state.
    """

    def __init__(self, cfg: Any | None = None) -> None:
        s_raw: Dict[str, Any] = {}
        if cfg is not None and hasattr(cfg, "raw"):
            s_raw = getattr(cfg, "raw", {}).get("sensors", {}).get("pressure", {}) or {}

        self._noise_kPa:   float = float(s_raw.get("dp_noise_kPa",               0.25))
        self._drift_rate:  float = float(s_raw.get("dp_drift_rate_kPa_per_step", 0.0005))
        self._drift_max:   float = float(s_raw.get("dp_drift_max_kPa",           2.0))
        self._fouling_gain: float = float(s_raw.get("dp_fouling_gain",           0.5))
        self._latency:     int   = int(s_raw.get("dp_latency_steps",             1))

        # Internal state — reset on each episode
        self._drift: float = 0.0
        # Circular buffer: maxlen = latency + 1 so buf[0] is always `latency` steps behind
        self._buf: deque[float] = deque([0.0] * (self._latency + 1), maxlen=self._latency + 1)

    # ------------------------------------------------------------------
    # Public API  (mirrors OpticalSensorArray interface)
    # ------------------------------------------------------------------

    def reset(
        self,
        truth_state: Dict[str, float],
        sensor_state: Optional[Dict[str, float]] = None,
    ) -> None:
        """Reset drift accumulator and latency buffer; zero sensor_state key."""
        self._drift = 0.0
        self._buf = deque([0.0] * (self._latency + 1), maxlen=self._latency + 1)
        if sensor_state is not None:
            sensor_state["dp_sensor_kPa"] = 0.0

    def update(
        self,
        truth_state: Dict[str, float],
        sensor_state: Dict[str, float],
        dt: float = 0.1,
    ) -> None:
        """
        Compute noisy/drifted ΔP reading and write to sensor_state only.

        Must be called after HydraulicsModel.update() so dp_total_pa is current.
        """
        # 1. Read truth ΔP (Pa → kPa)
        truth_dp_kPa = float(truth_state.get("dp_total_pa", 0.0)) / 1000.0

        # 2. Additive Gaussian noise
        noise = float(np.random.randn()) * self._noise_kPa

        # 3. Advance random-walk drift; cap within ±drift_max
        drift_step = float(np.random.randn()) * self._drift_rate
        self._drift = float(np.clip(self._drift + drift_step, -self._drift_max, self._drift_max))

        # 4. Fouling-induced sensor offset (proportional to average mesh loading)
        #    Mechanism confirmed in literature; gain requires field calibration.
        mesh_avg = float(truth_state.get("mesh_loading_avg", 0.0))
        fouling_offset = self._fouling_gain * mesh_avg

        # 5. Instantaneous measurement (clipped to non-negative)
        measured = float(max(0.0, truth_dp_kPa + noise + self._drift + fouling_offset))

        # 6. Push through latency buffer; report the value from `latency` steps ago
        self._buf.append(measured)
        delayed = float(self._buf[0])  # oldest element = `latency` steps behind

        # 7. Write to sensor_state ONLY — never touch truth_state
        sensor_state["dp_sensor_kPa"] = float(max(0.0, delayed))
