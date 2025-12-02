# hydrion/sensors/optical.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class OpticalSensorParams:
    """
    Simple optical sensor model for micro-particle detection.

    Outputs (all in [0, 1]):
        sensor_turbidity   ~ optical density / cloudiness
        sensor_scatter     ~ scattered light intensity
        sensor_camera      ~ normalized camera pixel mean
    """

    # Reference scales
    Q_ref_Lmin: float = 20.0   # flow at which signal is "nominal"
    turbidity_gain_clog: float = 0.8
    turbidity_gain_capture: float = 0.4
    turbidity_gain_particles: float = 0.6

    scatter_gain_flow: float = 0.4
    scatter_gain_particles: float = 0.6

    # Noise
    turbidity_noise_std: float = 0.01
    scatter_noise_std: float = 0.01
    camera_noise_std: float = 0.02

    # Clamp to [0, 1]
    eps: float = 1e-8


class OpticalSensorArray:
    """
    Optical sensor "bar" that looks at the outflow.

    Reads from env state:
        Q_out_Lmin
        mesh_loading_avg
        capture_eff
        C_out (downstream particle concentration, optional)

    Writes:
        sensor_turbidity  in [0, 1]
        sensor_scatter    in [0, 1]
        sensor_camera     in [0, 1]
    """

    def __init__(self, cfg: Any | None = None) -> None:
        # Allow future config-driven params, but defaults are fine for now
        s_raw: Dict[str, float] = {}
        if cfg is not None and hasattr(cfg, "raw"):
            s_raw = getattr(cfg, "raw", {}).get("sensors", {}).get("optical", {}) or {}

        self.params = OpticalSensorParams(
            Q_ref_Lmin=float(s_raw.get("Q_ref_Lmin", 20.0)),
            turbidity_gain_clog=float(s_raw.get("turbidity_gain_clog", 0.8)),
            turbidity_gain_capture=float(s_raw.get("turbidity_gain_capture", 0.4)),
            turbidity_gain_particles=float(s_raw.get("turbidity_gain_particles", 0.6)),
            scatter_gain_flow=float(s_raw.get("scatter_gain_flow", 0.4)),
            scatter_gain_particles=float(s_raw.get("scatter_gain_particles", 0.6)),
            turbidity_noise_std=float(s_raw.get("turbidity_noise_std", 0.01)),
            scatter_noise_std=float(s_raw.get("scatter_noise_std", 0.01)),
            camera_noise_std=float(s_raw.get("camera_noise_std", 0.02)),
            eps=float(s_raw.get("eps", 1e-8)),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self, state: Dict[str, float]) -> None:
        state["sensor_turbidity"] = 0.0
        state["sensor_scatter"] = 0.0
        state["sensor_camera"] = 0.0

    def update(self, state: Dict[str, float], dt: float) -> None:
        p = self.params

        Q = float(state.get("Q_out_Lmin", 0.0))
        mesh_avg = float(state.get("mesh_loading_avg", 0.0))
        capture_eff = float(state.get("capture_eff", 0.8))
        C_out = float(state.get("C_out", 0.5))  # downstream particle conc in [0,1] ish

        # Normalize flow
        flow_norm = float(np.clip(Q / max(p.Q_ref_Lmin, p.eps), 0.0, 2.0))

        # --- Turbidity -------------------------------------------------
        # More clogging, more particles, and *lower* capture efficiency
        turbidity = (
            p.turbidity_gain_clog * mesh_avg
            + p.turbidity_gain_particles * C_out
            + p.turbidity_gain_capture * (1.0 - capture_eff)
        )
        turbidity = float(np.clip(turbidity + np.random.randn() * p.turbidity_noise_std, 0.0, 1.0))

        # --- Scatter ---------------------------------------------------
        scatter = (
            p.scatter_gain_flow * flow_norm
            + p.scatter_gain_particles * C_out
        )
        scatter = float(np.clip(scatter + np.random.randn() * p.scatter_noise_std, 0.0, 1.0))

        # --- Camera "pixel mean" --------------------------------------
        # Think of this as a fused brightness + turbidity metric
        camera = 0.5 * scatter + 0.5 * (1.0 - turbidity)
        camera = float(np.clip(camera + np.random.randn() * p.camera_noise_std, 0.0, 1.0))

        state["sensor_turbidity"] = turbidity
        state["sensor_scatter"] = scatter
        state["sensor_camera"] = camera
