# hydrion/sensors/optical.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class OpticalParams:
    """
    Optical sensor model (v2).

    Uses size-resolved particle outputs from ParticleModel v2.1:

        C_out_ultra, C_out_fine, C_out_small, C_out_medium, C_out_large
        charge_mean

    Produces:
        sensor_turbidity
        sensor_scatter
        camera_signal
    """

    # Weight contributions to turbidity (fine particles dominate)
    w_ultra_turb: float = 1.20
    w_fine_turb: float = 1.00
    w_small_turb: float = 0.70
    w_medium_turb: float = 0.30
    w_large_turb: float = 0.10

    # Weight contributions to scatter (fibers dominate)
    w_ultra_scat: float = 0.10
    w_fine_scat: float = 0.25
    w_small_scat: float = 0.70
    w_medium_scat: float = 1.00
    w_large_scat: float = 1.20

    # Charge effects (adds brightness + reduces apparent turbidity)
    charge_brightness_boost: float = 0.25
    charge_turbidity_reduction: float = 0.15

    # Noise parameters
    noise_turb: float = 0.02
    noise_scat: float = 0.02
    noise_cam: float = 0.02

    eps: float = 1e-6


class OpticalSensorArray:
    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        
        # FIX: HydrionConfig stores data under cfg.raw
        sensors_cfg = cfg.raw.get("sensors", {}) if hasattr(cfg, "raw") else {}

        self.params = OpticalParams(
            w_ultra_turb=float(sensors_cfg.get("w_ultra_turb", 1.20)),
            w_fine_turb=float(sensors_cfg.get("w_fine_turb", 1.00)),
            w_small_turb=float(sensors_cfg.get("w_small_turb", 0.70)),
            w_medium_turb=float(sensors_cfg.get("w_medium_turb", 0.30)),
            w_large_turb=float(sensors_cfg.get("w_large_turb", 0.10)),
            w_ultra_scat=float(sensors_cfg.get("w_ultra_scat", 0.10)),
            w_fine_scat=float(sensors_cfg.get("w_fine_scat", 0.25)),
            w_small_scat=float(sensors_cfg.get("w_small_scat", 0.70)),
            w_medium_scat=float(sensors_cfg.get("w_medium_scat", 1.00)),
            w_large_scat=float(sensors_cfg.get("w_large_scat", 1.20)),
            charge_brightness_boost=float(sensors_cfg.get("charge_brightness_boost", 0.25)),
            charge_turbidity_reduction=float(sensors_cfg.get("charge_turbidity_reduction", 0.15)),
            noise_turb=float(sensors_cfg.get("noise_turb", 0.02)),
            noise_scat=float(sensors_cfg.get("noise_scat", 0.02)),
            noise_cam=float(sensors_cfg.get("noise_cam", 0.02)),
        )


    def reset(self, state: Dict[str, float]) -> None:
        state["sensor_turbidity"] = 0.0
        state["sensor_scatter"] = 0.0
        state["camera_signal"] = 0.0

    def update(self, state: Dict[str, float], dt: float) -> None:
        p = self.params

        # Read per-bin concentrations
        C = {
            "ultra": float(state.get("C_out_ultra", 0.0)),
            "fine": float(state.get("C_out_fine", 0.0)),
            "small": float(state.get("C_out_small", 0.0)),
            "medium": float(state.get("C_out_medium", 0.0)),
            "large": float(state.get("C_out_large", 0.0)),
        }

        charge_mean = float(state.get("charge_mean", 0.0))

        # -------------------------------
        # TURBIDITY MODEL (v2)
        # -------------------------------
        turb_base = (
            p.w_ultra_turb * C["ultra"]
            + p.w_fine_turb * C["fine"]
            + p.w_small_turb * C["small"]
            + p.w_medium_turb * C["medium"]
            + p.w_large_turb * C["large"]
        )

        # Charge makes turbidity appear lower (particles align, less scattering)
        turb = turb_base * (1.0 - p.charge_turbidity_reduction * charge_mean)
        turb += np.random.normal(0.0, p.noise_turb)
        turb = float(np.clip(turb, 0.0, 1.0))

        # -------------------------------
        # SCATTER MODEL (v2)
        # -------------------------------
        scat = (
            p.w_ultra_scat * C["ultra"]
            + p.w_fine_scat * C["fine"]
            + p.w_small_scat * C["small"]
            + p.w_medium_scat * C["medium"]
            + p.w_large_scat * C["large"]
        )
        scatter = scat + np.random.normal(0.0, p.noise_scat)
        scatter = float(np.clip(scatter, 0.0, 1.0))

        # -------------------------------
        # CAMERA SIGNAL (v2)
        # -------------------------------
        # Bright scatter increases camera activity
        # High turbidity darkens image
        # Charge adds subtle emissive brightness (for your glowing particles)
        camera_signal = (
            0.6 * scatter
            + 0.4 * (1.0 - turb)
            + p.charge_brightness_boost * charge_mean
        )
        camera_signal += np.random.normal(0.0, p.noise_cam)
        camera_signal = float(np.clip(camera_signal, 0.0, 1.0))

        state["sensor_turbidity"] = turb
        state["sensor_scatter"] = scatter
        state["camera_signal"] = camera_signal
