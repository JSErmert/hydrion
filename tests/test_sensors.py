# tests/test_sensors.py
import numpy as np

from hydrion.sensors.optical import OpticalSensorArray


def test_sensors_basic():
    sensors = OpticalSensorArray(cfg=None)
    state = {
        "Q_out_Lmin": 10.0,
        "mesh_loading_avg": 0.0,
        "capture_eff": 0.8,
        "C_out": 0.5,
    }

    sensors.reset(state)

    print("\n--- OpticalSensorArray Test ---")
    dt = 0.1

    for i in range(10):
        # Simulate slowly rising clogging & particles
        state["mesh_loading_avg"] = min(1.0, state["mesh_loading_avg"] + 0.05)
        state["C_out"] = min(1.0, state["C_out"] + 0.03)

        sensors.update(state, dt=dt)
        print(
            f"Step {i:02d}: turb={state['sensor_turbidity']:.3f}, "
            f"scatter={state['sensor_scatter']:.3f}, "
            f"camera={state['sensor_camera']:.3f}"
        )


if __name__ == "__main__":
    test_sensors_basic()
