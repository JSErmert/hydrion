# tests/test_electrostatics.py
import numpy as np

from hydrion.physics.electrostatics import ElectrostaticsModel


def test_electrostatics_basic():
    electro = ElectrostaticsModel(cfg=None)
    state = {"Q_out_Lmin": 10.0}

    electro.reset(state)

    print("\n--- ElectrostaticsModel Test ---")
    node_cmd = 0.8  # 80% of V_max
    dt = 0.1

    for i in range(10):
        electro.update(state, dt=dt, node_cmd=node_cmd)
        est = electro.get_state()
        print(
            f"Step {i:02d}: V_node={est['V_node']:7.1f} V, "
            f"E_norm={est['E_norm']:.3f}"
        )


if __name__ == "__main__":
    test_electrostatics_basic()
