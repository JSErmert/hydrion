# tests/test_electrostatics.py
# obs12_v2 — updated for ElectrostaticsModel v2 (M3)
import numpy as np

from hydrion.physics.electrostatics import ElectrostaticsModel


def test_electrostatics_basic():
    electro = ElectrostaticsModel(cfg=None)
    state = {"q_processed_lmin": 13.5}

    electro.reset(state)

    print("\n--- ElectrostaticsModel v2 Test ---")
    node_cmd = 0.8  # 80% of V_max_realism
    dt = 0.1

    for i in range(10):
        electro.update(state, dt=dt, node_cmd=node_cmd)
        est = electro.get_state()
        print(
            f"Step {i:02d}: V_node={est['V_node']:7.1f} V, "
            f"E_field_kVm={est['E_field_kVm']:.2f} kV/m, "
            f"E_field_norm={est['E_field_norm']:.3f}, "
            f"E_capture_gain={est['E_capture_gain']:.3f}"
        )

    # Assertions: v2 keys present, v1 keys absent
    assert "E_field_kVm"       in est, "E_field_kVm missing from state"
    assert "E_field_norm"      in est, "E_field_norm missing from state"
    assert "E_capture_gain"    in est, "E_capture_gain missing from state"
    assert "charge_factor"     in est, "charge_factor missing from state"
    assert "node_capture_gain" in est, "node_capture_gain missing from state"
    assert "E_norm"            not in est, "E_norm should be removed (obs12_v1 key)"

    # E_field_norm must be in [0, 1]
    assert 0.0 <= est["E_field_norm"] <= 1.0, f"E_field_norm out of range: {est['E_field_norm']}"

    # V_node must not exceed V_hard_clamp
    assert est["V_node"] <= 3000.0, f"V_node={est['V_node']} exceeded V_hard_clamp"

    # E_capture_gain must be positive when voltage is on
    assert est["E_capture_gain"] > 0.0, "E_capture_gain should be positive with node_cmd=0.8"


if __name__ == "__main__":
    test_electrostatics_basic()
