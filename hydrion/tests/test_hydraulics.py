import numpy as np
from hydrion.physics.hydraulics import HydraulicsModel
from hydrion.physics.clogging import CloggingModel
from hydrion.config import HydrionConfig
import yaml


def load_cfg():
    with open("hydrion/configs/default.yaml", "r") as f:
        raw = yaml.safe_load(f)
    return HydrionConfig(raw)


def test_hydraulics_basic():
    cfg = load_cfg()

    hydraulics = HydraulicsModel(cfg)
    clogging = CloggingModel(cfg)

    state = {}
    clogging.reset(state)

    print("\n--- HydraulicsModel Test ---")

    action = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)

    for step in range(5):
        hydraulics.update(state, dt=0.1, action=action, clogging_model=clogging)
        print(
            f"Step {step}: Q={state['Q_out_Lmin']:.2f} L/min, "
            f"P_in={state['P_in']:.1f} Pa"
        )


if __name__ == "__main__":
    test_hydraulics_basic()
