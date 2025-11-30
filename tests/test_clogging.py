import numpy as np
from hydrion.physics.clogging import CloggingModel
from hydrion.config import HydrionConfig
import yaml


def load_cfg():
    try:
        with open("configs/default.yaml", "r") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        raw = {"sim": {"dt": 0.1}, "clogging": {}}
    return HydrionConfig(raw)


def test_clogging_basic():
    cfg = load_cfg()
    clog = CloggingModel(cfg)

    state = {"Q_out_Lmin": 12.0}

    clog.reset(state)

    print("\n--- CloggingModel Test ---")

    for i in range(10):
        clog.update(state, dt=0.1)
        print(
            f"Step {i:02d}: "
            f"Mc1={state['Mc1']:.6f}, "
            f"mesh_avg={state['mesh_loading_avg']:.6f}, "
            f"eff={state['capture_eff']:.3f}"
        )


if __name__ == "__main__":
    test_clogging_basic()
