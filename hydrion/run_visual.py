"""
Run a single visualized Hydrion episode (Commit 5).
"""

import argparse

from hydrion.env import HydrionEnv
from hydrion.visualize_episode import visualize_episode
from hydrion.rendering.renderer_mpl import MatplotlibRenderer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=333)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Optional directory to save rendered frames",
    )
    args = parser.parse_args()

    # ----------------------------
    # Create environment
    # ----------------------------
    env = HydrionEnv(
        config_path="configs/default.yaml",
        seed=args.seed,
    )

    # ----------------------------
    # Create renderer
    # ----------------------------
    renderer = MatplotlibRenderer(
        save_dir=args.save_dir,
    )

    # ----------------------------
    # Run visualization
    # ----------------------------
    visualize_episode(
        env=env,
        renderer=renderer,
        seed=args.seed,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
