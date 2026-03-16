#!/usr/bin/env python3
"""
Render unified robot motion visualization (GIF + HTML).

Supports two sources:
- experiments seed directory: .../level_x/seed_y
- deploy episode directory: .../episodes/ep_XXXX_*
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

if "MUJOCO_GL" not in os.environ and os.name == "posix":
    os.environ.setdefault("MUJOCO_GL", "egl")

from genedynamics.viz.motion_episode import MotionEpisode
from genedynamics.viz.motion_renderer import MotionRenderer


def main() -> int:
    parser = argparse.ArgumentParser(description="Render unified robot motion HTML/GIF.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed-dir", type=str, default=None, help="Experiments seed dir (contains trajectory/trajectory.json)")
    group.add_argument("--episode-dir", type=str, default=None, help="Deploy episode dir (contains states.npy/actions.npy)")
    parser.add_argument("--output-dir", type=str, default="results/visualizations/motion")
    parser.add_argument("--name", type=str, default="motion")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--best-idx", type=int, default=None, help="Override best_idx for experiment trajectory selection")
    args = parser.parse_args()

    if args.seed_dir:
        episode = MotionEpisode.from_experiment_seed_dir(
            args.seed_dir,
            best_idx=args.best_idx,
            fps=args.fps,
        )
    else:
        episode = MotionEpisode.from_deploy_episode_dir(args.episode_dir, fps=args.fps)

    renderer = MotionRenderer(args.output_dir)
    html_path = renderer.render_html(
        episode,
        name=args.name,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    print(f"Rendered: {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
