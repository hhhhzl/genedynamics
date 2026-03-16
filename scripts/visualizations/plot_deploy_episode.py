#!/usr/bin/env python3
"""
Plot deploy episode trajectory (3D).

Visualizes deploy simulation from telemetry.json or states.npy/actions.npy.
Use this to inspect deploy simulation results.

Usage:
    python scripts/visualizations/plot_deploy_episode.py results/deploy/uav3d_sim/episodes/ep_0001_20260303-144543
    python scripts/visualizations/plot_deploy_episode.py results/deploy/uav3d_mbd_sim/episodes/ep_0001_* --save
    python scripts/visualizations/plot_deploy_episode.py results/deploy/uav3d_sim --episodes 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def load_episode(ep_dir: Path) -> dict:
    """Load episode from telemetry.json or states.npy."""
    data = {}
    if (ep_dir / "telemetry.json").exists():
        with open(ep_dir / "telemetry.json") as f:
            data = json.load(f)
    elif (ep_dir / "states.npy").exists():
        import numpy as np
        states = np.load(ep_dir / "states.npy")
        actions = np.load(ep_dir / "actions.npy")
        data["states"] = states.tolist()
        data["actions"] = actions.tolist()
    if (ep_dir / "meta.json").exists():
        with open(ep_dir / "meta.json") as f:
            data["meta"] = json.load(f)
    return data


def plot_uav_3d(ep_dir: Path, data: dict, target: tuple, save: bool = False,
                show: bool = True) -> None:
    """Plot UAV 3D trajectory from deploy episode."""
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    states = data.get("states", [])
    if not states:
        print(f"  No states in {ep_dir.name}")
        return

    pos = np.array([s[:3] for s in states])
    target = np.asarray(target, dtype=np.float32)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color="#1f77b4", linewidth=2, label="Trajectory")
    ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color="green", s=100, marker="o", label="Start")
    ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], color="#1f77b4", s=150, marker="*", label="End")
    ax.scatter(target[0], target[1], target[2], color="red", s=200, marker="*", label="Target")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Auto bounds
    margin = 0.5
    for idx, setter in [(0, ax.set_xlim), (1, ax.set_ylim), (2, ax.set_zlim)]:
        lo = min(pos[:, idx].min(), target[idx]) - margin
        hi = max(pos[:, idx].max(), target[idx]) + margin
        setter(lo, hi)

    meta = data.get("meta", {})
    steps = meta.get("steps", len(states) - 1)
    ax.set_title(f"Deploy episode: {ep_dir.name} ({steps} steps)")

    if save:
        out = ep_dir / "trajectory_3d.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot deploy episode trajectory (3D)")
    parser.add_argument("path", type=str, help="Episode dir or results/deploy/{robot}_sim")
    parser.add_argument("--target", type=str, default=None, help="Target as 'x,y,z'")
    parser.add_argument("--save", action="store_true", help="Save trajectory_3d.png to episode dir")
    parser.add_argument("--no-show", action="store_true", help="Don't show interactive plot (use with --save)")
    parser.add_argument("--episodes", type=int, default=1, help="Max episodes to plot when path is a dir")
    args = parser.parse_args()

    path = Path(args.path).resolve()
    if not path.exists():
        print(f"Path not found: {path}")
        return 1

    target = (0.0, 0.0, 1.0)
    if args.target:
        target = tuple(float(x) for x in args.target.split(","))

    if path.is_dir() and path.name.startswith("ep_"):
        ep_dirs = [path]
    elif path.is_dir() and (path / "episodes").exists():
        ep_dirs = sorted((path / "episodes").iterdir(), key=lambda p: p.name)
        ep_dirs = [d for d in ep_dirs if d.is_dir() and d.name.startswith("ep_")]
        ep_dirs = ep_dirs[: args.episodes]
    else:
        parent = path.parent
        pattern = path.name
        ep_dirs = list(parent.glob(pattern)) if "*" in pattern else [path]
        ep_dirs = ep_dirs[: args.episodes]

    if not ep_dirs:
        print(f"No episodes found at {path}")
        return 1

    for ep_dir in ep_dirs:
        if not ep_dir.is_dir():
            continue
        print(f"\n=== {ep_dir.name} ===")
        data = load_episode(ep_dir)
        if not data:
            print("  No data found")
            continue
        plot_uav_3d(ep_dir, data, target, save=args.save, show=not args.no_show)

    return 0


if __name__ == "__main__":
    sys.exit(main())
