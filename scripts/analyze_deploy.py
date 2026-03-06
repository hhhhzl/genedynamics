#!/usr/bin/env python3
"""
Analyze Deploy episode results and validate against expectations.

Usage:
    python scripts/analyze_deploy.py results/deploy/uav3d_sim/episodes/ep_0001_20260303-144543
    python scripts/analyze_deploy.py results/deploy/quadruped_sim/episodes/ep_0001_*
    python scripts/analyze_deploy.py results/deploy --robot uav3d
"""

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
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
        data["states"] = np.load(ep_dir / "states.npy").tolist()
        data["actions"] = np.load(ep_dir / "actions.npy").tolist()
    if (ep_dir / "meta.json").exists():
        with open(ep_dir / "meta.json") as f:
            data["meta"] = json.load(f)
    return data


def analyze_uav(data: dict, target: tuple = (0.0, 0.0, 1.0)) -> dict:
    """Analyze UAV episode: state [x,y,z, vx,vy,vz, roll,pitch,yaw, wx,wy,wz]."""
    states = data.get("states", [])
    if not states:
        return {"error": "No states"}
    target = tuple(target)
    first = states[0][:3]
    last = states[-1][:3]
    dist = sum((last[i] - target[i]) ** 2 for i in range(3)) ** 0.5
    z_min = min(s[2] for s in states)
    z_max = max(s[2] for s in states)
    return {
        "start_pos": first,
        "end_pos": last,
        "target": target,
        "distance_to_target": round(dist, 4),
        "z_range": (round(z_min, 3), round(z_max, 3)),
        "steps": len(states) - 1,
        "reached_target": dist < 0.35,
        "crashed": z_min < -0.5,
    }


def analyze_quadruped(data: dict, target: tuple = (2.0, 0.0, 0.5)) -> dict:
    """Analyze quadruped episode: state[:3] = base xyz."""
    states = data.get("states", [])
    if not states:
        return {"error": "No states"}
    target = tuple(target)
    first = [s[0] if hasattr(s, "__getitem__") else s for s in states[0][:3]]
    last = [s[0] if hasattr(s, "__getitem__") else s for s in states[-1][:3]]
    first = [float(x) for x in first]
    last = [float(x) for x in last]
    dist = sum((last[i] - target[i]) ** 2 for i in range(3)) ** 0.5
    return {
        "start_pos": first,
        "end_pos": last,
        "target": target,
        "distance_to_target": round(dist, 4),
        "steps": len(states) - 1,
        "reached_target": dist < 0.35,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze Deploy episode results")
    parser.add_argument("path", type=str, help="Episode dir or results/deploy/{robot}_sim")
    parser.add_argument("--robot", type=str, default="auto", choices=["auto", "uav3d", "quadruped", "humanoid"])
    parser.add_argument("--target", type=str, default=None, help="Target as 'x,y,z'")
    args = parser.parse_args()

    path = Path(args.path).resolve()
    if not path.exists():
        print(f"Path not found: {path}")
        return 1

    # Resolve episode dir(s)
    if path.is_dir() and path.name.startswith("ep_"):
        ep_dirs = [path]
    elif path.is_dir() and (path / "episodes").exists():
        ep_dirs = sorted((path / "episodes").iterdir(), key=lambda p: p.name)
        ep_dirs = [d for d in ep_dirs if d.is_dir() and d.name.startswith("ep_")]
    else:
        parent = path.parent
        pattern = path.name
        ep_dirs = list(parent.glob(pattern)) if "*" in pattern else [path]

    if not ep_dirs:
        print(f"No episodes found at {path}")
        return 1

    target = None
    if args.target:
        target = tuple(float(x) for x in args.target.split(","))

    for ep_dir in ep_dirs:
        if not ep_dir.is_dir():
            continue
        print(f"\n=== {ep_dir.name} ===")
        data = load_episode(ep_dir)
        if not data:
            print("  No data found")
            continue

        meta = data.get("meta", {})
        print(f"  Steps: {meta.get('steps', len(data.get('states', [])) - 1)}")
        print(f"  State dim: {meta.get('state_dim', '?')}")

        robot = args.robot
        if robot == "auto":
            if "uav3d" in str(path):
                robot = "uav3d"
            elif "quadruped" in str(path):
                robot = "quadruped"
            else:
                robot = "uav3d"

        if robot == "uav3d":
            tgt = target or (0.0, 0.0, 1.0)
            result = analyze_uav(data, tgt)
        elif robot == "quadruped":
            tgt = target or (2.0, 0.0, 0.5)
            result = analyze_quadruped(data, tgt)
        else:
            print("  Humanoid analysis not implemented")
            continue

        if "error" in result:
            print(f"  Error: {result['error']}")
            continue

        print(f"  Start: {result['start_pos']}")
        print(f"  End:   {result['end_pos']}")
        print(f"  Target: {result['target']}")
        print(f"  Distance to target: {result['distance_to_target']}")
        if "z_range" in result:
            print(f"  Z range: {result['z_range']}")
        print(f"  Reached target: {'✓' if result['reached_target'] else '✗'}")
        if result.get("crashed"):
            print("  WARNING: Drone may have crashed (z < -0.5)")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
