#!/usr/bin/env python3
"""
Render deploy episode as MuJoCo simulation GIF.

Uses drone_full_3d_mujoco to render each state from a deploy episode,
producing a true MuJoCo 3D visualization (not matplotlib).

Usage:
    python scripts/visualizations/render_deploy_mujoco_gif.py results/deploy/uav3d_mbd_sim/episodes/ep_0001_*
    python scripts/visualizations/render_deploy_mujoco_gif.py results/deploy/uav3d_sim --episodes 1
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
    return data


def render_episode_mujoco_gif(ep_dir: Path, data: dict, output_path: Path | None = None,
                              width: int = 640, height: int = 480, fps: float = 20.0,
                              duration_ms: float | None = None,
                              drone_scale: float = 3.0,
                              fixed_camera: bool = True,
                              draw_trajectory: bool = True) -> Path:
    """Render episode states as MuJoCo GIF.
    drone_scale: Scale up drone geometry for visibility (default 3x).
    fixed_camera: If True, use fixed camera showing full trajectory; else track drone.
    draw_trajectory: If True, draw trajectory line in scene.
    """
    import os
    import tempfile
    import numpy as np
    import mujoco
    from genedynamics.envs.factories import make_env
    from genedynamics.envs.utils.state_converter import state_12d_to_mujoco
    from genedynamics.envs.utils.mujoco_model_generator import create_render_xml_with_trajectory

    states = data.get("states", [])
    if not states:
        raise ValueError(f"No states in {ep_dir.name}")

    positions = np.asarray([s[:3] for s in states], dtype=np.float64)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        tmp_xml = f.name
    create_render_xml_with_trajectory(
        tmp_xml,
        trajectory_positions=positions if draw_trajectory else [],
        drone_scale=drone_scale,
    )
    env = make_env(
        "drone_full_3d_mujoco",
        dt=0.05,
        horizon=len(states),
        target=(0.0, 0.0, 1.0),
        control_limit=1.0,
        model_path=tmp_xml,
    )

    backend = env._physics_backend_instance
    if backend is None:
        if hasattr(env, "close"):
            env.close()
        raise RuntimeError("drone_full_3d_mujoco requires MuJoCo. Install: pip install mujoco")

    model = backend.model
    data_obj = backend.data

    # Use mujoco.Renderer (high-level API, more reliable for offscreen)
    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
    except Exception:
        renderer = None

    nq_free = 7  # 3 pos + 4 quat for freejoint
    nv_free = 6  # 3 vel + 3 ang_vel

    # Camera: fixed (show full trajectory) or tracking
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.azimuth = 135
    camera.elevation = -20
    if fixed_camera:
        traj_center = np.mean(positions, axis=0)
        max_dist = float(np.max(np.linalg.norm(positions - traj_center, axis=1)))
        camera.lookat[:] = traj_center
        camera.distance = max(2.5, max_dist * 2.2)  # Show full trajectory
    else:
        camera.distance = 1.5
        camera.lookat[:] = positions[0]  # Will update per frame

    frames = []
    for s in states:
        state_12d = np.asarray(s, dtype=np.float32)
        mj_state = state_12d_to_mujoco(state_12d)
        data_obj.qpos[:nq_free] = mj_state["qpos"]
        data_obj.qvel[:nv_free] = mj_state["qvel"]
        mujoco.mj_forward(model, data_obj)

        if renderer is not None:
            if not fixed_camera:
                camera.lookat[:] = state_12d[:3]
            renderer.update_scene(data_obj, camera=camera)
            rgb = renderer.render()
            if rgb is not None:
                frames.append(rgb)

    if renderer is not None:
        try:
            renderer.close()
        except Exception:
            pass
    if hasattr(env, "close"):
        env.close()
    if os.path.exists(tmp_xml):
        try:
            os.unlink(tmp_xml)
        except Exception:
            pass

    if not frames:
        raise RuntimeError("No frames rendered")

    try:
        import imageio
    except ImportError:
        raise ImportError("imageio required. Install: pip install imageio")

    out = output_path or (ep_dir / "trajectory_mujoco.gif")
    if duration_ms is not None:
        # duration per frame in ms for GIF
        imageio.v3.imwrite(out, frames, duration=duration_ms, loop=0)
    else:
        # use fps
        imageio.v3.imwrite(out, frames, fps=fps, loop=0)

    return Path(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render deploy episode as MuJoCo GIF")
    parser.add_argument(
        "paths",
        nargs="+",
        type=str,
        help="Episode dir(s) or results/deploy/{robot}_sim (shell glob like ep_0005_* expands to multiple)",
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Output GIF path")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--duration-ms", type=float, default=None, help="Frame duration in ms (overrides fps for GIF)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--drone-scale", type=float, default=3.0, help="Scale drone geometry for visibility (default 3x)")
    parser.add_argument("--track-camera", action="store_true", help="Track drone with camera (default: fixed camera)")
    parser.add_argument("--no-trajectory-line", action="store_true", help="Disable trajectory line overlay")
    args = parser.parse_args()

    ep_dirs = []
    for path_arg in args.paths:
        path = Path(path_arg).resolve()
        if not path.exists():
            print(f"Path not found: {path}")
            continue
        if path.is_dir() and path.name.startswith("ep_"):
            ep_dirs.append(path)
        elif path.is_dir() and (path / "episodes").exists():
            dirs = sorted((path / "episodes").iterdir(), key=lambda p: p.name)
            dirs = [d for d in dirs if d.is_dir() and d.name.startswith("ep_")]
            ep_dirs.extend(dirs[: args.episodes])
        else:
            parent = path.parent
            pattern = path.name
            if "*" in str(pattern):
                ep_dirs.extend(parent.glob(pattern))
            else:
                ep_dirs.append(path)

    ep_dirs = ep_dirs[: args.episodes]
    if not ep_dirs:
        print("No episodes found")
        return 1

    for ep_dir in ep_dirs:
        if not ep_dir.is_dir():
            continue
        print(f"Rendering {ep_dir.name}...")
        data = load_episode(ep_dir)
        if not data:
            print(f"  No data found")
            continue
        try:
            out_path = Path(args.output) if args.output else None
            result = render_episode_mujoco_gif(
                ep_dir,
                data,
                output_path=out_path,
                width=args.width,
                height=args.height,
                fps=args.fps,
                duration_ms=args.duration_ms,
                drone_scale=args.drone_scale,
                fixed_camera=not args.track_camera,
                draw_trajectory=not args.no_trajectory_line,
            )
            print(f"  Saved {result}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
