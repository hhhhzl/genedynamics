#!/usr/bin/env python3
"""
Render quadruped (Go2) deploy episode as MuJoCo simulation GIF.

Uses Unitree Go2 model from mujoco_menagerie. Requires MUJOCO_MENAGERIE_PATH
or pip install mujoco-menagerie.

Usage:
    python scripts/render_deploy_quadruped_gif.py results/deploy/quadruped_go2_mbd_sim --episodes 3
    python scripts/render_deploy_quadruped_gif.py results/deploy/quadruped_go2_mbd_sim/episodes/ep_0001_*
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

if "MUJOCO_GL" not in os.environ:
    import platform
    if platform.system() == "Linux":
        os.environ.setdefault("MUJOCO_GL", "egl")


def load_episode(ep_dir: Path) -> dict:
    """Load episode from states.npy (preferred) or telemetry.json."""
    data = {}
    if (ep_dir / "states.npy").exists():
        import numpy as np
        states = np.load(ep_dir / "states.npy")
        actions = np.load(ep_dir / "actions.npy")
        data["states"] = states.tolist()
        data["actions"] = actions.tolist()
    elif (ep_dir / "telemetry.json").exists():
        with open(ep_dir / "telemetry.json") as f:
            data = json.load(f)
    return data


def render_quadruped_episode_gif(
    ep_dir: Path,
    data: dict,
    output_path: Path | None = None,
    width: int = 640,
    height: int = 480,
    fps: float = 20.0,
    duration_ms: float | None = None,
    fixed_camera: bool = True,
    draw_trajectory: bool = True,
    model: str = "go2",
) -> Path:
    """Render quadruped episode states as MuJoCo GIF using Go2 or ant model."""
    import tempfile
    import numpy as np
    import mujoco
    from genedynamics.envs.utils.mujoco_model_generator import (
        create_go2_render_xml_with_trajectory,
        create_ant_render_xml_with_trajectory,
    )

    states = data.get("states", [])
    if not states:
        raise ValueError(f"No states in {ep_dir.name}")

    # Forward-fill NaN (e.g. from failed Go2 sim) so render doesn't produce black frames
    states = [np.asarray(s, dtype=np.float64).ravel() for s in states]
    last_valid = None
    for i in range(len(states)):
        if np.any(~np.isfinite(states[i])):
            if last_valid is not None:
                states[i] = last_valid.copy()
        else:
            last_valid = states[i].copy()
    if last_valid is None:
        raise ValueError(f"All states have NaN in {ep_dir.name}")

    # Auto-detect model from state size: Ant=29, Go2=37
    if model == "auto":
        state_dim = len(states[0])
        model = "go2" if state_dim >= 35 else "ant"

    positions = np.asarray([s[:3] for s in states], dtype=np.float64)
    # Write XML to same dir as model so mesh paths (assets/*.obj) resolve
    model_dir = None
    if model == "go2":
        try:
            from genedynamics.robots.registry import _get_go2_path
            p = _get_go2_path()
            if p:
                model_dir = Path(p).parent
        except Exception:
            pass
    else:
        try:
            from genedynamics.robots.registry import _get_ant_path
            p = _get_ant_path()
            if p:
                model_dir = Path(p).parent
        except Exception:
            pass
    if model_dir and model_dir.exists():
        tmp_xml = str(model_dir / "_render_traj_temp.xml")
    else:
        fd, tmp_xml = tempfile.mkstemp(suffix=".xml")
        os.close(fd)

    used_model = model
    try:
        if model == "go2":
            try:
                create_go2_render_xml_with_trajectory(
                    tmp_xml,
                    trajectory_positions=positions if draw_trajectory else [],
                )
            except FileNotFoundError:
                create_ant_render_xml_with_trajectory(
                    tmp_xml,
                    trajectory_positions=positions if draw_trajectory else [],
                )
                used_model = "ant"
        else:
            create_ant_render_xml_with_trajectory(
                tmp_xml,
                trajectory_positions=positions if draw_trajectory else [],
            )
    except FileNotFoundError as e:
        if os.path.exists(tmp_xml):
            os.unlink(tmp_xml)
        raise RuntimeError(
            f"Model not found: {e}. "
            "For go2: set MUJOCO_MENAGERIE_PATH or pip install mujoco-menagerie. "
            "For ant: pip install gymnasium."
        ) from e

    model_obj = mujoco.MjModel.from_xml_path(tmp_xml)
    data_obj = mujoco.MjData(model_obj)

    nq = model_obj.nq
    nv = model_obj.nv

    try:
        renderer = mujoco.Renderer(model_obj, height=height, width=width)
    except Exception:
        renderer = None

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.azimuth = 135
    camera.elevation = -20
    if fixed_camera:
        traj_center = np.nanmean(positions, axis=0)
        if np.any(~np.isfinite(traj_center)):
            traj_center = positions[0]
        dists = np.linalg.norm(positions - traj_center, axis=1)
        max_dist = float(np.nanmax(dists)) if np.any(np.isfinite(dists)) else 2.0
        camera.lookat[:] = traj_center
        camera.distance = max(3.0, max_dist * 2.5)
    else:
        camera.distance = 2.5
        camera.lookat[:] = positions[0]

    frames = []
    for s in states:
        state = np.asarray(s, dtype=np.float64).ravel()
        if np.any(~np.isfinite(state)):
            continue  # skip invalid frames (should not happen after forward-fill)
        # Handle state/model size mismatch (e.g. ant episode with Go2 model or vice versa)
        qpos = np.zeros(nq, dtype=np.float64)
        qvel = np.zeros(nv, dtype=np.float64)
        qpos[: min(nq, len(state))] = state[: min(nq, len(state))]
        if len(state) > nq:
            qvel[: min(nv, len(state) - nq)] = state[nq : nq + min(nv, len(state) - nq)]
        data_obj.qpos[:] = qpos
        data_obj.qvel[:] = qvel
        mujoco.mj_forward(model_obj, data_obj)

        if renderer is not None:
            if not fixed_camera:
                camera.lookat[:] = state[:3]
            renderer.update_scene(data_obj, camera=camera)
            rgb = renderer.render()
            if rgb is not None:
                frames.append(rgb)

    if renderer is not None:
        try:
            renderer.close()
        except Exception:
            pass
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
        imageio.v3.imwrite(out, frames, duration=duration_ms, loop=0)
    else:
        imageio.v3.imwrite(out, frames, fps=fps, loop=0)

    return Path(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render quadruped deploy episode as MuJoCo GIF")
    parser.add_argument(
        "paths",
        nargs="+",
        type=str,
        help="Episode dir(s) or results/deploy/quadruped_*_sim",
    )
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--duration-ms", type=float, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--track-camera", action="store_true")
    parser.add_argument("--no-trajectory-line", action="store_true")
    parser.add_argument("--model", type=str, default="auto",
                        help="Robot model: auto (detect from state size), go2, ant")
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
            print("  No data found")
            continue
        try:
            out_path = Path(args.output) if args.output else None
            result = render_quadruped_episode_gif(
                ep_dir,
                data,
                output_path=out_path,
                width=args.width,
                height=args.height,
                fps=args.fps,
                duration_ms=args.duration_ms,
                fixed_camera=not args.track_camera,
                draw_trajectory=not args.no_trajectory_line,
                model=args.model,
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
