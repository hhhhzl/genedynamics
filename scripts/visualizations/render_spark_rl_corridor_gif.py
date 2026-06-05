#!/usr/bin/env python
"""Render a MuJoCo GIF of the spark-RL sport-mode controller walking the
14D corridor plan (with the upper-body mapper wired in).

Reads the ``qpos`` / ``qvel`` timeseries dumped by
:mod:`scripts.tasks.robot.humanoid.sport_mode_corridor` and feeds them
into :func:`genedynamics.deploy.observers.mujoco_render.render_episode_to_gif`,
which is the same renderer used by the legacy deploy CLI.

Run inside the dev-cpu Docker image (mujoco + torch needed)::

    docker run --rm -v "$PWD:/work" -w /work genedynamics/dev-cpu:local bash -c \\
      "pip install --quiet torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu && \\
       python scripts/tasks/robot/humanoid/sport_mode_corridor.py \\
           --max-steps 400 --out-dir results/deploy/spark_rl --quiet && \\
       python scripts/visualizations/render_spark_rl_corridor_gif.py"

The defaults assume the diagnose script wrote to
``results/deploy/spark_rl/sport_mode.npz``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


# Path: scripts/visualizations/<file>.py — parents[2] is the project root.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def render(
    npz_path: Path,
    *,
    output_path: Optional[Path] = None,
    width: int = 720,
    height: int = 540,
    fps: float = 50.0,
    speed: float = 1.0,
    draw_trajectory: bool = True,
    cam_distance: float = 3.5,
    cam_azimuth: float = 150.0,
    cam_elevation: float = -14.0,
    follow: bool = True,
    cam_mode: str = "follow",
    every_n: int = 1,
) -> Path:
    """Render the diagnose .npz to a GIF with a follow camera.

    Args:
        npz_path: Path to ``sport_mode.npz`` (must contain
            ``qpos`` and ``qvel`` arrays).
        output_path: Where to write the .gif. Defaults to
            ``<npz_path.parent>/spark_rl_corridor.gif``.
        width / height: Frame size in pixels.
        fps: Output GIF frames-per-second.
        speed: Real-time speed multiplier.
        draw_trajectory: Overlay the pelvis xy trace as a thin tube.
        cam_distance / cam_azimuth / cam_elevation: Free-camera params.
            Defaults match the WBC follower stills (side, ~3.5 m).
        follow: If True, lookat tracks the pelvis every frame so the
            robot stays centred even when it walks for many metres.
            If False, lookat is the trajectory midpoint (the legacy
            :func:`render_episode_to_gif` behaviour, which makes the
            robot tiny on long drifts).
        every_n: Skip every Nth frame to shrink the GIF / speed it up.

    Returns:
        The path to the written .gif.
    """
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Diagnose npz not found: {npz_path}\n"
            "Run scripts/tasks/robot/humanoid/sport_mode_corridor.py first "
            "with --out-dir results/deploy/spark_rl"
        )
    data = np.load(npz_path, allow_pickle=True)
    if "qpos" not in data.files or "qvel" not in data.files:
        raise KeyError(
            f"{npz_path} does not contain qpos / qvel arrays. "
            "Re-run the diagnose script after pulling the latest changes."
        )

    qpos = np.asarray(data["qpos"], dtype=np.float64)
    qvel = np.asarray(data["qvel"], dtype=np.float64)
    if qpos.ndim != 2 or qvel.ndim != 2 or qpos.shape[0] != qvel.shape[0]:
        raise ValueError(
            f"qpos {qpos.shape} and qvel {qvel.shape} have inconsistent shapes"
        )
    if every_n > 1:
        qpos = qpos[::int(every_n)]
        qvel = qvel[::int(every_n)]
    n = qpos.shape[0]
    print(
        f"[render] loaded {n} frames "
        f"(qpos {qpos.shape[1]}, qvel {qvel.shape[1]}) from {npz_path}; "
        f"follow={follow}, every_n={every_n}"
    )

    out = output_path or (npz_path.parent / "spark_rl_corridor.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    # cam_mode supersedes the legacy `follow` boolean when explicit.
    mode = (cam_mode or "follow").lower()
    if mode not in {"follow", "global", "midpoint"}:
        raise ValueError(f"cam_mode must be follow|global|midpoint, got {cam_mode!r}")
    return _render_follow(
        episode_dir=npz_path.parent,
        qpos=qpos,
        qvel=qvel,
        output_path=out,
        width=width,
        height=height,
        fps=float(fps) * float(speed),
        draw_trajectory=draw_trajectory,
        cam_distance=cam_distance,
        cam_azimuth=cam_azimuth,
        cam_elevation=cam_elevation,
        cam_mode=mode,
    )


def _load_corridor_scene(episode_dir: Path) -> Optional[dict]:
    p = episode_dir / "corridor_scene.json"
    if not p.exists():
        return None
    try:
        import json
        with open(p, "r", encoding="utf-8") as f:
            blob = json.load(f) or {}
        if "corridor_width" in blob and "obstacles" in blob:
            return blob
    except Exception:
        pass
    return None


def _render_follow(
    *,
    episode_dir: Path,
    qpos: np.ndarray,
    qvel: np.ndarray,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    draw_trajectory: bool,
    cam_distance: float,
    cam_azimuth: float,
    cam_elevation: float,
    cam_mode: str,
) -> Path:
    """Render with a configurable camera. Builds the same XML the deploy
    renderer builds (G1 scene + corridor obstacles + traj line), but does
    its own render loop so we can override the camera."""
    import mujoco
    import tempfile
    from genedynamics.envs.utils.mujoco_model_generator import (
        create_g1_render_xml_with_trajectory,
    )

    corridor_scene = _load_corridor_scene(episode_dir)
    if corridor_scene is None:
        print(f"[render] WARNING: no corridor_scene.json in {episode_dir}; obstacles won't be drawn")
    else:
        # Hide the dark blue-gray floor highlight box that the planner uses
        # to show the corridor footprint — looks like a "black block" near
        # the start in deploy renders.
        corridor_scene = dict(corridor_scene)
        corridor_scene["hide_floor_patch"] = True

    pelvis_xyz = qpos[:, :3]
    # The G1 base XML uses relative <include>s to mesh files, so the temp
    # XML must live in the same directory as g1.xml. Resolve it the same
    # way render_episode_to_gif does.
    g1_xml_path = _resolve_g1_xml_path()
    if g1_xml_path is not None:
        tmp_xml = Path(g1_xml_path).parent / "_render_spark_rl_temp.xml"
        cleanup_path = tmp_xml
    else:
        fd, tmp_path_str = tempfile.mkstemp(suffix=".xml")
        os.close(fd)
        tmp_xml = Path(tmp_path_str)
        cleanup_path = tmp_xml
    create_g1_render_xml_with_trajectory(
        str(tmp_xml),
        trajectory_positions=pelvis_xyz.tolist() if draw_trajectory else [],
        corridor_scene=corridor_scene,
    )

    try:
        mj_model = mujoco.MjModel.from_xml_path(str(tmp_xml))
        mj_data = mujoco.MjData(mj_model)
        gw = int(mj_model.vis.global_.offwidth)
        gh = int(mj_model.vis.global_.offheight)
        mj_model.vis.global_.offwidth = max(gw, int(width))
        mj_model.vis.global_.offheight = max(gh, int(height))
        renderer = mujoco.Renderer(mj_model, height=height, width=width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = float(cam_distance)
        cam.azimuth = float(cam_azimuth)
        cam.elevation = float(cam_elevation)

        # Resolve fixed lookat / distance for non-follow modes up-front.
        # global mode: frame the whole corridor scene from corridor_scene.json,
        # which is what the user wants when they say "覆盖完全局".
        if cam_mode == "global" and corridor_scene is not None:
            try:
                length = float(corridor_scene.get("corridor_length", 8.0))
                width_m = float(corridor_scene.get("corridor_width", 1.6))
                start = corridor_scene.get("start_pos", [0.0, 0.0])
                goal = corridor_scene.get("goal_pos", [length, 0.0])
                cx = 0.5 * (float(start[0]) + float(goal[0]))
                cy = 0.5 * (float(start[1]) + float(goal[1]))
            except Exception:
                cx, cy, length, width_m = float(pelvis_xyz[:, 0].mean()), 0.0, 8.0, 1.6
            # Camera distance must fit corridor_length horizontally given
            # mujoco's default vertical fov (~45°). At azimuth=90° the
            # corridor's x-extent is the camera's horizontal extent, so
            # required distance ≈ (length/2) / tan(hfov/2). hfov = vfov *
            # aspect for mujoco. Use a generous 1.15x safety so walls + first
            # frame robot fit even when the policy drifts a bit.
            aspect = float(width) / max(1.0, float(height))
            vfov_rad = np.deg2rad(45.0)
            hfov_rad = 2.0 * np.arctan(np.tan(0.5 * vfov_rad) * aspect)
            need_dist = 0.5 * float(length) / max(np.tan(0.5 * hfov_rad), 1e-3)
            cam.distance = max(float(cam_distance), need_dist * 1.15)
            cam.lookat[:] = (cx, cy, 0.6 * 1.6)  # mid-height of room
            print(f"[render] global cam: lookat=({cx:.2f}, {cy:.2f}, 0.96), distance={cam.distance:.2f}")
        elif cam_mode == "midpoint":
            mid = pelvis_xyz.mean(axis=0)
            cam.lookat[:] = (float(mid[0]), float(mid[1]), 0.9)

        nq = mj_model.nq
        nv = mj_model.nv
        frames: List[np.ndarray] = []
        n = qpos.shape[0]
        for i in range(n):
            mj_data.qpos[:] = qpos[i, :nq]
            mj_data.qvel[:] = qvel[i, :nv] if qvel.shape[1] >= nv else 0.0
            mujoco.mj_forward(mj_model, mj_data)
            if cam_mode == "follow":
                cam.lookat[0] = float(mj_data.qpos[0])
                cam.lookat[1] = float(mj_data.qpos[1])
                cam.lookat[2] = 0.9
            renderer.update_scene(mj_data, camera=cam)
            frames.append(renderer.render())

        import imageio
        imageio.mimsave(str(output_path), frames, fps=fps, loop=0)
        print(f"[render] wrote {output_path}")
        return output_path
    finally:
        try:
            cleanup_path.unlink()
        except OSError:
            pass


def _resolve_g1_xml_path() -> Optional[str]:
    """Mirror the resolution that :func:`render_episode_to_gif` does for G1.

    The temp XML must sit next to ``g1.xml`` so that mesh ``<include>``s resolve.
    """
    try:
        from genedynamics.robots.registry import _get_g1_path
        p = _get_g1_path()
        if p:
            return str(p)
    except Exception:
        pass
    menagerie = os.environ.get("MUJOCO_MENAGERIE_PATH")
    if menagerie:
        cand = Path(menagerie) / "unitree_g1" / "g1.xml"
        if cand.exists():
            return str(cand)
    proj = Path(__file__).resolve().parents[2]
    for d in (proj / "third_party" / "mujoco_menagerie", proj / "mujoco_menagerie"):
        cand = d / "unitree_g1" / "g1.xml"
        if cand.exists():
            return str(cand)
    return None


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--npz",
        type=Path,
        default=_ROOT / "results" / "deploy" / "spark_rl" / "sport_mode.npz",
        help="Diagnose npz file (default: results/deploy/spark_rl/sport_mode.npz)",
    )
    p.add_argument("--out", type=Path, default=None, help="Output gif path")
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--fps", type=float, default=50.0, help="Recorded control rate")
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    p.add_argument("--no-trajectory", action="store_true", help="Disable pelvis trace overlay")
    p.add_argument("--cam-distance", type=float, default=3.5)
    p.add_argument("--cam-azimuth", type=float, default=150.0)
    p.add_argument("--cam-elevation", type=float, default=-14.0)
    p.add_argument(
        "--cam-mode",
        choices=("follow", "global", "midpoint"),
        default="follow",
        help=(
            "follow: per-frame lookat=robot. "
            "global: fixed lookat=corridor center, distance auto-fit from corridor_scene.json. "
            "midpoint: legacy lookat=trajectory midpoint."
        ),
    )
    p.add_argument(
        "--no-follow",
        action="store_true",
        help="Deprecated. Equivalent to --cam-mode midpoint.",
    )
    p.add_argument(
        "--every-n",
        type=int,
        default=1,
        help="Render only every Nth frame to shrink the GIF",
    )
    args = p.parse_args(argv)

    cam_mode = args.cam_mode
    if args.no_follow and cam_mode == "follow":
        cam_mode = "midpoint"

    try:
        render(
            args.npz,
            output_path=args.out,
            width=args.width,
            height=args.height,
            fps=args.fps,
            speed=args.speed,
            draw_trajectory=not args.no_trajectory,
            cam_distance=args.cam_distance,
            cam_azimuth=args.cam_azimuth,
            cam_elevation=args.cam_elevation,
            cam_mode=cam_mode,
            every_n=args.every_n,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
