#!/usr/bin/env python
"""Render a MuJoCo GIF of the spark-RL sport-mode controller walking the
14D corridor plan (with the upper-body mapper wired in).

Reads the ``qpos`` / ``qvel`` timeseries dumped by
:mod:`scripts.tasks.robot.diagnose_sport_mode_corridor` and feeds them
into :func:`genedynamics.deploy.observers.mujoco_render.render_episode_to_gif`,
which is the same renderer used by the legacy deploy CLI.

Run inside the dev-cpu Docker image (mujoco + torch needed)::

    docker run --rm -v "$PWD:/work" -w /work genedynamics/dev-cpu:local bash -c \\
      "pip install --quiet torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu && \\
       python scripts/tasks/robot/diagnose_sport_mode_corridor.py \\
           --max-steps 400 --out-dir results/deploy/diagnose_spark_rl --quiet && \\
       python scripts/visualizations/render_spark_rl_corridor_gif.py"

The defaults assume the diagnose script wrote to
``results/deploy/diagnose_spark_rl/diagnose_sport_mode.npz``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

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
) -> Path:
    """Render the diagnose .npz to a GIF.

    Args:
        npz_path: Path to ``diagnose_sport_mode.npz`` (must contain
            ``qpos`` and ``qvel`` arrays — added in the diagnose script).
        output_path: Where to write the .gif. Defaults to
            ``<npz_path.parent>/spark_rl_corridor.gif``.
        width / height: Frame size in pixels.
        fps: Output GIF frames-per-second. Defaults to the diagnose
            script's control rate (50 Hz).
        speed: Real-time speed multiplier. ``1.0`` plays at the recorded
            control rate; ``2.0`` doubles the playback fps; ``0.5`` halves it.
        draw_trajectory: Overlay the pelvis xy trace on the rendered scene.

    Returns:
        The path to the written .gif.
    """
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Diagnose npz not found: {npz_path}\n"
            "Run scripts/tasks/robot/diagnose_sport_mode_corridor.py first "
            "with --out-dir results/deploy/diagnose_spark_rl"
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

    # render_episode_to_gif expects a single (N, nq + nv) tensor
    states = np.concatenate([qpos, qvel], axis=1)
    print(
        f"[render] loaded {states.shape[0]} frames "
        f"(qpos {qpos.shape[1]}, qvel {qvel.shape[1]}) from {npz_path}"
    )

    out = output_path or (npz_path.parent / "spark_rl_corridor.gif")

    from genedynamics.deploy.observers.mujoco_render import render_episode_to_gif

    render_episode_to_gif(
        episode_dir=npz_path.parent,
        states=states,
        actions=None,
        output_path=out,
        model="g1",
        width=width,
        height=height,
        fps=float(fps) * float(speed),
        draw_trajectory=draw_trajectory,
    )
    print(f"[render] wrote {out}")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--npz",
        type=Path,
        default=_ROOT / "results" / "deploy" / "diagnose_spark_rl" / "diagnose_sport_mode.npz",
        help="Diagnose npz file (default: results/deploy/diagnose_spark_rl/diagnose_sport_mode.npz)",
    )
    p.add_argument("--out", type=Path, default=None, help="Output gif path")
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--fps", type=float, default=50.0, help="Recorded control rate")
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    p.add_argument("--no-trajectory", action="store_true", help="Disable pelvis trace overlay")
    args = p.parse_args(argv)

    try:
        render(
            args.npz,
            output_path=args.out,
            width=args.width,
            height=args.height,
            fps=args.fps,
            speed=args.speed,
            draw_trajectory=not args.no_trajectory,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
