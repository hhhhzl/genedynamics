"""Render canonical contact-task trajectories from saved executed q/qd states.

This shared experiment utility never reruns or replays the controller. It reconstructs Brax
pipeline states from the q/qd saved by the unified experiment runner and writes
the standard ``trajectory_best.gif`` and ``trajectory_best.png`` artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import jax.numpy as jnp

from genedynamics.experiments.plugins.environments import (
    HumanoidBoxPushPlugin,
    ManipulatorPegInsertPlugin,
    ManipulatorSurfaceScanPlugin,
)


PLUGINS = {
    "manipulator_surface_scan": ManipulatorSurfaceScanPlugin,
    "manipulator_peg_insert": ManipulatorPegInsertPlugin,
    "humanoid_box_push": HumanoidBoxPushPlugin,
}


def render_result(
    result_path: Path,
    *,
    stride: int,
    fps: float,
    width: int,
    height: int,
    force: bool,
) -> bool:
    with open(result_path) as handle:
        result = json.load(handle)
    config = result.get("config_snapshot") or {}
    env_name = config.get("env_name")
    if env_name not in PLUGINS:
        return False
    trajectory_path = result_path.parent / "trajectory" / "trajectory.json"
    if not trajectory_path.is_file():
        raise FileNotFoundError(trajectory_path)
    with open(trajectory_path) as handle:
        trajectory = json.load(handle)
    compact_states = [
        state for state in trajectory.get("states", [])
        if state.get("q") is not None and state.get("qd") is not None
    ]
    if not compact_states:
        raise ValueError(f"trajectory has no executed q/qd states: {trajectory_path}")
    output_dir = trajectory_path.parent
    gif_path = output_dir / "trajectory_best.gif"
    png_path = output_dir / "trajectory_best.png"
    if not force and gif_path.is_file() and png_path.is_file():
        return True

    adapter = PLUGINS[env_name]()
    method_params = config.get("method_params") or {}
    env = adapter.create_env({
        **(config.get("env_params") or {}),
        "_experiment_seed": int(result["seed"]),
        "_controller_method": method_params.get(
            "controller_method", config.get("method", "mdac")
        ),
        "_execution_env_params": config.get("execution_env_params") or {},
    })
    render_env = adapter.execution_env(env)
    selected = compact_states[::max(1, stride)]
    if selected[-1] is not compact_states[-1]:
        selected.append(compact_states[-1])
    pipeline_states = [
        render_env.pipeline_init(jnp.asarray(state["q"]), jnp.asarray(state["qd"]))
        for state in selected
    ]
    frames = render_env.render(pipeline_states, height=height, width=width)
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    imageio.imwrite(png_path, frames[len(frames) // 2])
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    rendered = 0
    failures = []
    for root_text in args.roots:
        for path in sorted(Path(root_text).rglob("results.json")):
            try:
                rendered += int(render_result(
                    path,
                    stride=args.stride,
                    fps=args.fps,
                    width=args.width,
                    height=args.height,
                    force=args.force,
                ))
            except Exception as exc:
                failures.append(f"{path}: {type(exc).__name__}: {exc}")
    print(f"rendered_or_present={rendered} failures={len(failures)}")
    for failure in failures:
        print(f"ERROR {failure}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
