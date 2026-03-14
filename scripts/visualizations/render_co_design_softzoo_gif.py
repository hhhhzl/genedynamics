#!/usr/bin/env python3
"""
Render SoftZoo co-design result to GIF.

Loads x, phi from results.json, runs one rollout, records particle positions,
and creates a matplotlib 3D scatter animation saved as GIF.
(Does not use GGUI to avoid Vulkan dependency in headless environments.)

Usage:
  python scripts/visualizations/render_co_design_softzoo_gif.py results/phase_a/smoke
  python scripts/visualizations/render_co_design_softzoo_gif.py results/phase_a/smoke --output motion.gif
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def main() -> int:
    parser = argparse.ArgumentParser(description="Render SoftZoo co-design to GIF")
    parser.add_argument("results_dir", type=str, help="Results dir (contains results.json)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output GIF path")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--max-frames", type=int, default=200, help="Max frames to render")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.is_dir():
        print(f"Error: Not a directory: {results_dir}")
        return 1

    results_path = results_dir / "results.json"
    if not results_path.exists():
        print(f"Error: results.json not found in {results_dir}")
        return 1

    with open(results_path) as f:
        results = json.load(f)
    if not results:
        print("Error: results.json is empty")
        return 1

    r = results[0]
    res = r.get("result", r)
    x = res.get("x")
    phi = res.get("phi")
    if x is None or phi is None:
        print("Error: results.json missing x or phi")
        return 1

    import numpy as np
    from genedynamics.envs.external.softzoo.adapters import (
        make_softzoo_env,
        encode_morphology,
        encode_controller,
    )
    from genedynamics.envs.external.softzoo.task_registry import get_task_spec
    from genedynamics.envs.external.softzoo.config import SoftZooRuntimeConfig

    task_id = res.get("task_id", "crawling_ground")
    task_spec = get_task_spec(task_id)
    mode_spec = task_spec.modes[0] if task_spec.modes else None
    fid_spec = task_spec.fidelity_levels[0] if task_spec.fidelity_levels else None

    runtime_config = SoftZooRuntimeConfig(
        project_root=str(_root),
        use_renderer=False,
        suppress_init_print=True,
        ti_arch="cpu",
    )

    env = make_softzoo_env(
        task_spec=task_spec,
        mode_spec=mode_spec,
        fidelity_spec=fid_spec,
        runtime_config=runtime_config,
    )

    controller = encode_controller(np.array(phi), task_spec, env)
    design = encode_morphology(np.array(x), task_spec, env=env)

    np.random.seed(args.seed)
    obs = env.reset(design)
    controller.reset()

    # Record particle positions each step
    positions_list = []
    for step in range(min(args.max_frames, task_spec.max_steps)):
        s = env.sim.solver.current_s
        robot_x = env.design_space.get_x(s)
        if hasattr(robot_x, "numpy"):
            pos = np.asarray(robot_x)
        else:
            pos = np.asarray(robot_x)
        positions_list.append(pos)
        act = controller(env.sim.solver.current_s, obs)
        obs, reward, done, info = env.step(act)
        if done:
            break

    env.close()

    if not positions_list:
        print("Error: No frames recorded")
        return 1

    # Create matplotlib 3D animation and save as GIF
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        pos = positions_list[frame]
        if pos.size > 0:
            ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1], s=8, c="steelblue", alpha=0.8)
        ax.set_xlim(0.2, 0.6)
        ax.set_ylim(0.2, 0.6)
        ax.set_zlim(0.0, 0.15)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        ax.set_title(f"Step {frame}")
        ax.view_init(elev=20, azim=45 + frame * 0.5)

    anim = matplotlib.animation.FuncAnimation(
        fig, update, frames=len(positions_list), interval=int(1000 / args.fps)
    )

    gif_path = Path(args.output) if args.output else results_dir / "motion_replay.gif"
    if not gif_path.is_absolute():
        gif_path = results_dir / gif_path.name
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    writer = PillowWriter(fps=args.fps)
    anim.save(str(gif_path), writer=writer)
    plt.close()
    print(f"GIF saved: {gif_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
