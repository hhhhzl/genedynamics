#!/usr/bin/env python3
"""
Offline walk follower for stepping-stones plans.

Generates first full-body stepping rollout for Go2:
  - qpos.npy / qvel.npy / ctrl.npy
  - HTML/GIF replay
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from genedynamics.deploy.sim_plan.stepping_walk_follower import (
    SteppingWalkFollower,
    WalkFollowerConfig,
    load_stepping_plan_from_seed_dir,
)
from genedynamics.tasks.stepping_stones import sample_stepping_stones_scene, stepping_scene_to_dict
from genedynamics.viz.motion_episode import MotionEpisode
from genedynamics.viz.motion_renderer import MotionRenderer


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _resolve_stepping_scene(seed_dir: Path, step_width: float) -> Optional[Dict[str, Any]]:
    ex = _load_json(seed_dir / "execution_results.json")
    if isinstance(ex.get("stepping_scene"), dict):
        return ex["stepping_scene"]

    res = _load_json(seed_dir / "results.json")
    level = int(res.get("level", 1))
    seed = int(res.get("seed", 0))
    scene = sample_stepping_stones_scene(
        level=level,
        seed=seed,
        l_max=0.35,
        stance_width=step_width,
        start_mid=(-1.25, 0.0),
        goal_mid=(1.25, 0.0),
    )
    return stepping_scene_to_dict(scene)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Go2 walk follower for stepping-stones")
    parser.add_argument(
        "--seed-dir",
        type=str,
        default="results/quadruped/stepping_stones_2d/smoke/mbd/level_1/seed_0",
    )
    parser.add_argument("--step-width", type=float, default=0.30)
    parser.add_argument("--gait", type=str, default="walk", choices=["walk", "trot"])
    parser.add_argument("--phase-steps", type=int, default=20)
    parser.add_argument("--swing-height", type=float, default=0.07)
    parser.add_argument("--sim-dt", type=float, default=0.01)
    parser.add_argument("--base-height-offset", type=float, default=0.02)
    parser.add_argument("--min-foot-z", type=float, default=0.015)
    parser.add_argument("--lock-base-pose", action="store_true")
    parser.add_argument("--base-z-min", type=float, default=0.22)
    parser.add_argument("--max-render-frames", type=int, default=140)
    args = parser.parse_args()

    if "MUJOCO_GL" not in os.environ:
        os.environ.setdefault("MUJOCO_GL", "osmesa")

    seed_dir = Path(args.seed_dir)
    method_name = seed_dir.parent.parent.name if seed_dir.parent.parent.exists() else "run"
    out_dir = (
        seed_dir.parent.parent.parent.parent
        / "stepping_stones_walk_follow"
        / method_name
        / seed_dir.parent.name
        / seed_dir.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = load_stepping_plan_from_seed_dir(seed_dir, step_width=float(args.step_width))
    follower = SteppingWalkFollower(
        cfg=WalkFollowerConfig(
            gait=str(args.gait),
            sim_dt=float(args.sim_dt),
            phase_steps=int(args.phase_steps),
            swing_height=float(args.swing_height),
            step_width=float(args.step_width),
            base_height_offset=float(args.base_height_offset),
            min_foot_z=float(args.min_foot_z),
            lock_base_pose=bool(args.lock_base_pose),
            base_z_min=float(args.base_z_min),
        )
    )
    qpos, qvel, ctrl = follower.rollout(
        mid=plan["mid"],
        yaw=plan["yaw"],
        step_width=float(args.step_width),
        feet_ref=plan["feet"],
        mode_ref=plan["mode"],
        tau_ref=plan["tau"],
    )

    np.save(out_dir / "qpos.npy", qpos)
    np.save(out_dir / "qvel.npy", qvel)
    np.save(out_dir / "ctrl.npy", ctrl)

    scene_dict = _resolve_stepping_scene(seed_dir, float(args.step_width))
    if scene_dict:
        with open(out_dir / "stepping_scene.json", "w", encoding="utf-8") as f:
            json.dump(scene_dict, f, indent=2)

    with open(out_dir / "walk_follow_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed_dir": str(seed_dir),
                "frames": int(qpos.shape[0]),
                "nq": int(qpos.shape[1]),
                "nv": int(qvel.shape[1]),
                "nu": int(ctrl.shape[1]) if ctrl.ndim == 2 else 0,
                "step_width": float(args.step_width),
                "gait": str(args.gait),
                "phase_steps": int(args.phase_steps),
                "base_height_offset": float(args.base_height_offset),
                "min_foot_z": float(args.min_foot_z),
                "lock_base_pose": bool(args.lock_base_pose),
                "base_z_min": float(args.base_z_min),
            },
            f,
            indent=2,
        )

    # Render preview with downsampling for practical size.
    n = qpos.shape[0]
    step = max(1, int(np.ceil(n / max(1, int(args.max_render_frames)))))
    idx = np.arange(0, n, step, dtype=np.int32)
    states = np.concatenate([qpos[idx], qvel[idx]], axis=1)
    aidx = idx[:-1]
    if ctrl.shape[0] > 0 and aidx.size > 0:
        aidx = np.clip(aidx, 0, ctrl.shape[0] - 1)
        actions = ctrl[aidx]
    else:
        actions = None
    fps = 1.0 / (float(args.sim_dt) * float(step))
    ep = MotionEpisode(states=states, actions=actions, robot_type="quadruped", model_id="go2", fps=fps)
    renderer = MotionRenderer(out_dir)
    html = renderer.render_html(ep, name="go2_walk_follow", width=960, height=720, fps=fps)
    print("Output dir:", out_dir)
    print("HTML:", html)


if __name__ == "__main__":
    main()
