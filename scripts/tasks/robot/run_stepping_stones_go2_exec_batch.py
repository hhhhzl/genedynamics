#!/usr/bin/env python3
"""
Path B batch: run stepping-stones planners from YAML, then write Go2 Path-B artifacts
(qpos.npy, qvel.npy, ctrl.npy, execution_results.json) next to each seed's trajectory.

Usage:
  export MUJOCO_GL=egl   # Linux headless; use osmesa if EGL drivers are missing
  python scripts/tasks/robot/run_stepping_stones_go2_exec_batch.py \\
    --config configs/quadruped/stepping_stones_2d_exec/smoke/batch_smoke.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_mujoco_gl() -> None:
    if "MUJOCO_GL" not in os.environ:
        import platform

        if platform.system() == "Linux":
            os.environ.setdefault("MUJOCO_GL", "egl")


def _load_plan_dt(plan_yaml: Path) -> float:
    cfg = yaml.safe_load(plan_yaml.read_text(encoding="utf-8")) or {}
    env_params = cfg.get("env_params") or {}
    return float(env_params.get("dt", 1.0))


def _load_step_width(plan_yaml: Path) -> float:
    cfg = yaml.safe_load(plan_yaml.read_text(encoding="utf-8")) or {}
    env_params = cfg.get("env_params") or {}
    return float(env_params.get("step_width", env_params.get("stance_width", 0.30)))


def _load_plan_cfg(plan_yaml: Path) -> Dict[str, Any]:
    return yaml.safe_load(plan_yaml.read_text(encoding="utf-8")) or {}


def _stepping_scene_from_plan_cfg(plan_cfg: Dict[str, Any], level: int, seed: int) -> Dict[str, Any]:
    from genedynamics.tasks.stepping_stones import sample_stepping_stones_scene

    env_params = plan_cfg.get("env_params") or {}
    scene = sample_stepping_stones_scene(
        level=int(level),
        seed=int(seed),
        l_max=float(env_params.get("l_max", 0.35)),
        stance_width=float(env_params.get("stance_width", 0.30)),
        start_mid=tuple(env_params.get("start_mid", [-1.25, 0.0])),
        goal_mid=tuple(env_params.get("goal_mid", [1.25, 0.0])),
    )
    return {
        "level": int(scene.level),
        "difficulty": str(scene.difficulty),
        "map_x": [float(scene.map_x[0]), float(scene.map_x[1])],
        "map_y": [float(scene.map_y[0]), float(scene.map_y[1])],
        "river_x": [float(scene.river_x[0]), float(scene.river_x[1])],
        "stones_centers": np.asarray(scene.stones_centers, dtype=np.float32).tolist(),
        "stones_radii": np.asarray(scene.stones_radii, dtype=np.float32).tolist(),
    }


def _run_planner(root: Path, plan_yaml: Path, output_dir: Path) -> None:
    cfg = yaml.safe_load(plan_yaml.read_text(encoding="utf-8")) or {}
    cfg["output_dir"] = str(output_dir.relative_to(root))
    tmp = output_dir.parent / f"._pathb_plan_{plan_yaml.stem}.yaml"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    try:
        subprocess.run(
            [sys.executable, "-m", "genedynamics.experiments.runner", str(tmp.relative_to(root))],
            cwd=str(root),
            check=True,
        )
    finally:
        if tmp.is_file():
            tmp.unlink()


def _process_seed_dir(
    seed_dir: Path,
    plan_dt: float,
    render_fps: float,
    step_width: float,
    results_meta: Dict[str, Any],
) -> Dict[str, Any]:
    from genedynamics.experiments.common.stepping_stones_go2_pathb import (
        go2_kinematic_states_from_stepping,
        load_best_stepping_traj_from_seed_dir,
        write_pathb_artifacts,
    )

    traj4, _blob = load_best_stepping_traj_from_seed_dir(seed_dir)
    qpos, qvel, ctrl = go2_kinematic_states_from_stepping(
        traj4,
        plan_dt=plan_dt,
        render_fps=render_fps,
        step_width=step_width,
    )
    write_pathb_artifacts(
        seed_dir,
        qpos,
        qvel,
        ctrl,
        extra=results_meta,
    )
    return {
        "seed_dir": str(seed_dir),
        "frames": int(qpos.shape[0]),
        "nq": int(qpos.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Path B: stepping stones plan + Go2 exec artifacts")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/quadruped/stepping_stones_2d_exec/smoke/batch_smoke.yaml",
        help="Batch YAML (base_output_dir, runs, render_fps)",
    )
    parser.add_argument(
        "--exec-only",
        action="store_true",
        help="Skip planning; only lift existing trajectory/trajectory.json under output dirs",
    )
    args = parser.parse_args()

    _ensure_mujoco_gl()
    root = _repo_root()
    batch_path = root / args.config
    if not batch_path.is_file():
        raise SystemExit(f"Missing batch config: {batch_path}")

    batch = yaml.safe_load(batch_path.read_text(encoding="utf-8")) or {}
    base_out = root / str(batch.get("base_output_dir", "results/quadruped/stepping_stones_2d_exec/smoke"))
    render_fps = float(batch.get("render_fps", 50.0))
    runs: List[Dict[str, Any]] = batch.get("runs") or []

    summary: Dict[str, Any] = {"base_output_dir": str(base_out.relative_to(root)), "runs": []}

    for run in runs:
        name = str(run.get("name", "run"))
        plan_rel = run.get("plan_config")
        if not plan_rel:
            raise SystemExit(f"run {name!r} missing plan_config")
        plan_yaml = root / str(plan_rel)
        if not plan_yaml.is_file():
            raise SystemExit(f"Missing plan yaml: {plan_yaml}")

        out_dir = base_out / name
        plan_dt = _load_plan_dt(plan_yaml)
        step_width = _load_step_width(plan_yaml)
        plan_cfg = _load_plan_cfg(plan_yaml)

        if not args.exec_only:
            out_dir.mkdir(parents=True, exist_ok=True)
            _run_planner(root, plan_yaml, out_dir)

        run_summary: Dict[str, Any] = {"name": name, "output_dir": str(out_dir.relative_to(root)), "seeds": []}

        for level_dir in sorted(out_dir.glob("level_*")):
            for seed_dir in sorted(level_dir.glob("seed_*")):
                results_path = seed_dir / "results.json"
                results_blob: Dict[str, Any] = {}
                if results_path.is_file():
                    with open(results_path, "r", encoding="utf-8") as f:
                        results_blob = json.load(f) or {}
                meta = {
                    "level": results_blob.get("level"),
                    "seed": results_blob.get("seed"),
                    "metrics": results_blob.get("metrics"),
                    "planning_time": results_blob.get("planning_time"),
                }
                if meta["level"] is not None and meta["seed"] is not None:
                    try:
                        meta["stepping_scene"] = _stepping_scene_from_plan_cfg(
                            plan_cfg,
                            int(meta["level"]),
                            int(meta["seed"]),
                        )
                    except Exception:
                        pass
                try:
                    rec = _process_seed_dir(seed_dir, plan_dt, render_fps, step_width, meta)
                    rec["level"] = results_blob.get("level")
                    rec["seed"] = results_blob.get("seed")
                    run_summary["seeds"].append(rec)
                except Exception as e:
                    run_summary["seeds"].append(
                        {
                            "seed_dir": str(seed_dir),
                            "error": str(e),
                        }
                    )
        summary["runs"].append(run_summary)

    summary_path = base_out / "batch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Wrote", summary_path)


if __name__ == "__main__":
    main()
