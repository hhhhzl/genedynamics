#!/usr/bin/env python3
"""
Offline G1 corridor WBC pipeline for humanoid corridor plans.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from genedynamics.deploy.followers.humanoid.mujoco import (
    HumanoidMujocoPipeline,
    HumanoidMujocoPipelineConfig,
)
from genedynamics.envs.humanoid_corridor_2d import (
    corridor_scene_to_dict,
    resolve_corridor_scene_preset,
)
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


def _default_output_dir(seed_dir: Path) -> Path:
    method_name = seed_dir.parent.parent.name if seed_dir.parent.parent.exists() else "run"
    return (
        seed_dir.parent.parent.parent.parent
        / "g1_corridor_wbc"
        / method_name
        / seed_dir.parent.name
        / seed_dir.name
    )


def _infer_config_path(seed_dir: Path) -> Optional[Path]:
    repo_root = Path(__file__).resolve().parents[3]
    seed_path = seed_dir.resolve() if seed_dir.is_absolute() else (repo_root / seed_dir).resolve()
    try:
        rel = seed_path.relative_to((repo_root / "results").resolve())
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 3:
        return None
    stem_parts = list(parts[:-2])
    config_path = (repo_root / "configs").joinpath(*stem_parts).with_suffix(".yaml")
    return config_path if config_path.exists() else None


def _load_corridor_scene_metadata(seed_dir: Path) -> Optional[Dict[str, Any]]:
    config_path = _infer_config_path(seed_dir)
    if config_path is None:
        return None
    try:
        import yaml
    except Exception:
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return None
    env_params = cfg.get("env_params") or {}
    obstacle_config = cfg.get("obstacle_config") or {}
    scene_preset = env_params.get("scene_preset") or obstacle_config.get("scene_preset")
    if not scene_preset:
        return None
    try:
        scene = resolve_corridor_scene_preset(str(scene_preset))
    except Exception:
        return None
    scene_meta = corridor_scene_to_dict(scene, scene_preset=str(scene_preset))
    scene_meta["config_path"] = str(config_path)
    return scene_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline G1 corridor WBC pipeline for humanoid corridor plans")
    parser.add_argument(
        "--seed-dir",
        type=str,
        default="results/humanoid/corridor_2d/smoke/twogo_zone_a/level_1/seed_0",
    )
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--model-xml-path", type=str, default=None)
    parser.add_argument("--best-idx", type=int, default=None)
    parser.add_argument("--source-dt", type=float, default=0.25)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--sim-dt", type=float, default=0.002)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-render-frames", type=int, default=180)
    parser.add_argument(
        "--render-mode",
        type=str,
        default="both",
        choices=("dynamic", "kinematic", "both"),
        help="Which state stream to render: physical rollout, kinematic preview, or both.",
    )
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    if "MUJOCO_GL" not in os.environ:
        os.environ.setdefault("MUJOCO_GL", "osmesa")

    seed_dir = Path(args.seed_dir)
    out_dir = Path(args.out_dir) if args.out_dir else _default_output_dir(seed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = HumanoidMujocoPipeline(
        model_xml_path=args.model_xml_path,
        cfg=HumanoidMujocoPipelineConfig(
            source_dt=float(args.source_dt),
            control_dt=float(args.control_dt),
            sim_dt=float(args.sim_dt),
        ),
    )
    rollout = pipeline.rollout_plan_from_seed_dir(
        str(seed_dir),
        best_idx=args.best_idx,
        max_frames=args.max_frames,
    )

    states = np.asarray(rollout["states"], dtype=np.float64)
    qpos = np.asarray(rollout["qpos"], dtype=np.float64)
    qvel = np.asarray(rollout["qvel"], dtype=np.float64)
    ctrl = np.asarray(rollout["ctrl"], dtype=np.float64)
    tau_ff = np.asarray(rollout.get("tau_ff", np.zeros((0, 0))), dtype=np.float64)
    ddq = np.asarray(rollout.get("ddq", np.zeros((0, 0))), dtype=np.float64)
    lam = np.asarray(rollout.get("lambda", np.zeros((0, 0))), dtype=np.float64)
    render_states = np.asarray(rollout.get("render_states", states), dtype=np.float64)
    plan_states = np.asarray(rollout["plan_states"], dtype=np.float64)
    corridor_scene = _load_corridor_scene_metadata(seed_dir)

    np.save(out_dir / "states.npy", states)
    np.save(out_dir / "qpos.npy", qpos)
    np.save(out_dir / "qvel.npy", qvel)
    np.save(out_dir / "ctrl.npy", ctrl)
    np.save(out_dir / "tau_ff.npy", tau_ff)
    np.save(out_dir / "ddq.npy", ddq)
    np.save(out_dir / "lambda.npy", lam)
    np.save(out_dir / "render_states.npy", render_states)
    np.save(out_dir / "plan_states.npy", plan_states)

    with open(out_dir / "debug.json", "w", encoding="utf-8") as f:
        json.dump(rollout["debug"], f, indent=2, default=_json_default)

    seed_results = _load_json(seed_dir / "results.json")
    follow_results: Dict[str, Any] = {
        "seed_dir": str(seed_dir),
        "out_dir": str(out_dir),
        "frames": int(states.shape[0]),
        "nq": int(qpos.shape[1]) if qpos.ndim == 2 and qpos.size > 0 else 0,
        "nv": int(qvel.shape[1]) if qvel.ndim == 2 and qvel.size > 0 else 0,
        "nu": int(ctrl.shape[1]) if ctrl.ndim == 2 and ctrl.size > 0 else 0,
        "n_tau": int(tau_ff.shape[1]) if tau_ff.ndim == 2 and tau_ff.size > 0 else 0,
        "n_ddq": int(ddq.shape[1]) if ddq.ndim == 2 and ddq.size > 0 else 0,
        "n_lambda": int(lam.shape[1]) if lam.ndim == 2 and lam.size > 0 else 0,
        "model_xml_path": str(rollout["metadata"].get("model_xml_path", "")),
        "control_dt": float(rollout["metadata"].get("control_dt", args.control_dt)),
        "sim_dt": float(rollout["metadata"].get("sim_dt", args.sim_dt)),
        "best_idx": int(rollout["metadata"].get("best_idx", args.best_idx if args.best_idx is not None else -1)),
        "schema_version": str(rollout["metadata"].get("schema_version", "")),
        "render_state_mode": str(rollout["metadata"].get("render_state_mode", "")),
        "solver": str(rollout["metadata"].get("solver", "")),
        "source_results": seed_results,
    }
    with open(out_dir / "g1_corridor_wbc_results.json", "w", encoding="utf-8") as f:
        json.dump(follow_results, f, indent=2, default=_json_default)

    html_paths: Dict[str, Path] = {}
    if not args.no_render:
        renderer = MotionRenderer(out_dir)
        render_specs = []
        if args.render_mode in {"dynamic", "both"} and states.ndim == 2 and states.shape[0] > 0:
            render_specs.append(("dynamic", states, "g1_corridor_wbc_dynamic"))
        if args.render_mode in {"kinematic", "both"} and render_states.ndim == 2 and render_states.shape[0] > 0:
            render_specs.append(("kinematic", render_states, "g1_corridor_wbc_kinematic"))

        for render_kind, render_source_states, render_name in render_specs:
            n = render_source_states.shape[0]
            step = max(1, int(np.ceil(n / max(1, int(args.max_render_frames)))))
            idx = np.arange(0, n, step, dtype=np.int32)
            render_states_sub = render_source_states[idx]
            render_actions = None
            if ctrl.ndim == 2 and ctrl.shape[0] > 0:
                aidx = np.clip(idx[:-1], 0, ctrl.shape[0] - 1) if idx.size > 1 else np.zeros((0,), dtype=np.int32)
                render_actions = ctrl[aidx] if aidx.size > 0 else None
            fps = 1.0 / (float(args.sim_dt) * float(step))
            episode = MotionEpisode(
                states=render_states_sub,
                actions=render_actions,
                robot_type="humanoid",
                model_id="g1",
                fps=fps,
                metadata={
                    "seed_dir": str(seed_dir),
                    "best_idx": int(args.best_idx) if args.best_idx is not None else None,
                    "corridor_scene": corridor_scene,
                    "render_kind": render_kind,
                    "render_state_mode": str(rollout["metadata"].get("render_state_mode", "")),
                },
            )
            html_paths[render_kind] = renderer.render_html(
                episode,
                name=render_name,
                width=960,
                height=720,
                fps=fps,
            )

    print("Output dir:", out_dir)
    print("Frames:", states.shape[0])
    for render_kind in ("dynamic", "kinematic"):
        html_path = html_paths.get(render_kind)
        if html_path is not None:
            print(f"HTML ({render_kind}):", html_path)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
