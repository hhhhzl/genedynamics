#!/usr/bin/env python3
"""
Offline G1 corridor WBC pipeline for humanoid corridor plans.

Loads a seed directory (from the planner), runs the WBC controller closed-loop
in MuJoCo, saves states + renders.

Configuration is taken from :class:`G1CorridorMujocoWBCPreset`.

Execution order per tick::

    CorridorPlanFrame → TraversalIntent → HumanoidContactScheduler
        → HumanoidFootstepPlanner → HumanoidUpperBodyMapper
        → HumanoidTaskBuilder → HumanoidWBCController (act)
        → MujocoRobotIO.send_control / step
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from genedynamics.deploy.presets.g1_corridor_mujoco_wbc import G1CorridorMujocoWBCPreset
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
    repo_root = Path(__file__).resolve().parents[4]
    seed_path = seed_dir.resolve() if seed_dir.is_absolute() else (repo_root / seed_dir).resolve()
    try:
        rel = seed_path.relative_to((repo_root / "results").resolve())
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 3:
        return None
    config_path = (repo_root / "configs").joinpath(*list(parts[:-2])).with_suffix(".yaml")
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


def _quat_wxyz_from_yaw(yaw: float) -> np.ndarray:
    cy, sy = float(np.cos(0.5 * yaw)), float(np.sin(0.5 * yaw))
    return np.array([cy, 0.0, 0.0, sy], dtype=np.float64)


def rollout(
    seed_dir: Path,
    *,
    best_idx: Optional[int] = None,
    source_dt: float = 0.25,
    control_dt: float = 0.02,
    sim_dt: float = 1.0 / 500.0,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the WBC pipeline on a seed directory, return dict of history arrays."""
    from genedynamics.deploy.followers.common.plan_adapter import CorridorTrajectoryAdapter
    from genedynamics.deploy.followers.common.traversal_intent import TraversalIntent
    from genedynamics.deploy.followers.humanoid.contact_scheduler import (
        HumanoidContactScheduler,
        HumanoidContactSchedulerConfig,
    )
    from genedynamics.deploy.followers.humanoid.footstep_planner import HumanoidFootstepPlanner
    from genedynamics.deploy.followers.humanoid.task_builder import HumanoidTaskBuilder
    from genedynamics.deploy.followers.humanoid.task_spec import (
        ContactObservations,
        FootContactObservation,
    )
    from genedynamics.deploy.followers.humanoid.upper_body_mapper import HumanoidUpperBodyMapper
    from genedynamics.deploy.controllers.wbc import HumanoidWBCController, WBCResult
    from genedynamics.deploy.controllers.wbc.config import WBCConfig, TaskGainsConfig
    from genedynamics.deploy.interfaces.messages import Intent
    from genedynamics.deploy.io.mujoco_io import MujocoRobotIO, FootContactSnapshot
    import mujoco as _mj

    # 1. Load plan
    adapter = CorridorTrajectoryAdapter(source_dt=source_dt, target_dt=control_dt)
    traj = adapter.load_seed_dir(seed_dir, best_idx=best_idx)
    frames = traj.decode_frames()
    if max_frames is not None:
        frames = frames[: int(max_frames)]
    if not frames:
        return {}

    # 2. Build stack
    io = MujocoRobotIO(sim_dt=sim_dt)
    # Reduced gains to avoid torque saturation at swing-leg liftoff. The
    # default swing_foot_position_kp=20 demanded 21+ N·m of bound violation
    # in the first gait cycle, collapsing the robot at t≈1.56 s. Halving
    # the swing-foot tracking gain (and damping) lets the QP find a torque
    # solution that respects G1's joint limits at the cost of looser foot
    # tracking.
    wbc_cfg = WBCConfig()
    wbc_cfg.gains.swing_foot_position_kp = 10.0
    wbc_cfg.gains.swing_foot_position_kd = 4.0
    wbc_cfg.gains.com_kp = 25.0
    wbc_cfg.gains.com_kd = 10.0
    wbc_cfg.gains.pelvis_orientation_kp = 25.0
    wbc_cfg.gains.pelvis_orientation_kd = 8.0
    controller = HumanoidWBCController(io=io, cfg=wbc_cfg)
    # Slower gait — gives the WBC more time per phase, lower torque demand,
    # more time for new support foot to load before pelvis collapses.
    contact_cfg = HumanoidContactSchedulerConfig(
        swing_time=0.50,
        double_support_time=0.20,
    )
    contact_scheduler = HumanoidContactScheduler(cfg=contact_cfg)
    footstep_planner = HumanoidFootstepPlanner()
    upper_body_mapper = HumanoidUpperBodyMapper()
    task_builder = HumanoidTaskBuilder()

    # 3. Anchor to first plan frame
    f0 = frames[0]
    height_offset = float(io.data.qpos[2]) - 0.75
    io.data.qpos[0] = float(f0.x)
    io.data.qpos[1] = float(f0.y)
    io.data.qpos[2] = float(f0.h) + height_offset
    io.data.qpos[3:7] = _quat_wxyz_from_yaw(float(f0.psi))
    _mj.mj_forward(io.model, io.data)
    stand_qpos = io.data.qpos.copy()

    def _foot_positions() -> tuple[np.ndarray, np.ndarray]:
        snaps = io.foot_contact_observations()
        return snaps["left"].position_world.copy(), snaps["right"].position_world.copy()

    def _contact_obs() -> ContactObservations:
        snaps = io.foot_contact_observations()
        def _snap(s: FootContactSnapshot) -> FootContactObservation:
            return FootContactObservation(
                position_world=s.position_world,
                velocity_world=s.velocity_world,
                rotation_world=s.rotation_world,
                angular_velocity_world=s.angular_velocity_world,
                in_contact=s.in_contact,
                contact_count=s.contact_count,
                support_load=s.support_load,
            )
        return ContactObservations(left=_snap(snaps["left"]), right=_snap(snaps["right"]))

    left_pos, right_pos = _foot_positions()
    intent0 = TraversalIntent.from_plan_frame(f0)
    contact_scheduler.reset(initial_frame=f0)
    footstep_planner.reset(initial_intent=intent0)
    footstep_planner.seed_from_current_feet(left_pos, right_pos)
    controller.reset()

    # 4. Rollout loop
    n = len(frames)
    nq = int(io.model.nq)
    nv = int(io.model.nv)
    nu = int(io.spec.num_actuated)

    qpos_hist, qvel_hist, ctrl_hist = [], [], []
    tau_hist, ddq_hist, lam_hist = [], [], []
    state_hist, render_state_hist, plan_states = [], [], []
    debug = []

    step_dt = float(max(traj.dt, control_dt))

    for i, frame in enumerate(frames):
        traversal_intent = TraversalIntent.from_plan_frame(frame)
        contact_obs = _contact_obs()
        phase = contact_scheduler.advance(traversal_intent, step_dt, contact_obs)
        footsteps = footstep_planner.update(traversal_intent, phase, contact_obs, dt=step_dt)
        upper_body = upper_body_mapper.map(frame)
        tasks = task_builder.build(frame, traversal_intent, phase, footsteps, upper_body)

        state = io.get_state()
        intent = Intent(
            t=float(frame.time_sec),
            base_yaw=float(traversal_intent.yaw),
            base_height=float(traversal_intent.body_height),
            base_lin_vel=traversal_intent.planar_velocity.copy(),
            base_yaw_rate=float(traversal_intent.yaw_rate),
            base_pos_xy=traversal_intent.planar_position.copy(),
            torso_yaw=float(tasks.torso_yaw),
            contact_phase=phase.phase.value,
            extras={"humanoid_tasks": tasks, "phase": phase, "dt": step_dt},
        )
        cmd = controller.act(state, intent)
        wbc: WBCResult = cmd.extras.get("wbc")

        # Kinematic render state: WBC q_ref at the task's target pelvis pose
        q_render = stand_qpos.copy()
        pelvis = np.asarray(tasks.pelvis.position_world, dtype=np.float64)
        q_render[0] = float(pelvis[0])
        q_render[1] = float(pelvis[1])
        q_render[2] = float(pelvis[2]) + height_offset
        cy, sy = float(np.cos(0.5 * tasks.pelvis.yaw_world)), float(np.sin(0.5 * tasks.pelvis.yaw_world))
        q_render[3:7] = [cy, 0.0, 0.0, sy]
        if wbc is not None and wbc.q_ref.size == nu:
            q_render = io.spec.apply_actuated_qpos(q_render, wbc.q_ref)
        render_state_hist.append(np.concatenate([q_render, np.zeros(nv, dtype=np.float64)]))

        io.send_control(cmd)
        io.step(step_dt)

        qpos = np.asarray(io.data.qpos, dtype=np.float64).copy()
        qvel = np.asarray(io.data.qvel, dtype=np.float64).copy()
        qpos_hist.append(qpos)
        qvel_hist.append(qvel)
        ctrl_hist.append(wbc.q_ref.copy() if wbc is not None and wbc.q_ref.size else np.zeros(nu))
        tau_hist.append(wbc.tau_ff.copy() if wbc is not None and wbc.tau_ff.size else np.zeros(nu))
        ddq_hist.append(wbc.ddq_full.copy() if wbc is not None and wbc.ddq_full.size else np.zeros(nv))
        lam_hist.append(_pad_lambda(wbc))
        state_hist.append(np.concatenate([qpos, qvel]))
        plan_states.append(frame.raw_state.copy())
        debug.append({
            "frame_index": i,
            "time_sec": float(frame.time_sec),
            "phase": phase.phase.value,
            "solver_metadata": {
                "method": wbc.method if wbc else "",
                "eq_residual_norm": float(wbc.eq_residual) if wbc else 0.0,
                "ineq_violation_max": float(wbc.ineq_violation) if wbc else 0.0,
                "task_errors": dict(wbc.task_errors) if wbc else {},
                "contact_feet": list(wbc.contact_feet) if wbc else [],
                "torque_saturation_max": float(wbc.torque_saturation_max) if wbc else 0.0,
                "torque_bound_violation_max": float(wbc.torque_bound_violation_max) if wbc else 0.0,
                "startup_blend": float(wbc.startup_blend) if wbc else 0.0,
            },
        })

    io.close()

    return {
        "states": np.asarray(state_hist, dtype=np.float64),
        "qpos": np.asarray(qpos_hist, dtype=np.float64),
        "qvel": np.asarray(qvel_hist, dtype=np.float64),
        "ctrl": np.asarray(ctrl_hist, dtype=np.float64),
        "tau_ff": np.asarray(tau_hist, dtype=np.float64),
        "lambda": np.asarray(lam_hist, dtype=np.float64),
        "ddq": np.asarray(ddq_hist, dtype=np.float64),
        "render_states": np.asarray(render_state_hist, dtype=np.float64),
        "plan_states": np.asarray(plan_states, dtype=np.float64),
        "debug": debug,
        "metadata": {
            "model_xml_path": "",
            "control_dt": step_dt,
            "sim_dt": sim_dt,
            "num_frames": n,
            "best_idx": int(traj.best_idx),
            "schema_version": adapter.schema.version,
            "render_state_mode": "wbc_inverse_dynamics_root_preview",
            "solver": "wbc_inverse_dynamics_qp_hard_contact",
            "preset": "G1CorridorMujocoWBCPreset",
        },
    }


def _pad_lambda(wbc: Any) -> np.ndarray:
    out = np.zeros(6, dtype=np.float64)
    if wbc is None or wbc.lambda_ref.size == 0:
        return out
    raw = np.asarray(wbc.lambda_ref, dtype=np.float64).reshape(-1)
    for i, name in enumerate(wbc.contact_feet):
        src = raw[3 * i : 3 * (i + 1)]
        if src.size != 3:
            continue
        dst = slice(0, 3) if name == "left_foot" else slice(3, 6)
        out[dst] = src
    return out


def main() -> None:
    _preset = G1CorridorMujocoWBCPreset()

    parser = argparse.ArgumentParser(description="Offline G1 corridor WBC pipeline")
    parser.add_argument(
        "--seed-dir",
        type=str,
        default="results/humanoid/corridor_2d/plan/twogo_zone_a/level_1/seed_0",
    )
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--best-idx", type=int, default=None)
    parser.add_argument("--source-dt", type=float, default=0.25)
    parser.add_argument("--control-dt", type=float, default=1.0 / _preset.control_hz)
    parser.add_argument("--sim-dt", type=float, default=_preset.sim_dt)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-render-frames", type=int, default=180)
    parser.add_argument(
        "--render-mode",
        type=str,
        default="both",
        choices=("dynamic", "kinematic", "both"),
    )
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    if "MUJOCO_GL" not in os.environ:
        os.environ.setdefault("MUJOCO_GL", "osmesa")

    seed_dir = Path(args.seed_dir)
    out_dir = Path(args.out_dir) if args.out_dir else _default_output_dir(seed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = rollout(
        seed_dir,
        best_idx=args.best_idx,
        source_dt=float(args.source_dt),
        control_dt=float(args.control_dt),
        sim_dt=float(args.sim_dt),
        max_frames=args.max_frames,
    )

    if not result:
        print("ERROR: empty rollout — check seed_dir contains a trajectory")
        return

    states = result["states"]
    qpos = result["qpos"]
    qvel = result["qvel"]
    ctrl = result["ctrl"]
    tau_ff = result["tau_ff"]
    ddq = result["ddq"]
    lam = result["lambda"]
    render_states = result["render_states"]
    plan_states = result["plan_states"]
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
        json.dump(result["debug"], f, indent=2, default=_json_default)

    seed_results = _load_json(seed_dir / "results.json")
    meta = result["metadata"]
    follow_results: Dict[str, Any] = {
        "seed_dir": str(seed_dir),
        "out_dir": str(out_dir),
        "frames": int(states.shape[0]),
        "nq": int(qpos.shape[1]) if qpos.ndim == 2 and qpos.size > 0 else 0,
        "nv": int(qvel.shape[1]) if qvel.ndim == 2 and qvel.size > 0 else 0,
        "nu": int(ctrl.shape[1]) if ctrl.ndim == 2 and ctrl.size > 0 else 0,
        "control_dt": float(meta.get("control_dt", args.control_dt)),
        "sim_dt": float(meta.get("sim_dt", args.sim_dt)),
        "best_idx": int(meta.get("best_idx", args.best_idx if args.best_idx is not None else -1)),
        "schema_version": str(meta.get("schema_version", "")),
        "render_state_mode": str(meta.get("render_state_mode", "")),
        "solver": str(meta.get("solver", "")),
        "preset": str(meta.get("preset", "")),
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
            render_sub = render_source_states[idx]
            render_actions = None
            if ctrl.ndim == 2 and ctrl.shape[0] > 0 and idx.size > 1:
                aidx = np.clip(idx[:-1], 0, ctrl.shape[0] - 1)
                render_actions = ctrl[aidx] if aidx.size > 0 else None
            fps = 1.0 / (float(args.sim_dt) * float(step))
            episode = MotionEpisode(
                states=render_sub,
                actions=render_actions,
                robot_type="humanoid",
                model_id="g1",
                fps=fps,
                metadata={
                    "seed_dir": str(seed_dir),
                    "best_idx": int(args.best_idx) if args.best_idx is not None else None,
                    "corridor_scene": corridor_scene,
                    "render_kind": render_kind,
                    "render_state_mode": str(meta.get("render_state_mode", "")),
                    "preset": str(meta.get("preset", "")),
                },
            )
            html_paths[render_kind] = renderer.render_html(
                episode, name=render_name, width=960, height=720, fps=fps,
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
