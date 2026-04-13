#!/usr/bin/env python
"""Diagnose the WBC controller on a real corridor trajectory.

Sibling of :mod:`scripts.tasks.robot.diagnose_sport_mode_corridor`. Produces the same
metric schema (steps, fell-over flag, pelvis z range, endpoint distance,
mean/max lateral offset, real-time factor) so the WBC and sport-mode
results can be compared directly with ``diff``-friendly numbers.

Path C' from the user's plan: validate whether the WBC controller can do
on the same ``trajectory.json`` what spark's RL policy already does. If
WBC matches or beats the RL baseline, you've shipped a *first-principles*
controller that doesn't depend on a third-party trained checkpoint.

What it does
------------
1. Load the plan via :class:`CorridorTrajectoryAdapter`.
2. Build :class:`MujocoRobotIO` + :class:`HumanoidWBCController` directly
   (no :class:`HumanoidMujocoPipeline` wrapper).
3. Wire up the follower stack: :class:`HumanoidContactScheduler` +
   :class:`HumanoidFootstepPlanner` + :class:`HumanoidUpperBodyMapper` +
   :class:`HumanoidTaskBuilder`.
4. Step the loop: derive :class:`Intent` with ``extras["humanoid_tasks"]``
   from the follower stack and call ``controller.act(state, intent)``.
5. Extract pelvis xy from qpos, run the same metric helpers as the
   sport-mode diagnose, and emit the same report.

Output schema
-------------
Identical to ``diagnose_sport_mode_corridor.py``:

* stdout text report with verdict
* ``<out-dir>/diagnose_wbc.npz`` — full per-step timeseries
* ``<out-dir>/diagnose_wbc.json`` — JSON summary

Usage
-----
::

    docker run --rm -v "$PWD:/work" -w /work genedynamics/dev-cpu:local \\
        python scripts/tasks/robot/diagnose_wbc_corridor.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# Path: scripts/tasks/robot/<file>.py — parents[3] is the project root.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


DEFAULT_PLAN = (
    "results/humanoid/corridor_2d/smoke/twogo_zone_a/level_1/seed_0/"
    "trajectory/trajectory.json"
)


# ---------------------------------------------------------------------------
# Result container — mirrors sport-mode diagnose
# ---------------------------------------------------------------------------


@dataclass
class WBCDiagnoseResult:
    plan_path: str
    n_steps: int
    sim_time_s: float
    wall_time_s: float
    fell_over: bool
    final_pelvis_xy: np.ndarray
    final_plan_xy: np.ndarray
    endpoint_distance_m: float
    mean_lateral_offset_m: float
    max_lateral_offset_m: float
    pelvis_z_min: float
    pelvis_z_max: float
    pelvis_xy: np.ndarray  # (n, 2)
    plan_xy: np.ndarray  # (n, 2)
    qp_solve_failures: int
    torque_saturations: int

    def to_json(self) -> dict:
        return {
            "plan_path": self.plan_path,
            "controller": "wbc",
            "n_steps": self.n_steps,
            "sim_time_s": self.sim_time_s,
            "wall_time_s": self.wall_time_s,
            "fell_over": self.fell_over,
            "endpoint_distance_m": float(self.endpoint_distance_m),
            "mean_lateral_offset_m": float(self.mean_lateral_offset_m),
            "max_lateral_offset_m": float(self.max_lateral_offset_m),
            "pelvis_z_min": float(self.pelvis_z_min),
            "pelvis_z_max": float(self.pelvis_z_max),
            "qp_solve_failures": int(self.qp_solve_failures),
            "torque_saturations": int(self.torque_saturations),
        }


# ---------------------------------------------------------------------------
# Geometry helpers (same logic as sport-mode diagnose)
# ---------------------------------------------------------------------------


def _lateral_offsets(actual_xy: np.ndarray, plan_xy: np.ndarray) -> np.ndarray:
    if actual_xy.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    if plan_xy.shape[0] < 2:
        return np.linalg.norm(actual_xy - plan_xy[0], axis=1)
    out = np.zeros(actual_xy.shape[0], dtype=np.float64)
    seg_starts = plan_xy[:-1]
    seg_ends = plan_xy[1:]
    seg_vecs = seg_ends - seg_starts
    seg_lens_sq = np.sum(seg_vecs * seg_vecs, axis=1)
    seg_lens_sq = np.where(seg_lens_sq < 1e-12, 1e-12, seg_lens_sq)
    for i, p in enumerate(actual_xy):
        diffs = p - seg_starts
        t = np.clip(np.sum(diffs * seg_vecs, axis=1) / seg_lens_sq, 0.0, 1.0)
        proj = seg_starts + (t[:, None] * seg_vecs)
        d = np.linalg.norm(p - proj, axis=1)
        out[i] = float(d.min())
    return out


# ---------------------------------------------------------------------------
# Core diagnose
# ---------------------------------------------------------------------------


def diagnose(
    plan_path: Path,
    *,
    control_hz: float = 50.0,
    sim_dt: float = 1.0 / 500.0,
    max_steps: Optional[int] = None,
    fall_threshold_m: float = 0.40,
    out_dir: Optional[Path] = None,
    quiet: bool = False,
    cap_vx: Optional[float] = None,
    cap_vy: Optional[float] = None,
    cap_omega: Optional[float] = None,
) -> WBCDiagnoseResult:
    if not quiet:
        print(f"[wbc-diagnose] loading plan: {plan_path}")

    # ----- 1. Imports (lazy so ImportError is surfaced cleanly) -----------
    from genedynamics.deploy.followers.common.plan_adapter import (
        CorridorTrajectoryAdapter,
    )
    from genedynamics.deploy.followers.common.traversal_intent import TraversalIntent
    from genedynamics.deploy.followers.humanoid.contact_scheduler import (
        HumanoidContactScheduler,
    )
    from genedynamics.deploy.followers.humanoid.footstep_planner import (
        HumanoidFootstepPlanner,
    )
    from genedynamics.deploy.followers.humanoid.task_builder import HumanoidTaskBuilder
    from genedynamics.deploy.followers.humanoid.task_spec import (
        ContactObservations,
        FootContactObservation,
    )
    from genedynamics.deploy.followers.humanoid.upper_body_mapper import (
        HumanoidUpperBodyMapper,
    )
    from genedynamics.deploy.controllers.wbc import HumanoidWBCController, WBCResult
    from genedynamics.deploy.interfaces.messages import Intent
    from genedynamics.deploy.io.mujoco_io import MujocoRobotIO, FootContactSnapshot

    # ----- 2. Load plan ---------------------------------------------------
    adapter = CorridorTrajectoryAdapter(
        source_dt=0.25,
        target_dt=1.0 / control_hz,
    )
    plan = adapter.load_trajectory_json(plan_path)
    schema = plan.schema

    if cap_vx is not None or cap_vy is not None or cap_omega is not None:
        if not quiet:
            print(
                f"[wbc-diagnose] capping plan velocities: "
                f"vx≤{cap_vx} vy≤{cap_vy} omega≤{cap_omega}"
            )
        states = plan.states.copy()
        if cap_vx is not None:
            i = schema.field_names.index("v_x")
            states[:, i] = np.clip(states[:, i], -float(cap_vx), float(cap_vx))
        if cap_vy is not None:
            i = schema.field_names.index("v_y")
            states[:, i] = np.clip(states[:, i], -float(cap_vy), float(cap_vy))
        if cap_omega is not None:
            i = schema.field_names.index("omega")
            states[:, i] = np.clip(states[:, i], -float(cap_omega), float(cap_omega))
        plan.states = states

    frames = plan.decode_frames()
    n_plan = len(frames)
    if not quiet:
        print(
            f"[wbc-diagnose] plan loaded: {n_plan} frames @ "
            f"{1.0 / control_hz:.4f}s"
        )

    plan_xy_full = np.asarray([[f.x, f.y] for f in frames], dtype=np.float64)
    step_dt = 1.0 / control_hz

    # ----- 3. Build IO + controller + follower stack ----------------------
    if not quiet:
        print("[wbc-diagnose] building MujocoRobotIO + HumanoidWBCController + follower stack")

    io = MujocoRobotIO(sim_dt=sim_dt)
    controller = HumanoidWBCController(io=io)
    contact_scheduler = HumanoidContactScheduler()
    footstep_planner = HumanoidFootstepPlanner()
    upper_body_mapper = HumanoidUpperBodyMapper()
    task_builder = HumanoidTaskBuilder()

    # Anchor to first plan frame.
    f0 = frames[0]
    import mujoco  # noqa: PLC0415 — inside Docker only
    io.data.qpos[0] = float(f0.x)
    io.data.qpos[1] = float(f0.y)
    io.data.qpos[2] = float(f0.h) + (float(io.data.qpos[2]) - 0.75)
    cy, sy = float(np.cos(0.5 * f0.psi)), float(np.sin(0.5 * f0.psi))
    io.data.qpos[3:7] = [cy, 0.0, 0.0, sy]
    mujoco.mj_forward(io.model, io.data)

    # Seed footstep planner from current foot positions.
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

    # ----- 4. Replay loop -------------------------------------------------
    n_steps = min(n_plan, int(max_steps) if max_steps else n_plan)
    nq_full = int(io.model.nq)
    nv_full = int(io.model.nv)

    pelvis_xy_log = np.zeros((n_steps, 2), dtype=np.float64)
    pelvis_z_log = np.zeros(n_steps, dtype=np.float64)
    qpos_log = np.zeros((n_steps, nq_full), dtype=np.float64)
    debug: list[dict] = []

    wall_t0 = time.monotonic()
    for k in range(n_steps):
        f = frames[k]
        traversal_intent = TraversalIntent.from_plan_frame(f)
        contact_obs = _contact_obs()
        phase = contact_scheduler.advance(traversal_intent, step_dt, contact_obs)
        footsteps = footstep_planner.update(traversal_intent, phase, contact_obs, dt=step_dt)
        upper_body = upper_body_mapper.map(f)
        tasks = task_builder.build(f, traversal_intent, phase, footsteps, upper_body)

        state = io.get_state()
        intent = Intent(
            t=float(f.time_sec),
            base_yaw=float(traversal_intent.yaw),
            base_height=float(traversal_intent.body_height),
            base_lin_vel=traversal_intent.planar_velocity.copy(),
            base_yaw_rate=float(traversal_intent.yaw_rate),
            base_pos_xy=traversal_intent.planar_position.copy(),
            torso_yaw=float(tasks.torso_yaw),
            contact_phase=phase.phase.value,
            extras={
                "humanoid_tasks": tasks,
                "phase": phase,
                "dt": step_dt,
            },
        )
        cmd = controller.act(state, intent)
        wbc: WBCResult = cmd.extras.get("wbc")

        io.send_control(cmd)
        io.step(step_dt)

        qpos = np.asarray(io.data.qpos, dtype=np.float64)
        pelvis_xy_log[k] = qpos[:2]
        pelvis_z_log[k] = float(qpos[2])
        qpos_log[k] = qpos

        debug.append({
            "frame_index": k,
            "solver_metadata": {
                "method": wbc.method if wbc else "",
                "eq_residual_norm": float(wbc.eq_residual) if wbc else 0.0,
                "ineq_violation_max": float(wbc.ineq_violation) if wbc else 0.0,
                "task_errors": dict(wbc.task_errors) if wbc else {},
                "contact_feet": list(wbc.contact_feet) if wbc else [],
                "torque_saturation_max": float(wbc.torque_saturation_max) if wbc else 0.0,
                "torque_bound_violation_max": float(wbc.torque_bound_violation_max) if wbc else 0.0,
            },
        })

        if pelvis_z_log[k] < fall_threshold_m:
            n_steps = k + 1
            pelvis_xy_log = pelvis_xy_log[:n_steps]
            pelvis_z_log = pelvis_z_log[:n_steps]
            qpos_log = qpos_log[:n_steps]
            break

    wall_t1 = time.monotonic()
    if not quiet:
        print(f"[wbc-diagnose] rollout finished: {n_steps} steps in {wall_t1 - wall_t0:.2f}s")

    # ----- 5. Metrics ------------------------------------------------------
    plan_xy_actual = plan_xy_full[:n_steps]
    final_plan = plan_xy_full[n_steps - 1] if n_steps > 0 else plan_xy_full[-1]
    final_actual = pelvis_xy_log[-1] if n_steps > 0 else np.zeros(2)
    endpoint_d = float(np.linalg.norm(final_actual - final_plan))
    fell = bool(pelvis_z_log.min() < fall_threshold_m) if n_steps > 0 else True
    laterals = _lateral_offsets(pelvis_xy_log, plan_xy_full)

    def _meta(entry: dict) -> dict:
        return entry.get("solver_metadata") or {}

    qp_failures = sum(
        1
        for d in debug
        if float(_meta(d).get("torque_bound_violation_max", 0.0)) > 1e6
    )
    torque_sat_count = sum(
        1
        for d in debug
        if float(_meta(d).get("torque_saturation_max", 0.0)) > 0.99
    )

    result = WBCDiagnoseResult(
        plan_path=str(plan_path),
        n_steps=n_steps,
        sim_time_s=n_steps * step_dt,
        wall_time_s=wall_t1 - wall_t0,
        fell_over=fell,
        final_pelvis_xy=final_actual,
        final_plan_xy=final_plan,
        endpoint_distance_m=endpoint_d,
        mean_lateral_offset_m=float(laterals.mean()) if laterals.size else float("nan"),
        max_lateral_offset_m=float(laterals.max()) if laterals.size else float("nan"),
        pelvis_z_min=float(pelvis_z_log.min()) if n_steps else float("nan"),
        pelvis_z_max=float(pelvis_z_log.max()) if n_steps else float("nan"),
        pelvis_xy=pelvis_xy_log,
        plan_xy=plan_xy_actual,
        qp_solve_failures=qp_failures,
        torque_saturations=torque_sat_count,
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

        # Extract per-step task-error arrays from solver_metadata.
        task_err_keys: set[str] = set()
        for d in debug:
            for k in (_meta(d).get("task_errors") or {}):
                task_err_keys.add(k)
        task_err_arrays: dict[str, np.ndarray] = {}
        for k in sorted(task_err_keys):
            arr = np.full(n_steps, np.nan, dtype=np.float64)
            for i, d in enumerate(debug[:n_steps]):
                val = (_meta(d).get("task_errors") or {}).get(k)
                if val is not None:
                    arr[i] = float(val)
            task_err_arrays[f"task_err_{k}"] = arr

        torque_sat_arr = np.array(
            [float(_meta(d).get("torque_saturation_max", 0.0)) for d in debug[:n_steps]],
            dtype=np.float64,
        )
        contact_count_arr = np.array(
            [int(len(_meta(d).get("contact_feet", []))) for d in debug[:n_steps]],
            dtype=np.int32,
        )
        eq_resid_arr = np.array(
            [float(_meta(d).get("eq_residual_norm", 0.0)) for d in debug[:n_steps]],
            dtype=np.float64,
        )
        ineq_viol_arr = np.array(
            [float(_meta(d).get("ineq_violation_max", 0.0)) for d in debug[:n_steps]],
            dtype=np.float64,
        )
        np.savez(
            out_dir / "diagnose_wbc.npz",
            pelvis_xy=result.pelvis_xy,
            pelvis_z=pelvis_z_log,
            plan_xy=result.plan_xy,
            laterals=laterals,
            qpos=qpos_log,
            torque_saturation=torque_sat_arr,
            contact_count=contact_count_arr,
            eq_residual=eq_resid_arr,
            ineq_violation=ineq_viol_arr,
            **task_err_arrays,
        )
        (out_dir / "diagnose_wbc.json").write_text(
            json.dumps(result.to_json(), indent=2)
        )
        if not quiet:
            print(f"[wbc-diagnose] dumped {out_dir / 'diagnose_wbc.npz'}")

    io.close()
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(result: WBCDiagnoseResult) -> None:
    print()
    print("=" * 64)
    print(" WBC corridor diagnose")
    print("=" * 64)
    print(f"  Plan:                   {result.plan_path}")
    print(f"  Steps run:              {result.n_steps}")
    print(f"  Sim time:               {result.sim_time_s:.2f} s")
    print(f"  Wall time:              {result.wall_time_s:.2f} s")
    print(f"  Real-time factor:       {result.sim_time_s / max(result.wall_time_s, 1e-6):.2f}x")
    print()
    fall = "YES — robot fell over" if result.fell_over else "no"
    print(f"  Fell over:              {fall}")
    print(f"  Pelvis z range:         [{result.pelvis_z_min:.3f}, {result.pelvis_z_max:.3f}] m")
    print(f"  Endpoint distance:      {result.endpoint_distance_m:.3f} m")
    print(f"  Mean lateral offset:    {result.mean_lateral_offset_m:.3f} m")
    print(f"  Max  lateral offset:    {result.max_lateral_offset_m:.3f} m")
    print(f"  Final pelvis xy:        ({result.final_pelvis_xy[0]:+.3f}, {result.final_pelvis_xy[1]:+.3f})")
    print(f"  Final plan    xy:       ({result.final_plan_xy[0]:+.3f}, {result.final_plan_xy[1]:+.3f})")
    print(f"  WBC torque saturations: {result.torque_saturations}")
    print(f"  WBC QP failures:        {result.qp_solve_failures}")
    print()

    if result.fell_over:
        verdict = "FAIL — WBC could not keep G1 upright on this plan."
        recommendation = (
            "    Likely WBC tuning issue. Inspect torque saturation count\n"
            "    (>0 means hitting force bounds). Try lowering task weights\n"
            "    in deploy/controllers/wbc/config.py, or see README.md."
        )
    elif result.endpoint_distance_m > 1.0:
        verdict = "MARGINAL — WBC stayed up but missed endpoint by >1 m."
        recommendation = "    Check follower velocity scaling and contact phase scheduling."
    elif result.max_lateral_offset_m > 0.5:
        verdict = "MARGINAL — endpoint OK but lateral drift exceeds 0.5 m."
        recommendation = "    Tune the WBC pelvis lateral task weight."
    else:
        verdict = "PASS — WBC traces the corridor cleanly. Ship it."
        recommendation = "    Next: compare frame-by-frame against the spark RL baseline."
    print(f"  Verdict: {verdict}")
    print(recommendation)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--plan", type=Path, default=_ROOT / DEFAULT_PLAN)
    p.add_argument("--control-hz", type=float, default=50.0)
    p.add_argument("--sim-dt", type=float, default=1.0 / 500.0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--cap-vx", type=float, default=None,
                   help="Clip plan v_x channel to ±this value (m/s)")
    p.add_argument("--cap-vy", type=float, default=None,
                   help="Clip plan v_y channel to ±this value (m/s)")
    p.add_argument("--cap-omega", type=float, default=None,
                   help="Clip plan omega channel to ±this value (rad/s)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "results" / "deploy" / "diagnose_wbc",
    )
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if not args.plan.exists():
        print(f"ERROR: plan not found: {args.plan}", file=sys.stderr)
        return 2

    try:
        result = diagnose(
            args.plan,
            control_hz=args.control_hz,
            sim_dt=args.sim_dt,
            max_steps=args.max_steps,
            out_dir=args.out_dir,
            quiet=args.quiet,
            cap_vx=args.cap_vx,
            cap_vy=args.cap_vy,
            cap_omega=args.cap_omega,
        )
    except ImportError as exc:
        print(f"ERROR: missing dependency: {exc}", file=sys.stderr)
        print(
            "  Run inside the genedynamics/dev-cpu Docker image:\n"
            "    docker run --rm -v $PWD:/work -w /work genedynamics/dev-cpu \\\n"
            "        python scripts/tasks/robot/diagnose_wbc_corridor.py",
            file=sys.stderr,
        )
        return 3

    print_report(result)
    return 0 if not result.fell_over else 1


if __name__ == "__main__":
    sys.exit(main())
