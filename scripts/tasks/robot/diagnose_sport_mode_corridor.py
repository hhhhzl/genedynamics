#!/usr/bin/env python
"""Diagnose sport-mode (spark RL motion.pt) on a real corridor trajectory.

This is the highest-information test we have for the sport-mode path:
does spark's pretrained RL policy trace your 14D corridor plan when
plugged in via :class:`SportModeController` + :class:`SparkRLLocoClient`?

What it does
------------
1. Load a planner output JSON (default: a smoke-test 14D corridor) via
   :class:`CorridorTrajectoryAdapter` and resample to the control rate.
2. Construct :class:`MujocoRobotIO` against the G1 scene.
3. Construct :class:`SportModeController` driven by
   :class:`SparkRLLocoClient` (loads ``spark/.../motion.pt``).
4. Build a :class:`HumanoidUpperBodyMapper` so the 14D plan's
   ``psi_torso`` / ``a_left/right`` / ``p_left/right`` fields are
   decoded into 17 upper-body joint targets each control tick.
5. Replay the plan: at every control step, derive an
   :class:`Intent` from the current plan frame (``v_x``, ``v_y``, yaw rate,
   target height, torso yaw) and call ``controller.act(state, intent)``.
6. Step the IO, record per-step pelvis xy + intent + cmd, and compute
   summary metrics.

Output
------
* A short text report on stdout summarising:
    - whether the robot fell over (pelvis z < 0.40 m)
    - final tracking error against the plan endpoint
    - mean and max lateral offset from the plan path
    - control loop wall time and any rate-limiter overruns
* A ``.npz`` dump under ``--out-dir`` with the full per-step timeseries so
  follow-up scripts can plot.
* (optional) A pyplot PNG ``--plot`` showing plan vs actual pelvis trace.

Usage
-----
::

    docker run --rm -it -v $PWD:/work -w /work genedynamics/dev-cpu \\
        python scripts/tasks/robot/diagnose_sport_mode_corridor.py

    # On Apple Silicon hosts, run inside the dev-cpu image — the script
    # imports mujoco at top-of-file via MujocoRobotIO.

    # Pick a different plan file:
    python scripts/tasks/robot/diagnose_sport_mode_corridor.py \\
        --plan results/humanoid/corridor_2d/smoke/twogo_zone_a/level_1/seed_0/trajectory/trajectory.json

    # Save a plot:
    python scripts/tasks/robot/diagnose_sport_mode_corridor.py --plot

Why this exists
---------------
Before this script, "does sport-mode work?" was answered by spinning up
the full deploy pipeline and eyeballing logs. This collapses that into
30 seconds of wall-clock time and three numbers (fall / drift / endpoint
distance) that decide whether the WBC tuning sprint or the 24D state
upgrade are actually needed.
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


# Project root added to sys.path so the script runs from anywhere.
# Path: scripts/tasks/robot/<file>.py — parents[3] is the project root.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


DEFAULT_PLAN = (
    "results/humanoid/corridor_2d/smoke/twogo_zone_a/level_1/seed_0/"
    "trajectory/trajectory.json"
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class DiagnoseResult:
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
    intent_lin_vel: np.ndarray  # (n, 2)
    intent_yaw_rate: np.ndarray  # (n,)
    rate_overruns: int

    def to_json(self) -> dict:
        """Compact JSON-friendly summary (no big arrays)."""
        return {
            "plan_path": self.plan_path,
            "n_steps": self.n_steps,
            "sim_time_s": self.sim_time_s,
            "wall_time_s": self.wall_time_s,
            "fell_over": self.fell_over,
            "endpoint_distance_m": float(self.endpoint_distance_m),
            "mean_lateral_offset_m": float(self.mean_lateral_offset_m),
            "max_lateral_offset_m": float(self.max_lateral_offset_m),
            "pelvis_z_min": float(self.pelvis_z_min),
            "pelvis_z_max": float(self.pelvis_z_max),
            "rate_overruns": int(self.rate_overruns),
        }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _lateral_offsets(actual_xy: np.ndarray, plan_xy: np.ndarray) -> np.ndarray:
    """Per-step nearest-segment perpendicular distance from actual to plan polyline."""
    n_actual = actual_xy.shape[0]
    if plan_xy.shape[0] < 2:
        return np.linalg.norm(actual_xy - plan_xy[0], axis=1)
    out = np.zeros(n_actual, dtype=np.float64)
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
    loco_kwargs: Optional[dict] = None,
    cap_vx: Optional[float] = None,
) -> DiagnoseResult:
    """Run the diagnostic and return the result struct.

    Args:
        loco_kwargs: Forwarded to :class:`SparkRLLocoClient.__init__` so
            callers can override ``policy_path``, ``nominal_step_period``
            or ``cmd_clip``.
        cap_vx: Optional cap on ``intent.base_lin_vel`` magnitude. Useful
            for testing whether slowing the commanded velocity keeps the
            walker on track when the policy's training-time speed limit
            (±0.3 m/s) is below the plan's commanded speed.
    """

    # ----- 1. Load the plan ------------------------------------------------
    if not quiet:
        print(f"[diagnose] loading plan: {plan_path}")
    from genedynamics.deploy.followers.common.plan_adapter import (
        CorridorTrajectoryAdapter,
    )

    adapter = CorridorTrajectoryAdapter(
        source_dt=0.25,
        target_dt=1.0 / control_hz,
    )
    plan = adapter.load_trajectory_json(plan_path)
    n_plan = plan.states.shape[0]
    if not quiet:
        print(f"[diagnose] plan loaded: {n_plan} frames @ {1.0 / control_hz:.4f}s")

    # Decode every frame once so the loop is just an array index lookup.
    schema = plan.schema
    frames = [schema.decode_state(plan.states[i], time_sec=float(plan.times[i])) for i in range(n_plan)]
    plan_xy_full = np.asarray([[f.x, f.y] for f in frames], dtype=np.float64)

    # ----- 2. Build IO + controller + upper-body mapper -------------------
    if not quiet:
        print(
            "[diagnose] building MujocoRobotIO + SportModeController + "
            "SparkRLLocoClient + HumanoidUpperBodyMapper"
        )
    from genedynamics.deploy.io.mujoco_io import MujocoRobotIO
    from genedynamics.deploy.controllers.sport_mode import (
        SparkRLLocoClient,
        SportModeController,
    )
    from genedynamics.deploy.followers.humanoid.upper_body_mapper import (
        HumanoidUpperBodyMapper,
    )
    from genedynamics.envs.robots.g1 import G1RobotModel

    io = MujocoRobotIO(sim_dt=sim_dt)
    robot = G1RobotModel()
    loco = SparkRLLocoClient(robot=robot, **(loco_kwargs or {}))
    controller = SportModeController(io=io, loco_client=loco)
    upper_body_mapper = HumanoidUpperBodyMapper()

    # Anchor the robot to the plan's first XY so the diagnostic actually
    # tracks rather than diverging from the moment t=0.
    f0 = frames[0]
    state = io.reset_to(
        base_xyz=(f0.x, f0.y, f0.h),
        base_quat_wxyz=(np.cos(f0.psi / 2), 0.0, 0.0, np.sin(f0.psi / 2)),
    )
    controller.reset(io)

    # ----- 3. Replay loop -------------------------------------------------
    from genedynamics.deploy.interfaces.messages import Intent
    from genedynamics.deploy.runtime.rate_limiter import RateLimiter

    rate = RateLimiter(hz=control_hz)
    n_steps = min(n_plan, int(max_steps) if max_steps else n_plan)

    pelvis_xy_log = np.zeros((n_steps, 2), dtype=np.float64)
    pelvis_z_log = np.zeros(n_steps, dtype=np.float64)
    intent_lin_vel_log = np.zeros((n_steps, 2), dtype=np.float64)
    intent_yaw_rate_log = np.zeros(n_steps, dtype=np.float64)
    # Full state history for the MuJoCo renderer (needs nq + nv per step).
    nq_full = int(io.model.nq)
    nv_full = int(io.model.nv)
    qpos_log = np.zeros((n_steps, nq_full), dtype=np.float64)
    qvel_log = np.zeros((n_steps, nv_full), dtype=np.float64)

    wall_t0 = time.monotonic()
    for k in range(n_steps):
        rate.tick()
        f = frames[k]
        vxy = np.array([f.v_x, f.v_y], dtype=np.float64)
        if cap_vx is not None:
            speed = float(np.linalg.norm(vxy))
            if speed > cap_vx:
                vxy = vxy * (cap_vx / max(speed, 1e-9))
        # Decode the 14D plan frame's torso/arm fields into 17 upper-body
        # joint targets (waist x3 + arms x14). The SportModeController will
        # merge these with the leg targets coming from the loco client.
        upper_targets = upper_body_mapper.map(f)
        intent = Intent(
            t=float(f.time_sec),
            base_yaw=float(f.psi),
            base_height=float(f.h),
            base_lin_vel=vxy,
            base_yaw_rate=float(f.omega),
            torso_yaw=float(f.psi_torso),
            extras={
                "dt": 1.0 / control_hz,
                "upper_body_targets": upper_targets,
            },
        )
        cmd = controller.act(state, intent)
        io.send_control(cmd)
        state = io.step(1.0 / control_hz)

        if state.qpos is not None and state.qpos.size >= 3:
            pelvis_xy_log[k] = state.qpos[:2]
            pelvis_z_log[k] = float(state.qpos[2])
        if state.qpos is not None and state.qpos.size == nq_full:
            qpos_log[k] = state.qpos
        if state.qvel is not None and state.qvel.size == nv_full:
            qvel_log[k] = state.qvel
        intent_lin_vel_log[k] = intent.base_lin_vel
        intent_yaw_rate_log[k] = intent.base_yaw_rate

        if pelvis_z_log[k] < fall_threshold_m:
            n_steps = k + 1  # truncate logs
            pelvis_xy_log = pelvis_xy_log[:n_steps]
            pelvis_z_log = pelvis_z_log[:n_steps]
            qpos_log = qpos_log[:n_steps]
            qvel_log = qvel_log[:n_steps]
            intent_lin_vel_log = intent_lin_vel_log[:n_steps]
            intent_yaw_rate_log = intent_yaw_rate_log[:n_steps]
            break
    wall_t1 = time.monotonic()

    # ----- 4. Metrics ------------------------------------------------------
    plan_xy_actual = plan_xy_full[:n_steps]
    final_plan = plan_xy_full[n_steps - 1] if n_steps > 0 else plan_xy_full[-1]
    final_actual = pelvis_xy_log[-1] if n_steps > 0 else np.zeros(2)
    endpoint_d = float(np.linalg.norm(final_actual - final_plan))

    laterals = _lateral_offsets(pelvis_xy_log, plan_xy_full)
    fell = bool(pelvis_z_log.min() < fall_threshold_m) if n_steps > 0 else True

    result = DiagnoseResult(
        plan_path=str(plan_path),
        n_steps=n_steps,
        sim_time_s=n_steps / control_hz,
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
        intent_lin_vel=intent_lin_vel_log,
        intent_yaw_rate=intent_yaw_rate_log,
        rate_overruns=int(rate.stats.overruns),
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / "diagnose_sport_mode.npz"
        np.savez(
            npz_path,
            pelvis_xy=result.pelvis_xy,
            pelvis_z=pelvis_z_log,
            plan_xy=result.plan_xy,
            intent_lin_vel=result.intent_lin_vel,
            intent_yaw_rate=result.intent_yaw_rate,
            laterals=laterals,
            qpos=qpos_log,
            qvel=qvel_log,
        )
        (out_dir / "diagnose_sport_mode.json").write_text(
            json.dumps(result.to_json(), indent=2)
        )
        if not quiet:
            print(f"[diagnose] dumped {npz_path}")

    io.close()
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(result: DiagnoseResult) -> None:
    print()
    print("=" * 64)
    print(" Sport-mode corridor diagnose")
    print("=" * 64)
    print(f"  Plan:                   {result.plan_path}")
    print(f"  Steps run:              {result.n_steps}")
    print(f"  Sim time:               {result.sim_time_s:.2f} s")
    print(f"  Wall time:              {result.wall_time_s:.2f} s")
    print(f"  Real-time factor:       {result.sim_time_s / max(result.wall_time_s, 1e-6):.2f}x")
    print(f"  Rate-limiter overruns:  {result.rate_overruns}")
    print()
    fall = "YES — robot fell over" if result.fell_over else "no"
    print(f"  Fell over:              {fall}")
    print(f"  Pelvis z range:         [{result.pelvis_z_min:.3f}, {result.pelvis_z_max:.3f}] m")
    print(f"  Endpoint distance:      {result.endpoint_distance_m:.3f} m")
    print(f"  Mean lateral offset:    {result.mean_lateral_offset_m:.3f} m")
    print(f"  Max  lateral offset:    {result.max_lateral_offset_m:.3f} m")
    print(f"  Final pelvis xy:        ({result.final_pelvis_xy[0]:+.3f}, {result.final_pelvis_xy[1]:+.3f})")
    print(f"  Final plan    xy:       ({result.final_plan_xy[0]:+.3f}, {result.final_plan_xy[1]:+.3f})")
    print()

    # Pass/fail call
    if result.fell_over:
        verdict = "FAIL — sport-mode + SparkRLLocoClient cannot keep G1 upright."
        recommendation = (
            "    The pretrained policy diverged. Check obs scaling, joint\n"
            "    permutation, or whether plan velocities exceed the policy's\n"
            "    training-time range (default cmd_clip = ±0.3 m/s)."
        )
    elif result.endpoint_distance_m > 1.0:
        verdict = "MARGINAL — robot stayed up but missed endpoint by >1 m."
        recommendation = (
            "    Likely the plan is faster than the policy was trained on.\n"
            "    Either extend the rollout (--max-steps) or relax cmd_clip\n"
            "    in SparkRLLocoClient."
        )
    elif result.max_lateral_offset_m > 0.5:
        verdict = "MARGINAL — endpoint OK but lateral drift exceeds 0.5 m."
        recommendation = (
            "    Next step: verify the follower → controller heading channel\n"
            "    (intent.base_yaw vs intent.base_yaw_rate)."
        )
    else:
        verdict = "PASS — sport-mode tracks the corridor well enough to skip WBC tuning."
        recommendation = (
            "    Next step: ship the sport-mode preset and move to Phase 7\n"
            "    on real hardware when available."
        )
    print(f"  Verdict: {verdict}")
    print(recommendation)
    print()


# ---------------------------------------------------------------------------
# Plotting (optional)
# ---------------------------------------------------------------------------


def make_plot(result: DiagnoseResult, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[diagnose] matplotlib not installed; skipping plot")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(result.plan_xy[:, 0], result.plan_xy[:, 1], "k--", lw=2, label="plan")
    ax.plot(result.pelvis_xy[:, 0], result.pelvis_xy[:, 1], "C0-", lw=1.5, label="pelvis")
    ax.scatter(result.final_plan_xy[0], result.final_plan_xy[1], s=80, c="k", marker="x", label="plan end")
    ax.scatter(result.final_pelvis_xy[0], result.final_pelvis_xy[1], s=80, c="C0", marker="o", label="pelvis end")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"sport-mode corridor diagnose — {Path(result.plan_path).parent.parent.name}\n"
        f"endpoint err {result.endpoint_distance_m:.2f} m, max lat {result.max_lateral_offset_m:.2f} m"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"[diagnose] plot saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--plan",
        type=Path,
        default=_ROOT / DEFAULT_PLAN,
        help=f"Path to a planner trajectory.json. Default: {DEFAULT_PLAN}",
    )
    p.add_argument("--control-hz", type=float, default=50.0)
    p.add_argument("--sim-dt", type=float, default=1.0 / 500.0)
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap the rollout length (default: full plan)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "results" / "deploy" / "diagnose_sport_mode",
        help="Where to dump the .npz / .json artifacts",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Save a pelvis-vs-plan PNG to <out-dir>/diagnose_sport_mode.png",
    )
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if not args.plan.exists():
        print(f"ERROR: plan file not found: {args.plan}", file=sys.stderr)
        return 2

    try:
        result = diagnose(
            args.plan,
            control_hz=args.control_hz,
            sim_dt=args.sim_dt,
            max_steps=args.max_steps,
            out_dir=args.out_dir,
            quiet=args.quiet,
        )
    except ImportError as exc:
        print(f"ERROR: missing dependency: {exc}", file=sys.stderr)
        print(
            "       Run inside the genedynamics/dev-cpu Docker image:\n"
            "         docker run --rm -it -v $PWD:/work -w /work \\\n"
            "             genedynamics/dev-cpu \\\n"
            "             python scripts/tasks/robot/diagnose_sport_mode_corridor.py",
            file=sys.stderr,
        )
        return 3

    print_report(result)
    if args.plot:
        make_plot(result, args.out_dir / "diagnose_sport_mode.png")
    return 0 if not result.fell_over else 1


if __name__ == "__main__":
    sys.exit(main())
