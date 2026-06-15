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
        python scripts/tasks/robot/humanoid/sport_mode_corridor.py

    # On Apple Silicon hosts, run inside the dev-cpu image — the script
    # imports mujoco at top-of-file via MujocoRobotIO.

    # Pick a different plan file:
    python scripts/tasks/robot/humanoid/sport_mode_corridor.py \\
        --plan results/humanoid/corridor_2d/main/twogo_zone_a/level_1/seed_0/trajectory/trajectory.json

    # Save a plot:
    python scripts/tasks/robot/humanoid/sport_mode_corridor.py --plot

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
from typing import Any, Optional

import numpy as np


# Project root added to sys.path so the script runs from anywhere.
# Path: scripts/tasks/robot/humanoid/<file>.py — parents[4] is the project root.
_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


DEFAULT_PLAN = (
    "results/humanoid/corridor_2d/main/twogo_long/level_1/seed_0/"
    "trajectory/trajectory.json"
)


# Convenience aliases so callers / batch drivers can pass a short name.
PLAN_ALIASES = {
    "twogo_long":   "results/humanoid/corridor_2d/main/twogo_long/level_1/seed_0/trajectory/trajectory.json",
    "twogo_zone_a": "results/humanoid/corridor_2d/main/twogo_zone_a/level_1/seed_0/trajectory/trajectory.json",
    "twogo_zone_b": "results/humanoid/corridor_2d/main/twogo_zone_b/level_1/seed_0/trajectory/trajectory.json",
    "twogo_zone_c": "results/humanoid/corridor_2d/main/twogo_zone_c/level_1/seed_0/trajectory/trajectory.json",
    "twogo_zone_d": "results/humanoid/corridor_2d/main/twogo_zone_d/level_1/seed_0/trajectory/trajectory.json",
}


def resolve_plan_path(plan: str | Path) -> Path:
    """Resolve a plan alias / relative path to an absolute trajectory.json."""
    s = str(plan)
    if s in PLAN_ALIASES:
        return _ROOT / PLAN_ALIASES[s]
    p = Path(plan)
    if not p.is_absolute():
        p = _ROOT / p
    return p


def _infer_corridor_scene_meta(plan_path: Path) -> Optional[dict]:
    """Try to resolve the corridor scene preset for ``plan_path``.

    The diagnose script accepts any ``trajectory.json`` under
    ``results/humanoid/corridor_2d/.../seed_X/trajectory/``. The matching
    planner config lives at ``configs/humanoid/corridor_2d/<group>/<run>.yaml``
    where ``<group>/<run>`` mirrors the path between ``corridor_2d`` and
    ``level_X``. If we can find that config and it names a ``scene_preset``,
    we resolve it via :func:`resolve_corridor_scene_preset` and return a
    dict suitable for writing to ``corridor_scene.json`` (which the deploy
    renderer reads to draw obstacles + walls in the GIF).
    """
    try:
        seed_dir = plan_path.resolve().parent.parent  # .../seed_X/trajectory/trajectory.json -> .../seed_X
        rel = seed_dir.relative_to((_ROOT / "results").resolve())
    except (ValueError, OSError):
        return None
    parts = rel.parts
    if len(parts) < 3:
        return None
    config_path = (_ROOT / "configs").joinpath(*list(parts[:-2])).with_suffix(".yaml")
    if not config_path.exists():
        return None
    try:
        import yaml
    except ImportError:
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except OSError:
        return None
    env_params = cfg.get("env_params") or {}
    obstacle_config = cfg.get("obstacle_config") or {}
    scene_preset = env_params.get("scene_preset") or obstacle_config.get("scene_preset")
    if not scene_preset:
        return None
    try:
        from genedynamics.envs.humanoid_corridor_2d import (
            corridor_scene_to_dict,
            resolve_corridor_scene_preset,
        )
    except ImportError:
        return None
    try:
        scene = resolve_corridor_scene_preset(str(scene_preset))
    except KeyError:
        return None
    blob = corridor_scene_to_dict(scene, scene_preset=str(scene_preset))
    blob["config_path"] = str(config_path)
    return blob


# ---------------------------------------------------------------------------
# Deploy-config loader
# ---------------------------------------------------------------------------


# Maps deploy-yaml field name -> argparse dest name.
# spark_pd_gains is inverted because the CLI flag is --no-spark-pd-gains.
_DEPLOY_FIELD_MAP = {
    "plan": "plan",
    "control_hz": "control_hz",
    "sim_dt": "sim_dt",
    "xy_kp": "xy_kp",
    "xy_correction_cap": "xy_correction_cap",
    "yaw_kp": "yaw_kp",
    "warmup_sec": "warmup_sec",
    "cmd_lpf_tau": "cmd_lpf_tau",
    "best_idx": "best_idx",
}


def load_deploy_config(path: Path) -> dict:
    """Parse ``configs/humanoid/corridor_2d/deploy/*.yaml`` into argparse-style
    overrides. Unknown fields are ignored. ``render:`` is left untouched —
    it's consumed by the renderer, not this script."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    overrides: dict = {}
    for yaml_key, dest_key in _DEPLOY_FIELD_MAP.items():
        if yaml_key in data and data[yaml_key] is not None:
            overrides[dest_key] = data[yaml_key]
    if "spark_pd_gains" in data and data["spark_pd_gains"] is not None:
        overrides["no_spark_pd_gains"] = not bool(data["spark_pd_gains"])
    return overrides


def _format_render_hint(deploy_yaml: dict, npz_path: Path, gif_path: Path) -> str:
    """Build the render command (render_deploy_humanoid) that regenerates the full deploy
    figure set — tracking / sport_mode / motion_strip / trajectory_mujoco.gif — next to the npz."""
    r = (deploy_yaml or {}).get("render") or {}
    parts = [
        "python scripts/visualizations/render_deploy_humanoid.py",
        f"  --npz {npz_path}",
        "  --which all",
    ]
    if "every_n" in r:
        parts.append(f"  --every-n {r['every_n']}")
    return " \\\n".join(parts)


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
    # Full executed qpos per step (n, nq_full); [:, 3:7] is the base quaternion
    # (w, x, y, z) -> the faithful executed base yaw for body-SDF evaluation.
    qpos: np.ndarray
    intent_lin_vel: np.ndarray  # (n, 2)
    intent_yaw_rate: np.ndarray  # (n,)
    rate_overruns: int
    # Planned arm/torso references (per-step, decoded from the 14D plan).
    plan_arm_torso: np.ndarray  # (n, 5): a_left, a_right, p_left, p_right, psi_torso
    # Commanded actuated-joint position vector (per-step, what the controller asked).
    cmd_joint_pos: np.ndarray  # (n, num_actuated)
    actuated_joint_names: tuple  # length num_actuated
    # A-posteriori safety certificate (Step 4): executed minimum body-SDF
    # clearance min_t g(x_t) over the rollout (elliptical torso + arm tips vs
    # the scene obstacles/walls, at the EXECUTED base pose), and whether it
    # certifies safety (min clearance >= 0  <=>  executed g(x) <= 0).
    executed_min_body_sdf: float = float("nan")
    certified_safe: bool = False

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
            "executed_min_body_sdf": float(self.executed_min_body_sdf),
            "certified_safe": bool(self.certified_safe),
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
    goal_hold_sec: float = 0.0,
    plan_speed: float = 1.0,
    fall_threshold_m: float = 0.40,
    out_dir: Optional[Path] = None,
    quiet: bool = False,
    loco_kwargs: Optional[dict] = None,
    io: Optional[Any] = None,
    loco_client: Optional[Any] = None,
    cap_vx: Optional[float] = None,
    body_frame_track: bool = True,
    yaw_kp: float = 0.0,
    xy_kp: float = 1.5,
    xy_correction_cap: float = 0.25,
    best_idx: Optional[int] = None,
    warmup_sec: float = 0.8,
    cmd_lpf_tau: float = 0.0,
    spark_pd_gains: bool = True,
    mapper_cfg: Optional[dict] = None,
    use_governor: bool = False,
    corridor_half_width: Optional[float] = None,
    governor_cfg: Optional[dict] = None,
    gov_cmd_lpf: float = 1.0,  # <1.0 => first-order LPF alpha on governor output velocity (kills cmd jerk)
    use_body_sdf_governor: bool = False,
    body_sdf_scene: Optional[dict] = None,
    body_sdf_activation_band: float = 0.30,
    body_sdf_lookahead: float = 0.30,
) -> DiagnoseResult:
    """Run the diagnostic and return the result struct.

    Args:
        loco_kwargs: Forwarded to :class:`SparkRLLocoClient.__init__` so
            callers can override ``policy_path``, ``nominal_step_period``
            or ``cmd_clip``. Ignored when ``loco_client`` is supplied.
        io: Pre-built RobotIO to drive. Defaults to a fresh
            :class:`MujocoRobotIO` (the verbatim sim path). Inject e.g.
            :class:`UnitreeG1RobotIO` to run this same governor-aware
            diagnostic on hardware / AR. The Spark MuJoCo PD-gain patch is
            auto-skipped for non-MuJoCo IOs.
        loco_client: Pre-built loco client. Defaults to
            :class:`SparkRLLocoClient`. Inject :class:`RealLocoClient` for the
            stock G1 sport-mode loco interface on hardware.
        cap_vx: Optional cap on ``intent.base_lin_vel`` magnitude. Useful
            for testing whether slowing the commanded velocity keeps the
            walker on track when the policy's training-time speed limit
            (±0.3 m/s) is below the plan's commanded speed.
        body_frame_track: When True (the default), rotate the plan's
            world-frame ``(v_x, v_y)`` into the robot's body frame each
            step before passing them to the loco client. The SparkRL
            policy expects body-frame command velocities, so without this
            rotation any small yaw drift causes monotonic lateral runaway.
        yaw_kp: Proportional gain on yaw error. Adds
            ``yaw_kp * wrap(plan.psi - actual_yaw)`` to ``base_yaw_rate``
            so heading is regulated to the plan, not just the open-loop
            ``omega`` feed-forward. Default 0 because the SparkRL policy
            clips yaw_rate to ±0.3 — any meaningful kp saturates and
            destabilises the gait.
        xy_kp: Proportional gain on planar-position error in WORLD frame.
            Closes the open-loop tracking loop: at each tick the velocity
            command is ``plan.v_xy + xy_kp * (plan.xy - actual.xy)``
            before the body-frame rotation. This is the key change that
            turns SparkRL into a position tracker instead of a velocity
            replayer.
        xy_correction_cap: Magnitude cap (m/s) on the ``xy_kp * err``
            correction term so a transient ~1 m error doesn't blow the
            command past the policy's ±0.3 m/s training distribution.
        warmup_sec: Seconds of zero-velocity commands sent before plan
            playback begins. Without it the robot sees a step from 0 to
            full v_x at t=0 and the first few frames look like a teleport
            while the policy spins up its gait clock. Logs from warmup
            steps are PRE-PENDED to the per-step arrays so plot vs. plan
            timing remains correct.
        cmd_lpf_tau: First-order low-pass filter time-constant (seconds)
            applied to the **feed-forward** plan velocity / yaw_rate only,
            BEFORE the XY P-correction is added. Smooths the noisy plan
            resampling output without lagging the closed-loop correction
            (which we tried earlier and made the robot fall — correction
            lag → error accumulates → cmd saturates → fall). 0 disables.
    """

    # ----- 1. Load the plan ------------------------------------------------
    if not quiet:
        print(f"[diagnose] loading plan: {plan_path}")
    from genedynamics.deploy.followers.common.plan_adapter import (
        CorridorTrajectoryAdapter,
    )

    # plan_speed<1 stretches the plan in time (slower playback) so the commanded
    # pace stays within the SparkRL policy's ~0.3 m/s envelope and the tracker
    # stops lagging ~0.7 m mid-corridor (the main reason twogo stalls short in
    # tight zones). Execution-only: same trajectory, just played slower; the FF
    # velocity is scaled by plan_speed below to match.
    _plan_speed = max(1e-3, float(plan_speed))
    adapter = CorridorTrajectoryAdapter(
        source_dt=0.25 / _plan_speed,
        target_dt=1.0 / control_hz,
    )
    plan = adapter.load_trajectory_json(plan_path, best_idx=best_idx)
    n_plan = plan.states.shape[0]
    if not quiet:
        print(f"[diagnose] plan loaded: {n_plan} frames @ {1.0 / control_hz:.4f}s")

    # Decode every frame once so the loop is just an array index lookup.
    schema = plan.schema
    frames = [schema.decode_state(plan.states[i], time_sec=float(plan.times[i])) for i in range(n_plan)]
    # Goal-hold: the resampled plan ENDS at the goal, but the SparkRL tracker lags
    # (warmup + tracking), so without holding the goal the rollout stops ~0.5 m
    # short. Append goal-frame copies so the robot finishes the traverse
    # (execution-only; the planner is fixed). The XY position P-control keeps
    # pulling the lagging robot toward the held goal frame until it arrives.
    if goal_hold_sec and goal_hold_sec > 0.0:
        n_hold = int(round(float(goal_hold_sec) * control_hz))
        if n_hold > 0:
            frames = frames + [frames[-1]] * n_hold
            n_plan = len(frames)
            if not quiet:
                print(f"[diagnose] goal-hold: +{n_hold} frames ({goal_hold_sec:.1f}s) holding the goal")
    plan_xy_full = np.asarray([[f.x, f.y] for f in frames], dtype=np.float64)

    # ----- 2. Build IO + controller + upper-body mapper -------------------
    # IO and loco client are injectable so the SAME governor-aware diagnostic
    # can drive a different backend (e.g. UnitreeG1RobotIO + RealLocoClient for
    # hardware / AR) without touching any governor logic below. When both are
    # left as None the defaults reproduce the verbatim MuJoCo + SparkRL sim path.
    from genedynamics.deploy.io.mujoco_io import MujocoRobotIO
    from genedynamics.deploy.controllers.sport_mode import (
        SparkRLLocoClient,
        SportModeController,
    )
    from genedynamics.deploy.followers.humanoid.upper_body_mapper import (
        HumanoidUpperBodyMapper,
        HumanoidUpperBodyMapperConfig,
    )
    from genedynamics.envs.robots.g1 import G1RobotModel

    robot = G1RobotModel()
    if io is None:
        io = MujocoRobotIO(sim_dt=sim_dt)
    if loco_client is None:
        loco_client = SparkRLLocoClient(robot=robot, **(loco_kwargs or {}))
    controller = SportModeController(io=io, loco_client=loco_client)
    # mapper_cfg lets the deploy tune arm tucking (execution-only) so the wrists
    # don't swing into side-wall obstacles — the planner is fixed.
    _mcfg = HumanoidUpperBodyMapperConfig(**mapper_cfg) if mapper_cfg else None
    upper_body_mapper = HumanoidUpperBodyMapper(_mcfg)
    if not quiet:
        print(
            f"[diagnose] building {type(io).__name__} + SportModeController + "
            f"{type(loco_client).__name__} + HumanoidUpperBodyMapper"
        )

    # Optional: override PD gains to match spark's MuJoCo training distribution.
    # The G1 XML uses <position kp="500" dampratio="1"> uniformly across all
    # actuators. Spark trained at hip=100/kd=2, knee=150/kd=4, ankle=40/kd=2 —
    # a 5-12× stiffness mismatch that makes the deploy interpret policy
    # joint-targets at much higher gain than training. We patch the live
    # MjModel actuator params (gainprm / biasprm) since SportModeController's
    # cmd.kp/kd is silently ignored by MujocoRobotIO (it just writes joint_pos
    # to data.ctrl and lets the XML actuators apply their own gains).
    if spark_pd_gains and isinstance(io, MujocoRobotIO):
        gains_by_role = {
            "hip": (100.0, 2.0),
            "knee": (150.0, 4.0),
            "ankle": (40.0, 2.0),
        }
        n_overridden = 0
        for name, idx in controller._joint_index.items():
            role = next((r for r in gains_by_role if r in name), None)
            if role is None:
                continue
            kp, kd = gains_by_role[role]
            # MuJoCo position actuator: gainprm[0] = kp, biasprm[1] = -kp,
            # biasprm[2] = -kd (mujoco docs).
            io.model.actuator_gainprm[idx, 0] = kp
            io.model.actuator_biasprm[idx, 1] = -kp
            io.model.actuator_biasprm[idx, 2] = -kd
            n_overridden += 1
        if not quiet:
            print(f"[diagnose] patched MjModel actuator_gainprm for "
                  f"{n_overridden} leg joints (hip 100/2, knee 150/4, ankle 40/2)")

    # Anchor the robot to the plan's first XY so the diagnostic actually
    # tracks rather than diverging from the moment t=0. Sim IOs can teleport
    # (reset_to); real hardware can't — it just reads its current (localized)
    # pose, so the robot must be physically placed at the plan start.
    f0 = frames[0]
    if hasattr(io, "reset_to"):
        state = io.reset_to(
            base_xyz=(f0.x, f0.y, f0.h),
            base_quat_wxyz=(np.cos(f0.psi / 2), 0.0, 0.0, np.sin(f0.psi / 2)),
        )
    else:
        state = io.reset()
    controller.reset(io)

    # Optional reference governor (off by default -> existing path unchanged).
    # Acts as a per-step admissibility / rate / posture limiter ON TOP of the
    # xy_kp position feedback below (spark needs that feedback; the governor
    # adds principled rate limiting, a pelvis-roll posture margin, and the
    # m_track safety tightening). See genedynamics/deploy/followers/governor/.
    governor = None
    body_sdf_set = None  # set when use_body_sdf_governor (obstacle-aware safety)
    if use_governor:
        from genedynamics.deploy.followers.governor import (
            ContinuousBaseGovernor,
            CorridorAdmissibleSet,
            GovernorConfig,
        )
        _gcfg = GovernorConfig(dt=1.0 / control_hz, **(governor_cfg or {}))
        if use_body_sdf_governor:
            # Obstacle-aware governor: build a BodySdfAdmissibleSet from the
            # run's corridor scene (elliptical torso + arm tips, height-gated
            # against the actual obstacles). Reuses the planner body SDF.
            from genedynamics.deploy.followers.governor import (
                BodyConfig,
                BodySdfAdmissibleSet,
            )
            _scene = body_sdf_scene
            if _scene is None:
                _scene = _infer_corridor_scene_meta(plan_path) or {}
            body_sdf_set = BodySdfAdmissibleSet.from_scene_dict(
                _scene,
                body=BodyConfig(),
                activation_band=float(body_sdf_activation_band),
                lookahead=float(body_sdf_lookahead),
            )
            governor = ContinuousBaseGovernor(body_sdf_set, _gcfg)
            if not quiet:
                n_obs = len(_scene.get("obstacles", []) or [])
                print(f"[diagnose] body-SDF governor: {n_obs} obstacles, "
                      f"corridor_width={_scene.get('corridor_width')}, "
                      f"activation_band={body_sdf_activation_band}")
        else:
            governor = ContinuousBaseGovernor(CorridorAdmissibleSet(), _gcfg)
        governor.reset(state)

    # ----- 3. Replay loop -------------------------------------------------
    from genedynamics.deploy.interfaces.messages import Intent
    from genedynamics.deploy.runtime.rate_limiter import RateLimiter

    # First-order LPF state for the FEED-FORWARD plan velocity / yaw_rate.
    # We filter only the FF (plan output) so the XY P-correction stays
    # reactive — earlier we tried filtering the full cmd and the resulting
    # correction lag caused the policy to fall.
    dt_ctrl = 1.0 / control_hz
    lpf_alpha = float(dt_ctrl) / (float(cmd_lpf_tau) + float(dt_ctrl)) if cmd_lpf_tau > 0 else 1.0
    vxy_ff_filt = np.zeros(2, dtype=np.float64)
    omega_ff_filt = 0.0
    _gov_v_prev = None  # LPF state for governor output-velocity smoothing (gov_cmd_lpf)

    # Silent warmup. Ramps the cmd velocity from 0 to plan[0].v over
    # ``warmup_sec`` so the policy enters the gait smoothly. Earlier we tried
    # zero-cmd warmup and the policy stepped backwards ~0.5 m (zero is a
    # singular point in its training distribution); a linear ramp keeps
    # the cmd inside the trained regime the whole way.
    n_warmup = max(0, int(round(float(warmup_sec) * control_hz)))
    if n_warmup > 0:
        if not quiet:
            print(
                f"[diagnose] warming up {n_warmup} steps ({warmup_sec:.2f} s) "
                f"with cmd ramp 0 → ({f0.v_x:+.3f}, {f0.v_y:+.3f}) m/s"
            )
        upper0 = upper_body_mapper.map(f0)
        target_vx = float(f0.v_x)
        target_vy = float(f0.v_y)
        target_yaw_rate = float(f0.omega)
        for i in range(n_warmup):
            alpha = float(i + 1) / float(n_warmup)  # 1/n .. 1.0
            raw_vxy = np.array([alpha * target_vx, alpha * target_vy], dtype=np.float64)
            raw_omega = alpha * target_yaw_rate
            vxy_ff_filt = lpf_alpha * raw_vxy + (1.0 - lpf_alpha) * vxy_ff_filt
            omega_ff_filt = lpf_alpha * raw_omega + (1.0 - lpf_alpha) * omega_ff_filt
            warm_intent = Intent(
                t=0.0,
                base_yaw=float(f0.psi),
                base_height=float(f0.h),
                base_lin_vel=vxy_ff_filt.copy(),
                base_yaw_rate=float(omega_ff_filt),
                torso_yaw=float(f0.psi_torso),
                extras={
                    "dt": 1.0 / control_hz,
                    "upper_body_targets": upper0,
                },
            )
            cmd = controller.act(state, warm_intent)
            io.send_control(cmd)
            state = io.step(1.0 / control_hz)

    # Fix B: after warmup the robot has drifted slightly off (f0.x, f0.y) —
    # PPO's "stand" isn't perfectly stationary. If we feed the un-shifted
    # plan straight into XY P-control, the very first step sees a non-zero
    # xy_err which adds correction velocity that, combined with plan v,
    # saturates cmd_clip and destabilises the gait. Shift the entire plan
    # reference by the warmup drift so the first xy_err is zero.
    plan_xy_offset = np.zeros(2, dtype=np.float64)
    if n_warmup > 0 and state.qpos is not None and state.qpos.size >= 2:
        plan_xy_offset = (
            np.asarray(state.qpos[:2], dtype=np.float64)
            - np.array([float(f0.x), float(f0.y)], dtype=np.float64)
        )
        if not quiet:
            print(
                f"[diagnose] plan shifted by warmup drift "
                f"({plan_xy_offset[0]:+.4f}, {plan_xy_offset[1]:+.4f}) m"
            )

    # Hand the body-SDF governor the planner's (collision-free) lateral lane,
    # offset-corrected to the executed world frame, so its safety half-space can
    # steer toward the globally-consistent safe side instead of a myopic local
    # gradient. Additive: only when the obstacle-aware governor is active.
    if body_sdf_set is not None:
        body_sdf_set.set_plan_lane(plan_xy_full + plan_xy_offset)

    rate = RateLimiter(hz=control_hz)
    n_steps = min(n_plan, int(max_steps) if max_steps else n_plan)

    pelvis_xy_log = np.zeros((n_steps, 2), dtype=np.float64)
    pelvis_z_log = np.zeros(n_steps, dtype=np.float64)
    intent_lin_vel_log = np.zeros((n_steps, 2), dtype=np.float64)
    intent_yaw_rate_log = np.zeros(n_steps, dtype=np.float64)
    # Full executed state history (qpos_log feeds the body-SDF certificate +
    # the MuJoCo renderer). Sim exposes io.model.nq/nv; real hardware has no
    # MuJoCo model, so size from the spec: qpos = 7 (floating base) + actuated.
    if hasattr(io, "model"):
        nq_full = int(io.model.nq)
        nv_full = int(io.model.nv)
    else:
        nq_full = 7 + int(io.spec.num_actuated)
        nv_full = 6 + int(io.spec.num_actuated)
    nu_full = int(io.spec.num_actuated)
    qpos_log = np.zeros((n_steps, nq_full), dtype=np.float64)
    qvel_log = np.zeros((n_steps, nv_full), dtype=np.float64)
    # Planned arm-shrink + torso references straight off the 14D plan.
    plan_arm_torso_log = np.zeros((n_steps, 5), dtype=np.float64)
    # What the SportModeController actually commanded each tick (29-D joint pos).
    cmd_joint_pos_log = np.zeros((n_steps, nu_full), dtype=np.float64)

    wall_t0 = time.monotonic()
    for k in range(n_steps):
        rate.tick()
        f = frames[k]
        # FF velocity from plan, filtered (plan resampling causes ripple).
        # Scale by plan_speed: a slower playback needs proportionally lower FF vel.
        vxy_world_ff_raw = np.array([f.v_x, f.v_y], dtype=np.float64) * _plan_speed
        vxy_ff_filt = lpf_alpha * vxy_world_ff_raw + (1.0 - lpf_alpha) * vxy_ff_filt
        vxy_world_ff = vxy_ff_filt.copy()
        omega_ff_filt = lpf_alpha * float(f.omega) * _plan_speed + (1.0 - lpf_alpha) * omega_ff_filt

        # XY P-control on planar position. Open-loop velocity replay is
        # what makes SparkRL drift off-plan; adding this closes the loop.
        actual_xy = (
            np.asarray(state.qpos[:2], dtype=np.float64)
            if state.qpos is not None and state.qpos.size >= 2
            else np.zeros(2, dtype=np.float64)
        )
        plan_xy = np.array([float(f.x), float(f.y)], dtype=np.float64) + plan_xy_offset
        xy_err = plan_xy - actual_xy
        xy_correction = float(xy_kp) * xy_err
        # Cap the correction term so a transient ~1 m error doesn't push
        # the velocity command past the policy's ±0.3 m/s training range.
        corr_norm = float(np.linalg.norm(xy_correction))
        if corr_norm > xy_correction_cap:
            xy_correction = xy_correction * (xy_correction_cap / max(corr_norm, 1e-9))
        vxy_world = vxy_world_ff + xy_correction
        if governor is not None:
            from genedynamics.deploy.followers.governor import CorridorContext
            _gyaw = 0.0
            if state.qpos is not None and state.qpos.size >= 7:
                _q = np.asarray(state.qpos[3:7], dtype=np.float64)
                _gyaw = float(np.arctan2(
                    2.0 * (_q[0] * _q[3] + _q[1] * _q[2]),
                    1.0 - 2.0 * (_q[2] * _q[2] + _q[3] * _q[3]),
                ))
            governor.set_anchor(float(actual_xy[0]), float(actual_xy[1]), _gyaw)
            if body_sdf_set is not None:
                # Obstacle-aware: refresh the per-step body config from the plan
                # frame (torso yaw + arm tuck + height) so the SDF reflects the
                # follower's commanded posture, then govern in the world frame.
                from genedynamics.deploy.followers.governor import (
                    BodyConfig,
                    BodySdfContext,
                )
                _body = BodyConfig(
                    h=float(f.h),
                    psi_torso=float(f.psi_torso),
                    a_left=float(f.a_left),
                    a_right=float(f.a_right),
                )
                _ctx = BodySdfContext(
                    obstacles=body_sdf_set._ctx.obstacles,
                    wall_y_min=body_sdf_set._ctx.wall_y_min,
                    wall_y_max=body_sdf_set._ctx.wall_y_max,
                    body=_body,
                    plan_x=body_sdf_set._ctx.plan_x,
                    plan_y=body_sdf_set._ctx.plan_y,
                )
            else:
                _ctx = CorridorContext(
                    center=plan_xy, tangent_yaw=float(f.psi), half_width=corridor_half_width,
                )
            _ref = Intent(
                t=float(f.time_sec), base_yaw=float(f.psi), base_height=float(f.h),
                base_lin_vel=vxy_world, base_yaw_rate=omega_ff_filt,
                base_pos_xy=actual_xy + vxy_world / float(control_hz),
            )
            governor.govern(_ref, state, _ctx)
            _gp = governor.governed_pose
            if _gp is not None:
                vxy_world = (_gp[:2] - actual_xy) * float(control_hz)
                # Command smoothing: first-order LPF on the governor's output
                # velocity to suppress the per-step jerk introduced by the
                # intermittent body-SDF correction + measured-pose re-anchor.
                if gov_cmd_lpf < 1.0:
                    if _gov_v_prev is None:
                        _gov_v_prev = vxy_world.copy()
                    vxy_world = gov_cmd_lpf * vxy_world + (1.0 - gov_cmd_lpf) * _gov_v_prev
                    _gov_v_prev = vxy_world.copy()
        if cap_vx is not None:
            speed = float(np.linalg.norm(vxy_world))
            if speed > cap_vx:
                vxy_world = vxy_world * (cap_vx / max(speed, 1e-9))

        # Body-frame conversion + yaw P-correction.
        # SparkRL's PPO policy treats LocoCommand.{vx,vy} as body-frame
        # velocities. The 14D plan stores world-frame (v_x, v_y), so any
        # actual yaw drift makes the deploy walk crooked. Rotate
        # world→body each tick using the live qpos quaternion, and add a
        # P-term on yaw error to regulate heading to plan.psi.
        actual_yaw = 0.0
        if body_frame_track and state.qpos is not None and state.qpos.size >= 7:
            quat = np.asarray(state.qpos[3:7], dtype=np.float64)
            w, x, y, z = quat
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            actual_yaw = float(np.arctan2(siny_cosp, cosy_cosp))
            c, s = np.cos(-actual_yaw), np.sin(-actual_yaw)
            vxy_cmd = np.array([
                c * vxy_world[0] - s * vxy_world[1],
                s * vxy_world[0] + c * vxy_world[1],
            ], dtype=np.float64)
        else:
            vxy_cmd = vxy_world

        yaw_err = ((float(f.psi) - actual_yaw + np.pi) % (2.0 * np.pi)) - np.pi
        # Use FF-filtered omega + reactive yaw correction (correction unfiltered).
        yaw_rate_cmd = omega_ff_filt + float(yaw_kp) * yaw_err

        # Decode the 14D plan frame's torso/arm fields into 17 upper-body
        # joint targets (waist x3 + arms x14). The SportModeController will
        # merge these with the leg targets coming from the loco client.
        upper_targets = upper_body_mapper.map(f)
        intent = Intent(
            t=float(f.time_sec),
            base_yaw=float(f.psi),
            base_height=float(f.h),
            base_lin_vel=vxy_cmd,
            base_yaw_rate=yaw_rate_cmd,
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
        plan_arm_torso_log[k] = (f.a_left, f.a_right, f.p_left, f.p_right, f.psi_torso)
        if cmd.joint_pos is not None and cmd.joint_pos.size == nu_full:
            cmd_joint_pos_log[k] = cmd.joint_pos

        if pelvis_z_log[k] < fall_threshold_m:
            n_steps = k + 1  # truncate logs
            pelvis_xy_log = pelvis_xy_log[:n_steps]
            pelvis_z_log = pelvis_z_log[:n_steps]
            qpos_log = qpos_log[:n_steps]
            qvel_log = qvel_log[:n_steps]
            intent_lin_vel_log = intent_lin_vel_log[:n_steps]
            intent_yaw_rate_log = intent_yaw_rate_log[:n_steps]
            plan_arm_torso_log = plan_arm_torso_log[:n_steps]
            cmd_joint_pos_log = cmd_joint_pos_log[:n_steps]
            break
    wall_t1 = time.monotonic()

    # ----- 4. Metrics ------------------------------------------------------
    # Apply the warmup drift offset to the metric reference so endpoint /
    # lateral_offset are computed against the same shifted plan the
    # XY P-control is tracking.
    plan_xy_full = plan_xy_full + plan_xy_offset
    plan_xy_actual = plan_xy_full[:n_steps]
    final_plan = plan_xy_full[n_steps - 1] if n_steps > 0 else plan_xy_full[-1]
    final_actual = pelvis_xy_log[-1] if n_steps > 0 else np.zeros(2)
    endpoint_d = float(np.linalg.norm(final_actual - final_plan))

    laterals = _lateral_offsets(pelvis_xy_log, plan_xy_full)
    fell = bool(pelvis_z_log.min() < fall_threshold_m) if n_steps > 0 else True

    # ----- 4b. A-posteriori safety certificate (Step 4) -------------------
    # Evaluate the executed minimum body-SDF clearance g(x) over the rollout,
    # using the EXECUTED base pose (qpos) + per-step commanded posture
    # (plan_arm_torso, executed pelvis height). This is the operative safety
    # guarantee — certified iff min_t g(x_t) >= 0 (body never penetrates an
    # obstacle/wall). Computed independently of the governor so the baseline
    # and the governed run are certified on the same footing. Best-effort:
    # a missing scene / import never breaks the rollout.
    executed_body_sdf = np.full(max(n_steps, 1), np.nan, dtype=np.float64)
    executed_min_body_sdf = float("nan")
    certified_safe = False
    try:
        from genedynamics.deploy.followers.governor import (  # noqa: PLC0415
            BodyConfig as _CertBodyConfig,
            BodySdfAdmissibleSet as _CertBodySdfSet,
        )
        from genedynamics.deploy.followers.governor.reference_selector import (  # noqa: PLC0415
            certify_safety as _certify_safety,
        )
        _cert_scene = body_sdf_scene if body_sdf_scene is not None else (_infer_corridor_scene_meta(plan_path) or {})
        _has_scene = bool(_cert_scene.get("obstacles")) or (_cert_scene.get("corridor_width") is not None)
        if n_steps > 0 and _has_scene:
            _cert_set = _CertBodySdfSet.from_scene_dict(_cert_scene, body=_CertBodyConfig())
            _cert_ctx = _cert_set._ctx
            for _i in range(n_steps):
                _x = float(qpos_log[_i, 0])
                _y = float(qpos_log[_i, 1])
                _q = qpos_log[_i, 3:7]
                _yaw = float(np.arctan2(
                    2.0 * (_q[0] * _q[3] + _q[1] * _q[2]),
                    1.0 - 2.0 * (_q[2] * _q[2] + _q[3] * _q[3]),
                ))
                _aL, _aR, _pL, _pR, _psiT = plan_arm_torso_log[_i]
                _cert_ctx.body = _CertBodyConfig(
                    h=float(pelvis_z_log[_i]), psi_torso=float(_psiT),
                    a_left=float(_aL), a_right=float(_aR),
                )
                executed_body_sdf[_i] = _cert_set.body_min_sdf(_x, _y, _yaw, _cert_ctx)
            executed_min_body_sdf = float(np.nanmin(executed_body_sdf[:n_steps]))
            certified_safe = bool(_certify_safety(executed_min_body_sdf))
    except Exception as _cert_exc:  # noqa: BLE001 — certificate is best-effort
        if not quiet:
            print(f"[diagnose] body-SDF certificate unavailable: {_cert_exc}")

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
        qpos=qpos_log,
        intent_lin_vel=intent_lin_vel_log,
        intent_yaw_rate=intent_yaw_rate_log,
        rate_overruns=int(rate.stats.overruns),
        plan_arm_torso=plan_arm_torso_log,
        cmd_joint_pos=cmd_joint_pos_log,
        actuated_joint_names=tuple(io.spec.actuated_joints),
        executed_min_body_sdf=executed_min_body_sdf,
        certified_safe=certified_safe,
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / "sport_mode.npz"
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
            plan_arm_torso=plan_arm_torso_log,
            cmd_joint_pos=cmd_joint_pos_log,
            actuated_joint_names=np.asarray(result.actuated_joint_names, dtype=object),
            executed_body_sdf=executed_body_sdf[:n_steps],
        )
        (out_dir / "sport_mode.json").write_text(
            json.dumps(result.to_json(), indent=2)
        )
        # Persist the corridor scene so the deploy renderer can draw obstacles.
        scene_meta = _infer_corridor_scene_meta(plan_path)
        if scene_meta is not None:
            (out_dir / "corridor_scene.json").write_text(
                json.dumps(scene_meta, indent=2, default=str)
            )
            if not quiet:
                print(
                    f"[diagnose] wrote corridor_scene.json "
                    f"(preset={scene_meta.get('scene_preset', '?')})"
                )
        elif not quiet:
            print("[diagnose] no scene_preset found for plan; obstacles not rendered")
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
    if not np.isnan(result.executed_min_body_sdf):
        _cert = "CERTIFIED  g(x)<=0" if result.certified_safe else "VIOLATED — body penetrates"
        print(f"  Exec min body-SDF:      {result.executed_min_body_sdf:+.3f} m  →  {_cert}")
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
    """Draw a 2x2 figure: xy trace, arm tuck (commanded vs achieved),
    arm posture, and torso yaw."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[diagnose] matplotlib not installed; skipping plot")
        return

    n = result.n_steps
    t = np.arange(n) / 50.0  # control_hz default; only used for x-axis labelling

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Top-left: xy trace.
    ax = axes[0, 0]
    ax.plot(result.plan_xy[:, 0], result.plan_xy[:, 1], "k--", lw=2, label="plan")
    ax.plot(result.pelvis_xy[:, 0], result.pelvis_xy[:, 1], "C0-", lw=1.5, label="pelvis")
    ax.scatter(result.final_plan_xy[0], result.final_plan_xy[1], s=80, c="k", marker="x", label="plan end")
    ax.scatter(result.final_pelvis_xy[0], result.final_pelvis_xy[1], s=80, c="C0", marker="o", label="pelvis end")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("XY trace")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: arm tuck reference (a_left/right ∈ [0,1]) and the commanded
    # elbow joint (the mapper's primary tuck channel) for both arms.
    ax = axes[0, 1]
    a_left = result.plan_arm_torso[:n, 0]
    a_right = result.plan_arm_torso[:n, 1]
    ax.plot(t, a_left, "C0--", lw=1.2, label="plan a_left")
    ax.plot(t, a_right, "C3--", lw=1.2, label="plan a_right")
    le_idx = result.actuated_joint_names.index("left_elbow_joint") if "left_elbow_joint" in result.actuated_joint_names else None
    re_idx = result.actuated_joint_names.index("right_elbow_joint") if "right_elbow_joint" in result.actuated_joint_names else None
    if le_idx is not None:
        ax.plot(t, result.cmd_joint_pos[:n, le_idx], "C0-", lw=1.2, label="cmd L elbow (rad)")
    if re_idx is not None:
        ax.plot(t, result.cmd_joint_pos[:n, re_idx], "C3-", lw=1.2, label="cmd R elbow (rad)")
    ax.set_xlabel("t (s)")
    ax.set_title("Arm tuck (plan ratio vs commanded elbow)")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8)

    # Bottom-left: shoulder pitch as proxy for arm posture.
    ax = axes[1, 0]
    p_left = result.plan_arm_torso[:n, 2]
    p_right = result.plan_arm_torso[:n, 3]
    ax.plot(t, p_left, "C0--", lw=1.2, label="plan p_left")
    ax.plot(t, p_right, "C3--", lw=1.2, label="plan p_right")
    lsp = "left_shoulder_pitch_joint"
    rsp = "right_shoulder_pitch_joint"
    if lsp in result.actuated_joint_names:
        ax.plot(t, result.cmd_joint_pos[:n, result.actuated_joint_names.index(lsp)], "C0-", lw=1.2, label="cmd L shoulder pitch")
    if rsp in result.actuated_joint_names:
        ax.plot(t, result.cmd_joint_pos[:n, result.actuated_joint_names.index(rsp)], "C3-", lw=1.2, label="cmd R shoulder pitch")
    ax.set_xlabel("t (s)")
    ax.set_title("Arm posture (plan p_* vs commanded shoulder pitch)")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8)

    # Bottom-right: torso yaw plan vs commanded waist_yaw.
    ax = axes[1, 1]
    psi_torso = result.plan_arm_torso[:n, 4]
    ax.plot(t, psi_torso, "k--", lw=1.5, label="plan psi_torso")
    wyj = "waist_yaw_joint"
    if wyj in result.actuated_joint_names:
        ax.plot(t, result.cmd_joint_pos[:n, result.actuated_joint_names.index(wyj)], "C2-", lw=1.5, label="cmd waist_yaw")
    ax.set_xlabel("t (s)"); ax.set_ylabel("rad")
    ax.set_title("Torso yaw (plan vs commanded)")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"sport-mode diagnose — {Path(result.plan_path).parent.parent.name}  "
        f"(endpoint err {result.endpoint_distance_m:.2f} m, max lat {result.max_lateral_offset_m:.2f} m)",
        fontsize=11,
    )
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
        "--deploy-config",
        type=Path,
        default=None,
        help=(
            "Optional path to a configs/humanoid/corridor_2d/deploy/*.yaml. "
            "Values from the yaml are applied as defaults; CLI flags still "
            "override them. The yaml's ``render:`` block is printed at the "
            "end as a ready-to-paste render command."
        ),
    )
    p.add_argument(
        "--plan",
        type=str,
        default=DEFAULT_PLAN,
        help=(
            "Path to a planner trajectory.json, or a short alias "
            f"({', '.join(sorted(PLAN_ALIASES))}). Default: {DEFAULT_PLAN}"
        ),
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
        default=_ROOT / "results" / "deploy" / "sport_mode",
        help="Where to dump the .npz / .json artifacts",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Save a pelvis-vs-plan PNG to <out-dir>/sport_mode.png",
    )
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--no-body-frame",
        action="store_true",
        help=(
            "Disable plan→body-frame velocity rotation. SparkRL expects "
            "body-frame; without rotation any yaw drift makes the robot "
            "walk crooked. Default keeps the rotation on."
        ),
    )
    p.add_argument(
        "--yaw-kp",
        type=float,
        default=0.0,
        help="P-gain on yaw error added to base_yaw_rate. Default 0 because "
             "SparkRL clips yaw_rate to ±0.3 — any meaningful kp saturates.",
    )
    p.add_argument(
        "--xy-kp",
        type=float,
        default=1.5,
        help="P-gain on planar position error in WORLD frame, added to the "
             "velocity command before the body-frame rotation. This is what "
             "turns SparkRL from open-loop velocity replay into a position "
             "tracker. 0 disables.",
    )
    p.add_argument(
        "--xy-correction-cap",
        type=float,
        default=0.25,
        help="Magnitude cap (m/s) on the xy_kp*err term so a transient ~1 m "
             "error doesn't push the cmd outside the policy's training range.",
    )
    p.add_argument(
        "--best-idx",
        type=int,
        default=None,
        help="Override which candidate trajectory to follow (multimodal "
             "planners output ~8 modes; default uses the planner's pick).",
    )
    p.add_argument(
        "--warmup-sec",
        type=float,
        default=0.8,
        help="Seconds of zero-velocity commands sent BEFORE plan playback "
             "starts so the policy locks into its standing gait. Eliminates "
             "the apparent 'teleport' in the first few rendered frames.",
    )
    p.add_argument(
        "--no-spark-pd-gains",
        action="store_true",
        help="Skip the runtime patch of MjModel.actuator_gainprm to spark's "
             "MuJoCo training PD gains (hip 100/2, knee 150/4, ankle 40/2). "
             "Without this flag the patch is applied (default behaviour) — "
             "verified to convert medium from FALL → no-fall.",
    )
    p.add_argument(
        "--cmd-lpf-tau",
        type=float,
        default=0.0,
        help="Time-constant (s) of the 1st-order LPF on the FEED-FORWARD "
             "plan velocity / yaw_rate (XY correction is NOT filtered). "
             "Default 0 (off) — empirically anything above 0.05 s makes the "
             "plan signal too laggy and tracking worsens; the visible wobble "
             "comes from PPO gait, not cmd noise.",
    )
    # Two-pass parse so yaml values become defaults that CLI flags can still
    # override. First pass peeks at --deploy-config; second pass uses
    # ``set_defaults`` to layer the yaml under the CLI.
    boot, _ = p.parse_known_args(argv)
    deploy_yaml: dict = {}
    if boot.deploy_config is not None:
        if not boot.deploy_config.exists():
            print(f"ERROR: --deploy-config not found: {boot.deploy_config}", file=sys.stderr)
            return 2
        try:
            import yaml
            with open(boot.deploy_config, "r", encoding="utf-8") as f:
                deploy_yaml = yaml.safe_load(f) or {}
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: failed to read deploy-config: {exc}", file=sys.stderr)
            return 2
        p.set_defaults(**load_deploy_config(boot.deploy_config))

    args = p.parse_args(argv)

    plan_path = resolve_plan_path(args.plan)
    if not plan_path.exists():
        print(f"ERROR: plan file not found: {plan_path}", file=sys.stderr)
        return 2

    try:
        result = diagnose(
            plan_path,
            control_hz=args.control_hz,
            sim_dt=args.sim_dt,
            max_steps=args.max_steps,
            out_dir=args.out_dir,
            quiet=args.quiet,
            body_frame_track=not args.no_body_frame,
            yaw_kp=float(args.yaw_kp),
            xy_kp=float(args.xy_kp),
            xy_correction_cap=float(args.xy_correction_cap),
            best_idx=args.best_idx,
            warmup_sec=float(args.warmup_sec),
            cmd_lpf_tau=float(args.cmd_lpf_tau),
            spark_pd_gains=not bool(args.no_spark_pd_gains),
        )
    except ImportError as exc:
        print(f"ERROR: missing dependency: {exc}", file=sys.stderr)
        print(
            "       Run inside the genedynamics/dev-cpu Docker image:\n"
            "         docker run --rm -it -v $PWD:/work -w /work \\\n"
            "             genedynamics/dev-cpu \\\n"
            "             python scripts/tasks/robot/humanoid/sport_mode_corridor.py",
            file=sys.stderr,
        )
        return 3

    print_report(result)
    if args.plot:
        make_plot(result, args.out_dir / "sport_mode.png")
    if deploy_yaml.get("render"):
        npz = Path(args.out_dir) / "sport_mode.npz"
        gif = Path(args.out_dir) / "spark_rl_final_03x.gif"
        print("\n[deploy-config] render command (yaml render: block):")
        print(_format_render_hint(deploy_yaml, npz, gif))
    return 0 if not result.fell_over else 1


if __name__ == "__main__":
    sys.exit(main())
