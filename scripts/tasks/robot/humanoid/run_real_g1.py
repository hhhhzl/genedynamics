#!/usr/bin/env python
"""Run a certified corridor plan on the REAL Unitree G1 (M5 entry point).

Reuses the validated governor pipeline (``sport_mode_corridor.diagnose``) via the
M0.5 IO-injection point — the governor logic is untouched. Swaps in:
  * ``UnitreeG1RobotIO`` (real SDK) instead of MujocoRobotIO,
  * ``RealLocoClient`` (Unitree stock sport-mode) instead of SparkRLLocoClient,
  * ``SceneFrameLocalization(vicon, T_world_scene)`` so the robot's tracked WORLD
    pose is reported in the plan's SCENE frame,
  * the AR obstacle scene as ``body_sdf_scene`` (governor avoids them at runtime).

SDK imports are lazy → this file imports and ``--dry-run``s WITHOUT the robot or
``unitree_sdk2py``. A real run needs the G1, the SDK, and a running Vicon writer.

    # dry run (no hardware): validate wiring + frame glue + plan knobs
    python scripts/tasks/robot/humanoid/run_real_g1.py --plan <trajectory.json> --dry-run
    # real run (on the robot machine, Vicon up, robot at the plan start):
    python scripts/tasks/robot/humanoid/run_real_g1.py --plan <trajectory.json> \
        --preset zone_d --t-world-scene 1.2 0.4 0.0 --network-interface eth0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Governor knob defaults (mirror run_sport_mode_zones).
GOVERNOR_CFG = {"v_max_lat": 0.55, "yaw_rate_max": 0.65}
GOV_CMD_LPF = 0.3
GOV_ACTIVATION_BAND = 0.08
GOV_LOOKAHEAD = 0.15
M_TRACK_FALLBACK = 0.08  # the swept zone_d optimum; calibration overrides via the plan


def _plan_field(plan_path: Path, key: str, default: float) -> float:
    try:
        return float(json.loads(Path(plan_path).read_text()).get(key, default))
    except Exception:
        return float(default)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", required=True, help="Calibrated trajectory.json (from replan_from_scene).")
    p.add_argument("--preset", default="zone_d", help="Corridor preset for the governor body_sdf_scene.")
    p.add_argument("--scene-file", default=None, help="AR scene contract JSON (overrides --preset for obstacles).")
    p.add_argument("--t-world-scene", type=float, nargs=3, metavar=("X", "Y", "YAW"), default=[0.0, 0.0, 0.0])
    p.add_argument("--localization", default="vicon", choices=["vicon", "ros2", "mock"])
    p.add_argument("--network-interface", default="eth0")
    p.add_argument("--domain-id", type=int, default=0)
    p.add_argument("--control-hz", type=float, default=50.0)
    p.add_argument("--loco-rate-hz", type=float, default=10.0,
                   help="Max rate to republish the velocity command to the loco RPC. The call is "
                        "BLOCKING; republishing every control step floods the channel and stalls the "
                        "loop (the walk-3s/stop-5s symptom). 10 Hz + the 1.5s command duration is smooth.")
    p.add_argument("--max-steps", type=int, default=None)
    # Validated execution config (matches run_sport_mode_zones defaults): time-scale
    # the plan so the tracker keeps up, and hold the goal so it finishes the traverse.
    p.add_argument("--plan-speed", type=float, default=0.5,
                   help="plan playback speed (<1 slows it so the tracker keeps up; sim-validated 0.5). Start SMALLER on first real runs.")
    p.add_argument("--goal-hold-sec", type=float, default=5.0,
                   help="seconds to hold the goal after the plan ends so the lagging robot finishes (sim-validated 5.0)")
    p.add_argument("--out-dir", default="results/g1_corridor/real")
    p.add_argument("--dry-run", action="store_true", help="Assemble + validate without the SDK/robot.")
    args = p.parse_args(argv)

    plan = Path(args.plan)
    t_world_scene = {"x": args.t_world_scene[0], "y": args.t_world_scene[1], "yaw": args.t_world_scene[2]}
    m_track = _plan_field(plan, "m_track", M_TRACK_FALLBACK)
    lookahead = _plan_field(plan, "gov_lookahead", GOV_LOOKAHEAD)

    # ----- body_sdf_scene (the AR obstacles the governor avoids) --------------
    from genedynamics.deploy.ar.scene_source import SceneSource
    from genedynamics.envs.domains.humanoid.corridor import corridor_scene_to_dict
    if args.scene_file:
        src = SceneSource.from_file(args.scene_file, T_world_scene=t_world_scene)
    else:
        src = SceneSource.from_preset(args.preset, T_world_scene=t_world_scene)
    body_sdf_scene = corridor_scene_to_dict(src.scene, scene_preset=src.scene_preset)

    # ----- scene-frame localization (world→scene via T_world_scene) ----------
    from genedynamics.deploy.localization.scene_frame_plugin import SceneFrameLocalization

    def _build_inner_localization():
        if args.localization == "vicon":
            from genedynamics.deploy.localization.vicon_shm_plugin import ViconShmPlugin
            return ViconShmPlugin({})
        if args.localization == "ros2":
            from genedynamics.deploy.localization.ros2_odometry_plugin import ROS2OdometryPlugin
            return ROS2OdometryPlugin({"odom_topic": "/odom"})
        from scripts.tasks.robot.humanoid.run_twin_server import _FixedPoseLocalization
        # z = G1 nominal standing pelvis height: mock has no real base height, and the
        # fall detector reads pelvis z — z=0 would false-trigger "fell over" on a robot
        # that is actually standing (sport MainMode). Real localization (vicon) supplies
        # the true height.
        return _FixedPoseLocalization(src.scene and [*src.scene.start_pos, 0.793, 1.0, 0.0, 0.0, 0.0])

    if args.dry_run:
        import numpy as np
        # Validate the frame glue with a fixed pose at the plan start.
        from scripts.tasks.robot.humanoid.run_twin_server import _FixedPoseLocalization
        Xs, Ys, _ = src.T_world_scene.pose_to_world(float(src.scene.start_pos[0]), float(src.scene.start_pos[1]), 0.0)
        inner = _FixedPoseLocalization([Xs, Ys, 0.0, 1.0, 0.0, 0.0, 0.0])
        loc = SceneFrameLocalization(inner, t_world_scene)
        pose, _twist = loc.get_state()
        print("[run_real_g1] DRY RUN — no SDK/robot constructed.")
        print(f"  plan            : {plan}")
        print(f"  obstacles       : {len(body_sdf_scene.get('obstacles', []))}  (preset={src.scene_preset})")
        print(f"  governor        : m_track={m_track} lookahead={lookahead} cfg={GOVERNOR_CFG}")
        print(f"  T_world_scene   : {t_world_scene}")
        print(f"  frame glue check: world start → scene pose "
              f"({pose[0]:.3f}, {pose[1]:.3f}) (expect ≈ {tuple(src.scene.start_pos)})")
        ok = abs(pose[0] - src.scene.start_pos[0]) < 1e-3 and abs(pose[1] - src.scene.start_pos[1]) < 1e-3
        print(f"  RESULT          : {'wiring OK ✅' if ok else 'FRAME MISMATCH ❌'}")
        return 0 if ok else 1

    # ----- real hardware path (needs unitree_sdk2py + the G1 + Vicon) --------
    from genedynamics.deploy.io.unitree_g1_io import UnitreeG1RobotIO
    from genedynamics.deploy.controllers.sport_mode.real_loco_client import RealLocoClient
    from scripts.tasks.robot.humanoid.sport_mode_corridor import diagnose, print_report

    loc = SceneFrameLocalization(_build_inner_localization(), t_world_scene)
    io = UnitreeG1RobotIO(
        network_interface=args.network_interface, domain_id=args.domain_id,
        msc_mode="sport", control_period_s=1.0 / args.control_hz,
        torque_safety_margin=0.85, localization=loc,
    )
    loco = RealLocoClient(nominal_step_period=0.6, rate_limit_hz=args.loco_rate_hz)

    result = diagnose(
        plan, io=io, loco_client=loco, control_hz=args.control_hz,
        use_governor=True, use_body_sdf_governor=True,
        governor_cfg={**GOVERNOR_CFG, "m_track": m_track},
        gov_cmd_lpf=GOV_CMD_LPF, body_sdf_activation_band=GOV_ACTIVATION_BAND,
        body_sdf_lookahead=lookahead, body_sdf_scene=body_sdf_scene,
        spark_pd_gains=False, max_steps=args.max_steps,
        plan_speed=args.plan_speed, goal_hold_sec=args.goal_hold_sec,
        out_dir=Path(args.out_dir),
    )
    print_report(result)
    return 0 if result.certified_safe else 1


if __name__ == "__main__":
    raise SystemExit(main())
