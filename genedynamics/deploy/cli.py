"""
Unified deploy CLI.

Usage:
    genedynamics-deploy --config configs/quadruped/flat/mbd_deploy.yaml
    genedynamics-deploy --robot quadruped --model go2 --mode sim --planner mbd --episodes 2
    genedynamics-deploy --robot uav3d --mode sim --planner 2go
    genedynamics-deploy --mode replay --replay-dir results/.../ep_0001_*
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deploy pipeline: sim, real, shadow, replay",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "--config-path",
        type=str,
        default=None,
        help="Path to YAML config (overrides other args when set)",
    )

    # Identity
    parser.add_argument("--robot", type=str, default="quadruped", help="Robot type: quadruped, humanoid, uav3d")
    parser.add_argument("--model", type=str, default=None, help="Model ID: ant, go2, g1, drone, etc.")
    parser.add_argument("--mode", type=str, default="sim", choices=["sim", "real", "shadow", "replay"])
    parser.add_argument("--planner", type=str, default=None, help="Planner: stand, mbd, 2go, etc.")

    # Planning
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--plan-mode", type=str, default="plan_once", choices=["plan_once", "mpc"])

    # Execution
    parser.add_argument("--control-rate-hz", type=float, default=20.0)
    parser.add_argument("--plan-rate-hz", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    # Recording
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--record", action="store_true", help="Enable recording (default when not --no-record)")

    # Replay
    parser.add_argument("--replay-dir", "--episode-dir", type=str, default=None, help="For replay mode")

    # Visualization
    parser.add_argument("--viz", action="store_true", help="After run, render HTML and optionally serve")
    parser.add_argument("--viz-serve", action="store_true", help="Start Flask server for web viz")
    parser.add_argument("--viz-port", type=int, default=5000)

    # Start
    parser.add_argument("--start-mode", type=str, default=None, choices=["random", "near_target", "fixed"])
    parser.add_argument("--start-min-dist", type=float, default=None)
    parser.add_argument("--start-max-dist", type=float, default=None)

    args = parser.parse_args()

    # Load config
    if args.config and Path(args.config).exists():
        from genedynamics.deploy.config import DeployConfig
        config = DeployConfig.from_yaml(args.config)
    else:
        mode = args.mode
        if args.replay_dir:
            mode = "replay"
        from genedynamics.deploy.config import DeployConfig
        config = DeployConfig(
            robot_type=args.robot,
            model_id=args.model or _default_model(args.robot),
            mode=mode,
            planner=args.planner or _default_planner(args.robot),
            horizon=args.horizon,
            plan_mode=args.plan_mode,
            control_rate_hz=args.control_rate_hz,
            plan_rate_hz=args.plan_rate_hz,
            max_steps=args.max_steps,
            episodes=args.episodes,
            seed=args.seed,
            record=not args.no_record,
            output_dir=args.output_dir or _default_output_dir(args.robot, mode),
        )

    # Overrides from CLI
    overrides = {}
    if args.planner is not None:
        overrides["planner"] = args.planner
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.episodes is not None:
        overrides["episodes"] = args.episodes
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.no_record:
        overrides["record"] = False
    if args.start_mode is not None:
        overrides.setdefault("start", {})
        overrides["start"] = dict(overrides["start"]) if overrides["start"] else {}
        overrides["start"]["mode"] = args.start_mode
    if args.start_min_dist is not None:
        overrides.setdefault("start", {})
        overrides["start"] = dict(overrides["start"]) if overrides["start"] else {}
        overrides["start"]["near_target_min_dist"] = args.start_min_dist
    if args.start_max_dist is not None:
        overrides.setdefault("start", {})
        overrides["start"] = dict(overrides["start"]) if overrides["start"] else {}
        overrides["start"]["near_target_max_dist"] = args.start_max_dist
    if args.replay_dir:
        overrides["replay"] = overrides.get("replay") or {}
        overrides["replay"] = dict(overrides["replay"]) if overrides["replay"] else {}
        overrides["replay"]["episode_dir"] = args.replay_dir

    # Run
    from genedynamics.core.backends.runtime import RuntimeBackendManager
    from genedynamics.deploy.pipeline import run_pipeline

    RuntimeBackendManager.set_backend("jax", device="cpu")

    result = run_pipeline(config, overrides=overrides or None)

    if result.get("error"):
        print(f"Error: {result['error']}")
        return 1

    if args.viz or args.viz_serve:
        _run_viz(result, args)

    print(f"Done. Results in {result.get('output_dir', config.output_dir)}")
    return 0


def _run_viz(result: dict, args) -> None:
    """Run web visualization after deploy."""
    from genedynamics.deploy.viz import WebVizService
    output_dir = result.get("output_dir", "results/deploy")
    episode_paths = result.get("episode_paths", [])
    if not episode_paths:
        ep_dir = Path(output_dir) / "episodes"
        if ep_dir.exists():
            episode_paths = sorted(ep_dir.glob("ep_*"))[:5]
        else:
            print("No episodes to visualize")
            return
    svc = WebVizService(output_dir=output_dir, port=args.viz_port)
    for ep_path in episode_paths[:3]:
        try:
            svc.render_episode(ep_path, robot_type=result.get("robot_type", "quadruped"), model_id=result.get("model_id", "go2"))
        except Exception as e:
            print(f"Viz failed for {ep_path}: {e}")
    if args.viz_serve:
        svc.serve(block=True)


def _default_model(robot_type: str) -> str:
    if robot_type == "uav3d":
        return "drone"
    if robot_type == "quadruped":
        return "flat"
    if robot_type == "humanoid":
        return "humanoid"
    return "flat"


def _default_planner(robot_type: str) -> str:
    if robot_type == "uav3d":
        return "2go"
    return "stand"


def _default_output_dir(robot_type: str, mode: str) -> str:
    return f"results/deploy/{robot_type}_{mode}"


if __name__ == "__main__":
    sys.exit(main())
