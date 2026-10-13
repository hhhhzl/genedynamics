"""Collect tiny, episode-grouped CPU fixtures from actual robot execution.

PYTHONPATH=. python scripts/cross_embodiment/collect_obstacle_transfer_data.py --backend numpy
PYTHONPATH=. python scripts/cross_embodiment/collect_obstacle_transfer_data.py --backend jax \
    --role response_development --microbatch-size 2 --steps 20

Default 20-step episodes are explicitly truncated plumbing checks. Use
--full-length for the two complete scripted route families; these remain
scripted development demonstrations, not a trained generative policy.
"""
import argparse
import json
from pathlib import Path

from genedynamics.learning.cross_embodiment import ExecutionSpec, MechanismSpec, ObstacleTransferConfig
from genedynamics.learning.cross_embodiment.data import collect_dataset, make_episode_plan


def _mechanisms(text):
    values = []
    try:
        for item in text.split(";"):
            gain, delay, mass = item.split(",")
            values.append(MechanismSpec(float(gain), int(delay), float(mass)))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("mechanisms use 'gain,delay_steps,mass;gain,delay_steps,mass'") from error
    return tuple(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--role", choices=("source_demonstrations", "response_development","response_validation"), default="source_demonstrations")
    parser.add_argument("--robots", nargs="+", choices=("panda", "xarm7"))
    parser.add_argument("--mechanisms", type=_mechanisms)
    parser.add_argument("--episodes-per-route", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--full-length", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.)
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    robots = tuple(args.robots or (("panda",) if args.role == "source_demonstrations" else ("panda", "xarm7")))
    mechanisms = args.mechanisms or ((MechanismSpec(),) if args.role == "source_demonstrations" else (
        MechanismSpec(), MechanismSpec(servo_gain=.8, delay_steps=2, payload_mass=.8),
    ))
    episodes = make_episode_plan(seed=args.seed, episodes_per_route=args.episodes_per_route,
                                 robots=robots, mechanisms=mechanisms)
    execution = ExecutionSpec(backend=args.backend, device=args.device,
                              physics_backend="mujoco" if args.backend == "numpy" else "mjx",
                              microbatch_size=args.microbatch_size)
    output = args.output or Path("tmp/cross_embodiment/obstacle_transfer_data") / args.role
    _, metadata = collect_dataset(episodes, role=args.role, execution=execution,
                                 task=ObstacleTransferConfig(timeout_seconds=args.timeout),
                                 chunk_length=args.chunk, steps=None if args.full_length else args.steps,
                                 output=output)
    print(json.dumps({"output": str(output), **{name: metadata[name] for name in (
        "role", "episodes_attempted", "episodes_successful", "episodes_failed", "episodes_truncated",
        "status_counts", "valid_control_steps", "gpu_qualification",
    )}}, indent=2))
    return 0  # Failed/truncated episodes are valid retained dataset outcomes.


if __name__ == "__main__":
    raise SystemExit(main())
