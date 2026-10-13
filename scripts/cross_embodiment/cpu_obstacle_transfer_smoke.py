"""Run actual 3D robot rollouts and save a portable CPU replay fixture.

python scripts/cross_embodiment/cpu_obstacle_transfer_smoke.py --backend numpy --robot panda
python scripts/cross_embodiment/cpu_obstacle_transfer_smoke.py --backend jax --robot xarm7 --batch 2
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import numpy as np

from genedynamics.learning.cross_embodiment import (
    CrossEmbodiment, ObstacleTransferConfig, ExecutionSpec, MechanismSpec, RandomRequest, SkillSpec,
)
from genedynamics.learning.cross_embodiment.contracts import Status
from genedynamics.learning.cross_embodiment.core.artifacts import save_bundle
from genedynamics.learning.cross_embodiment.reference import reference_commands


def run(args):
    task, skill = ObstacleTransferConfig(robot=args.robot, reset_joint_noise=0., timeout_seconds=args.timeout), SkillSpec()
    if args.batch < 1 or args.chunk < 1 or task.timeout_steps % args.chunk:
        raise ValueError("batch/chunk must be positive; chunk must divide timeout_steps for a fixed JIT shape")
    mechanism = MechanismSpec(servo_gain=args.gain, delay_steps=args.delay, payload_mass=args.mass)
    execution = ExecutionSpec(backend=args.backend, device=args.device,
                              physics_backend="mujoco" if args.backend == "numpy" else "mjx")
    runner = CrossEmbodiment(execution=execution, task=task, skill=skill, mechanism=mechanism)
    noise = np.zeros((args.batch, 7), dtype=np.float32)
    request = RandomRequest(args.seed, tuple(range(args.batch)))
    initial = runner.reset_batch(request, noise_batch=noise)
    runner.synchronize(initial)
    commands = np.broadcast_to(reference_commands(task, skill, args.route)[None], (args.batch, task.timeout_steps, 13)).copy()
    # Compile using the same shape as all measured chunks; discard warmup state.
    start = time.perf_counter()
    warmup = runner.collect_chunk(initial, commands[:, :args.chunk])
    runner.synchronize(warmup)
    compile_seconds = time.perf_counter() - start
    state, traces = initial, []
    start = time.perf_counter()
    for offset in range(0, task.timeout_steps, args.chunk):
        result = runner.evaluate_chunk(state, commands[:, offset:offset+args.chunk])
        state = result.state
        traces.append(result.trace)
    runner.synchronize(result)
    elapsed = time.perf_counter() - start
    tensors = {"commands": commands, "reset_noise": noise}
    for i, name in enumerate(result.trace._fields):
        tensors[f"trace.{name}"] = np.concatenate([np.asarray(t[i]) for t in traces], axis=1)
    tensors.update(final_status=np.asarray(state.status), final_steps=np.asarray(state.steps))
    report = {"cpu_functional": "passed" if np.all(tensors["final_status"] == Status.SUCCESS) else "failed",
              "gpu_qualification": "not_run", "scope": "ObstacleTransfer scripted execution smoke; learned evaluation has a separate entry", "route": args.route,
              "scenario": runner.describe(), "execution": asdict(execution), "skill": asdict(skill),
              "seed": args.seed, "sample_ids": request.sample_ids, "warmup_seconds": compile_seconds,
              "runtime_environment": {name: os.environ.get(name) for name in ("JAX_PLATFORMS", "XLA_FLAGS", "MUJOCO_GL")},
              "rollout_seconds": elapsed,
              "scheduled_control_steps_per_second": args.batch * task.timeout_steps / elapsed,
              "valid_control_steps_per_second": int(tensors["trace.valid"].sum()) / elapsed,
              "status": [Status(int(s)).name for s in tensors["final_status"]],
              "terminal_events": tensors["trace.terminal"].sum(axis=1).tolist(),
              "last_position_error": tensors["trace.position_error"][:,-1].tolist(),
              "last_orientation_error": tensors["trace.orientation_error"][:,-1].tolist(),
              "note": "Deterministic controller/scene smoke; no learned prior, calibration or transport claim."}
    save_bundle(args.output, tensors, report)
    print(json.dumps({key: report[key] for key in ("cpu_functional", "gpu_qualification", "status",
                                                  "terminal_events", "last_position_error", "last_orientation_error",
                                                  "warmup_seconds", "rollout_seconds")}, indent=2))
    return 0 if report["cpu_functional"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("jax", "numpy"), default="jax")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--robot", choices=("panda", "xarm7"), default="panda")
    parser.add_argument("--route", choices=("upper", "side"), default="upper")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gain", type=float, default=1.)
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--mass", type=float, default=.5)
    parser.add_argument("--timeout", type=float, default=30.)
    parser.add_argument("--output", type=Path, default=Path("tmp/cross_embodiment/obstacle_transfer_smoke"))
    raise SystemExit(run(parser.parse_args()))
