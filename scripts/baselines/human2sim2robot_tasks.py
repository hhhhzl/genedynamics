"""Qualify the original H2S2R learner on an explicitly adapted real CPU task.

Requires a successful, independently replayed source robot demonstration.
This preserves trajectory reward + native recurrent PPO; it does not stand in
for the author's human-video/hand/FABRICS/IsaacGym asset qualification.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from genedynamics.experiments.artifacts import atomic_json
from genedynamics.experiments.framework.config import ExperimentConfig
from genedynamics.experiments.baselines import instantiate_task
from genedynamics.learning.baselines.human2sim2robot.task_env import ObjectTrajectoryTaskEnv
from genedynamics.learning.baselines.human2sim2robot.training import (
    build_learner, capture_training_state, restore_training_state, train_epoch,
)


def _state_equal(left, right):
    if torch.is_tensor(left):
        return left.device == right.device and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_state_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_state_equal(a, b) for a, b in zip(left, right))
    return left == right


def load_demonstration(path, *, expected_task, expected_robot="panda"):
    path = Path(path)
    receipt = json.loads((path.parent / "report.json").read_text())
    if not receipt["valid_source_demo"]:
        raise ValueError("source demonstration must pass independent task evaluation and exact replay")
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["episode_sha256"]:
        raise ValueError("source demonstration SHA changed")
    task = receipt["contract"]["task"]
    config = task["config"]
    if task["name"] != expected_task or config["robot"] != expected_robot:
        raise ValueError("source demonstration task/robot differs from the declared training contract")
    demo_dt = float(config["control_dt"])
    if not np.isfinite(demo_dt) or demo_dt <= 0:
        raise ValueError("source demonstration has an invalid sampling clock")
    with np.load(path, allow_pickle=False) as data:
        positions, rotations = data["public.tcp_position"], data["public.tcp_rotation"]
        times = data["public.elapsed_seconds"].reshape(-1)
        if len(times) != len(positions) or not np.allclose(np.diff(times), demo_dt, rtol=1e-4, atol=1e-6):
            raise ValueError("source demonstration timestamps differ from its declared sampling clock")
    receipt = {**receipt, "demonstration_contract": {"task": expected_task, "robot": expected_robot, "control_dt": demo_dt}}
    poses = np.tile(np.eye(4, dtype=np.float32), (len(positions), 1, 1))
    poses[:, :3, :3], poses[:, :3, 3] = rotations, positions
    return poses, receipt


def algorithm_state(native_state):
    """Irreversible charge metadata must grow across a physical branch replay.

    Compare every learning, physics, history and logical accounting field,
    keeping this audit annotation separate from resumable algorithm state.
    """
    value = deepcopy(native_state)
    value["env_state"]["task"].pop("lifetime_accounting_at_export", None)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--demonstration", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, default=Path("results/_development/p3_20261014"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=18300)
    parser.add_argument("--training-robot", choices=("panda", "xarm7"), default="xarm7")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 2:
        parser.error("at least two epochs are required for the continuation audit")
    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.use_development_output_root(args.development_root)
    output = cfg.output_dir / "level_source" / f"seed_{args.seed}"
    from scripts.baselines import require_non_paper_output
    output = require_non_paper_output(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "attempt.json").exists():
        raise RuntimeError("this charged qualification attempt already exists; use a new seed/output for a new attempt")
    atomic_json(output / "attempt.json", {"status": "running", "seed": args.seed,
                "config": str(args.config), "scope": "public nominal model training qualification; not hidden target interaction",
                "training_robot": args.training_robot})
    (output / "config_snapshot.yaml").write_text(args.config.read_text())
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    poses, demo_receipt = load_demonstration(args.demonstration, expected_task=cfg.env_name)
    demo_dt = demo_receipt["demonstration_contract"]["control_dt"]
    # Source demonstration supplies the reward. RL runs in the publicly known
    # nominal TARGET body model, as H2S2R's digital-twin mechanism requires.
    # Actual hidden execution deviations below are never used to build it.
    task = instantiate_task(cfg.env_name, robot=args.training_robot, seed=args.seed)
    env = ObjectTrajectoryTaskEnv(task, poses, num_envs=2, seed=args.seed, demo_dt=demo_dt)
    started = time.perf_counter()
    agent, network = build_learner(env, output / "native_learner", device=args.device, num_actors=2)
    before = deepcopy(agent.model.state_dict())
    first_metrics = train_epoch(agent)
    atomic_json(output / "progress.json", {"stage": "first_epoch", "source_lifetime_calls": task.lifetime_accounting,
                "metrics": first_metrics})
    changed = any(not torch.equal(before[k], value) for k, value in agent.model.state_dict().items())
    continuation = capture_training_state(agent)
    checkpoint = output / "training_continuation.pt"
    torch.save(continuation, checkpoint)
    uninterrupted = train_epoch(agent)
    atomic_json(output / "progress.json", {"stage": "uninterrupted_second_epoch", "source_lifetime_calls": task.lifetime_accounting,
                "metrics": uninterrupted})
    expected = capture_training_state(agent)
    lifetime_before_restore = dict(task.lifetime_accounting)
    # This is our own locally generated full-state file, containing NumPy scene
    # state. Published external assets use their restricted source loaders.
    # Keep Adam's non-capturable scalar step on CPU; optimizer.load_state_dict
    # moves its moment tensors to parameters, while our bridge places rollout
    # memory on the policy device. Blanket map_location=cuda changes that state.
    restore_training_state(agent, torch.load(checkpoint, map_location="cpu", weights_only=False))
    resumed = train_epoch(agent)
    atomic_json(output / "progress.json", {"stage": "resumed_second_epoch", "source_lifetime_calls": task.lifetime_accounting,
                "metrics": resumed})
    actual = capture_training_state(agent)
    resumed_equal = _state_equal(algorithm_state(expected["native"]), algorithm_state(actual["native"]))
    continued_equal = _state_equal(expected["continuation"], actual["continuation"])
    metric_keys = ("actor_loss", "critic_loss", "entropy", "kl", "learning_rate", "epoch", "frame")
    metrics_equal = all(uninterrupted[key] == resumed[key] for key in metric_keys)
    if not (changed and resumed_equal and continued_equal and metrics_equal):
        atomic_json(output / "resume_failure.json", {"weights_changed": changed, "native_equal": resumed_equal,
                    "continuation_equal": continued_equal, "metrics_equal": metrics_equal,
                    "first": first_metrics, "uninterrupted": uninterrupted, "resumed": resumed})
        raise RuntimeError("native PPO full continuation qualification failed")
    learning_curve = [first_metrics, resumed]
    for _ in range(2, args.epochs):
        metrics = train_epoch(agent)
        learning_curve.append(metrics)
        atomic_json(output / "progress.json", {"stage": "training", "completed_epochs": len(learning_curve),
                    "source_lifetime_calls": task.lifetime_accounting, "metrics": metrics})
    atomic_json(output / "learning_curve.json", learning_curve)
    torch.save(capture_training_state(agent), output / "training_completed.pt")
    deploy = agent.get_weights()
    torch.save(deploy, output / "native_policy.pt")
    agent.writer.close()
    from genedynamics.solvers.single.human2sim2robot import Human2Sim2RobotSolver
    solver = Human2Sim2RobotSolver(network_config=network, observation_shape=env.observation_space.shape,
                action_low=env.action_space.low, action_high=env.action_space.high,
                device=args.device, normalize_input=True, normalize_value=True)
    solver.restore(deploy)
    profiles = []
    for batch in (1, 4):
        profile_task = instantiate_task(cfg.env_name, robot=args.training_robot, seed=args.seed + 10000 + batch)
        profile_env = ObjectTrajectoryTaskEnv(profile_task, poses, num_envs=batch, seed=args.seed + 10000 + batch, demo_dt=demo_dt)
        obs = profile_env.reset()
        solver.reset(batch)
        durations, inference_durations = [], []
        for _ in range(11):
            if torch.device(args.device).type == "cuda": torch.cuda.synchronize()
            start = time.perf_counter(); action = solver.act(obs)
            if torch.device(args.device).type == "cuda": torch.cuda.synchronize()
            assert torch.isfinite(action).all()
            inference_durations.append(time.perf_counter() - start)
            obs, _, profile_dones, _ = profile_env.step(action.detach().cpu().numpy())
            solver.reset(dones=profile_dones)
            durations.append(time.perf_counter() - start)
        profiles.append(dict(batch=batch, cold_seconds=durations[0], p50_seconds=float(np.median(durations[1:])),
                             p95_seconds=float(np.percentile(durations[1:], 95)),
                             inference_p50_seconds=float(np.median(inference_durations[1:])),
                             physical_calls=dict(profile_task.lifetime_accounting),
                             scope="actual policy+history+CPU physics in public nominal model; separate from target evaluation"))
    # Qualification actors get complete target episodes even if early training
    # yields failures. Failures remain rows; this is not a scored P4 comparison.
    from genedynamics.learning.cross_embodiment.contact import ContactMechanismSpec
    from genedynamics.learning.cross_embodiment.contracts import MechanismSpec
    execution = cfg.execution_env_params
    mechanism_cls = MechanismSpec if cfg.env_name == "obstacle_transfer" else ContactMechanismSpec
    target_mechanism = mechanism_cls(**execution)
    target_task = instantiate_task(cfg.env_name, robot="xarm7", mechanism_spec=target_mechanism, seed=args.seed + 100)
    target = ObjectTrajectoryTaskEnv(target_task, poses, num_envs=2, seed=args.seed + 100, demo_dt=demo_dt)
    observation = target.reset(); solver.reset(2); ended = np.zeros(2, bool)
    for _ in range(target_task.config.timeout_steps):
        actions = solver.act(observation).detach().cpu().numpy()
        # No auto-reset here: exact first requested episode per slot is scored.
        observation, _, dones, _ = target.step(actions, auto_reset=False)
        ended |= dones.astype(bool)
        if ended.all(): break
    evaluation = target_task.evaluate()
    if evaluation["status_mismatches"]:
        raise RuntimeError("independent CPU evaluator differs from task execution")
    replays = [target_task.replay_episode(record) for record in target_task.records]
    if not all(row["matches_status"] and row["max_pose_error"] == 0 and row["trace_exact"] for row in replays):
        raise RuntimeError("target qualification replay differs")
    report = {
        "status": "adapted_training_and_execution_chain_qualified", "native_method_qualified": False,
        "scope": f"{args.epochs} native PPO epochs and exact resumed second epoch; source trajectory reward and public nominal target-body model RL; held-object/shared-control adaptation, not converged benchmark performance",
        "training_epochs": args.epochs, "resume_audit_extra_epochs": 1,
        "training_robot": args.training_robot,
        "training_model_provenance": "known nominal body/geometry model; no hidden execution deviations or target data used for model construction",
        "source_demo_robot": "panda", "target_robot": "xarm7",
        "training_model_calls": dict(task.lifetime_accounting),
        "target_physics_audit_only": execution,
        "weights_changed": changed, "resume_native_equal": resumed_equal,
        "resume_comparison_excludes": ["env_state.task.lifetime_accounting_at_export (irreversible actual interaction audit; must increase)"],
        "resume_rollout_equal": continued_equal, "resume_metrics_equal": metrics_equal,
        "source_demo": str(args.demonstration), "source_demo_sha256": demo_receipt["episode_sha256"],
        "demonstration_contract": demo_receipt["demonstration_contract"],
        "first_epoch": first_metrics, "uninterrupted_second_epoch": uninterrupted,
        "resumed_second_epoch": resumed, "device": str(next(agent.model.parameters()).device),
        "source_calls_before_restore": lifetime_before_restore,
        "source_lifetime_calls": dict(task.lifetime_accounting),
        "target_first_episode_evaluation": evaluation, "target_audit_replays": replays,
        "profiles": profiles, "wall_seconds": time.perf_counter() - started,
        "training_checkpoint_sha256": hashlib.file_digest(checkpoint.open("rb"), "sha256").hexdigest(),
        "completed_training_checkpoint_sha256": hashlib.file_digest((output / "training_completed.pt").open("rb"), "sha256").hexdigest(),
        "adaptations": ["robot-state demonstration replaces human RGBD processing", "held-object reset replaces pre-manipulation human-hand exploration initialization",
                        "existing CPU task model replaces original digital twin assets", "shared skill controller replaces FABRICS",
                        "task observation schema and 13D action replace native KUKA/Allegro interface",
                        "common physical task termination replaces native reference-end/fallen-object reset; reference holds its final pose"],
        "pending_native": ["author assets/data/pretrained models download URL", "IsaacGym Preview4/FABRICS native environment and original closed-loop qualification"],
    }
    atomic_json(output / "qualification.json", report)
    atomic_json(output / "attempt.json", {"status": "complete", "seed": args.seed, "qualified_scope": report["scope"]})
    print(json.dumps({key: report[key] for key in ("status", "weights_changed", "resume_native_equal", "resume_metrics_equal", "device")}, indent=2))


if __name__ == "__main__":
    main()
