"""Bounded nominal evaluation and paired termination-learning diagnostics."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

import numpy as np
import torch

from genedynamics.core.artifacts import atomic_json
from .diagnostics import _learning_curve_diagnostics, _load_demonstration
from .pilot import _monitor_checkpoint


VERSION = "human2sim2robot-nominal-learning-diagnostic-v1"
TERMINATION_VERSION = "human2sim2robot-reference-termination-diagnostic-v1"
LEARNING_VERSION = "human2sim2robot-paired-termination-learning-v1"


def _checked(root: Path, binding: dict) -> Path:
    path = root / binding["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]:
        raise ValueError(f"nominal diagnostic input changed: {path}")
    return path


def learning_summary(training: dict, curve: list[dict]) -> dict:
    signals = _learning_curve_diagnostics(curve)
    # A falling critic loss is not a convergence test for task learning.
    signals.pop("optimizer_signal_stabilized", None)
    monitors = [{key: row[key] for key in (
        "epoch", "requested", "successes", "constraint_violations", "outcomes",
        "median_reference_completion_fraction", "median_tracking_error_final_m",
        "median_action_saturation_fraction",
    )} for row in training["monitor_curve"]]
    return {
        "training_transitions": training["training_transitions"],
        "selected_epoch": training["selected_epoch"],
        "monitor_curve": monitors, "optimizer_signals": signals,
        "last_learning_rate": curve[-1]["learning_rate"],
        "any_nominal_success": any(row["successes"] for row in monitors),
        "convergence_proven": False,
        "interpretation": "One million transitions is a budget, not evidence of convergence; small PPO losses and a vanishing scheduled learning rate do not establish task mastery.",
    }


def run_nominal_diagnostic(config: dict, root: Path) -> dict:
    if config.get("version") != VERSION or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU nominal diagnostic requires its explicit config and CUDA_VISIBLE_DEVICES=''")
    if len(config["checkpoints"]) != 2 or len(config["seeds"]) != 1:
        raise ValueError("only selected/final checkpoint at one fixed nominal seed are admitted")
    torch.set_num_threads(1)
    recipe = json.loads(_checked(root, config["training_config"]).read_text())
    training = json.loads(_checked(root, config["training_report"]).read_text())
    curve = json.loads(_checked(root, config["learning_curve"]).read_text())
    demonstration, _, demo_summary = _load_demonstration(_checked(root, recipe["demonstration"]))
    recipe = deepcopy(recipe)
    recipe["pilot"]["process_device"] = "cpu"
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "plan.json", config)
    started = time.monotonic()
    results = []
    for binding in config["checkpoints"]:
        if time.monotonic() - started >= config["budget"]["maximum_wall_seconds"]:
            raise TimeoutError("CPU nominal diagnostic wall-time budget exhausted")
        packet = torch.load(_checked(root, binding["checkpoint"]), map_location="cpu", weights_only=False)
        if packet["public_task_adapter"]["task"] != config["task"]:
            raise ValueError("nominal checkpoint task mismatch")
        row = _monitor_checkpoint(packet, recipe, demonstration, demo_summary["demo_dt"],
                                  config["seeds"], recipe["audit"]["action_saturation_threshold"])
        row.update({"role": binding["role"], "checkpoint": binding["checkpoint"]})
        results.append(row)
        atomic_json(output / f"{binding['role']}.json", row)
    report = {
        "version": VERSION, "status": "complete", "task": config["task"],
        "scope": "fresh nominal model seed using original training reset and controller; no shifted target evaluation",
        "learning": learning_summary(training, curve), "results": results,
        "new_training_updates": 0, "gpu_used": False,
        "wall_seconds": time.monotonic() - started,
        "interpretation": "Nominal failures diagnose this robot-source adaptation; they do not prove the original human-video method fails under transfer.",
    }
    atomic_json(output / "report.json", report)
    return report


def run_reference_termination_diagnostic(config: dict, root: Path) -> dict:
    """Compare only episode termination and smoke-test native PPO on CPU.

The shortened reference in the optimizer smoke is explicitly an integration
fixture.  No policy from this check is admitted to a task-performance table.
"""
    from genedynamics.envs.adapters.transfer_task import instantiate_task
    from genedynamics.solvers.single.human2sim2robot import Human2Sim2RobotSolver
    from genedynamics.solvers.single.human2sim2robot.backends.human2sim2robot_torch import native_lstm_config
    from .task_env import ObjectTrajectoryTaskEnv
    from .training_v2 import build_learner, train_epoch

    if config.get("version") != TERMINATION_VERSION or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("reference-termination diagnostic is CPU only")
    budget = config["budget"]
    if (budget["nominal_episodes"], budget["ppo_epochs"], budget["ppo_actors"],
            budget["ppo_horizon"]) != (2, 2, 4, 16):
        raise ValueError("only two rollouts and 128 PPO transitions are authorized by this diagnostic")
    torch.set_num_threads(1)
    recipe = json.loads(_checked(root, config["training_config"]).read_text())
    packet = torch.load(_checked(root, config["checkpoint"]), map_location="cpu", weights_only=False)
    demo, _, meta = _load_demonstration(_checked(root, recipe["demonstration"]))
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "plan.json", config)
    started = time.monotonic()
    results, traces = [], []
    for terminate in (False, True):
        task = instantiate_task(recipe["task"], robot=recipe["body"], seed=config["seed"])
        env = ObjectTrajectoryTaskEnv(task, demo, num_envs=1, seed=config["seed"],
                                      demo_dt=meta["demo_dt"], terminate_on_reference_end=terminate)
        obs = env.reset()
        solver = Human2Sim2RobotSolver(network_config=native_lstm_config(),
            observation_shape=env.observation_space.shape, action_low=env.action_space.low,
            action_high=env.action_space.high, device="cpu", normalize_input=True, normalize_value=True)
        solver.restore(packet["native_policy"])
        solver.reset(1)
        actions, rewards, indices = [], [], []
        for _ in range(task.config.timeout_steps):
            if time.monotonic() - started >= budget["maximum_wall_seconds"]:
                raise TimeoutError("reference diagnostic wall budget exhausted")
            action = solver.act(obs).detach().cpu().numpy().astype(np.float32)
            indices.append(float(env.reference_index[0]))
            obs, reward, done, info = env.step(action, auto_reset=False)
            actions.append(action[0]); rewards.append(float(reward[0]))
            if bool(done[0]):
                break
        arrays = {"actions": np.asarray(actions), "rewards": np.asarray(rewards),
                  "reference_indices": np.asarray(indices)}
        traces.append(arrays)
        role = "reference_terminal" if terminate else "legacy_task_terminal"
        np.savez_compressed(output / f"{role}.npz", **arrays)
        row = {
            "role": role, "steps": len(rewards), "reward_sum": float(sum(rewards)),
            "discounted_return_gamma_0_998": float(np.sum(np.asarray(rewards) * .998 ** np.arange(len(rewards)))),
            "final_reference_index": float(env.reference_index[0]),
            "reference_motion_done": bool(info.get("reference_motion_done", [False])[0]),
            "shared_task_status": int(info["task_status"][0]),
            "shared_task_success": bool(info["task_status"][0] == 1),
            "time_outs": bool(info["time_outs"][0]),
            "final_tracking_error_m": float(info["reference_tracking_error"][0]),
            "accounting": dict(task.accounting),
        }
        results.append(row)
        atomic_json(output / f"{role}.json", row)
    prefix = min(len(row["actions"]) for row in traces)
    prefix_exact = all(np.array_equal(traces[0][key][:prefix], traces[1][key][:prefix])
                       for key in ("actions", "rewards", "reference_indices"))
    if not prefix_exact:
        raise AssertionError("changing termination altered the shared rollout prefix")
    # Two recorded first frames force reference-end resets inside a PPO batch.
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    task = instantiate_task(recipe["task"], robot=recipe["body"], seed=config["seed"])
    env = ObjectTrajectoryTaskEnv(task, demo[:2], num_envs=4, seed=config["seed"],
                                  demo_dt=meta["demo_dt"], terminate_on_reference_end=True)
    agent = None
    try:
        agent, _ = build_learner(env, output / "integration_smoke", device="cpu",
                                 num_actors=4, horizon=16, epochs=2)
        smoke = [train_epoch(agent) for _ in range(2)]
        finite = all(np.isfinite(np.asarray(row[key])).all() for row in smoke
                     for key in ("actor_loss", "critic_loss", "entropy", "kl"))
        if not finite:
            raise AssertionError("native PPO termination smoke produced nonfinite updates")
    finally:
        if agent is not None:
            agent.writer.close()
    report = {
        "version": TERMINATION_VERSION, "status": "complete", "task": config["task"],
        "checkpoint": config["checkpoint"], "results": results,
        "paired_prefix_exact": prefix_exact, "paired_prefix_steps": prefix,
        "ppo_integration_smoke": {"fixture": "first two recorded reference frames only",
            "epochs": 2, "transitions": 128, "finite_metrics": bool(finite), "curve": smoke,
            "task_performance_evidence": False},
        "wall_seconds": time.monotonic() - started, "gpu_used": False,
        "remaining_adaptation": "Shared task success and safety still terminate training; native force-tracking suppresses success early termination. Reward and task success predicates were not changed.",
        "interpretation": "This verifies episode-accounting alignment, not improved learning or transfer performance.",
    }
    atomic_json(output / "report.json", report)
    return report


def paired_learning_plan(config: dict, root: Path) -> dict:
    if config.get("version") != LEARNING_VERSION or config["task"] != "T2":
        raise ValueError("explicit bounded T2 paired-learning recipe required")
    execution, budget = config["execution"], config["budget"]
    cpu = execution == {"device": "cpu", "cuda_visible_devices": ""}
    expected_uuid = execution.get("physical_uuid")
    gpu = (
        isinstance(expected_uuid, str)
        and re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", expected_uuid) is not None
        and execution.get("cuda_visible_devices") in {"0", "1"}
        and execution == {"device": "cuda:0", "cuda_visible_devices": execution["cuda_visible_devices"], "physical_uuid": expected_uuid}
    )
    # 64 * 16 control ticks include both a 503-tick reference terminal
    # and the legacy 1000-tick task timeout; 32 epochs barely reach the first.
    expected = (8, 16, 64, [0, 16, 32, 64]) if cpu else (64, 16, 128, [0, 32, 64, 128])
    if not (cpu or gpu) or (budget["actors"], budget["horizon"], budget["epochs"], config["monitor_epochs"]) != expected:
        raise ValueError("paired-learning resources differ from bounded CPU/GPU recipe")
    if len(config["monitor_seeds"]) != 1:
        raise ValueError("this diagnostic admits one common nominal monitor seed")
    for key in ("checkpoint", "training_config", "termination_diagnostic"):
        _checked(root, config[key])
    previous = json.loads(_checked(root, config["termination_diagnostic"]).read_text())
    if not previous["paired_prefix_exact"] or not previous["ppo_integration_smoke"]["finite_metrics"]:
        raise ValueError("termination alignment and native PPO smoke are prerequisites")
    if (root / config["output"]).exists():
        raise FileExistsError("paired-learning output already exists")
    return {**config, "status": "ready_not_started", "gpu_started": False,
        "training_transitions_per_branch": budget["actors"] * budget["horizon"] * budget["epochs"],
        "monitor_episodes_total": 2 * len(config["monitor_epochs"]),
        "scope": "bounded nominal learning diagnostic; one warm checkpoint, new identical optimizers, full shared-task monitoring"}


def run_paired_learning(config: dict, root: Path) -> dict:
    from genedynamics.envs.adapters.transfer_task import instantiate_task
    from .task_env import ObjectTrajectoryTaskEnv
    from .training_v2 import build_learner, train_epoch
    from .pilot import _atomic_torch, _packet

    plan = paired_learning_plan(config, root)
    device = config["execution"]["device"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != config["execution"]["cuda_visible_devices"]:
        raise RuntimeError("paired-learning visible device binding differs")
    if device != "cpu":
        if torch.cuda.device_count() != 1:
            raise RuntimeError("paired diagnostic requires exactly one visible physical GPU")
        properties = torch.cuda.get_device_properties(0)
        expected_uuid = config["execution"]["physical_uuid"]
        if str(properties.uuid).removeprefix("GPU-").lower() != expected_uuid.removeprefix("GPU-").lower():
            raise RuntimeError("paired diagnostic exposed wrong physical GPU")
        torch.cuda.set_per_process_memory_fraction(config["budget"]["maximum_gpu_gib"] * 1024**3 / properties.total_memory, 0)
    torch.set_num_threads(1)
    recipe = json.loads(_checked(root, config["training_config"]).read_text())
    packet = torch.load(_checked(root, config["checkpoint"]), map_location="cpu", weights_only=False)
    demo, _, meta = _load_demonstration(_checked(root, recipe["demonstration"]))
    recipe = deepcopy(recipe)
    recipe["pilot"]["process_device"] = device
    budget = config["budget"]
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "plan.json", plan)
    started = time.monotonic()
    branches, initial_digests = [], []
    try:
        for reference_terminal in (False, True):
            torch.manual_seed(config["seed"]); np.random.seed(config["seed"]); random.seed(config["seed"])
            if device != "cpu": torch.cuda.manual_seed_all(config["seed"])
            role = "reference_terminal" if reference_terminal else "legacy_task_terminal"
            directory = output / role
            directory.mkdir()
            task = instantiate_task(recipe["task"], robot=recipe["body"], seed=config["seed"])
            env = ObjectTrajectoryTaskEnv(task, demo, num_envs=budget["actors"], seed=config["seed"],
                demo_dt=meta["demo_dt"], terminate_on_reference_end=reference_terminal)
            agent, _ = build_learner(env, directory / "native_learner", device=device,
                num_actors=budget["actors"], horizon=budget["horizon"], epochs=budget["epochs"])
            try:
                agent.set_weights(packet["native_policy"])
                digest = hashlib.sha256()
                for name, value in sorted(agent.model.state_dict().items()):
                    digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
                initial_digests.append(digest.hexdigest())
                learning, monitors = [], []
                for epoch in range(budget["epochs"] + 1):
                    if time.monotonic() - started > budget["maximum_wall_seconds"]:
                        raise TimeoutError("paired-learning wall-time budget exhausted")
                    if epoch:
                        learning.append(train_epoch(agent))
                    if epoch in config["monitor_epochs"]:
                        current = _packet(agent, recipe, env, demo, meta["demo_dt"])
                        current["public_task_adapter"]["training_termination"] = {
                            "reference_end": reference_terminal, "shared_task_success_and_safety": True,
                            "deployment_termination": "unchanged shared task contract"}
                        _atomic_torch(directory / f"epoch_{epoch}.pt", current)
                        row = _monitor_checkpoint(current, recipe, demo, meta["demo_dt"],
                            config["monitor_seeds"], recipe["audit"]["action_saturation_threshold"])
                        row.update(epoch=epoch, training_transitions=epoch * budget["actors"] * budget["horizon"])
                        monitors.append(row)
                        atomic_json(directory / "monitor_curve.json", monitors)
                    if epoch and epoch % 8 == 0:
                        atomic_json(output / "progress.json", {"role": role, "epoch": epoch,
                            "epochs_per_branch": budget["epochs"], "wall_seconds": time.monotonic()-started})
                atomic_json(directory / "learning_curve.json", learning)
                branches.append({"role": role, "initial_model_sha256": initial_digests[-1],
                    "monitor_curve": monitors, "completed_transitions": plan["training_transitions_per_branch"]})
            finally:
                agent.writer.close()
                del agent
                if device != "cpu": torch.cuda.empty_cache()
        if len(set(initial_digests)) != 1:
            raise AssertionError("paired learners did not start with identical policy and normalization weights")
        report = {"version": LEARNING_VERSION, "status": "complete", "task": config["task"],
            "branches": branches, "same_initial_model": True, "scope": plan["scope"],
            "wall_seconds": time.monotonic()-started, "gpu_used": device != "cpu",
            "learning_improvement_proven": False,
            "interpretation": "A small finite budget curve, not a convergence or SOTA test; retain failed nominal monitors."}
    except BaseException as error:
        atomic_json(output / "report.json", {"status": "failed", "error": repr(error),
            "branches": branches, "wall_seconds": time.monotonic()-started})
        raise
    atomic_json(output / "report.json", report)
    return report
