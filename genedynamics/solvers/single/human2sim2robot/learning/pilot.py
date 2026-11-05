"""Bounded GPU pilot for the H2S2R robot-source task adaptation.

Training keeps the pinned upstream recurrent PPO implementation.  Nominal
model monitoring adds the continuous trajectory diagnostics required by the
CPU audit, so a zero-success checkpoint is no longer selected blindly.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

import numpy as np
import torch

from genedynamics.core.artifacts import artifact_write, atomic_json
from genedynamics.envs.adapters.transfer_task import FLAT_FIELDS, instantiate_task
from genedynamics.solvers.single.human2sim2robot import Human2Sim2RobotSolver
from genedynamics.solvers.single.human2sim2robot.backends.human2sim2robot_torch import (
    native_lstm_config,
)

from .diagnostics import (
    VERSION as AUDIT_VERSION,
    _load_demonstration,
    normalize_physical_actions,
)
from .task_env import ObjectTrajectoryTaskEnv
from .training_v2 import build_learner, train_epoch


PILOT_VERSION = "human2sim2robot-robot-source-pilot-v1"
PUBLIC_ADAPTER_VERSION = "public-held-object-native-PPO-v1"


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_write(path, 512 * 1024**2):
        # PyTorch's inline zip writer rejects some hidden extensionless names.
        # Keep the temporary in the destination directory for atomic replace,
        # while using a regular checkpoint-looking filename.
        fd, temporary = tempfile.mkstemp(
            dir=path.parent, prefix="torch-checkpoint-", suffix=".pt"
        )
        os.close(fd)
        try:
            torch.save(value, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def pilot_plan(config: dict[str, Any], root: Path) -> dict[str, Any]:
    if config.get("version") != AUDIT_VERSION:
        raise ValueError(f"expected audit config {AUDIT_VERSION}")
    pilot = deepcopy(config["pilot"])
    if pilot["physical_gpu"] != "GPU1" or pilot["cuda_visible_devices"] != "1":
        raise ValueError("this pilot is frozen to physical GPU1")
    expected_uuid = str(pilot.get("expected_physical_uuid", ""))
    if expected_uuid != "GPU-58b54684-3577-11d3-f039-0e8469190739":
        raise ValueError("this pilot must bind the frozen physical GPU1 UUID")
    expected = int(pilot["num_actors"]) * int(pilot["horizon_steps"]) * int(pilot["epochs"])
    if expected != int(pilot["training_transitions"]):
        raise ValueError("pilot transition budget does not match actors × horizon × epochs")
    audit_report = root / config["output"] / "report.json"
    if not audit_report.is_file():
        raise FileNotFoundError(f"CPU learning-chain report is required: {audit_report}")
    audited = json.loads(audit_report.read_text())
    if audited.get("status") != "cpu_learning_chain_audited" or audited.get("task") != config["task"]:
        raise ValueError("CPU learning-chain report does not match the pilot task")
    if audited.get("gpu1_pilot_plan") != pilot:
        raise ValueError("pilot config changed after the CPU learning-chain audit")
    pilot_output = root / config["pilot_output"]
    if pilot_output.exists():
        raise FileExistsError(f"pilot output already exists; refusing to overwrite: {pilot_output}")
    historical = json.loads((root / config["historical_run"]["path"] / "report.json").read_text())
    historical_controls = historical["accounting"]["known_model_training"]["control_steps"]
    estimated = float(historical["wall_seconds"]) * expected / historical_controls
    monitor_multiplier = (
        len(pilot["monitor_epochs"]) * len(pilot["monitor_seeds"])
        / sum(row["requested"] for row in historical["monitor_curve"].values())
    )
    # Historical wall time already includes 16 monitor episodes.  Add a
    # conservative fractional allowance for the larger development monitor.
    estimated *= 1.0 + 0.15 * max(0.0, monitor_multiplier - 1.0)
    return {
        "version": PILOT_VERSION,
        "status": "ready_not_started",
        "task": config["task"],
        "body": config["body"],
        "mode": config["mode"],
        "output": config["pilot_output"],
        "physical_gpu": "GPU1",
        "expected_physical_uuid": expected_uuid,
        "cuda_visible_devices": "1",
        "process_device": pilot["process_device"],
        "num_actors": pilot["num_actors"],
        "horizon_steps": pilot["horizon_steps"],
        "epochs": pilot["epochs"],
        "training_transitions": expected,
        "monitor_episodes": len(pilot["monitor_epochs"]) * len(pilot["monitor_seeds"]),
        "estimated_wall_seconds": estimated,
        "maximum_wall_seconds": pilot["maximum_wall_seconds"],
        "maximum_gpu_memory_gib": pilot["maximum_gpu_memory_gib"],
        "gpu_started": False,
    }


def _verify_gpu_binding(pilot: dict[str, Any]):
    """Bind the one visible CUDA device to the frozen physical GPU1 UUID."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != pilot["cuda_visible_devices"]:
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not match the frozen pilot binding")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the GPU1-only process must expose exactly one CUDA device")
    properties = torch.cuda.get_device_properties(0)
    actual_uuid = str(getattr(properties, "uuid", ""))
    expected_uuid = str(pilot["expected_physical_uuid"])
    if actual_uuid.removeprefix("GPU-").lower() != expected_uuid.removeprefix("GPU-").lower():
        raise RuntimeError(
            "visible CUDA device UUID differs from frozen physical GPU1: "
            f"{actual_uuid or '<unavailable>'} != {expected_uuid}"
        )
    return properties


def _packet(agent, config: dict[str, Any], env: ObjectTrajectoryTaskEnv,
            demonstration: np.ndarray, demo_dt: float) -> dict[str, Any]:
    network = native_lstm_config()
    metadata = {
        "version": PUBLIC_ADAPTER_VERSION,
        "task": config["task"],
        "body": config["body"],
        "mode": int(config["mode"]),
        "flat_fields": list(FLAT_FIELDS),
        "demonstration": demonstration.tolist(),
        "demonstration_sha256": env.demonstration_sha256,
        "demo_dt": demo_dt,
        "network_config": asdict(network),
        "observation_shape": list(env.observation_space.shape),
        "action_low": env.action_space.low.tolist(),
        "action_high": env.action_space.high.tolist(),
        "source_demonstration": config["demonstration"],
        "human_assets_used": False,
        "original_native_method_qualified": False,
        "adaptation": "robot-source object trajectory and shared task action chart",
    }
    return {"public_task_adapter": metadata, "native_policy": deepcopy(agent.get_weights())}


def _monitor_checkpoint(
    packet: dict[str, Any], config: dict[str, Any], demonstration: np.ndarray,
    demo_dt: float, seeds: list[int], saturation_threshold: float,
) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        task = instantiate_task(config["task"], robot=config["body"], seed=seed)
        env = ObjectTrajectoryTaskEnv(task, demonstration, num_envs=1, seed=seed, demo_dt=demo_dt)
        observation = env.reset()
        solver = Human2Sim2RobotSolver(
            network_config=native_lstm_config(),
            observation_shape=env.observation_space.shape,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            device=config["pilot"]["process_device"],
            normalize_input=True,
            normalize_value=True,
        )
        solver.restore(packet["native_policy"])
        solver.reset(1)
        tracking, actions = [], []
        for _ in range(task.config.timeout_steps):
            action = solver.act(observation).detach().cpu().numpy().astype(np.float32)
            observation, _, done, info = env.step(action, auto_reset=False)
            tracking.append(float(info["reference_tracking_error"][0]))
            actions.append(action[0])
            if bool(done[0]):
                break
        evaluation = task.evaluate()
        if evaluation["status_mismatches"] or len(evaluation["episodes"]) != 1:
            raise RuntimeError("nominal monitor evaluator or episode denominator differs")
        evaluated = evaluation["episodes"][0]
        normalized, active = normalize_physical_actions(
            np.asarray(actions), env.action_space.low, env.action_space.high,
        )
        rows.append({
            "seed": seed,
            "success": bool(evaluated["success"]),
            "constraint_violation": bool(evaluated["constraint_violation"]),
            "outcome": evaluated["outcome"],
            "control_steps": int(evaluated["physical_control_steps"]),
            "final_position_error_m": evaluated["final_position_error_m"],
            "reference_completion_fraction": min(
                float(env.reference_index[0]) / max(1, len(demonstration) - 1), 1.0,
            ),
            "tracking_error_final_m": tracking[-1],
            "tracking_error_mean_m": float(np.mean(tracking)),
            "action_saturation_fraction": float(
                np.mean(np.abs(normalized[:, active]) >= saturation_threshold)
            ),
        })
    return {
        "requested": len(rows),
        "successes": sum(row["success"] for row in rows),
        "constraint_violations": sum(row["constraint_violation"] for row in rows),
        "median_reference_completion_fraction": float(np.median(
            [row["reference_completion_fraction"] for row in rows]
        )),
        "median_tracking_error_final_m": float(np.median(
            [row["tracking_error_final_m"] for row in rows]
        )),
        "median_action_saturation_fraction": float(np.median(
            [row["action_saturation_fraction"] for row in rows]
        )),
        "outcomes": {name: sum(row["outcome"] == name for row in rows)
                     for name in sorted({row["outcome"] for row in rows})},
        "episodes": rows,
    }


def _rank_monitor(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        row["successes"] / row["requested"],
        -row["constraint_violations"] / row["requested"],
        row["median_reference_completion_fraction"],
        -row["median_tracking_error_final_m"],
        -row["median_action_saturation_fraction"],
        -row["epoch"],
    )


def run_training_pilot(config: dict[str, Any], root: Path) -> dict[str, Any]:
    plan = pilot_plan(config, root)
    properties = _verify_gpu_binding(config["pilot"])
    output = root / config["pilot_output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "recipe.json", config)
    atomic_json(output / "plan.json", plan)
    pilot = config["pilot"]
    memory_fraction = pilot["maximum_gpu_memory_gib"] * 1024**3 / properties.total_memory
    torch.cuda.set_per_process_memory_fraction(min(float(memory_fraction), 1.0), 0)
    seed = int(config["seed"]) + 100000
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    demo_path = root / config["demonstration"]["path"]
    demonstration, _, demo_summary = _load_demonstration(demo_path)
    task = instantiate_task(config["task"], robot=config["body"], seed=seed)
    env = ObjectTrajectoryTaskEnv(
        task, demonstration, num_envs=int(pilot["num_actors"]), seed=seed,
        demo_dt=demo_summary["demo_dt"],
    )
    agent = None
    curve: list[dict[str, Any]] = []
    monitors: list[dict[str, Any]] = []
    started = time.monotonic()
    final: dict[str, Any] = {"version": PILOT_VERSION, "status": "running", "plan": plan}
    try:
        agent, _ = build_learner(
            env, output / "native_learner", device=pilot["process_device"],
            num_actors=int(pilot["num_actors"]), horizon=int(pilot["horizon_steps"]),
            epochs=int(pilot["epochs"]),
        )
        monitor_epochs = set(int(value) for value in pilot["monitor_epochs"])
        for epoch in range(1, int(pilot["epochs"]) + 1):
            if time.monotonic() - started > float(pilot["maximum_wall_seconds"]):
                raise TimeoutError("predeclared H2S2R pilot wall-time ceiling reached")
            curve.append(train_epoch(agent))
            if epoch % 32 == 0:
                atomic_json(output / "progress.json", {
                    "status": "training", "epoch": epoch,
                    "training_transitions": epoch * pilot["num_actors"] * pilot["horizon_steps"],
                    "latest_metrics": curve[-1], "wall_seconds": time.monotonic() - started,
                })
            if epoch in monitor_epochs:
                packet = _packet(agent, config, env, demonstration, demo_summary["demo_dt"])
                checkpoint = output / "checkpoints" / f"epoch_{epoch}.pt"
                _atomic_torch(checkpoint, packet)
                row = _monitor_checkpoint(
                    packet, config, demonstration, demo_summary["demo_dt"],
                    [int(value) for value in pilot["monitor_seeds"]],
                    float(config["audit"]["action_saturation_threshold"]),
                )
                row.update({"epoch": epoch, "checkpoint": str(checkpoint.relative_to(root))})
                monitors.append(row)
                atomic_json(output / "monitor_curve.json", monitors)
        if {row["epoch"] for row in monitors} != monitor_epochs:
            raise AssertionError("not every frozen monitor checkpoint was executed")
        selected = max(monitors, key=_rank_monitor)
        selected_packet = torch.load(root / selected["checkpoint"], map_location="cpu", weights_only=False)
        _atomic_torch(output / "selected_policy.pt", selected_packet)
        atomic_json(output / "learning_curve.json", curve)
        final.update({
            "status": "complete",
            "task": config["task"], "body": config["body"], "mode": config["mode"],
            "epochs": len(curve),
            "training_transitions": len(curve) * pilot["num_actors"] * pilot["horizon_steps"],
            "monitor_curve": monitors,
            "selected_epoch": selected["epoch"],
            "selection_rule": pilot["checkpoint_selection"],
            "selected_has_nominal_success": bool(selected["successes"]),
            "selected_reference_completion_fraction": selected["median_reference_completion_fraction"],
            "selected_action_saturation_fraction": selected["median_action_saturation_fraction"],
            "gpu_started": True,
        })
    except BaseException as exc:
        final.update({"status": "failed", "error": repr(exc), "gpu_started": True})
        raise
    finally:
        if agent is not None:
            agent.writer.close()
        final.update({
            "wall_seconds": time.monotonic() - started,
            "maximum_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0),
        })
        atomic_json(output / "report.json", final)
    return final


__all__ = ["PILOT_VERSION", "pilot_plan", "run_training_pilot"]
