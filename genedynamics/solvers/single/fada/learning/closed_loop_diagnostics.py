"""Offline attribution for a task-adapted FADA Planner--IDM checkpoint.

The diagnostic consumes a sealed public rollout.  It never resets or steps a
simulator.  Planner prediction, teacher-forced IDM prediction, normalization,
and deployed actor replay are reported separately so a closed-loop failure is
not automatically attributed to the native model.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from genedynamics.solvers.single.fada import FADASolver
from genedynamics.solvers.single.fada.learning.task_training import (
    TaskTrajectory,
    TaskWindowDataset,
    portable_observation,
)
from genedynamics.tasks.transfer.controllers.fada.joint import joint_context
from genedynamics.tasks.transfer.specs import task_spec


PUBLIC_FIELDS = frozenset((
    "joint_position", "joint_velocity", "tcp_position", "tcp_rotation", "tcp_twist",
    "measured_wrench", "goal_position", "goal_rotation", "previous_action",
    "previous_request", "reference_position", "reference_rotation", "elapsed_seconds",
    "body_position", "body_twist", "controller_feedback",
))


ACTION_GROUPS = {
    "linear_m_per_s": slice(0, 3),
    "angular_rad_per_s": slice(3, 6),
    "log_stiffness": slice(6, 12),
    "force_N": slice(12, 13),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_path(path, expected_sha256: str, *, label: str) -> Path:
    value = Path(path)
    if not value.is_file():
        raise FileNotFoundError(f"{label} does not exist: {value}")
    actual = _sha256(value)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA changed: expected {expected_sha256}, got {actual}")
    return value


def _public_episode(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        public = {
            name: np.asarray(archive["public." + name]).copy()
            for name in PUBLIC_FIELDS
            if "public." + name in archive.files
        }
        if set(public) != PUBLIC_FIELDS:
            missing = sorted(PUBLIC_FIELDS - set(public))
            raise ValueError(f"sealed FADA rollout lacks public fields: {missing}")
        actions = np.asarray(archive["actions"], dtype=np.float32).copy()
        traces = {
            name[6:]: np.asarray(archive[name]).copy()
            for name in archive.files
            if name.startswith("trace.")
        }
    count = len(actions)
    if actions.shape != (count, 13) or any(len(value) != count + 1 for value in public.values()):
        raise ValueError("FADA rollout must contain one public transition per 13D action")
    return public, actions, traces


def _anchored_public(public: dict[str, np.ndarray], task: str) -> dict[str, np.ndarray]:
    result = {key: value.copy() for key, value in public.items()}
    if task == "T4":
        result["body_position"] -= result["body_position"][0:1]
    return result


def _memory_context(public, spec, mode: int, metadata: dict) -> np.ndarray:
    version = metadata.get("public_memory_version")
    if spec.alias == "T1" and version == "public-T1-common-planned-FADA-memory-v1":
        from genedynamics.tasks.transfer.controllers.fada.obstacle_planned import planned_context_sequence

        path = metadata.get("public_known_body_path")
        if path is None:
            raise ValueError("planned T1 checkpoint lacks its public body path")
        return planned_context_sequence(public, spec, mode, path)
    if spec.alias == "T2" and version == "public-T2-corridor-FADA-memory-v2":
        from genedynamics.tasks.transfer.controllers.fada.guide_corridor_v2 import corridor_context_sequence

        return corridor_context_sequence(public, spec, mode)
    from genedynamics.tasks.transfer.guide_observation import context_sequence

    return context_sequence(public, spec, mode)


def task_trajectory_from_rollout(public, actions, *, task: str, robot: str, mode: int, payload) -> TaskTrajectory:
    """Reconstruct the exact public chart used by the deployed 199D actor."""
    public = _anchored_public(public, task)
    spec = task_spec(task, robot=robot)
    count = len(actions) + 1
    command = np.concatenate(
        (
            public["goal_position"],
            public["goal_rotation"].reshape(count, 9),
            np.repeat(np.eye(2, dtype=np.float32)[mode : mode + 1], count, axis=0),
            _memory_context(public, spec, mode, payload.get("genedynamics_task_adapter", {})),
        ),
        axis=-1,
    )
    if int(payload["cmd_dim"]) == 199:
        command = np.concatenate((command, joint_context(public, robot)), axis=-1)
    if command.shape != (count, int(payload["cmd_dim"])):
        raise ValueError(
            f"reconstructed command has shape {command.shape}, checkpoint expects (*,{payload['cmd_dim']})"
        )
    return TaskTrajectory(
        portable_observation(public),
        actions.copy(),
        command.astype(np.float32),
        public["previous_action"].astype(np.float32),
        "sealed_fada_closed_loop_rollout",
    )


def _stack(dataset, indices):
    samples = [dataset[int(index)] for index in indices]
    return {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}


def _metric_groups(prediction: np.ndarray, target: np.ndarray) -> dict:
    return {
        key: {
            "rmse": float(np.sqrt(np.mean(np.square(prediction[..., selection] - target[..., selection])))),
            "maximum_abs_error": float(np.max(np.abs(prediction[..., selection] - target[..., selection]))),
        }
        for key, selection in ACTION_GROUPS.items()
    }


def _normalization_roundtrip(batch, backend) -> dict:
    result = {"enabled": bool(backend.io_normalization), "finite_positive_scales": True, "fields": {}}
    values = {
        "obs": batch["history_obs"],
        "act": batch["history_act"],
        "cmd": batch["current_command"],
    }
    for key, value in values.items():
        stats = backend.stats[key]
        mean, std = stats["mean"].cpu(), stats["std"].cpu()
        finite = bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all())
        normalized = (value - mean) / std
        restored = normalized * std + mean
        result["fields"][key] = {
            "dimension": int(value.shape[-1]),
            "maximum_abs_roundtrip_error": float((restored - value).abs().max()),
            "finite_positive_scale": finite,
        }
        result["finite_positive_scales"] &= finite
    return result


def _actor(payload, *, task: str, robot: str, mode: int, device: str):
    spec = task_spec(task, robot=robot)
    if task == "T2":
        from genedynamics.tasks.transfer.controllers.fada.guide_corridor_v2 import PublicCorridorGuideFADAActor

        return PublicCorridorGuideFADAActor(spec, payload, mode=mode, device=device)
    from genedynamics.tasks.transfer.controllers.fada.joint import PublicJointFADATaskActor

    return PublicJointFADATaskActor(spec, payload, mode=mode, device=device)


def _first_over(errors: np.ndarray, threshold: float):
    indices = np.flatnonzero(errors > threshold)
    return None if not len(indices) else int(indices[0])


def _closed_loop_replay(public, actions, traces, payload, *, task: str, robot: str, mode: int, device: str,
                        replay_threshold: float, teacher_threshold: float, control_limit: int | None = None) -> dict:
    deployed = _actor(payload, task=task, robot=robot, mode=mode, device=device)
    spec = task_spec(task, robot=robot)
    teacher_task = SimpleNamespace(spec=spec, config=spec.config)
    from genedynamics.tasks.transfer.teachers import PublicTeacher

    teacher = PublicTeacher(teacher_task, mode=mode)
    replay, teacher_actions = [], []
    controls = len(actions) if control_limit is None else min(len(actions), int(control_limit))
    if controls < 1:
        raise ValueError("closed-loop replay control limit must be positive")
    for index in range(controls):
        packet = {key: value[index : index + 1] for key, value in public.items()}
        replay.append(np.asarray(deployed.act(packet), dtype=np.float32).reshape(13))
        teacher_actions.append(np.asarray(teacher.act(deepcopy(packet)), dtype=np.float32).reshape(13))
    replay = np.asarray(replay)
    teacher_actions = np.asarray(teacher_actions)
    replay_error = np.max(np.abs(replay - actions[:controls]), axis=1)
    scale = np.asarray(
        [.12] * 3 + [.6] * 3 + [1.] * 6 + [max(float(getattr(spec.config, "force_limit", 0.0)), 1.)],
        dtype=np.float32,
    )
    teacher_error = np.max(np.abs(replay - teacher_actions) / scale, axis=1)
    constraint_fields = [
        name for name in ("valid", "finite", "forbidden_collision", "joint_limit") if name in traces
    ]
    first_trace_issue = None
    for index in range(controls):
        bad = []
        for name in constraint_fields:
            value = traces[name][index]
            failed = (not bool(value)) if name in {"valid", "finite"} else bool(value)
            if failed:
                bad.append(name)
        if bad:
            first_trace_issue = {"control_index": index, "fields": bad}
            break
    return {
        "scope": "offline replay of a recorded FADA closed-loop rollout; no new simulator call",
        "recorded_controls": int(len(actions)),
        "replayed_controls": int(controls),
        "actor_replay": {
            "first_divergence_control_index": _first_over(replay_error, replay_threshold),
            "threshold_abs_action": float(replay_threshold),
            "maximum_abs_action_error": float(replay_error.max(initial=0.0)),
            "matching_checkpoint_and_interface": bool(np.all(replay_error <= replay_threshold)),
        },
        "teacher_reference_on_fada_visited_states": {
            "first_divergence_control_index": _first_over(teacher_error, teacher_threshold),
            "threshold_normalized_linf": float(teacher_threshold),
            "maximum_normalized_linf": float(teacher_error.max(initial=0.0)),
            "interpretation": "policy-action divergence along FADA's recorded closed loop; not a counterfactual teacher rollout",
        },
        "first_recorded_trace_issue": first_trace_issue,
    }


def diagnose_closed_loop(config: dict, *, device: str = "cpu") -> dict:
    checkpoint = _checked_path(config["checkpoint"]["path"], config["checkpoint"]["sha256"], label="checkpoint")
    episode = _checked_path(config["episode"]["path"], config["episode"]["sha256"], label="episode")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    task, robot, mode = config["task"], config["robot"], int(config["mode"])
    public, actions, traces = _public_episode(episode)
    trajectory = task_trajectory_from_rollout(public, actions, task=task, robot=robot, mode=mode, payload=payload)
    dataset = TaskWindowDataset(
        [trajectory], history_len=int(payload["cfg"]["history_len"]),
        pred_horizon_k=int(payload["cfg"]["pred_horizon"]),
    )
    limit = min(int(config.get("sample_limit", 64)), len(dataset))
    if limit < 1:
        raise ValueError("sealed rollout is too short for one Planner--IDM window")
    indices = np.unique(np.linspace(0, len(dataset) - 1, limit, dtype=np.int64))
    solver = FADASolver.from_checkpoint(payload, device=device, task_action_binding=True)
    planner_actions, planner_futures, idm_actions, targets, futures = [], [], [], [], []
    first_batch = None
    for start in range(0, len(indices), int(config.get("batch_size", 32))):
        batch = _stack(dataset, indices[start : start + int(config.get("batch_size", 32))])
        if first_batch is None:
            first_batch = batch
        predicted_action, predicted_future = solver.predict(
            batch["history_obs"], batch["history_act"], batch["current_command"],
            history_valid_mask=batch["history_valid_mask"], return_obs=True,
        )
        backend = solver.backend
        obs = batch["history_obs"].to(backend.device)
        act = batch["history_act"].to(backend.device)
        cmd = batch["current_command"].to(backend.device)
        future = batch["future_observations"].to(backend.device)
        if backend.io_normalization:
            obs = (obs - backend.stats["obs"]["mean"]) / backend.stats["obs"]["std"]
            act = (act - backend.stats["act"]["mean"]) / backend.stats["act"]["std"]
            cmd = (cmd - backend.stats["cmd"]["mean"]) / backend.stats["cmd"]["std"]
            future = (future - backend.stats["obs"]["mean"]) / backend.stats["obs"]["std"]
        idm_command = cmd if solver.model.idm_use_current_command_for_history else None
        with torch.inference_mode():
            teacher_forced = solver.model.predict_idm_actions(
                obs, act, idm_command, future, history_valid_mask=batch["history_valid_mask"].to(backend.device),
            )
        if backend.io_normalization:
            teacher_forced = teacher_forced * backend.stats["act"]["std"] + backend.stats["act"]["mean"]
        planner_actions.append(predicted_action.detach().cpu().numpy())
        planner_futures.append(predicted_future.detach().cpu().numpy())
        idm_actions.append(teacher_forced.detach().cpu().numpy())
        targets.append(batch["target_actions"].numpy())
        futures.append(batch["future_observations"].numpy())
    planner_action = np.concatenate(planner_actions)
    planner_future = np.concatenate(planner_futures)
    idm_action = np.concatenate(idm_actions)
    target = np.concatenate(targets)
    true_future = np.concatenate(futures)
    act_scale = solver.backend.stats["act"]["std"].detach().cpu().numpy()
    obs_scale = solver.backend.stats["obs"]["std"].detach().cpu().numpy()
    return {
        "version": "fada-closed-loop-attribution-v1",
        "task": task,
        "robot": robot,
        "mode": mode,
        "device": str(solver.backend.device),
        "new_physics_calls": 0,
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "episode": {"path": str(episode), "sha256": _sha256(episode)},
        "sampled_windows": int(len(indices)),
        "planner_future_prediction": {
            "normalized_mse": float(np.mean(np.square((planner_future - true_future) / obs_scale))),
            "first_position_rmse_m": float(np.sqrt(np.mean(np.square(planner_future[:, 0, :3] - true_future[:, 0, :3])))),
            "last_position_rmse_m": float(np.sqrt(np.mean(np.square(planner_future[:, -1, :3] - true_future[:, -1, :3])))),
        },
        "idm_teacher_forced_prediction": {
            "normalized_action_mse": float(np.mean(np.square((idm_action - target) / act_scale))),
            "first_action": _metric_groups(idm_action[:, 0], target[:, 0]),
        },
        "end_to_end_planner_idm_prediction": {
            "normalized_action_mse": float(np.mean(np.square((planner_action - target) / act_scale))),
            "first_action": _metric_groups(planner_action[:, 0], target[:, 0]),
        },
        "normalization_and_interface_roundtrip": _normalization_roundtrip(first_batch, solver.backend),
        "closed_loop_divergence": _closed_loop_replay(
            public, actions, traces, payload, task=task, robot=robot, mode=mode, device=device,
            replay_threshold=float(config.get("replay_action_abs_tolerance", 2e-6)),
            teacher_threshold=float(config.get("teacher_action_normalized_linf_threshold", .1)),
            control_limit=config.get("replay_control_limit"),
        ),
        "limitations": [
            "teacher-forced IDM uses the recorded future and is not a closed-loop performance metric",
            "teacher action comparison is evaluated on FADA-visited states and is not a counterfactual teacher trajectory",
            "one sealed rollout diagnoses a failure path but does not estimate task success probability",
        ],
    }


__all__ = ["diagnose_closed_loop", "task_trajectory_from_rollout"]
