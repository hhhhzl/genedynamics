"""Mechanism-faithful FADA source preparation and frozen-planner target pilot.

The source phase either continues the admitted oracle/DAgger Planner--IDM
model with the released optional future-observation loss, or preserves an
independently requalified source checkpoint byte-for-byte.  The target phase
then freezes that planner and adapts only the IDM on sealed reward-free B-log
records.  Admission and evaluation are offline; this module never constructs
or steps a simulator.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from genedynamics.envs.adapters.transfer_task import FLAT_FIELDS
from genedynamics.learning.datasets.transfer.target import PublicTargetTrajectory
from genedynamics.learning.datasets.transfer.weak_arm import PublicArmWeakTrajectory
from genedynamics.solvers.single.fada import FADASolver
from genedynamics.tasks.transfer.controllers.fada.guide_corridor_v2 import corridor_checkpoint_binding
from genedynamics.tasks.transfer.controllers.fada.joint_target import lift_target_joint_checkpoint
from genedynamics.tasks.transfer.controllers.fada.obstacle_planned import planned_checkpoint_binding
from genedynamics.tasks.transfer.guide_observation import context_sequence
from genedynamics.tasks.transfer.specs import task_spec

from .closed_loop_diagnostics import task_trajectory_from_rollout
from .native_task_training import (
    ROUTING_VERSION,
    NativeRoutedTaskPlannerIDMTrainer,
    sample_native_source_batches,
)
from .task_training import TaskTrajectory, TaskWindowDataset, portable_observation, sample_dataset
from .trainer import FADAIDMAdapter, IDMAdaptationConfig


VERSION = "fada-planner-repair-idm-pilot-v1"
CONFIG_VERSION = "fada-planner-repair-idm-pilot-config-v1"
TASKS = {"T1", "T2", "T3"}
SOURCE_REPAIR = "repair_planner_idm"
SOURCE_PRESERVE = "preserve_qualified_checkpoint"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_file(reference: dict, label: str) -> Path:
    if set(reference) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain only path and sha256")
    path = Path(reference["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = _sha256(path)
    if actual != reference["sha256"]:
        raise ValueError(f"{label} SHA changed: expected {reference['sha256']}, got {actual}")
    return path


def _source_trajectory(reference: dict, *, task: str) -> TaskTrajectory:
    """Reconstruct the admitted 67D source chart from immutable public arrays."""
    path = Path(reference["path"])
    if _sha256(path) != reference["sha256"]:
        raise ValueError(f"source episode SHA changed: {path}")
    mode = int(reference["mode"])
    if mode not in (0, 1):
        raise ValueError("source mode must be zero or one")
    with np.load(path, allow_pickle=False) as archive:
        public = {
            name: np.asarray(archive["public." + name], dtype=np.float32).copy()
            for name in FLAT_FIELDS
        }
        actions = np.asarray(archive["actions"], dtype=np.float32).copy()
    count = len(actions) + 1
    if any(len(value) != count for value in public.values()) or actions.shape != (count - 1, 13):
        raise ValueError("source episode lacks one public transition per action")
    spec = task_spec(task, robot="panda")
    command = np.concatenate(
        (
            public["goal_position"],
            public["goal_rotation"].reshape(count, 9),
            np.repeat(np.eye(2, dtype=np.float32)[mode : mode + 1], count, axis=0),
            context_sequence(public, spec, mode),
        ),
        axis=-1,
    )
    if command.shape != (count, 67):
        raise ValueError("source controller memory must reconstruct the admitted 67D command")
    return TaskTrajectory(
        portable_observation(public), actions, command.astype(np.float32),
        public["previous_action"].astype(np.float32), "admitted_source_" + path.name,
    )


def _checked_shadows(references: list[dict]) -> list[dict[str, torch.Tensor]]:
    rows = []
    expected = {
        "history_obs": (30, 35), "history_act": (30, 13), "history_valid_mask": (30,),
        "current_command": (67,), "future_observations": (6, 35), "target_actions": (6, 13),
    }
    for reference in references:
        path = Path(reference["path"])
        if _sha256(path) != reference["sha256"]:
            raise ValueError(f"DAgger shadow SHA changed: {path}")
        batch = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(batch, list) or not batch:
            raise ValueError("each DAgger shadow artifact must contain at least one window")
        for row in batch:
            if set(row) != set(expected) or any(tuple(row[key].shape) != shape for key, shape in expected.items()):
                raise ValueError("DAgger shadow window differs from the released history30/horizon6 chart")
            if not all(torch.isfinite(value).all() for value in row.values() if value.dtype != torch.bool):
                raise ValueError("DAgger shadow window contains non-finite values")
        rows.extend(batch)
    return rows


def _load_public_target(path: Path):
    metadata = json.loads((path / "public.json").read_text())
    if metadata.get("version") == "public-arm-weak-task-data-v2":
        return PublicArmWeakTrajectory.load(path)
    return PublicTargetTrajectory.load(path)


def _target_records(manifest: dict, *, task: str) -> list:
    if (manifest.get("permission") != "B_log" or manifest.get("reward") is not False
            or manifest.get("success_filter") is not False or "FADA" not in manifest.get("available_to", [])):
        raise ValueError("target input must be a shared reward-free, unfiltered B-log available to FADA")
    records, controls = [], 0
    for entry in manifest.get("entries", []):
        row = _load_public_target(Path(entry["path"]))
        declared_task = row.metadata.get("task")
        if declared_task is not None and declared_task != task:
            raise ValueError("target B-log task differs from the pilot task")
        if row.metadata["robot"] != "xarm7" or int(row.metadata["mode"]) != int(entry["mode"]):
            raise ValueError("target B-log body or mode differs from its manifest")
        if row.digest != entry["parent_sha256"]:
            raise ValueError("target parent trajectory SHA changed")
        prefix = row.prefix(int(entry["controls"]))
        if prefix.digest != entry["prefix_sha256"]:
            raise ValueError("target prefix SHA changed")
        records.append(prefix)
        controls += len(prefix.requests)
    if controls != int(manifest.get("controls", -1)) or not records:
        raise ValueError("target B-log control accounting differs from its manifest")
    return records


def _bind_target_checkpoint(source: dict, config: dict) -> dict:
    payload = lift_target_joint_checkpoint(source)
    task = config["task"]
    if task == "T1":
        path = config.get("target_binding", {}).get("public_known_body_path")
        if path is None:
            raise ValueError("T1 target binding requires its predeclared public body path")
        payload = planned_checkpoint_binding(payload, path)
    elif task == "T2":
        payload = corridor_checkpoint_binding(payload)
    payload["genedynamics_task_adapter"]["task_name"] = task_spec(task, robot="xarm7").name
    return payload


def _datasets(config: dict, source_payload: dict):
    source_manifest_path = _checked_file(config["source_data_manifest"], "source data manifest")
    source_manifest = json.loads(source_manifest_path.read_text())
    if set(source_manifest) != {"offline", "validation", "real", "shadows"}:
        raise ValueError("source data manifest must retain oracle, validation, real DAgger, and shadow pools")
    pairs = {(int(row["stage"]), int(row["mode"])) for row in source_manifest["real"]}
    shadow_pairs = {(int(row["stage"]), int(row["mode"])) for row in source_manifest["shadows"]}
    if pairs != shadow_pairs or not pairs:
        raise ValueError("real DAgger and oracle-shadow stage/mode coverage must match")
    task = config["task"]
    offline_rows = [_source_trajectory(row, task=task) for row in source_manifest["offline"]]
    validation_rows = [_source_trajectory(row, task=task) for row in source_manifest["validation"]]
    real_rows = [_source_trajectory(row, task=task) for row in source_manifest["real"]]
    shadows = _checked_shadows(source_manifest["shadows"])
    history = int(source_payload["cfg"]["history_len"])
    horizon = int(source_payload["cfg"]["pred_horizon"])
    source = TaskWindowDataset(offline_rows, history_len=history, pred_horizon_k=horizon)
    source_validation = TaskWindowDataset(validation_rows, history_len=history, pred_horizon_k=horizon)
    real = TaskWindowDataset(real_rows, history_len=history, pred_horizon_k=horizon)

    target_manifest_path = _checked_file(config["target_data_manifest"], "target B-log manifest")
    target_manifest = json.loads(target_manifest_path.read_text())
    records = _target_records(target_manifest, task=task)
    target_payload = _bind_target_checkpoint(source_payload, config)
    target_rows = [
        task_trajectory_from_rollout(
            row.public, row.requests, task=task, robot="xarm7", mode=int(row.metadata["mode"]),
            payload=target_payload,
        )
        for row in records
    ]
    target = TaskWindowDataset(target_rows, history_len=history, pred_horizon_k=horizon)
    return {
        "source": source, "source_validation": source_validation, "real": real,
        "shadows": shadows, "target": target, "target_payload": target_payload,
        "accounting": {
            "oracle_episodes": len(offline_rows), "validation_episodes": len(validation_rows),
            "dagger_real_episodes": len(real_rows), "dagger_shadow_artifacts": len(source_manifest["shadows"]),
            "dagger_shadow_windows": len(shadows), "target_episodes": len(target_rows),
            "source_oracle_windows": len(source), "source_validation_windows": len(source_validation),
            "source_dagger_real_windows": len(real), "target_windows": len(target),
            "target_controls": int(target_manifest["controls"]), "new_physics_calls": 0,
        },
    }


def _stack(dataset, indices):
    rows = [dataset[int(index)] for index in indices]
    return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}


def _offline_metrics(payload: dict, dataset, *, device: str, sample_limit: int) -> dict:
    solver = FADASolver.from_checkpoint(payload, device=device, task_action_binding=True)
    count = min(int(sample_limit), len(dataset))
    indices = np.unique(np.linspace(0, len(dataset) - 1, count, dtype=np.int64))
    planned, future_pred, inverse, targets, futures = [], [], [], [], []
    for start in range(0, len(indices), 64):
        batch = _stack(dataset, indices[start : start + 64])
        action, future = solver.predict(
            batch["history_obs"], batch["history_act"], batch["current_command"],
            history_valid_mask=batch["history_valid_mask"], return_obs=True,
        )
        backend = solver.backend
        obs = batch["history_obs"].to(backend.device)
        act = batch["history_act"].to(backend.device)
        cmd = batch["current_command"].to(backend.device)
        true_future = batch["future_observations"].to(backend.device)
        if backend.io_normalization:
            obs = (obs - backend.stats["obs"]["mean"]) / backend.stats["obs"]["std"]
            act = (act - backend.stats["act"]["mean"]) / backend.stats["act"]["std"]
            cmd = (cmd - backend.stats["cmd"]["mean"]) / backend.stats["cmd"]["std"]
            true_future = (true_future - backend.stats["obs"]["mean"]) / backend.stats["obs"]["std"]
        idm_cmd = cmd if solver.model.idm_use_current_command_for_history else None
        with torch.inference_mode():
            tf_action = solver.model.predict_idm_actions(
                obs, act, idm_cmd, true_future,
                history_valid_mask=batch["history_valid_mask"].to(backend.device),
            )
        if backend.io_normalization:
            tf_action = tf_action * backend.stats["act"]["std"] + backend.stats["act"]["mean"]
        planned.append(action.detach().cpu().numpy())
        future_pred.append(future.detach().cpu().numpy())
        inverse.append(tf_action.detach().cpu().numpy())
        targets.append(batch["target_actions"].numpy())
        futures.append(batch["future_observations"].numpy())
    planned, future_pred, inverse, targets, futures = map(np.concatenate, (planned, future_pred, inverse, targets, futures))
    act_scale = solver.backend.stats["act"]["std"].detach().cpu().numpy()
    obs_scale = solver.backend.stats["obs"]["std"].detach().cpu().numpy()
    return {
        "windows": int(len(indices)),
        "planner_future_normalized_mse": float(np.mean(np.square((future_pred - futures) / obs_scale))),
        "teacher_forced_idm_normalized_action_mse": float(np.mean(np.square((inverse - targets) / act_scale))),
        "end_to_end_normalized_action_mse": float(np.mean(np.square((planned - targets) / act_scale))),
    }


def _source_repaired_checkpoint(trainer, initial: dict, config: dict) -> dict:
    trained = trainer.checkpoint()
    payload = deepcopy(initial)
    payload["model_state_dict"] = trained["model_state_dict"]
    payload.pop("planner_state_dict", None)
    payload.pop("idm_state_dict", None)
    old = int(initial.get("genedynamics_task_adapter", {}).get("updates", 0))
    payload["genedynamics_task_adapter"].update(
        planner_obs_loss_coef=float(config["source_training"]["planner_obs_loss_coef"]),
        updates=old + int(config["source_training"]["updates"]),
        repair_pilot={
            "version": VERSION, "phase": "source_oracle_and_dagger_planner_idm",
            "fresh_optimizers": True, "fixed_last_update": True,
        },
    )
    return payload


def _state_equal(left: dict, right: dict) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key].cpu(), right[key].cpu()) for key in left)


def _normalization_equal(left: dict, right: dict) -> bool:
    if left.keys() != right.keys():
        return False
    return all(
        torch.equal(left[key].cpu(), right[key].cpu()) if torch.is_tensor(left[key]) else left[key] == right[key]
        for key in left
    )


def _admission_inputs(config: dict):
    if config.get("version") != CONFIG_VERSION or config.get("task") not in TASKS:
        raise ValueError("unknown FADA repair pilot config or unsupported task")
    checkpoint_path = _checked_file(config["source_checkpoint"], "source checkpoint")
    qualification_path = _checked_file(config["source_qualification"], "source qualification")
    attribution_path = _checked_file(config["closed_loop_attribution"], "closed-loop attribution")
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = source.get("genedynamics_task_adapter", {})
    if (int(source.get("obs_dim", -1)) != 35 or int(source.get("cmd_dim", -1)) != 67
            or metadata.get("source_batch_routing") != ROUTING_VERSION
            or float(metadata.get("planner_obs_loss_coef", -1)) != 0.0):
        raise ValueError("pilot requires the admitted 35D/67D zero-future-loss native routed source")
    task_name = task_spec(config["task"], robot="panda").name
    if metadata.get("task_name") != task_name:
        raise ValueError("source checkpoint task differs from pilot task")
    qualification = json.loads(qualification_path.read_text())
    if qualification.get("status") != "complete" or qualification.get("routing_version") != ROUTING_VERSION:
        raise ValueError("source qualification is incomplete or uses another routing contract")
    source_strategy = str(config["source_training"].get("strategy", SOURCE_REPAIR))
    if source_strategy not in {SOURCE_REPAIR, SOURCE_PRESERVE}:
        raise ValueError("unsupported FADA source preparation strategy")
    if source_strategy == SOURCE_PRESERVE:
        positive_path = _checked_file(
            config["source_positive_control"], "source positive control",
        )
        positive = json.loads(positive_path.read_text())
        counts = positive.get("counts", {})
        if (
            positive.get("status") != "complete"
            or positive.get("version") != "transfer-source-positive-control-v1"
            or positive.get("task") != config["task"]
            or positive.get("method") != "fada"
            or positive.get("actor_role") != "qualified_source_reference"
            or positive.get("source_only") is not True
            or int(positive.get("target_physical_execution_count", -1)) != 0
            or positive.get("checkpoint", {}).get("sha256")
            != config["source_checkpoint"]["sha256"]
            or int(counts.get("requested", -1)) != 8
            or int(counts.get("passed", -1)) != 8
            or int(counts.get("successes", -1)) != 8
            or int(counts.get("constraint_violations", -1)) != 0
            or int(counts.get("execution_errors", -1)) != 0
            or int(counts.get("mode_matches", -1)) != 8
            or positive.get("evidence_audits_passed") is not True
        ):
            raise ValueError(
                "source-preserving target adaptation requires an exact fresh 8/8 source control"
            )
    attribution = json.loads(attribution_path.read_text())
    replay = attribution.get("closed_loop_divergence", {}).get("actor_replay", {})
    if (attribution.get("task") != config["task"] or attribution.get("robot") != "xarm7"
            or not replay.get("matching_checkpoint_and_interface")):
        raise ValueError("pilot requires a CPU exact-replay attribution for the same target task")
    return source, attribution


def admit_repair_pilot(config: dict) -> dict:
    """CPU-only SHA, schema, split, mechanism, and capacity admission."""
    source, attribution = _admission_inputs(config)
    data = _datasets(config, source)
    source_cfg, target_cfg = config["source_training"], config["target_adaptation"]
    source_strategy = str(source_cfg.get("strategy", SOURCE_REPAIR))
    if int(target_cfg.get("updates", 0)) <= 0:
        raise ValueError("pilot requires fixed positive target updates")
    if source_strategy == SOURCE_REPAIR and (
        float(source_cfg.get("planner_obs_loss_coef", 0.0)) <= 0
        or int(source_cfg.get("updates", 0)) <= 0
    ):
        raise ValueError("source repair requires positive planner supervision and updates")
    if source_strategy == SOURCE_PRESERVE and (
        float(source_cfg.get("planner_obs_loss_coef", -1.0)) != 0.0
        or int(source_cfg.get("updates", -1)) != 0
    ):
        raise ValueError("source preservation requires zero source updates and zero new planner loss")
    if target_cfg.get("finetune_method") not in {"lora", "full"}:
        raise ValueError("target phase must use the released IDM LoRA or full-IDM path")
    initial = {
        "planner_future_normalized_mse": attribution["planner_future_prediction"]["normalized_mse"],
        "teacher_forced_idm_normalized_action_mse": attribution["idm_teacher_forced_prediction"]["normalized_action_mse"],
        "end_to_end_normalized_action_mse": attribution["end_to_end_planner_idm_prediction"]["normalized_action_mse"],
    }
    return {
        "version": VERSION,
        "status": "admitted",
        "task": config["task"], "robot": "xarm7",
        "source_checkpoint": {**config["source_checkpoint"], "updates": source["genedynamics_task_adapter"]["updates"]},
        "diagnostic_rationale": initial,
        "data": data["accounting"],
        "source_strategy": source_strategy,
        "source_positive_control": deepcopy(config.get("source_positive_control")),
        "mechanism": (
            [
                "qualified oracle source checkpoint",
                "admitted real-trajectory and oracle-shadow DAgger pools train planner and IDM",
                "positive native planner future-observation supervision",
                "67D source checkpoint lifted to the task-bound 199D public target chart",
                "planner and normalization frozen while only IDM adapts on reward-free B-log",
            ]
            if source_strategy == SOURCE_REPAIR
            else [
                "qualified oracle source checkpoint",
                "fresh eight-request source closed-loop control passes without target access",
                "source checkpoint preserved byte-for-byte with zero source updates",
                "67D source checkpoint lifted to the task-bound 199D public target chart",
                "planner and normalization frozen while only IDM adapts on reward-free B-log",
            ]
        ),
        "training": {"source": deepcopy(source_cfg), "target": deepcopy(target_cfg)},
        "stop_rule": {
            "selection": "fixed last update; no reward, success, target evaluation, or retrospective checkpoint selection",
            "hard_abort": ["non-finite loss", "declared wall-time ceiling", "planner or normalization changes during target adaptation"],
            "source_updates": int(source_cfg["updates"]), "target_updates": int(target_cfg["updates"]),
        },
        "output": config["output"],
        "new_physics_calls": 0,
    }


def run_repair_pilot(config: dict, *, device: str = "cuda:0") -> dict:
    """Execute one fixed-compute candidate; caller owns any later rollout."""
    admitted = admit_repair_pilot(config)
    source, _ = _admission_inputs(config)
    data = _datasets(config, source)
    source_cfg = config["source_training"]
    source_strategy = str(source_cfg.get("strategy", SOURCE_REPAIR))
    torch.manual_seed(int(config["seed"])); np.random.seed(int(config["seed"]))
    before_source = _offline_metrics(source, data["source_validation"], device=device,
                                     sample_limit=int(config["evaluation_windows"]))
    source_curve, started = [], time.monotonic()
    if source_strategy == SOURCE_REPAIR:
        statistics = {
            name: deepcopy(source[name])
            for name in ("obs_norm_stats", "action_norm_stats", "command_norm_stats")
        }
        trainer = NativeRoutedTaskPlannerIDMTrainer(
            source["cfg"], statistics, device=device,
            learning_rate=float(source_cfg["learning_rate"]),
            weight_decay=float(source_cfg["weight_decay"]),
            planner_obs_loss_coef=float(source_cfg["planner_obs_loss_coef"]),
        )
        trainer.model.load_state_dict(source["model_state_dict"], strict=True)
        trainer.steps = int(source["genedynamics_task_adapter"].get("updates", 0))
        for update in range(1, int(source_cfg["updates"]) + 1):
            bundle = sample_native_source_batches(
                data["source"], data["real"], data["shadows"],
                int(source_cfg["batch_size"]),
                offline_ratio=float(source_cfg["offline_ratio"]),
                online_real_ratio=float(source_cfg["online_real_ratio"]),
            )
            metrics = trainer.step(bundle)
            if not all(np.isfinite(value) for value in metrics.values()):
                raise RuntimeError("non-finite FADA source repair loss")
            if time.monotonic() - started > float(config["wall_seconds"]):
                raise TimeoutError("FADA repair pilot exceeded its declared wall-time ceiling")
            if (
                update % int(source_cfg["log_interval"]) == 0
                or update == int(source_cfg["updates"])
            ):
                source_curve.append({
                    "update": update, "metrics": metrics,
                    "source_counts": trainer.last_source_counts,
                })
        repaired = _source_repaired_checkpoint(trainer, source, config)
    else:
        repaired = deepcopy(source)
    after_source = _offline_metrics(repaired, data["source_validation"], device=device,
                                    sample_limit=int(config["evaluation_windows"]))

    target_payload = _bind_target_checkpoint(repaired, config)
    before_target = _offline_metrics(target_payload, data["target"], device=device,
                                     sample_limit=int(config["evaluation_windows"]))
    target_cfg = config["target_adaptation"]
    adaptation = IDMAdaptationConfig(**{key: target_cfg[key] for key in (
        "finetune_method", "rank", "alpha", "dropout", "target_scope", "learning_rate",
        "weight_decay", "grad_clip", "horizon_gamma",
    )})
    adapter = FADAIDMAdapter(target_payload, device=device, config=adaptation)
    original_planner = deepcopy(adapter.model.planner.state_dict())
    normalization = deepcopy({name: target_payload[name] for name in
                              ("obs_norm_stats", "action_norm_stats", "command_norm_stats")})
    target_curve = []
    for update in range(1, int(target_cfg["updates"]) + 1):
        metrics = adapter.step(sample_dataset(data["target"], int(target_cfg["batch_size"])))
        if not all(np.isfinite(value) for value in metrics.values()):
            raise RuntimeError("non-finite FADA target IDM loss")
        if time.monotonic() - started > float(config["wall_seconds"]):
            raise TimeoutError("FADA repair pilot exceeded its declared wall-time ceiling")
        if update % int(target_cfg["log_interval"]) == 0 or update == int(target_cfg["updates"]):
            target_curve.append({"update": update, "metrics": metrics})
    final = adapter.export_checkpoint()
    planner_frozen = _state_equal(original_planner, adapter.model.planner.state_dict())
    normalization_frozen = all(_normalization_equal(normalization[name], final[name]) for name in normalization)
    if not planner_frozen or not normalization_frozen:
        raise RuntimeError("FADA target adaptation changed the frozen planner or normalization")
    provenance_key = (
        "repair_pilot" if source_strategy == SOURCE_REPAIR
        else "source_preserving_target_adaptation"
    )
    final["genedynamics_task_adapter"][provenance_key] = {
        "version": VERSION, "source_updates": int(source_cfg["updates"]),
        "source_strategy": source_strategy,
        "target_idm_updates": int(target_cfg["updates"]), "fixed_last_update": True,
    }
    after_target = _offline_metrics(final, data["target"], device=device,
                                    sample_limit=int(config["evaluation_windows"]))
    thresholds = config["acceptance"]
    checks = {
        "target_teacher_forced_idm_improved": after_target["teacher_forced_idm_normalized_action_mse"]
            <= before_target["teacher_forced_idm_normalized_action_mse"] * float(thresholds["maximum_target_idm_mse_ratio"]),
        "planner_frozen_during_target_adaptation": planner_frozen,
        "normalization_frozen_during_target_adaptation": normalization_frozen,
    }
    if source_strategy == SOURCE_REPAIR:
        checks.update({
            "source_planner_future_improved": after_source["planner_future_normalized_mse"]
                <= before_source["planner_future_normalized_mse"] * (1.0 - float(thresholds["minimum_source_planner_improvement_fraction"])),
            "source_teacher_forced_idm_preserved": after_source["teacher_forced_idm_normalized_action_mse"]
                <= before_source["teacher_forced_idm_normalized_action_mse"] * float(thresholds["maximum_source_idm_mse_ratio"]),
        })
    else:
        checks["source_model_preserved_before_target_binding"] = _state_equal(
            source["model_state_dict"], repaired["model_state_dict"],
        ) and all(
            _normalization_equal(source[name], repaired[name])
            for name in ("obs_norm_stats", "action_norm_stats", "command_norm_stats")
        )
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    if source_strategy == SOURCE_PRESERVE:
        shutil.copyfile(
            _checked_file(config["source_checkpoint"], "source checkpoint"),
            output / "source_checkpoint.pt",
        )
        checks["source_checkpoint_bytes_preserved"] = (
            _sha256(output / "source_checkpoint.pt")
            == config["source_checkpoint"]["sha256"]
        )
    else:
        torch.save(repaired, output / "source_checkpoint.pt")
    # Preserve the deployable target-chart checkpoint before any target IDM
    # update.  This is the true zero-target-data point for the fixed budget
    # curve; the 67D source checkpoint cannot be loaded by the 199D target
    # actor directly.
    torch.save(target_payload, output / "target_initial_checkpoint.pt")
    torch.save(final, output / "checkpoint.pt")
    report = {
        **admitted, "status": "complete", "device": str(device),
        "source": {
            "strategy": source_strategy,
            "before": before_source, "after": after_source, "curve": source_curve,
        },
        "target": {"before": before_target, "after": after_target, "curve": target_curve,
                   "planner_frozen": planner_frozen, "normalization_frozen": normalization_frozen},
        "acceptance": {"checks": checks, "candidate_for_closed_loop_evaluation": all(checks.values())},
        "artifacts": {
            "source_checkpoint": {"path": str(output / "source_checkpoint.pt"), "sha256": _sha256(output / "source_checkpoint.pt")},
            "target_initial_checkpoint": {
                "path": str(output / "target_initial_checkpoint.pt"),
                "sha256": _sha256(output / "target_initial_checkpoint.pt"),
            },
            "checkpoint": {"path": str(output / "checkpoint.pt"), "sha256": _sha256(output / "checkpoint.pt")},
        },
        "wall_seconds": time.monotonic() - started,
        "new_physics_calls": 0,
        "limitation": "offline pilot acceptance only; task performance requires a separately declared closed-loop rollout",
    }
    return report


__all__ = ["CONFIG_VERSION", "VERSION", "admit_repair_pilot", "run_repair_pilot"]
