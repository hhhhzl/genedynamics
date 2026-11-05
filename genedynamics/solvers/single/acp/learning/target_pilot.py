"""GPU training runner for an admitted ACP target-task demonstration bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import resource
import time
from typing import Any

import numpy as np
import torch

from genedynamics.core.artifacts import atomic_json
from genedynamics.learning.datasets.transfer.target import PublicTargetTrajectory
from .target_task_training import PublicACPTaskDataset
from .task_training import build_policy, model_configuration, sampling_profile
from .trainer import ACPTrainer, deterministic_runtime


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_plan(config: dict[str, Any], demonstration_report: dict[str, Any]) -> dict[str, Any]:
    profile = config["gpu_training"]
    same_task = demonstration_report.get("task") == config.get("task")
    admitted = bool(
        same_task
        and demonstration_report.get("gpu_training_admitted", False)
        and demonstration_report.get("train_validation_disjoint", False)
        and demonstration_report["splits"]["fit"]["windows"] > 0
        and demonstration_report["splits"]["validation"]["windows"] > 0
    )
    return {
        "task": config["task"], "admitted": admitted,
        "same_task": same_task,
        "fit_windows": demonstration_report.get("splits", {}).get("fit", {}).get("windows", 0),
        "validation_windows": demonstration_report.get("splits", {}).get("validation", {}).get("windows", 0),
        "device": profile["device"], "expected_physical_uuid": profile["expected_physical_uuid"],
        "updates": int(profile["max_updates"]), "batch_size": int(profile["batch_size"]),
        "estimated_wall_seconds": profile["estimated_wall_seconds"],
        "estimate_basis": profile["estimate_basis"],
    }


def _dataset(report: dict[str, Any], purpose: str, profile: dict[str, Any]) -> PublicACPTaskDataset:
    rows = []
    for receipt in report["splits"][purpose]["receipts"]:
        path = Path(receipt["path"])
        row = PublicTargetTrajectory.load(path)
        if row.digest != receipt["trajectory_sha256"]:
            raise ValueError("ACP demonstration trajectory changed after preflight")
        rows.append(row)
    return PublicACPTaskDataset(
        rows, purpose=purpose, history_len=int(profile["history_len"]),
        horizon=int(profile["horizon"]), action_chart=profile["action_chart"],
        action_down_sample_steps=int(profile.get("action_down_sample_steps", 1)),
    )


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {name: _to_device(child, device) for name, child in value.items()}
    return value


def heldout_reconstruction_plan(
    config: dict[str, Any], demonstration_report: dict[str, Any],
    training_report: dict[str, Any], *, output: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze the selected EMA checkpoint and disjoint validation windows."""
    base = training_plan(config, demonstration_report)
    if not base["admitted"] or training_report.get("status") != "complete":
        raise RuntimeError("ACP held-out profile requires admitted data and complete training")
    validations = list(training_report.get("validations", ()))
    if not validations:
        raise ValueError("ACP training report has no validation checkpoints")
    selected = min(validations, key=lambda row: (float(row["loss"]), int(row["update"])))
    checkpoint = Path(selected["checkpoint"]).resolve()
    if not checkpoint.is_file() or _sha256(checkpoint) != selected["checkpoint_sha256"]:
        raise ValueError("selected ACP checkpoint changed after training")
    profile = config["gpu_training"]
    count = min(int(profile["validation_windows"]), int(base["validation_windows"]))
    indices = np.unique(np.linspace(0, int(base["validation_windows"]) - 1, count, dtype=int))
    destination = Path(output or (Path(profile["output"]) / "heldout_native_reconstruction"))
    return {
        "version": "acp-heldout-native-reconstruction-v1",
        "status": "planned_unexecuted",
        "task": config["task"],
        "scope": "disjoint validation-window native19 physical reconstruction; not target task performance",
        "checkpoint": {
            "path": str(checkpoint), "sha256": selected["checkpoint_sha256"],
            "update": int(selected["update"]), "validation_loss": float(selected["loss"]),
        },
        "demonstration_report_sha256": training_report["demonstration_report_sha256"],
        "fit_windows": int(base["fit_windows"]),
        "validation_windows": int(base["validation_windows"]),
        "profiled_validation_indices": indices.tolist(),
        "profiled_windows": len(indices),
        "action_chart": profile["action_chart"],
        "action_down_sample_steps": int(profile.get("action_down_sample_steps", 1)),
        "seed_base": int(profile["seed"]) + 50000,
        "execution": {
            "device": profile["device"],
            "expected_physical_uuid": profile["expected_physical_uuid"],
        },
        "output": str(destination),
        "train_validation_disjoint": True,
    }


def profile_heldout_reconstruction(
    config: dict[str, Any], demonstration_report_path: str | Path,
    training_report_path: str | Path, *, output: str | Path | None = None,
) -> dict[str, Any]:
    """Run the frozen native19 reconstruction profile on physical GPU1."""
    demonstration_path = Path(demonstration_report_path).resolve()
    training_path = Path(training_report_path).resolve()
    demonstrations = json.loads(demonstration_path.read_text())
    training = json.loads(training_path.read_text())
    if _sha256(demonstration_path) != training.get("demonstration_report_sha256"):
        raise ValueError("ACP demonstration report changed after checkpoint training")
    plan = heldout_reconstruction_plan(
        config, demonstrations, training, output=output,
    )
    profile = config["gpu_training"]
    if torch.cuda.device_count() != 1 or not torch.cuda.is_available():
        raise RuntimeError("ACP held-out profile requires exactly one visible CUDA device")
    device = torch.device(profile["device"])
    actual_uuid = str(getattr(torch.cuda.get_device_properties(device), "uuid", ""))
    expected_uuid = str(profile["expected_physical_uuid"]).removeprefix("GPU-")
    if actual_uuid.lower() != expected_uuid.lower():
        raise RuntimeError(f"ACP profile exposed physical GPU {actual_uuid}, expected {expected_uuid}")
    destination = Path(plan["output"])
    if destination.exists():
        raise FileExistsError("ACP held-out profile output already exists; runs are immutable")
    destination.mkdir(parents=True)
    atomic_json(destination / "plan.json", plan)
    validation = _dataset(demonstrations, "validation", profile)
    payload = torch.load(plan["checkpoint"]["path"], map_location="cpu", weights_only=True)
    if (payload.get("task") != config["task"]
            or payload.get("action_chart") != plan["action_chart"]
            or int(payload.get("action_down_sample_steps", -1)) != plan["action_down_sample_steps"]):
        raise ValueError("selected ACP checkpoint metadata differs from held-out plan")
    policy = build_policy(
        payload["shape_meta"], int(payload["history_len"]),
        action_chart=payload["action_chart"],
    )
    policy.load_state_dict(payload["state_dicts"]["ema_model"], strict=True)
    policy.to(device).eval()
    statistics = sampling_profile(
        policy, validation, plan["profiled_validation_indices"],
        seed_base=plan["seed_base"],
    )
    report = {
        **plan, "status": "complete", "statistics": statistics,
        "validation_receipts": validation.receipts,
        "finite_metrics": True,
        "checkpoint_schema_loaded_strictly": True,
    }
    atomic_json(destination / "report.json", report)
    return report


def train_target_pilot(
    config: dict[str, Any], demonstration_report_path: str | Path,
    *, output: str | Path | None = None,
) -> dict[str, Any]:
    report_path = Path(demonstration_report_path).resolve()
    demonstrations = json.loads(report_path.read_text())
    plan = training_plan(config, demonstrations)
    if not plan["admitted"]:
        raise RuntimeError("ACP GPU pilot rejected by the qualified demonstration-data gate")
    profile = config["gpu_training"]
    if torch.cuda.device_count() != 1 or not torch.cuda.is_available():
        raise RuntimeError("ACP pilot requires exactly one visible CUDA device")
    device = torch.device(profile["device"])
    actual_uuid = str(getattr(torch.cuda.get_device_properties(device), "uuid", ""))
    expected_uuid = str(profile["expected_physical_uuid"]).removeprefix("GPU-")
    if actual_uuid.lower() != expected_uuid.lower():
        raise RuntimeError(f"ACP pilot exposed physical GPU {actual_uuid}, expected {expected_uuid}")
    capacity = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(
        float(profile["maximum_gpu_GiB"]) * 1024**3 / capacity, device,
    )
    output = Path(output or profile["output"])
    if output.exists():
        raise FileExistsError("ACP pilot output already exists; runs are immutable")
    output.mkdir(parents=True)
    deterministic_runtime()
    torch.set_num_threads(1)
    seed = int(profile["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    train = _dataset(demonstrations, "fit", profile)
    validation = _dataset(demonstrations, "validation", profile)
    meta, cfg = model_configuration(
        int(profile["history_len"]), int(profile["horizon"]), seed,
        shared_context=True,
        action_down_sample_steps=int(profile.get("action_down_sample_steps", 1)),
    )
    policy = build_policy(
        meta, int(profile["history_len"]), action_chart=profile["action_chart"],
    )
    policy.set_normalizer(train.normalizer(rotation_residual=True))
    trainer = ACPTrainer(
        policy, cfg, device=str(device), total_steps=int(profile["max_updates"]),
        optimizer_mode="all_trainable",
    )
    rng = np.random.default_rng(seed)
    begun = time.perf_counter()
    losses = []
    validations = []
    for update in range(1, int(profile["max_updates"]) + 1):
        indices = rng.integers(0, len(train), size=int(profile["batch_size"]))
        losses.append(trainer.train_step(_to_device(train.batch(indices), device)))
        if update % int(profile["evaluation_interval"]) == 0 or update == int(profile["max_updates"]):
            selected = np.unique(np.linspace(
                0, len(validation) - 1,
                min(int(profile["validation_windows"]), len(validation)), dtype=int,
            ))
            torch.manual_seed(seed + update)
            trainer.ema_model.eval()
            with torch.inference_mode():
                validation_loss = float(trainer.ema_model.compute_loss(
                    _to_device(validation.batch(selected), device), None,
                ).detach().cpu())
            checkpoint = output / f"ema_update_{update}.pt"
            torch.save({
                "state_dicts": {"ema_model": trainer.ema_model.state_dict()},
                "shape_meta": meta, "history_len": int(profile["history_len"]),
                "action_down_sample_steps": int(profile.get("action_down_sample_steps", 1)),
                "action_chart": profile["action_chart"], "variant": "target_public_teacher",
                "task": config["task"], "demonstration_report_sha256": _sha256(report_path),
            }, checkpoint)
            validations.append({
                "update": update, "loss": validation_loss,
                "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": _sha256(checkpoint),
            })
            atomic_json(output / "progress.json", {
                "status": "running", "plan": plan, "completed_updates": update,
                "recent_training_loss": losses[-1], "validations": validations,
            })
    trainer.save(output / "continuation.pt")
    final = {
        "status": "complete", "plan": plan, "completed_updates": trainer.global_step,
        "training_loss": {"first": losses[0], "last": losses[-1], "minimum": min(losses)},
        "validations": validations, "wall_seconds": time.perf_counter() - begun,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "rss_max_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "demonstration_report": str(report_path),
        "demonstration_report_sha256": _sha256(report_path),
    }
    atomic_json(output / "report.json", final)
    return final


__all__ = [
    "training_plan", "train_target_pilot", "heldout_reconstruction_plan",
    "profile_heldout_reconstruction",
]
