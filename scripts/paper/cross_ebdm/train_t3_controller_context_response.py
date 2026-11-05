#!/usr/bin/env python3
"""Train T3 physical-plus-controller-context response ensembles on GPU1."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from genedynamics.learning.datasets.transfer.target import (
    FOUR_TASK_VERSION, VERSION, PublicTargetTrajectory,
)
from genedynamics.solvers.single.cross_ebdm.learning.completion_value import (
    PublicQualifiedContextCompletionValueEnsemble,
    context_completion_value_features,
    response_endpoint_value,
)
from genedynamics.solvers.single.cross_ebdm.learning.context_response import (
    PublicControllerContextResponseEnsemble,
    PublicPhaseHeadControllerContextResponseEnsemble,
)
from genedynamics.solvers.single.cross_ebdm.learning.response import (
    DELTA_SCALE,
    PublicResponseSpec,
    response_windows,
)
from genedynamics.tasks.transfer.observation import PublicControllerMemory
from genedynamics.tasks.transfer.specs import task_spec


ROOT = Path(__file__).resolve().parents[3]
T3_MATING_CLEARANCE = float(
    task_spec("T3", robot="xarm7").config.mating_clearance
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "sha256": sha(path),
    }


def checked(value: dict[str, str], label: str) -> Path:
    path = Path(value["path"])
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file() or sha(path) != value["sha256"]:
        raise ValueError(f"{label} changed")
    return path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def save_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("version") not in {
        "t3-controller-context-response-training-v2",
        "t3-transition-aware-controller-context-response-training-v3",
        "t3-geometry-projected-controller-context-response-training-v4",
        "t3-phase-head-controller-context-response-training-v5",
    }:
        raise ValueError("unknown T3 response-training version")
    if config.get("status") != "frozen_unexecuted":
        raise ValueError("response-training plan is not frozen unexecuted")
    if config["data"]["episode_split"] != {
        "kind": "within_manifest_entry_modulo",
        "modulus": 4,
        "calibration_residue": 0,
        "fit_residues": [1, 2],
        "selection_residue": 3,
    }:
        raise ValueError("mode-complete response split changed")
    if set(config["training"]["instance_seeds"]) != {
        f"instance_{index:02d}" for index in range(1, 9)
    }:
        raise ValueError("eight response instance seeds required")
    normalized_completion = gate_aligned(config)
    if normalized_completion != (
        config["training"].get("completion_loss_units") == "horizon_fraction"
    ):
        raise ValueError("transition-aware completion loss units mismatch")
    if learned_phase_classifier(config) and (
        float(config["training"].get("phase_weight", 0.)) <= 0.
        or float(config["training"].get("transition_class_weight", 0.)) <= 1.
    ):
        raise ValueError("positive transition-aware phase weighting required")
    if geometry_projected(config) != (
        config["training"].get("endpoint_phase_projection")
        == "public_t3_geometry_v1"
    ):
        raise ValueError("public T3 endpoint phase projection contract mismatch")
    return config


def transition_aware(config: dict[str, Any]) -> bool:
    return (
        config["version"]
        == "t3-transition-aware-controller-context-response-training-v3"
    )


def geometry_projected(config: dict[str, Any]) -> bool:
    return (
        config["version"]
        == "t3-geometry-projected-controller-context-response-training-v4"
    )


def phase_head_model(config: dict[str, Any]) -> bool:
    return (
        config["version"]
        == "t3-phase-head-controller-context-response-training-v5"
    )


def learned_phase_classifier(config: dict[str, Any]) -> bool:
    return transition_aware(config) or phase_head_model(config)


def gate_aligned(config: dict[str, Any]) -> bool:
    return learned_phase_classifier(config) or geometry_projected(config)


def response_model_type(config: dict[str, Any]):
    return (
        PublicPhaseHeadControllerContextResponseEnsemble
        if phase_head_model(config) else PublicControllerContextResponseEnsemble
    )


def verify_device(config: dict[str, Any]) -> torch.device:
    expected = config["device"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected["process_visible_devices"]:
        raise RuntimeError("mode-complete response training requires CUDA_VISIBLE_DEVICES=1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one logical CUDA device required")
    properties = torch.cuda.get_device_properties(0)
    if str(properties.uuid).lower() != expected["physical_uuid"].removeprefix("GPU-").lower():
        raise RuntimeError("visible device is not admitted GPU1")
    return torch.device(expected["logical_device"])


def controller_trace(
    public: dict[str, np.ndarray], mode: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observer = PublicControllerMemory(task_spec("T3", robot="xarm7"), [mode])
    contexts, phases, counts = [], [], []
    for tick in range(len(public["tcp_position"])):
        context = observer.update({
            key: np.asarray(value)[tick:tick + 1]
            for key, value in public.items()
        })
        contexts.append(context[0])
        phases.append(int(observer.memory.phase[0]))
        counts.append(int(observer.memory.contact_count[0]))
    return (
        np.asarray(contexts, np.float32),
        np.asarray(phases, np.int64),
        np.asarray(counts, np.int64),
    )


def value_features(
    public: dict, ticks: np.ndarray, mode: int,
    contexts: np.ndarray, counts: np.ndarray,
) -> np.ndarray:
    result = context_completion_value_features(
        position=torch.from_numpy(np.asarray(public["tcp_position"])[ticks]),
        rotation=torch.from_numpy(np.asarray(public["tcp_rotation"])[ticks]),
        twist=torch.from_numpy(np.asarray(public["tcp_twist"])[ticks]),
        wrench=torch.from_numpy(np.asarray(public["measured_wrench"])[ticks]),
        goal_position=torch.from_numpy(np.asarray(public["goal_position"])[ticks]),
        goal_rotation=torch.from_numpy(np.asarray(public["goal_rotation"])[ticks]),
        controller_feedback=torch.from_numpy(
            np.asarray(public["controller_feedback"])[ticks]
        ),
        mode=torch.full((len(ticks),), mode, dtype=torch.long),
        controller_context=torch.from_numpy(contexts[ticks]),
        contact_count=torch.from_numpy(counts[ticks]),
    )
    return result.numpy().astype(np.float32, copy=False)


CURRENT_FIELDS = {
    "position": "tcp_position",
    "rotation": "tcp_rotation",
    "twist": "tcp_twist",
    "wrench": "measured_wrench",
    "goal_position": "goal_position",
    "goal_rotation": "goal_rotation",
    "controller_feedback": "controller_feedback",
}


def concatenate(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        raise ValueError("empty response dataset field")
    return np.concatenate(rows)


def load_dataset(config: dict[str, Any], spec: PublicResponseSpec) -> tuple[dict, dict]:
    split_spec = config["data"]["episode_split"]
    maximum = int(config["data"]["maximum_remaining_controls"])
    groups = {
        name: {
            "features": [], "commands": [], "targets": [], "modes": [],
            "phases": [], "controller_context": [], "value_features": [],
            **{field: [] for field in CURRENT_FIELDS},
        }
        for name in ("fit", "selection")
    }
    provenance = {"manifests": [], "episodes": {}, "counts": {}}
    for manifest_index, value in enumerate(config["data"]["public_log_manifests"]):
        manifest_path = checked(value, f"public-log manifest {manifest_index}")
        manifest = json.loads(manifest_path.read_text())
        if manifest["permission"] != "B_log" or manifest["reward"] or manifest["success_filter"]:
            raise ValueError("unfiltered public B_log required")
        for entry_index, entry in enumerate(manifest["entries"]):
            residue = entry_index % int(split_spec["modulus"])
            if residue == int(split_spec["calibration_residue"]):
                split = "calibration_excluded"
            elif residue == int(split_spec["selection_residue"]):
                split = "selection"
            elif residue in split_spec["fit_residues"]:
                split = "fit"
            else:
                raise ValueError("unassigned response episode residue")
            trajectory = PublicTargetTrajectory.load(entry["path"])
            if (trajectory.metadata["version"] not in {VERSION, FOUR_TASK_VERSION}
                    or trajectory.metadata.get("environment", "keyed_insertion_recovery") != "keyed_insertion_recovery"):
                raise ValueError("static T3 public trajectory required; fixture data needs its own training entry")
            if (
                trajectory.digest != entry["parent_sha256"]
                or trajectory.digest != entry["prefix_sha256"]
                or trajectory.task != "T3"
                or len(trajectory.requests) != int(entry["controls"])
            ):
                raise ValueError("public response trajectory changed")
            mode = int(trajectory.metadata["mode"])
            contexts, phases, counts = controller_trace(trajectory.public, mode)
            windows = response_windows(
                trajectory.public, trajectory.requests, body="xarm7", task="T3",
                calibration_context=[0, 0, 0], spec=spec,
            )
            end = len(trajectory.requests)
            keep = windows["ticks"] >= max(0, end - maximum)
            ticks = windows["ticks"][keep]
            key = f"manifest_{manifest_index:02d}/entry_{entry_index:04d}"
            provenance["episodes"][key] = {
                "digest": trajectory.digest,
                "mode": mode,
                "seed": int(trajectory.metadata["seed"]),
                "controls": end,
                "split": split,
                "response_pairs": len(ticks),
            }
            if split == "calibration_excluded":
                continue
            group = groups[split]
            group["features"].append(windows["features"][keep])
            group["commands"].append(windows["commands"][keep])
            future = ticks + spec.horizon
            context_delta = contexts[future] - contexts[ticks]
            normalized_contact = np.minimum(counts[future], 2).astype(np.float32) / 2.
            group["targets"].append(np.concatenate((
                windows["targets"][keep], context_delta,
                normalized_contact[:, None],
            ), axis=1).astype(np.float32, copy=False))
            group["modes"].append(np.full(len(ticks), mode, np.int64))
            group["phases"].append(phases[ticks])
            group["controller_context"].append(contexts[ticks])
            group["value_features"].append(value_features(
                trajectory.public, ticks, mode, contexts, counts,
            ))
            for target, source in CURRENT_FIELDS.items():
                group[target].append(np.asarray(trajectory.public[source])[ticks])
        provenance["manifests"].append(pointer(manifest_path))
    data = {}
    for split, rows in groups.items():
        data[split] = {name: concatenate(parts) for name, parts in rows.items()}
        provenance["counts"][split] = {
            "pairs": len(data[split]["modes"]),
            "pairs_by_mode": {
                str(mode): int((data[split]["modes"] == mode).sum())
                for mode in (0, 1)
            },
            "pairs_by_phase": {
                str(phase): int((data[split]["phases"] == phase).sum())
                for phase in range(4)
            },
        }
        if set(np.unique(data[split]["modes"])) != {0, 1}:
            raise ValueError(f"{split} response data is not mode-complete")
    return data, provenance


def to_device(data: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: torch.as_tensor(value, device=device) for name, value in data.items()}


def balanced_indices(
    modes: torch.Tensor, phases: torch.Tensor, count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if count % 8:
        raise ValueError("mode-and-phase-balanced response batch must divide by eight")
    selected = []
    for mode in (0, 1):
        for phase in range(4):
            available = torch.nonzero(
                (modes == mode) & (phases == phase), as_tuple=False,
            ).reshape(-1)
            if len(available) == 0:
                raise ValueError("empty mode-and-phase response cell")
            ids = torch.randint(
                len(available), (count // 8,), generator=generator,
                device=modes.device,
            )
            selected.append(available[ids])
    rows = torch.cat(selected)
    return rows[torch.randperm(len(rows), generator=generator, device=modes.device)]


def current(data: dict[str, torch.Tensor], ids: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: data[name][ids] for name in CURRENT_FIELDS}


def completion_loss(
    model: PublicControllerContextResponseEnsemble,
    prediction: torch.Tensor,
    data: dict[str, torch.Tensor],
    ids: torch.Tensor,
    value_model: PublicQualifiedContextCompletionValueEnsemble,
    horizon: int,
    *, horizon_normalized: bool,
    geometry_phase: bool,
) -> torch.Tensor:
    losses = []
    maximum = value_model.spec.maximum_remaining_controls
    current_values = value_model(data["value_features"][ids])
    physical, endpoint_context, endpoint_contact = (
        model.endpoint(
            prediction, data["controller_context"][ids],
            **(
                {
                    "current_position": data["position"][ids],
                    "current_rotation": data["rotation"][ids],
                    "mating_clearance": T3_MATING_CLEARANCE,
                }
                if geometry_phase else {}
            ),
        )
    )
    modes = data["modes"][ids]
    for mode in (0, 1):
        mask = modes == mode
        endpoint = response_endpoint_value(
            value_model, physical[:, mask],
            current={name: value[mask] for name, value in current(data, ids).items()},
            mode=mode,
            controller_context=endpoint_context[:, mask],
            contact_count=endpoint_contact[:, mask],
        )
        gains = current_values[:, None, mask] - endpoint
        if horizon_normalized:
            gains = gains * (maximum / horizon)
            truth = torch.ones_like(gains)
        else:
            truth = gains.new_full(gains.shape, horizon / maximum)
        losses.append(torch.nn.functional.smooth_l1_loss(gains, truth))
    return torch.stack(losses).mean()


def cell_metrics(
    model: PublicControllerContextResponseEnsemble,
    value_model: PublicQualifiedContextCompletionValueEnsemble,
    data: dict[str, torch.Tensor],
    mode: int,
    phase: int,
    config: dict[str, Any],
) -> dict[str, float | int]:
    ids = torch.nonzero(
        (data["modes"] == mode) & (data["phases"] == phase),
        as_tuple=False,
    ).reshape(-1)
    prediction = model(
        data["features"][ids], data["commands"][ids],
        data["controller_context"][ids],
    )
    target = data["targets"][ids]
    physical_endpoint, endpoint_context, endpoint_contact = model.endpoint(
        prediction, data["controller_context"][ids],
        **(
            {
                "current_position": data["position"][ids],
                "current_rotation": data["rotation"][ids],
                "mating_clearance": T3_MATING_CLEARANCE,
            }
            if geometry_projected(config) else {}
        ),
    )
    mean = physical_endpoint.mean(0)
    residual = mean - target[..., :20]
    scale = residual.new_tensor(DELTA_SCALE)
    physical = residual[..., :18] * scale
    endpoint = response_endpoint_value(
        value_model, physical_endpoint, current=current(data, ids), mode=mode,
        controller_context=endpoint_context,
        contact_count=endpoint_contact,
    )
    maximum = value_model.spec.maximum_remaining_controls
    current_values = value_model(data["value_features"][ids])
    pair_gain = (current_values[:, None, :] - endpoint) * maximum
    conservative = pair_gain.amin((0, 1))
    truth = conservative.new_full(conservative.shape, float(model.spec.horizon))
    optimism = (conservative - truth).clamp(min=0)
    quantile = float(config["selection"]["one_sided_optimism_quantile"])
    target_endpoint_context = (
        data["controller_context"][ids] + target[..., 20:73]
    )
    predicted_phase = endpoint_context.mean(0)[..., 22:26].argmax(-1)
    target_phase = target_endpoint_context[..., 22:26].argmax(-1)
    return {
        "pairs": int(len(ids)),
        "normalized_response_rmse": float(residual.square().mean().sqrt()),
        "position_delta_vector_mae_m": float(physical[..., :3].norm(dim=-1).mean()),
        "rotation_delta_vector_mae_rad": float(physical[..., 3:6].norm(dim=-1).mean()),
        "twist_delta_vector_mae": float(physical[..., 6:12].norm(dim=-1).mean()),
        "wrench_delta_vector_mae": float(physical[..., 12:18].norm(dim=-1).mean()),
        "controller_context_delta_mae": float(
            (prediction[..., 20:73].mean(0) - target[..., 20:73]).abs().mean()
        ),
        "controller_phase_accuracy": float(
            (predicted_phase == target_phase).float().mean()
        ),
        "normalized_contact_count_mae": float(
            (endpoint_contact.mean(0) - target[..., 73]).abs().mean()
        ),
        "completion_conservative_gain_mean_controls": float(conservative.mean()),
        "completion_conservative_mae_controls": float((conservative - truth).abs().mean()),
        "completion_conservative_direction_accuracy": float((conservative > 0).float().mean()),
        "completion_optimism_calibration_controls": float(torch.quantile(optimism, quantile)),
    }


def validation_metrics(
    model: PublicControllerContextResponseEnsemble,
    value_model: PublicQualifiedContextCompletionValueEnsemble,
    data: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> dict:
    model.eval()
    with torch.no_grad():
        by_cell = {
            f"mode_{mode}_phase_{phase}": cell_metrics(
                model, value_model, data, mode, phase, config,
            )
            for mode in (0, 1) for phase in range(4)
        }
    score = max(
        row["completion_conservative_mae_controls"]
        + 1000. * row["position_delta_vector_mae_m"]
        + 100. * row["rotation_delta_vector_mae_rad"]
        + 10. * row["controller_context_delta_mae"]
        for row in by_cell.values()
    )
    return {"by_cell": by_cell, "score": score}


def acceptance(config: dict, metrics: dict) -> dict:
    thresholds = config["acceptance"]
    checks = {}
    for cell, row in metrics["by_cell"].items():
        prefix = cell
        checks[prefix + "_position"] = (
            row["position_delta_vector_mae_m"]
            <= thresholds["maximum_position_delta_vector_mae_m"]
        )
        checks[prefix + "_rotation"] = (
            row["rotation_delta_vector_mae_rad"]
            <= thresholds["maximum_rotation_delta_vector_mae_rad"]
        )
        checks[prefix + "_context"] = (
            row["controller_context_delta_mae"]
            <= thresholds["maximum_controller_context_delta_mae"]
        )
        checks[prefix + "_phase"] = (
            row["controller_phase_accuracy"]
            >= thresholds["minimum_controller_phase_accuracy"]
        )
        checks[prefix + "_contact"] = (
            row["normalized_contact_count_mae"]
            <= thresholds["maximum_normalized_contact_count_mae"]
        )
        checks[prefix + "_completion_mae"] = (
            row["completion_conservative_mae_controls"]
            <= thresholds["maximum_completion_conservative_mae_controls"]
        )
        checks[prefix + "_completion_direction"] = (
            row["completion_conservative_direction_accuracy"]
            >= thresholds["minimum_completion_conservative_direction_accuracy"]
        )
        checks[prefix + "_completion_optimism"] = (
            row["completion_optimism_calibration_controls"]
            <= thresholds["maximum_completion_calibrated_optimism_controls"]
        )
    return {"passed": all(checks.values()), "checks": checks, "thresholds": thresholds}


def selection_score(config: dict, metrics: dict) -> float:
    thresholds = config["acceptance"]
    values = []
    for row in metrics["by_cell"].values():
        values.extend((
            row["position_delta_vector_mae_m"]
            / thresholds["maximum_position_delta_vector_mae_m"],
            row["rotation_delta_vector_mae_rad"]
            / thresholds["maximum_rotation_delta_vector_mae_rad"],
            row["controller_context_delta_mae"]
            / thresholds["maximum_controller_context_delta_mae"],
            row["normalized_contact_count_mae"]
            / thresholds["maximum_normalized_contact_count_mae"],
            row["completion_conservative_mae_controls"]
            / thresholds["maximum_completion_conservative_mae_controls"],
            row["completion_optimism_calibration_controls"]
            / thresholds["maximum_completion_calibrated_optimism_controls"],
            thresholds["minimum_completion_conservative_direction_accuracy"]
            / max(row["completion_conservative_direction_accuracy"], 1e-12),
            thresholds["minimum_controller_phase_accuracy"]
            / max(row["controller_phase_accuracy"], 1e-12),
        ))
    return float(max(values))


def one_step(
    model: PublicControllerContextResponseEnsemble,
    value_model: PublicQualifiedContextCompletionValueEnsemble,
    data: dict[str, torch.Tensor],
    ids: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    completion_weight: float,
    context_weight: float,
    phase_weight: float,
    transition_class_weight: float,
    horizon_normalized_completion: bool,
    geometry_phase_projection: bool,
) -> dict[str, float]:
    model.train()
    prediction = model(
        data["features"][ids], data["commands"][ids],
        data["controller_context"][ids],
    )
    target = data["targets"][ids]
    physical_losses = []
    context_losses = []
    for member in prediction:
        bootstrap = torch.randint(
            len(ids), (len(ids),), generator=generator, device=ids.device,
        )
        physical_losses.append(torch.nn.functional.mse_loss(
            member[bootstrap, :20], target[bootstrap, :20],
        ))
        context_losses.append(torch.nn.functional.mse_loss(
            member[bootstrap, 20:74], target[bootstrap, 20:74],
        ))
    response_loss = torch.stack(physical_losses).mean()
    context_loss = torch.stack(context_losses).mean()
    current_phase = data["controller_context"][ids, 22:26].argmax(-1)
    target_phase = (
        data["controller_context"][ids, 22:26] + target[:, 42:46]
    ).argmax(-1)
    endpoint_phase_logits = (
        prediction[..., 74:78]
        if prediction.shape[-1] == 78 else (
            data["controller_context"][ids, 22:26].unsqueeze(0)
            + prediction[..., 42:46]
        )
    )
    weights = torch.where(
        target_phase != current_phase,
        target_phase.new_full(target_phase.shape, transition_class_weight,
                              dtype=prediction.dtype),
        target_phase.new_ones(target_phase.shape, dtype=prediction.dtype),
    )
    phase_loss = torch.stack([
        (
            torch.nn.functional.cross_entropy(member, target_phase, reduction="none")
            * weights
        ).mean()
        for member in endpoint_phase_logits
    ]).mean()
    value_loss = completion_loss(
        model, prediction, data, ids, value_model, model.spec.horizon,
        horizon_normalized=horizon_normalized_completion,
        geometry_phase=geometry_phase_projection,
    )
    loss = (
        response_loss + context_weight * context_loss
        + phase_weight * phase_loss + completion_weight * value_loss
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "response_loss": float(response_loss.detach()),
        "context_loss": float(context_loss.detach()),
        "phase_loss": float(phase_loss.detach()),
        "completion_loss": float(value_loss.detach()),
    }


def run_preflight(config: dict[str, Any], config_path: Path) -> None:
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU response preflight must not initialize CUDA")
    spec = PublicResponseSpec(**{
        key: tuple(value) if key == "hidden" else value
        for key, value in config["model"].items()
    })
    data, provenance = load_dataset(config, spec)
    fit = to_device(data["fit"], torch.device("cpu"))
    value_path = checked(config["completion_value_checkpoint"], "completion checkpoint")
    value_model = PublicQualifiedContextCompletionValueEnsemble.from_checkpoint(
        torch.load(value_path, map_location="cpu", weights_only=True), "cpu",
    ).eval()
    for parameter in value_model.parameters():
        parameter.requires_grad_(False)
    seed = min(config["training"]["instance_seeds"].values())
    torch.manual_seed(seed)
    model = response_model_type(config)(spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    ids = balanced_indices(fit["modes"], fit["phases"], 32, generator)
    losses = one_step(
        model, value_model, fit, ids, optimizer, generator,
        float(config["training"]["completion_weight"]),
        float(config["training"]["context_weight"]),
        float(config["training"].get("phase_weight", 0.)),
        float(config["training"].get("transition_class_weight", 1.)),
        gate_aligned(config),
        geometry_projected(config),
    )
    passed = all(np.isfinite(value) for value in losses.values())
    output = ROOT / config["preflight_output"]
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": "t3-controller-context-response-preflight-v2",
        "status": "complete" if passed else "failed",
        "passed": passed,
        "gpu_calls": 0,
        "losses": losses,
        "data": provenance,
        "artifacts": {
            "config": pointer(config_path), "runner": pointer(Path(__file__).resolve()),
            "completion_value_checkpoint": pointer(value_path),
        },
        "target_reward_or_truth_opened": False,
        "confirmation_states_opened": 0,
    }
    write_json(output / "report.json", report)
    print(json.dumps({
        "status": report["status"], "output": str(output / "report.json"),
        "sha256": sha(output / "report.json"),
    }, indent=2))
    if not passed:
        raise AssertionError("mode-complete response preflight failed")


def run_training(config: dict[str, Any], config_path: Path) -> None:
    device = verify_device(config)
    preflight_path = checked(config["preflight_report"], "response preflight")
    preflight = json.loads(preflight_path.read_text())
    if not preflight.get("passed") or preflight.get("gpu_calls") != 0:
        raise ValueError("passed CPU response preflight required")
    spec = PublicResponseSpec(**{
        key: tuple(value) if key == "hidden" else value
        for key, value in config["model"].items()
    })
    raw, provenance = load_dataset(config, spec)
    fit = to_device(raw["fit"], device)
    selection = to_device(raw["selection"], device)
    value_path = checked(config["completion_value_checkpoint"], "completion checkpoint")
    value_model = PublicQualifiedContextCompletionValueEnsemble.from_checkpoint(
        torch.load(value_path, map_location="cpu", weights_only=True), device,
    ).eval()
    for parameter in value_model.parameters():
        parameter.requires_grad_(False)
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", config)
    begun = time.monotonic()
    instances = {}
    for name, seed in sorted(config["training"]["instance_seeds"].items()):
        seed = int(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = response_model_type(config)(spec).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate)
        generator = torch.Generator(device=device).manual_seed(seed)
        curves = []
        best_state = None
        best_metrics = None
        best_update = None
        best_score = None
        for update in range(1, int(config["training"]["updates"]) + 1):
            ids = balanced_indices(
                fit["modes"], fit["phases"],
                int(config["training"]["batch_size"]), generator,
            )
            losses = one_step(
                model, value_model, fit, ids, optimizer, generator,
                float(config["training"]["completion_weight"]),
                float(config["training"]["context_weight"]),
                float(config["training"].get("phase_weight", 0.)),
                float(config["training"].get("transition_class_weight", 1.)),
                gate_aligned(config),
                geometry_projected(config),
            )
            if update % int(config["training"]["validation_interval"]) == 0:
                current_metrics = validation_metrics(
                    model, value_model, selection, config,
                )
                score = (
                    selection_score(config, current_metrics)
                    if gate_aligned(config) else current_metrics["score"]
                )
                curve = {
                    "update": update, **losses,
                    "selection": current_metrics, "selection_score": score,
                }
                curves.append(curve)
                print(json.dumps({"instance": name, **curve}), flush=True)
                if best_score is None or score < best_score:
                    best_score = float(score)
                    best_state = deepcopy(model.state_dict())
                    best_metrics = current_metrics
                    best_update = update
        if best_state is None or best_metrics is None:
            raise RuntimeError(f"{name} response selection produced no checkpoint")
        model.load_state_dict(best_state, strict=True)
        decision = acceptance(config, best_metrics)
        checkpoint_path = output / name / "checkpoint.pt"
        save_torch(checkpoint_path, model.checkpoint())
        instances[name] = {
            "seed": seed,
            "selection_update": best_update,
            "selection_score": best_score,
            "selection_metrics": best_metrics,
            "decision": decision,
            "checkpoint": pointer(checkpoint_path),
            "curve": curves,
        }
    passed = all(row["decision"]["passed"] for row in instances.values())
    report = {
        "schema_version": config["version"],
        "status": "complete" if passed else "qualification_failed",
        "passed": passed,
        "instances": instances,
        "data": provenance,
        "device": {**config["device"], "observed_uuid": str(torch.cuda.get_device_properties(0).uuid)},
        "artifacts": {
            "config": pointer(config_path), "runner": pointer(Path(__file__).resolve()),
            "preflight": pointer(preflight_path),
            "completion_value_checkpoint": pointer(value_path),
        },
        "new_physics_calls": 0,
        "target_reward_or_truth_opened": False,
        "confirmation_states_opened": 0,
        "wall_seconds": time.monotonic() - begun,
    }
    write_json(output / "report.json", report)
    print(json.dumps({
        "status": report["status"], "passed": passed,
        "output": str(output / "report.json"),
        "sha256": sha(output / "report.json"),
    }, indent=2))
    if not passed:
        raise AssertionError("mode-complete response selection gate failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    if args.validate_only:
        run_preflight(config, config_path)
    else:
        run_training(config, config_path)


if __name__ == "__main__":
    main()
