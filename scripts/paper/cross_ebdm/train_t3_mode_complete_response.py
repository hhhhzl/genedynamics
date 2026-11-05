#!/usr/bin/env python3
"""Train mode-complete T3 public response ensembles with completion consistency."""

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
    PublicCompletionValueEnsemble,
    completion_value_features,
    response_endpoint_value,
)
from genedynamics.solvers.single.cross_ebdm.learning.issued_response import (
    PublicIssuedResponseEnsemble,
)
from genedynamics.solvers.single.cross_ebdm.learning.response import (
    DELTA_SCALE,
    PublicResponseSpec,
    response_windows,
)


ROOT = Path(__file__).resolve().parents[3]


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
    if config.get("version") != "t3-mode-complete-public-response-training-v1":
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
    return config


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


def value_features(public: dict, ticks: np.ndarray, mode: int) -> np.ndarray:
    result = completion_value_features(
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
            "value_features": [],
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
            group["targets"].append(windows["targets"][keep])
            group["modes"].append(np.full(len(ticks), mode, np.int64))
            group["value_features"].append(value_features(trajectory.public, ticks, mode))
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
        }
        if set(np.unique(data[split]["modes"])) != {0, 1}:
            raise ValueError(f"{split} response data is not mode-complete")
    return data, provenance


def to_device(data: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: torch.as_tensor(value, device=device) for name, value in data.items()}


def balanced_indices(
    modes: torch.Tensor, count: int, generator: torch.Generator,
) -> torch.Tensor:
    if count % 2:
        raise ValueError("mode-balanced response batch must be even")
    selected = []
    for mode in (0, 1):
        available = torch.nonzero(modes == mode, as_tuple=False).reshape(-1)
        ids = torch.randint(
            len(available), (count // 2,), generator=generator,
            device=modes.device,
        )
        selected.append(available[ids])
    rows = torch.cat(selected)
    return rows[torch.randperm(len(rows), generator=generator, device=modes.device)]


def current(data: dict[str, torch.Tensor], ids: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: data[name][ids] for name in CURRENT_FIELDS}


def completion_loss(
    prediction: torch.Tensor,
    data: dict[str, torch.Tensor],
    ids: torch.Tensor,
    value_model: PublicCompletionValueEnsemble,
    horizon: int,
) -> torch.Tensor:
    losses = []
    maximum = value_model.spec.maximum_remaining_controls
    current_values = value_model(data["value_features"][ids])
    modes = data["modes"][ids]
    for mode in (0, 1):
        mask = modes == mode
        endpoint = response_endpoint_value(
            value_model, prediction[:, mask],
            current={name: value[mask] for name, value in current(data, ids).items()},
            mode=mode,
        )
        gains = current_values[:, None, mask] - endpoint
        truth = gains.new_full(gains.shape, horizon / maximum)
        losses.append(torch.nn.functional.smooth_l1_loss(gains, truth))
    return torch.stack(losses).mean()


def mode_metrics(
    model: PublicIssuedResponseEnsemble,
    value_model: PublicCompletionValueEnsemble,
    data: dict[str, torch.Tensor],
    mode: int,
    config: dict[str, Any],
) -> dict[str, float | int]:
    ids = torch.nonzero(data["modes"] == mode, as_tuple=False).reshape(-1)
    prediction = model(data["features"][ids], data["commands"][ids])
    target = data["targets"][ids]
    mean = prediction.mean(0)
    residual = mean - target
    scale = residual.new_tensor(DELTA_SCALE)
    physical = residual[..., :18] * scale
    endpoint = response_endpoint_value(
        value_model, prediction, current=current(data, ids), mode=mode,
    )
    maximum = value_model.spec.maximum_remaining_controls
    current_values = value_model(data["value_features"][ids])
    pair_gain = (current_values[:, None, :] - endpoint) * maximum
    conservative = pair_gain.amin((0, 1))
    truth = conservative.new_full(conservative.shape, float(model.spec.horizon))
    optimism = (conservative - truth).clamp(min=0)
    quantile = float(config["selection"]["one_sided_optimism_quantile"])
    return {
        "pairs": int(len(ids)),
        "normalized_response_rmse": float(residual.square().mean().sqrt()),
        "position_delta_vector_mae_m": float(physical[..., :3].norm(dim=-1).mean()),
        "rotation_delta_vector_mae_rad": float(physical[..., 3:6].norm(dim=-1).mean()),
        "twist_delta_vector_mae": float(physical[..., 6:12].norm(dim=-1).mean()),
        "wrench_delta_vector_mae": float(physical[..., 12:18].norm(dim=-1).mean()),
        "completion_conservative_gain_mean_controls": float(conservative.mean()),
        "completion_conservative_mae_controls": float((conservative - truth).abs().mean()),
        "completion_conservative_direction_accuracy": float((conservative > 0).float().mean()),
        "completion_optimism_calibration_controls": float(torch.quantile(optimism, quantile)),
    }


def validation_metrics(
    model: PublicIssuedResponseEnsemble,
    value_model: PublicCompletionValueEnsemble,
    data: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> dict:
    model.eval()
    with torch.no_grad():
        by_mode = {
            str(mode): mode_metrics(model, value_model, data, mode, config)
            for mode in (0, 1)
        }
    score = max(
        row["completion_conservative_mae_controls"]
        + 1000. * row["position_delta_vector_mae_m"]
        + 100. * row["rotation_delta_vector_mae_rad"]
        for row in by_mode.values()
    )
    return {"by_mode": by_mode, "score": score}


def acceptance(config: dict, metrics: dict) -> dict:
    thresholds = config["acceptance"]
    checks = {}
    for mode, row in metrics["by_mode"].items():
        prefix = f"mode_{mode}"
        checks[prefix + "_position"] = (
            row["position_delta_vector_mae_m"]
            <= thresholds["maximum_position_delta_vector_mae_m"]
        )
        checks[prefix + "_rotation"] = (
            row["rotation_delta_vector_mae_rad"]
            <= thresholds["maximum_rotation_delta_vector_mae_rad"]
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


def one_step(
    model: PublicIssuedResponseEnsemble,
    value_model: PublicCompletionValueEnsemble,
    data: dict[str, torch.Tensor],
    ids: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    completion_weight: float,
) -> dict[str, float]:
    model.train()
    prediction = model(data["features"][ids], data["commands"][ids])
    target = data["targets"][ids]
    supervised = []
    for member in prediction:
        bootstrap = torch.randint(
            len(ids), (len(ids),), generator=generator, device=ids.device,
        )
        supervised.append(torch.nn.functional.mse_loss(
            member[bootstrap], target[bootstrap],
        ))
    response_loss = torch.stack(supervised).mean()
    value_loss = completion_loss(
        prediction, data, ids, value_model, model.spec.horizon,
    )
    loss = response_loss + completion_weight * value_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "response_loss": float(response_loss.detach()),
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
    value_model = PublicCompletionValueEnsemble.from_checkpoint(
        torch.load(value_path, map_location="cpu", weights_only=True), "cpu",
    ).eval()
    for parameter in value_model.parameters():
        parameter.requires_grad_(False)
    seed = min(config["training"]["instance_seeds"].values())
    torch.manual_seed(seed)
    model = PublicIssuedResponseEnsemble(spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    ids = balanced_indices(fit["modes"], 32, generator)
    losses = one_step(
        model, value_model, fit, ids, optimizer, generator,
        float(config["training"]["completion_weight"]),
    )
    passed = all(np.isfinite(value) for value in losses.values())
    output = ROOT / config["preflight_output"]
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": "t3-mode-complete-public-response-preflight-v1",
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
    value_model = PublicCompletionValueEnsemble.from_checkpoint(
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
        model = PublicIssuedResponseEnsemble(spec).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate)
        generator = torch.Generator(device=device).manual_seed(seed)
        curves = []
        best_state = None
        best_metrics = None
        best_update = None
        best_score = None
        for update in range(1, int(config["training"]["updates"]) + 1):
            ids = balanced_indices(
                fit["modes"], int(config["training"]["batch_size"]), generator,
            )
            losses = one_step(
                model, value_model, fit, ids, optimizer, generator,
                float(config["training"]["completion_weight"]),
            )
            if update % int(config["training"]["validation_interval"]) == 0:
                current_metrics = validation_metrics(
                    model, value_model, selection, config,
                )
                curve = {"update": update, **losses, "selection": current_metrics}
                curves.append(curve)
                print(json.dumps({"instance": name, **curve}), flush=True)
                if best_score is None or current_metrics["score"] < best_score:
                    best_score = float(current_metrics["score"])
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
