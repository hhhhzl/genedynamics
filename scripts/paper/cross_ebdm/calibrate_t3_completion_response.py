#!/usr/bin/env python3
"""Calibrate the composed T3 response-to-completion-value prediction on GPU1."""

from __future__ import annotations

import argparse
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
from genedynamics.solvers.single.cross_ebdm.learning.response import response_windows


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


def verify_device(config: dict[str, Any]) -> torch.device:
    expected = config["device"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected["process_visible_devices"]:
        raise RuntimeError("completion-response calibration requires CUDA_VISIBLE_DEVICES=1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one logical CUDA device required")
    properties = torch.cuda.get_device_properties(0)
    if str(properties.uuid).lower() != expected["physical_uuid"].removeprefix("GPU-").lower():
        raise RuntimeError("visible device is not admitted GPU1")
    return torch.device(expected["logical_device"])


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("version") != "t3-public-completion-response-calibration-v1":
        raise ValueError("unknown T3 completion-response calibration version")
    if config.get("status") != "frozen_unexecuted":
        raise ValueError("calibration plan is not frozen unexecuted")
    if config["data"]["episode_split"] != {
        "kind": "within_manifest_entry_modulo",
        "modulus": 4,
        "validation_residue": 0,
    }:
        raise ValueError("held-out episode split changed")
    if set(config["response_checkpoints"]) != {
        f"instance_{index:02d}" for index in range(1, 9)
    }:
        raise ValueError("eight frozen response instances required")
    return config


def public_value_features(
    public: dict[str, np.ndarray], ticks: np.ndarray, mode: int, device: torch.device,
) -> torch.Tensor:
    return completion_value_features(
        position=torch.as_tensor(public["tcp_position"][ticks], device=device),
        rotation=torch.as_tensor(public["tcp_rotation"][ticks], device=device),
        twist=torch.as_tensor(public["tcp_twist"][ticks], device=device),
        wrench=torch.as_tensor(public["measured_wrench"][ticks], device=device),
        goal_position=torch.as_tensor(public["goal_position"][ticks], device=device),
        goal_rotation=torch.as_tensor(public["goal_rotation"][ticks], device=device),
        controller_feedback=torch.as_tensor(
            public["controller_feedback"][ticks], device=device,
        ),
        mode=torch.full((len(ticks),), mode, device=device, dtype=torch.long),
    )


def current_public(
    public: dict[str, np.ndarray], ticks: np.ndarray, device: torch.device,
) -> dict[str, torch.Tensor]:
    names = {
        "position": "tcp_position",
        "rotation": "tcp_rotation",
        "twist": "tcp_twist",
        "wrench": "measured_wrench",
        "goal_position": "goal_position",
        "goal_rotation": "goal_rotation",
        "controller_feedback": "controller_feedback",
    }
    return {
        target: torch.as_tensor(public[source][ticks], device=device)
        for target, source in names.items()
    }


def load_validation_rows(config: dict[str, Any], response_spec: Any) -> tuple[dict, dict]:
    maximum = int(config["data"]["maximum_remaining_controls"])
    split = config["data"]["episode_split"]
    rows = {mode: [] for mode in (0, 1)}
    provenance = {"manifests": [], "episodes": {}, "counts_by_mode": {}}
    for manifest_index, value in enumerate(config["data"]["public_log_manifests"]):
        path = checked(value, f"public-log manifest {manifest_index}")
        manifest = json.loads(path.read_text())
        if manifest["permission"] != "B_log" or manifest["reward"] or manifest["success_filter"]:
            raise ValueError("unfiltered public B_log required")
        for entry_index, entry in enumerate(manifest["entries"]):
            if entry_index % int(split["modulus"]) != int(split["validation_residue"]):
                continue
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
                raise ValueError("held-out public trajectory changed")
            mode = int(trajectory.metadata["mode"])
            windows = response_windows(
                trajectory.public, trajectory.requests, body="xarm7", task="T3",
                calibration_context=[0, 0, 0], spec=response_spec,
            )
            end = len(trajectory.requests)
            keep = windows["ticks"] >= max(0, end - maximum)
            ticks = windows["ticks"][keep]
            if not len(ticks):
                raise ValueError("held-out episode has no late-phase response windows")
            rows[mode].append({
                "features": windows["features"][keep],
                "commands": windows["commands"][keep],
                "ticks": ticks,
                "public": trajectory.public,
            })
            key = f"manifest_{manifest_index:02d}/entry_{entry_index:04d}"
            provenance["episodes"][key] = {
                "digest": trajectory.digest,
                "mode": mode,
                "seed": int(trajectory.metadata["seed"]),
                "controls": end,
                "response_pairs": len(ticks),
            }
        provenance["manifests"].append(pointer(path))
    for mode in (0, 1):
        provenance["counts_by_mode"][str(mode)] = {
            "episodes": len(rows[mode]),
            "response_pairs": sum(len(row["ticks"]) for row in rows[mode]),
        }
        if not rows[mode]:
            raise ValueError(f"held-out mode {mode} is empty")
    return rows, provenance


def metrics(values: dict[str, list[torch.Tensor]], horizon: int, quantile: float) -> dict:
    data = {name: torch.cat(parts) for name, parts in values.items()}
    truth = torch.full_like(data["conservative"], float(horizon))
    residual = data["conservative"] - truth
    mean_residual = data["mean"] - truth
    optimism = residual.clamp(min=0)
    return {
        "pairs": int(len(truth)),
        "conservative_gain_mean_controls": float(data["conservative"].mean()),
        "conservative_gain_p05_controls": float(torch.quantile(data["conservative"], .05)),
        "conservative_mae_controls": float(residual.abs().mean()),
        "conservative_rmse_controls": float(residual.square().mean().sqrt()),
        "conservative_direction_accuracy": float((data["conservative"] > 0).float().mean()),
        "all_model_pairs_positive_fraction": float(data["all_positive"].float().mean()),
        "ensemble_mean_mae_controls": float(mean_residual.abs().mean()),
        "direct_value_conservative_mae_controls": float(
            (data["direct_conservative"] - truth).abs().mean()
        ),
        "response_projection_disagreement_mae_controls": float(
            (data["conservative"] - data["direct_conservative"]).abs().mean()
        ),
        "optimism_p95_controls": float(torch.quantile(optimism, .95)),
        "optimism_calibration_quantile": quantile,
        "optimism_calibration_controls": float(torch.quantile(optimism, quantile)),
        "optimism_max_controls": float(optimism.max()),
    }


def evaluate_instance(
    response: PublicIssuedResponseEnsemble,
    value_model: PublicCompletionValueEnsemble,
    rows: dict,
    config: dict[str, Any],
    device: torch.device,
) -> dict:
    maximum = value_model.spec.maximum_remaining_controls
    horizon = response.spec.horizon
    quantile = float(config["calibration"]["one_sided_optimism_quantile"])
    result = {"by_mode": {}}
    with torch.no_grad():
        for mode in (0, 1):
            values = {
                name: [] for name in (
                    "conservative", "mean", "direct_conservative", "all_positive",
                )
            }
            for row in rows[mode]:
                ticks = row["ticks"]
                features = torch.as_tensor(row["features"], device=device)
                commands = torch.as_tensor(row["commands"], device=device)
                prediction = response(features, commands)
                current = current_public(row["public"], ticks, device)
                endpoint = response_endpoint_value(
                    value_model, prediction, current=current, mode=mode,
                )
                current_value = value_model(public_value_features(
                    row["public"], ticks, mode, device,
                ))
                observed_future = value_model(public_value_features(
                    row["public"], ticks + horizon, mode, device,
                ))
                pair_gain = (current_value[:, None, :] - endpoint) * maximum
                direct_gain = (current_value - observed_future) * maximum
                values["conservative"].append(pair_gain.amin((0, 1)).cpu())
                values["mean"].append(pair_gain.mean((0, 1)).cpu())
                values["direct_conservative"].append(direct_gain.amin(0).cpu())
                values["all_positive"].append((pair_gain.amin((0, 1)) > 0).cpu())
            result["by_mode"][str(mode)] = metrics(values, horizon, quantile)
    return result


def acceptance(config: dict, instances: dict) -> dict:
    thresholds = config["acceptance"]
    checks = {}
    for instance, result in instances.items():
        for mode, row in result["by_mode"].items():
            prefix = f"{instance}_mode_{mode}"
            checks[prefix + "_mae"] = (
                row["conservative_mae_controls"]
                <= thresholds["maximum_conservative_mae_controls"]
            )
            checks[prefix + "_direction"] = (
                row["conservative_direction_accuracy"]
                >= thresholds["minimum_conservative_direction_accuracy"]
            )
            checks[prefix + "_optimism"] = (
                row["optimism_calibration_controls"]
                <= thresholds["maximum_calibrated_optimism_controls"]
            )
    return {"passed": all(checks.values()), "checks": checks, "thresholds": thresholds}


def run(config: dict[str, Any], config_path: Path) -> None:
    device = verify_device(config)
    value_path = checked(config["completion_value_checkpoint"], "completion-value checkpoint")
    value_payload = torch.load(value_path, map_location="cpu", weights_only=True)
    value_model = PublicCompletionValueEnsemble.from_checkpoint(value_payload, device).eval()
    if not value_payload.get("calibration"):
        raise ValueError("calibrated completion-value checkpoint required")
    responses = {}
    response_pointers = {}
    for name, value in sorted(config["response_checkpoints"].items()):
        path = checked(value, f"{name} response checkpoint")
        responses[name] = PublicIssuedResponseEnsemble.from_checkpoint(
            torch.load(path, map_location="cpu", weights_only=True), device,
        ).eval()
        response_pointers[name] = pointer(path)
    specs = {repr(response.spec) for response in responses.values()}
    if len(specs) != 1:
        raise ValueError("response instances do not share one frozen schema")
    first = responses[sorted(responses)[0]]
    rows, provenance = load_validation_rows(config, first.spec)
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", config)
    begun = time.monotonic()
    instances = {
        name: evaluate_instance(response, value_model, rows, config, device)
        for name, response in responses.items()
    }
    decision = acceptance(config, instances)
    margin = max(
        row["optimism_calibration_controls"]
        for result in instances.values()
        for row in result["by_mode"].values()
    )
    direct_margin = float(
        value_payload["calibration"]["minimum_admission_gain_controls"]
    )
    calibration = {
        "version": "public-T3-composed-completion-calibration-v1",
        "one_sided_optimism_quantile": config["calibration"]["one_sided_optimism_quantile"],
        "minimum_admission_gain_controls": max(margin, direct_margin),
        "composed_optimism_margin_controls": margin,
        "direct_value_margin_controls": direct_margin,
        "instances": instances,
    }
    report = {
        "schema_version": config["version"],
        "status": "complete" if decision["passed"] else "qualification_failed",
        "passed": decision["passed"],
        "decision": decision,
        "calibration": calibration,
        "data": provenance,
        "device": {
            **config["device"],
            "observed_uuid": str(torch.cuda.get_device_properties(0).uuid),
        },
        "artifacts": {
            "config": pointer(config_path),
            "runner": pointer(Path(__file__).resolve()),
            "completion_value_checkpoint": pointer(value_path),
            "response_checkpoints": response_pointers,
        },
        "new_physics_calls": 0,
        "target_reward_or_truth_opened": False,
        "confirmation_states_opened": 0,
        "wall_seconds": time.monotonic() - begun,
    }
    write_json(output / "report.json", report)
    print(json.dumps({
        "status": report["status"], "passed": report["passed"],
        "minimum_admission_gain_controls": calibration["minimum_admission_gain_controls"],
        "output": str(output / "report.json"),
        "sha256": sha(output / "report.json"),
    }, indent=2))
    if not report["passed"]:
        raise AssertionError("composed completion-response calibration gate failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    run(load_config(config_path), config_path)


if __name__ == "__main__":
    main()
