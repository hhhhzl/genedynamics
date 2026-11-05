#!/usr/bin/env python3
"""Train and calibrate eight T2 joint force/contact admission models on GPU1."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from genedynamics.solvers.single.cross_ebdm.learning.response import PublicResponseSpec
from genedynamics.solvers.single.cross_ebdm.learning.substep_risk import (
    PublicJointAdmissionEnsemble,
    PublicJointAdmissionTrainer,
    calibrated_joint_upper_bound,
    decode_joint_admission_risks,
    fit_joint_block_conformal_calibration,
    joint_admission_targets,
)


ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def checked(value: dict[str, str]) -> Path:
    path = ROOT / value["path"]
    if not path.is_file() or sha(path) != value["sha256"]:
        raise ValueError(f"artifact changed: {path}")
    return path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def save_torch(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("version") != "t2-joint-admission-model-training-v2":
        raise ValueError("unknown joint-admission training version")
    if config.get("status") != "frozen_unexecuted":
        raise ValueError("joint-admission training plan is not frozen unexecuted")
    for artifact in config["dataset"].values():
        checked(artifact)
    report = json.loads(checked(config["dataset"]["report"]).read_text())
    audit = json.loads(checked(config["dataset"]["independent_audit"]).read_text())
    if not report.get("passed") or not audit.get("passed"):
        raise ValueError("counterfactual dataset is not independently admitted")
    if len(config["training"]["seeds"]) != 8 or len(set(config["training"]["seeds"])) != 8:
        raise ValueError("exactly eight unique model seeds required")
    return config


def verify_device(config: dict[str, Any]) -> torch.device:
    expected = config["device"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected["process_visible_devices"]:
        raise RuntimeError("training requires CUDA_VISIBLE_DEVICES=1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one logical CUDA device is required")
    device = torch.device(expected["logical_device"])
    properties = torch.cuda.get_device_properties(device)
    actual = str(getattr(properties, "uuid", ""))
    required = expected["physical_uuid"].removeprefix("GPU-")
    if actual.lower() != required.lower():
        raise RuntimeError(f"wrong physical GPU: {actual}")
    return device


def materialize_dataset(config: dict[str, Any]):
    rows = [
        json.loads(line) for line in checked(config["dataset"]["progress"]).read_text().splitlines()
        if line
    ]
    limits = np.asarray(config["calibration"]["risk_limits"], dtype=float)
    splits = {
        name: {key: [] for key in (
            "features", "commands", "targets", "physical_risks", "episode_id",
            "high_risk", "contact_violation", "choice", "record_id",
        )}
        for name in ("fit", "calibration", "held_out_qualification")
    }
    for episode, row in enumerate(rows):
        split = row["request"]["split"]
        with np.load(checked(row["data"]), allow_pickle=False) as archive:
            checkpoints, choices = archive["commands"].shape[:2]
            features = np.repeat(archive["features"][:, None], choices, axis=1).reshape(
                checkpoints * choices, -1,
            )
            commands = archive["commands"].reshape(checkpoints * choices, 10, 13)
            force = archive["force_peaks_N"].reshape(checkpoints * choices, 2)
            loss = archive["contact_loss_seconds"].reshape(-1)
            coverage = archive["contact_coverage"].reshape(-1)
            missing = archive["outlet_without_contact"].reshape(-1)
            violation = archive["contact_violation"].reshape(-1)
        targets = joint_admission_targets(force, loss, coverage, missing)
        physical = decode_joint_admission_risks(targets)
        high = np.any(physical > limits, axis=1) | violation
        count = len(targets)
        splits[split]["features"].append(features.astype(np.float32))
        splits[split]["commands"].append(commands.astype(np.float32))
        splits[split]["targets"].append(targets)
        splits[split]["physical_risks"].append(physical.astype(np.float32))
        splits[split]["episode_id"].append(np.full(count, episode, np.int32))
        splits[split]["high_risk"].append(high)
        splits[split]["contact_violation"].append(violation)
        splits[split]["choice"].append(np.tile(np.arange(choices, dtype=np.int8), checkpoints))
        splits[split]["record_id"].append(np.full(count, episode, np.int32))
    return {
        split: {key: np.concatenate(values) for key, values in columns.items()}
        for split, columns in splits.items()
    }


def predict(model, data, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data["targets"]), batch_size):
            stop = min(start + batch_size, len(data["targets"]))
            values.append(model(
                torch.as_tensor(data["features"][start:stop], device=device),
                torch.as_tensor(data["commands"][start:stop], device=device),
            ).cpu().numpy())
    return np.concatenate(values, axis=1)


def validation_metrics(predicted: np.ndarray, data: dict[str, np.ndarray], calibration):
    upper = calibrated_joint_upper_bound(torch.as_tensor(predicted), calibration).numpy()
    truth = data["physical_risks"]
    covered = upper >= truth - 1e-7
    episodes = np.unique(data["episode_id"])
    block_coverage = [
        float(np.mean([
            bool(np.all(covered[data["episode_id"] == episode, channel]))
            for episode in episodes
        ])) for channel in range(4)
    ]
    limits = np.asarray(calibration.risk_limits)
    predicted_high = np.any(upper > limits, axis=1)
    true_high = np.asarray(data["high_risk"], dtype=bool)
    contact = np.asarray(data["contact_violation"], dtype=bool)
    choice_metrics = {}
    for choice in range(5):
        mask = data["choice"] == choice
        choice_metrics[str(choice)] = {
            "examples": int(mask.sum()),
            "true_high_risk": int(true_high[mask].sum()),
            "predicted_high_risk": int(predicted_high[mask].sum()),
        }
    return {
        "examples": len(truth),
        "episodes": len(episodes),
        "high_risk_examples": int(true_high.sum()),
        "contact_violation_examples": int(contact.sum()),
        "window_coverage": covered.mean(axis=0).tolist(),
        "episode_block_coverage": block_coverage,
        "high_risk_recall": float(predicted_high[true_high].mean()) if true_high.any() else None,
        "contact_violation_recall": float(predicted_high[contact].mean()) if contact.any() else None,
        "safe_rejection_rate": float(predicted_high[~true_high].mean()),
        "risk_mae": np.abs(upper - truth).mean(axis=0).tolist(),
        "maximum_true_risk": truth.max(axis=0).tolist(),
        "maximum_upper_risk": upper.max(axis=0).tolist(),
        "choices": choice_metrics,
    }


def gates(metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    values = config["validation_gates"]
    coverage = metrics["episode_block_coverage"]
    contact_recall = metrics["contact_violation_recall"]
    return {
        "total_force_block_coverage": coverage[0] >= values["minimum_episode_block_coverage"],
        "lateral_force_block_coverage": coverage[1] >= values["minimum_episode_block_coverage"],
        "contact_loss_block_coverage": coverage[2] >= values["minimum_episode_block_coverage"],
        "contact_coverage_block_coverage": coverage[3] >= values["minimum_episode_block_coverage"],
        "high_risk_recall": metrics["high_risk_recall"] is not None
        and metrics["high_risk_recall"] >= values["minimum_high_risk_recall"],
        "contact_violation_recall": contact_recall is not None
        and contact_recall >= values["minimum_contact_violation_recall"],
        "safe_rejection_rate": metrics["safe_rejection_rate"]
        <= values["maximum_safe_rejection_rate"],
        "all_candidates_present": set(metrics["choices"]) == {"0", "1", "2", "3", "4"},
    }


def train_instance(name: str, seed: int, config: dict[str, Any], dataset,
                   spec: PublicResponseSpec, device: torch.device, output: Path):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = PublicJointAdmissionEnsemble(spec).to(device)
    trainer = PublicJointAdmissionTrainer(
        model, underprediction_weight=config["training"]["underprediction_weight"],
    )
    fit = dataset["fit"]
    high = np.flatnonzero(fit["high_risk"])
    safe = np.flatnonzero(~fit["high_risk"])
    if not len(high) or not len(safe):
        raise ValueError("fit split requires both high-risk and safe candidate branches")
    batch_size = int(config["training"]["batch_size"])
    high_count = int(round(batch_size * config["training"]["high_risk_fraction"]))
    curve = []
    for update in range(1, int(config["training"]["updates"]) + 1):
        indices = np.concatenate((
            rng.choice(high, high_count, replace=True),
            rng.choice(safe, batch_size - high_count, replace=True),
        ))
        rng.shuffle(indices)
        metrics = trainer.step(
            torch.as_tensor(fit["features"][indices], device=device),
            torch.as_tensor(fit["commands"][indices], device=device),
            torch.as_tensor(fit["targets"][indices], device=device),
        )
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"nonfinite joint-admission training at {name}/{update}")
        if update % int(config["training"]["log_interval"]) == 0:
            curve.append({"update": update, **metrics})
    calibration_predictions = predict(model, dataset["calibration"], device)
    calibration = fit_joint_block_conformal_calibration(
        calibration_predictions, dataset["calibration"]["targets"],
        dataset["calibration"]["episode_id"],
        miscoverage=config["calibration"]["miscoverage"],
        risk_limits=tuple(config["calibration"]["risk_limits"]),
        interpolation_fractions=tuple(config["calibration"]["interpolation_fractions"]),
    )
    validation_predictions = predict(model, dataset["held_out_qualification"], device)
    metrics = validation_metrics(
        validation_predictions, dataset["held_out_qualification"], calibration,
    )
    gate_values = gates(metrics, config)
    folder = output / name
    folder.mkdir()
    checkpoint_path = folder / "checkpoint.pt"
    save_torch(checkpoint_path, model.checkpoint())
    calibration_path = folder / "calibration.json"
    write_json(calibration_path, calibration.__dict__)
    result = {
        "instance": name, "seed": seed,
        "passed": all(gate_values.values()), "gates": gate_values,
        "validation": metrics, "calibration": calibration.__dict__,
        "curve": curve,
        "checkpoint": pointer(checkpoint_path),
        "calibration_artifact": pointer(calibration_path),
    }
    write_json(folder / "report.json", result)
    del model, trainer, calibration_predictions, validation_predictions
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    device = verify_device(config)
    spec_values = dict(config["model"])
    spec_values["hidden"] = tuple(spec_values["hidden"])
    spec = PublicResponseSpec(**spec_values)
    output = ROOT / config["output"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite joint-admission model run: {output}")
    output.mkdir(parents=True)
    write_json(output / "plan.json", {
        **config, "status": "executing", "config": pointer(config_path),
        "runner": pointer(Path(__file__).resolve()),
    })
    begun = time.monotonic()
    report = {"status": "running", "passed": False, "instances": {}}
    try:
        dataset = materialize_dataset(config)
        report["dataset_counts"] = {
            split: {
                "examples": len(data["targets"]),
                "episodes": len(np.unique(data["episode_id"])),
                "high_risk_examples": int(data["high_risk"].sum()),
                "contact_violation_examples": int(data["contact_violation"].sum()),
            } for split, data in dataset.items()
        }
        for index, seed in enumerate(config["training"]["seeds"], 1):
            name = f"instance_{index:02d}"
            report["instances"][name] = train_instance(
                name, int(seed), config, dataset, spec, device, output,
            )
            write_json(output / "progress.json", report)
            print(json.dumps({
                "instance": name, "passed": report["instances"][name]["passed"],
                "gates": report["instances"][name]["gates"],
            }), flush=True)
        passed = all(row["passed"] for row in report["instances"].values())
        report.update(
            status="passed" if passed else "failed", passed=passed,
            wall_seconds=time.monotonic() - begun,
            device={
                **config["device"], "logical_name": torch.cuda.get_device_name(0),
                "torch_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "")),
            },
            data_policy=config["data_policy"], confirmation_states_opened=0,
            old_invalid_outcomes_used=False, performance_evidence=False,
            failure_replacement=False,
            artifacts={"config": pointer(config_path), "runner": pointer(Path(__file__).resolve())},
        )
    except BaseException as error:
        report.update(status="failed", error=repr(error), wall_seconds=time.monotonic() - begun)
        raise
    finally:
        write_json(output / "report.json", report)
    print(json.dumps({
        "status": report["status"], "passed": report["passed"],
        "output": str((output / "report.json").relative_to(ROOT)),
        "sha256": sha(output / "report.json"),
    }, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
