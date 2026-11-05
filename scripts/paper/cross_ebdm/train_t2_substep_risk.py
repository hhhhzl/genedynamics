#!/usr/bin/env python3
"""Train and calibrate eight public T2 substep-risk prototypes on GPU1."""

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

from genedynamics.solvers.single.cross_ebdm.learning.response import (
    PUBLIC_FIELDS,
    PublicResponseSpec,
)
from genedynamics.solvers.single.cross_ebdm.learning.substep_risk import (
    PublicSubstepRiskEnsemble,
    PublicSubstepRiskTrainer,
    calibrated_force_upper_bound,
    fit_block_conformal_calibration,
    substep_risk_windows,
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
    if config.get("version") != "t2-substep-risk-model-training-v1":
        raise ValueError("unknown substep-risk training version")
    if config.get("status") != "frozen_unexecuted":
        raise ValueError("training plan is not frozen unexecuted")
    for artifact in config["dataset"].values():
        checked(artifact)
    report = json.loads(checked(config["dataset"]["report"]).read_text())
    audit = json.loads(checked(config["dataset"]["independent_audit"]).read_text())
    if not report.get("passed") or not audit.get("passed"):
        raise ValueError("pooled substep-risk dataset is not independently admitted")
    if len(config["training"]["seeds"]) != 8 or len(set(config["training"]["seeds"])) != 8:
        raise ValueError("exactly eight unique prototype seeds required")
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


def load_episode(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        public = {name: archive[f"public.{name}"].copy() for name in PUBLIC_FIELDS}
        actions = archive["actions"].copy()
        total = archive["trace.force_peak"].copy()
        lateral = archive["trace.lateral_force_peak"].copy()
    return public, actions, total, lateral


def materialize_dataset(config: dict[str, Any], spec: PublicResponseSpec):
    manifest = json.loads(checked(config["dataset"]["episodes"]).read_text())
    with np.load(checked(config["dataset"]["window_index"]), allow_pickle=False) as archive:
        index = {name: archive[name].copy() for name in archive.files}
    selected = {}
    for split in ("fit", "calibration", "validation"):
        episode_ids = index[f"{split}.episode_index"]
        ticks = index[f"{split}.tick"]
        selected[split] = {
            int(episode): ticks[episode_ids == episode]
            for episode in np.unique(episode_ids)
        }
    arrays = {
        split: {name: [] for name in ("features", "commands", "targets", "episode_id", "high_risk")}
        for split in selected
    }
    for row in manifest["episodes"]:
        episode = int(row["episode_index"])
        split = row["split"]
        if episode not in selected[split]:
            continue
        data_path = checked(row["data"])
        public, actions, total, lateral = load_episode(data_path)
        windows = substep_risk_windows(
            public, actions, total_force_peaks_N=total,
            lateral_force_peaks_N=lateral, body=manifest["body"],
            calibration_context=manifest["calibration_context"], spec=spec,
        )
        ticks = selected[split][episode]
        if not np.array_equal(windows["ticks"][ticks], ticks):
            raise ValueError(f"window index mismatch: {row['record_id']}")
        targets = windows["targets"][ticks]
        force = np.expm1(targets)
        arrays[split]["features"].append(windows["features"][ticks])
        arrays[split]["commands"].append(windows["commands"][ticks])
        arrays[split]["targets"].append(targets)
        arrays[split]["episode_id"].append(np.full(len(ticks), episode, np.int32))
        arrays[split]["high_risk"].append(
            ((force[:, 0] >= config["calibration"]["soft_force_limits_N"][0])
             | (force[:, 1] >= config["calibration"]["soft_force_limits_N"][1]))
        )
    return {
        split: {name: np.concatenate(values) for name, values in columns.items()}
        for split, columns in arrays.items()
    }


def predict(model: PublicSubstepRiskEnsemble, data: dict[str, np.ndarray],
            device: torch.device, batch_size: int = 2048) -> np.ndarray:
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data["targets"]), batch_size):
            stop = min(start + batch_size, len(data["targets"]))
            features = torch.as_tensor(data["features"][start:stop], device=device)
            commands = torch.as_tensor(data["commands"][start:stop], device=device)
            values.append(model(features, commands).cpu().numpy())
    return np.concatenate(values, axis=1)


def validation_metrics(predicted: np.ndarray, data: dict[str, np.ndarray], calibration):
    raw = torch.as_tensor(predicted)
    upper = calibrated_force_upper_bound(raw, calibration).numpy()
    truth = np.expm1(data["targets"])
    covered = upper >= truth
    episodes = np.unique(data["episode_id"])
    block_coverage = [
        float(np.mean([
            bool(np.all(covered[data["episode_id"] == episode, channel]))
            for episode in episodes
        ])) for channel in range(2)
    ]
    limits = np.asarray(calibration.soft_force_limits_N)
    predicted_high = np.any(upper > limits, axis=1)
    true_high = np.asarray(data["high_risk"], dtype=bool)
    return {
        "windows": len(truth),
        "episodes": len(episodes),
        "high_risk_windows": int(true_high.sum()),
        "window_coverage": covered.mean(axis=0).tolist(),
        "episode_block_coverage": block_coverage,
        "high_risk_recall": float(predicted_high[true_high].mean()) if true_high.any() else None,
        "safe_window_rejection_rate": float(predicted_high[~true_high].mean()),
        "force_mae_N": np.abs(upper - truth).mean(axis=0).tolist(),
        "maximum_true_force_N": truth.max(axis=0).tolist(),
        "maximum_upper_force_N": upper.max(axis=0).tolist(),
    }


def gates(metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    values = config["validation_gates"]
    return {
        "total_force_episode_block_coverage": metrics["episode_block_coverage"][0]
        >= values["minimum_total_force_episode_block_coverage"],
        "lateral_force_episode_block_coverage": metrics["episode_block_coverage"][1]
        >= values["minimum_lateral_force_episode_block_coverage"],
        "high_risk_recall": metrics["high_risk_recall"]
        >= values["minimum_high_risk_recall"],
        "safe_window_rejection_rate": metrics["safe_window_rejection_rate"]
        <= values["maximum_safe_window_rejection_rate"],
    }


def train_instance(name: str, seed: int, config: dict[str, Any], dataset,
                   spec: PublicResponseSpec, device: torch.device, output: Path):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = PublicSubstepRiskEnsemble(spec).to(device)
    trainer = PublicSubstepRiskTrainer(
        model, underprediction_weight=config["training"]["underprediction_weight"],
    )
    fit = dataset["fit"]
    high = np.flatnonzero(fit["high_risk"])
    safe = np.flatnonzero(~fit["high_risk"])
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
            raise FloatingPointError(f"nonfinite risk training at {name}/{update}")
        if update % int(config["training"]["log_interval"]) == 0:
            curve.append({"update": update, **metrics})
    calibration_predictions = predict(model, dataset["calibration"], device)
    calibration = fit_block_conformal_calibration(
        calibration_predictions, dataset["calibration"]["targets"],
        dataset["calibration"]["episode_id"],
        miscoverage=config["calibration"]["miscoverage"],
        soft_force_limits_N=tuple(config["calibration"]["soft_force_limits_N"]),
        interpolation_fractions=tuple(config["calibration"]["interpolation_fractions"]),
    )
    validation_predictions = predict(model, dataset["validation"], device)
    metrics = validation_metrics(validation_predictions, dataset["validation"], calibration)
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
        raise FileExistsError(f"refusing to overwrite risk-model run: {output}")
    output.mkdir(parents=True)
    plan = {**config, "status": "executing", "config": pointer(config_path),
            "runner": pointer(Path(__file__).resolve())}
    write_json(output / "plan.json", plan)
    begun = time.monotonic()
    report = {"status": "running", "passed": False, "instances": {}}
    try:
        dataset = materialize_dataset(config, spec)
        report["dataset_counts"] = {
            split: {
                "windows": len(data["targets"]),
                "episodes": len(np.unique(data["episode_id"])),
                "high_risk_windows": int(data["high_risk"].sum()),
            } for split, data in dataset.items()
        }
        for index, seed in enumerate(config["training"]["seeds"], 1):
            name = f"instance_{index:02d}"
            report["instances"][name] = train_instance(
                name, int(seed), config, dataset, spec, device, output,
            )
            write_json(output / "progress.json", report)
            print(json.dumps({
                "instance": name,
                "passed": report["instances"][name]["passed"],
                "validation": report["instances"][name]["validation"],
            }), flush=True)
        passed = all(row["passed"] for row in report["instances"].values())
        report.update(
            status="passed" if passed else "failed", passed=passed,
            wall_seconds=time.monotonic() - begun,
            device={
                **config["device"], "logical_name": torch.cuda.get_device_name(0),
                "torch_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "")),
            },
            data_policy=config["data_policy"],
            confirmation_states_opened=0, old_invalid_outcomes_used=False,
            performance_evidence=False,
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
