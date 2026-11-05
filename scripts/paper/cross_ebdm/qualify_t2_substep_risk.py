#!/usr/bin/env python3
"""Calibrate on accessed development blocks and qualify on a fresh T2 cohort."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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


def load_episode(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        public = {name: archive[f"public.{name}"].copy() for name in PUBLIC_FIELDS}
        return (
            public, archive["actions"].copy(),
            archive["trace.force_peak"].copy(),
            archive["trace.lateral_force_peak"].copy(),
        )


def windows_for(path: Path, spec: PublicResponseSpec):
    public, actions, total, lateral = load_episode(path)
    return substep_risk_windows(
        public, actions, total_force_peaks_N=total,
        lateral_force_peaks_N=lateral, body="xarm7",
        calibration_context=(0.0, 0.0, 0.0), spec=spec,
    )


def concatenate(columns: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.concatenate(values) for name, values in columns.items()}


def materialize_development(config: dict[str, Any], spec: PublicResponseSpec):
    manifest = json.loads(checked(config["development_dataset"]["episodes"]).read_text())
    accepted = set(config["development_dataset"]["calibration_splits"])
    columns = {name: [] for name in ("features", "commands", "targets", "episode_id")}
    episodes = []
    for row in manifest["episodes"]:
        if row["split"] not in accepted:
            continue
        episode_id = int(row["episode_index"])
        windows = windows_for(checked(row["data"]), spec)
        count = len(windows["targets"])
        columns["features"].append(windows["features"])
        columns["commands"].append(windows["commands"])
        columns["targets"].append(windows["targets"])
        columns["episode_id"].append(np.full(count, episode_id, np.int32))
        episodes.append({
            "episode_id": episode_id, "record_id": row["record_id"],
            "split": row["split"], "windows": count,
        })
    return concatenate(columns), episodes


def materialize_fresh(config: dict[str, Any], spec: PublicResponseSpec):
    rows = [
        json.loads(line)
        for line in checked(config["fresh_cohort"]["progress"]).read_text().splitlines()
        if line
    ]
    learned = sorted(
        (row for row in rows if row["method_key"] != "public_geometry"),
        key=lambda row: row["record_id"],
    )
    columns = {name: [] for name in (
        "features", "commands", "targets", "episode_id", "high_risk",
    )}
    episodes = []
    limits = np.asarray(config["calibration"]["soft_force_limits_N"])
    for episode_id, row in enumerate(learned):
        windows = windows_for(checked(row["data"]), spec)
        targets = windows["targets"]
        force = np.expm1(targets)
        count = len(targets)
        columns["features"].append(windows["features"])
        columns["commands"].append(windows["commands"])
        columns["targets"].append(targets)
        columns["episode_id"].append(np.full(count, episode_id, np.int32))
        columns["high_risk"].append(np.any(force >= limits, axis=1))
        episodes.append({
            "episode_id": episode_id, "record_id": row["record_id"],
            "method_key": row["method_key"], "instance": row["instance"],
            "scale": row["scale"],
            "physical_initial_slot": row["request"]["physical_initial_slot"],
            "requested_mode": row["request"]["requested_mode"],
            "outcome": row["outcome"], "windows": count,
        })
    return concatenate(columns), episodes


def predict(model: PublicSubstepRiskEnsemble, data: dict[str, np.ndarray],
            device: torch.device, batch_size: int = 2048):
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


def metrics(predicted: np.ndarray, data: dict[str, np.ndarray], episodes,
            calibration) -> dict[str, Any]:
    upper = calibrated_force_upper_bound(
        torch.as_tensor(predicted), calibration,
    ).numpy()
    truth = np.expm1(data["targets"])
    covered = upper >= truth
    episode_ids = np.unique(data["episode_id"])
    block = {
        int(episode): np.all(covered[data["episode_id"] == episode], axis=0)
        for episode in episode_ids
    }
    limits = np.asarray(calibration.soft_force_limits_N)
    predicted_high = np.any(upper > limits, axis=1)
    true_high = np.asarray(data["high_risk"], dtype=bool)
    group_rows = {}
    for field in ("method_key", "scale", "physical_initial_slot", "requested_mode"):
        values = defaultdict(list)
        for row in episodes:
            values[str(row[field])].append(block[row["episode_id"]])
        group_rows[field] = {
            value: np.stack(rows).mean(axis=0).tolist()
            for value, rows in sorted(values.items())
        }
    block_values = np.stack([block[episode] for episode in episode_ids])
    return {
        "windows": len(truth), "episodes": len(episode_ids),
        "high_risk_windows": int(true_high.sum()),
        "window_coverage": covered.mean(axis=0).tolist(),
        "episode_block_coverage": block_values.mean(axis=0).tolist(),
        "joint_episode_block_coverage": float(np.all(block_values, axis=1).mean()),
        "high_risk_recall": float(predicted_high[true_high].mean()) if true_high.any() else None,
        "safe_window_rejection_rate": float(predicted_high[~true_high].mean()),
        "force_mae_N": np.abs(upper - truth).mean(axis=0).tolist(),
        "maximum_true_force_N": truth.max(axis=0).tolist(),
        "maximum_upper_force_N": upper.max(axis=0).tolist(),
        "episode_block_coverage_by": group_rows,
        "outcomes": dict(sorted(Counter(row["outcome"] for row in episodes).items())),
    }


def gate_values(values: dict[str, Any], config: dict[str, Any]):
    gates = config["validation_gates"]
    grouped = [
        channel
        for field in values["episode_block_coverage_by"].values()
        for channel in field.values()
    ]
    minimum_group = np.min(np.asarray(grouped), axis=0)
    return {
        "total_force_episode_block_coverage": bool(
            values["episode_block_coverage"][0]
            >= gates["minimum_total_force_episode_block_coverage"]
        ),
        "lateral_force_episode_block_coverage": bool(
            values["episode_block_coverage"][1]
            >= gates["minimum_lateral_force_episode_block_coverage"]
        ),
        "high_risk_recall": bool(
            values["high_risk_recall"] >= gates["minimum_high_risk_recall"]
        ),
        "safe_window_rejection_rate": bool(
            values["safe_window_rejection_rate"]
            <= gates["maximum_safe_window_rejection_rate"]
        ),
        "every_group_total_force_episode_block_coverage": bool(
            minimum_group[0] >= gates["minimum_group_episode_block_coverage"]
        ),
        "every_group_lateral_force_episode_block_coverage": bool(
            minimum_group[1] >= gates["minimum_group_episode_block_coverage"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text())
    if config.get("version") != "t2-substep-risk-fresh-qualification-v1":
        raise ValueError("unknown T2 risk qualification version")
    for section in ("source_models", "development_dataset", "fresh_cohort"):
        for value in config[section].values():
            if isinstance(value, dict) and set(value) == {"path", "sha256"}:
                checked(value)
    source_report = json.loads(checked(config["source_models"]["report"]).read_text())
    fresh_report = json.loads(checked(config["fresh_cohort"]["report"]).read_text())
    fresh_audit = json.loads(checked(config["fresh_cohort"]["independent_audit"]).read_text())
    if source_report.get("status") != "failed" or len(source_report.get("instances", {})) != 8:
        raise ValueError("the complete 7/8 development failure must be preserved")
    if not fresh_report.get("passed") or not fresh_audit.get("passed"):
        raise ValueError("fresh cohort is not complete and independently audited")
    device_config = config["device"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != device_config["process_visible_devices"]:
        raise RuntimeError("qualification requires CUDA_VISIBLE_DEVICES=1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one logical CUDA device required")
    properties = torch.cuda.get_device_properties(0)
    if str(getattr(properties, "uuid", "")).lower() != device_config[
            "physical_uuid"].removeprefix("GPU-").lower():
        raise RuntimeError("wrong physical GPU")
    device = torch.device(device_config["logical_device"])
    spec_values = dict(config["model"]); spec_values["hidden"] = tuple(spec_values["hidden"])
    spec = PublicResponseSpec(**spec_values)
    output = ROOT / config["output"]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_json(output / "plan.json", {**config, "config": pointer(config_path),
                                      "runner": pointer(Path(__file__).resolve())})
    begun = time.monotonic()
    report = {"status": "running", "passed": False, "instances": {}}
    try:
        development, development_episodes = materialize_development(config, spec)
        fresh, fresh_episodes = materialize_fresh(config, spec)
        report["dataset_counts"] = {
            "development_calibration": {
                "windows": len(development["targets"]),
                "episodes": len(development_episodes),
            },
            "fresh_qualification": {
                "windows": len(fresh["targets"]),
                "episodes": len(fresh_episodes),
                "high_risk_windows": int(fresh["high_risk"].sum()),
            },
        }
        for name, source in sorted(source_report["instances"].items()):
            checkpoint = checked(source["checkpoint"])
            model = PublicSubstepRiskEnsemble.from_checkpoint(
                torch.load(checkpoint, map_location="cpu", weights_only=True), device,
            )
            development_predictions = predict(model, development, device)
            calibration = fit_block_conformal_calibration(
                development_predictions, development["targets"],
                development["episode_id"],
                miscoverage=config["calibration"]["miscoverage"],
                soft_force_limits_N=tuple(config["calibration"]["soft_force_limits_N"]),
                interpolation_fractions=tuple(config["calibration"]["interpolation_fractions"]),
            )
            fresh_predictions = predict(model, fresh, device)
            values = metrics(fresh_predictions, fresh, fresh_episodes, calibration)
            gates = gate_values(values, config)
            folder = output / name; folder.mkdir()
            calibration_path = folder / "calibration.json"
            write_json(calibration_path, calibration.__dict__)
            result = {
                "instance": name, "seed": source["seed"],
                "passed": all(gates.values()), "gates": gates,
                "qualification": values,
                "calibration": calibration.__dict__,
                "checkpoint": pointer(checkpoint),
                "calibration_artifact": pointer(calibration_path),
            }
            report["instances"][name] = result
            write_json(folder / "report.json", result)
            write_json(output / "progress.json", report)
            print(json.dumps({
                "instance": name, "passed": result["passed"],
                "block": values["episode_block_coverage"],
                "recall": values["high_risk_recall"],
                "safe_rejection": values["safe_window_rejection_rate"],
            }), flush=True)
            del model, development_predictions, fresh_predictions
            torch.cuda.empty_cache()
        passed = all(row["passed"] for row in report["instances"].values())
        report.update(
            status="passed" if passed else "failed", passed=passed,
            wall_seconds=time.monotonic() - begun,
            device={**device_config, "logical_name": torch.cuda.get_device_name(0),
                    "torch_uuid": str(getattr(properties, "uuid", ""))},
            data_policy=config["data_policy"],
            confirmation_states_opened=0, old_invalid_outcomes_used=False,
            performance_evidence=False, failure_replacement=False,
            artifacts={"config": pointer(config_path), "runner": pointer(Path(__file__).resolve())},
        )
    except BaseException as error:
        report.update(status="failed", error=repr(error), wall_seconds=time.monotonic() - begun)
        raise
    finally:
        write_json(output / "report.json", report)
    print(json.dumps({
        "status": report["status"], "passed": report["passed"],
        "report": pointer(output / "report.json"),
    }, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
