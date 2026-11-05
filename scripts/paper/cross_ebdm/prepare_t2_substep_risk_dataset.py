#!/usr/bin/env python3
"""Freeze a pooled, episode-disjoint T2 substep-risk training index."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    ROOT / "configs/arm/guided_contact_alignment/training/pooled_substep_risk_dataset.json"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def checked(value: dict[str, str]) -> Path:
    path = ROOT / value["path"]
    if not path.is_file() or sha(path) != value["sha256"]:
        raise ValueError(f"source artifact changed: {path}")
    return path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("version") != "t2-pooled-substep-risk-dataset-v1":
        raise ValueError("unknown pooled T2 substep-risk dataset version")
    if config.get("device") != "cpu":
        raise ValueError("pooled substep-risk dataset preparation must use CPU")
    if config["inclusion"]["methods"] != ["cross_ebdm", "task_only"]:
        raise ValueError("pooled method inventory changed")
    if config["inclusion"]["horizon_control_steps"] != 10:
        raise ValueError("T2 substep-risk horizon changed")
    return config


def validate_source(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = {name: checked(value) for name, value in config["source"].items()}
    report = json.loads(paths["report"].read_text())
    audit = json.loads(paths["independent_audit"].read_text())
    if not audit.get("passed") or audit.get("audited_result_status") != "failed":
        raise ValueError("source must be independently audited with its failure preserved")
    if report.get("risk_tail_coverage_passed") is not False:
        raise ValueError("expected the declared per-instance tail-coverage failure")
    if (report.get("old_invalid_outcomes_used") is not False
            or report.get("counts", {}).get("confirmation_states_opened") != 0):
        raise ValueError("source violates data-access policy")
    rows = [
        json.loads(line) for line in paths["progress"].read_text().splitlines() if line
    ]
    plan = json.loads(paths["plan"].read_text())
    if {row["record_id"] for row in rows} != {row["record_id"] for row in plan["jobs"]}:
        raise ValueError("source rows do not exactly cover the frozen plan")
    return report, rows


def window_risk(data_path: Path, horizon: int, total_limit: float,
                lateral_limit: float) -> tuple[int, np.ndarray]:
    with np.load(data_path, allow_pickle=False) as archive:
        actions = archive["actions"]
        total = np.asarray(archive["trace.force_peak"], dtype=np.float32)
        lateral = np.asarray(archive["trace.lateral_force_peak"], dtype=np.float32)
        if (actions.shape != (len(actions), 13) or len(total) != len(actions)
                or len(lateral) != len(actions)):
            raise ValueError(f"invalid transition arrays: {data_path}")
        starts = np.arange(max(0, len(actions) - horizon + 1), dtype=np.int32)
        indices = starts[:, None] + np.arange(horizon)[None, :]
        high = ((total[indices].max(axis=1) >= total_limit)
                | (lateral[indices].max(axis=1) >= lateral_limit))
    return len(actions), high


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    source_report, rows = validate_source(config)
    output = ROOT / config["output"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen dataset: {output}")
    output.mkdir(parents=True)

    methods = set(config["inclusion"]["methods"])
    selected_rows = [row for row in rows if row["method_key"] in methods]
    horizon = int(config["inclusion"]["horizon_control_steps"])
    total_limit = float(config["risk_thresholds_N"]["total"])
    lateral_limit = float(config["risk_thresholds_N"]["lateral"])
    safe_stride = int(config["inclusion"]["fit_safe_window_stride"])
    episode_manifest = []
    split_arrays: dict[str, dict[str, list[np.ndarray]]] = {
        split: defaultdict(list) for split in ("fit", "calibration", "validation")
    }
    raw_windows = Counter()
    high_windows = Counter()
    selected_windows = Counter()
    episode_counts = Counter()
    slots: dict[str, set[int]] = defaultdict(set)

    for episode_index, row in enumerate(sorted(selected_rows, key=lambda value: value["record_id"])):
        data_path = checked(row["data"])
        control_steps, high = window_risk(
            data_path, horizon, total_limit, lateral_limit,
        )
        split = row["request"]["split"]
        ticks = np.arange(len(high), dtype=np.int32)
        if split == "fit":
            keep = high | (ticks % safe_stride == 0)
        else:
            keep = np.ones(len(high), dtype=bool)
        chosen = ticks[keep]
        split_arrays[split]["episode_index"].append(
            np.full(len(chosen), episode_index, dtype=np.int32)
        )
        split_arrays[split]["tick"].append(chosen)
        split_arrays[split]["high_risk"].append(high[keep].astype(np.bool_))
        raw_windows[split] += len(high)
        high_windows[split] += int(high.sum())
        selected_windows[split] += len(chosen)
        episode_counts[split] += 1
        slots[split].add(int(row["request"]["physical_initial_slot"]))
        episode_manifest.append({
            "episode_index": episode_index,
            "record_id": row["record_id"],
            "split": split,
            "physical_initial_slot": int(row["request"]["physical_initial_slot"]),
            "method_key": row["method_key"],
            "instance": row["instance"],
            "scale": float(row["scale"]),
            "outcome": row["outcome"],
            "control_steps": control_steps,
            "windows": len(high),
            "high_risk_windows": int(high.sum()),
            "selected_windows": len(chosen),
            "data": pointer(data_path),
        })

    packed = {}
    for split, arrays in split_arrays.items():
        for name, values in arrays.items():
            packed[f"{split}.{name}"] = np.concatenate(values)
    index_path = output / "window_index.npz"
    save_npz(index_path, packed)
    manifest_path = output / "episodes.json"
    write_json(manifest_path, {
        "schema_version": config["version"],
        "body": "xarm7", "task": "T2",
        "calibration_context": config["inclusion"]["calibration_context"],
        "episodes": episode_manifest,
    })

    pooled = config["pooled_admission"]
    expected_slots = {"fit": {0, 1, 2, 3}, "calibration": {4}, "validation": {5}}
    checks = {
        "source_audit_passed": bool(json.loads(checked(
            config["source"]["independent_audit"]
        ).read_text())["passed"]),
        "source_failure_preserved": source_report["status"] == "failed",
        "all_learned_episodes_included": len(selected_rows) == 576,
        "public_sentinel_excluded": all(row["method_key"] != "public_geometry" for row in selected_rows),
        "physical_slots_disjoint_and_frozen": dict(slots) == expected_slots,
        "fit_tail_gate": high_windows["fit"] >= int(pooled["minimum_fit_high_risk_windows"]),
        "calibration_tail_gate": high_windows["calibration"] >= int(pooled["minimum_calibration_high_risk_windows"]),
        "validation_tail_gate": high_windows["validation"] >= int(pooled["minimum_validation_high_risk_windows"]),
        "confirmation_states_unopened": config["confirmation_states_opened"] == 0,
        "old_invalid_outcomes_unused": config["old_invalid_outcomes_used"] is False,
    }
    report = {
        "schema_version": config["version"],
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "episodes": dict(episode_counts),
            "raw_windows": dict(raw_windows),
            "selected_windows": dict(selected_windows),
            "high_risk_windows": dict(high_windows),
            "physical_initial_slots": {
                split: sorted(values) for split, values in slots.items()
            },
        },
        "selection": {
            "fit": "all high-risk windows plus deterministic safe tick modulo stride",
            "fit_safe_window_stride": safe_stride,
            "calibration": "all windows, grouped by episode",
            "validation": "all windows, grouped by episode",
        },
        "data_policy": {
            "diagnostic_only": config["diagnostic_only"],
            "confirmation_states_opened": config["confirmation_states_opened"],
            "old_invalid_outcomes_used": config["old_invalid_outcomes_used"],
            "performance_evidence": config["performance_evidence"],
            "source_collection_status": source_report["status"],
            "source_collection_failure_reason": "per-instance tail coverage",
            "pooled_use_justification": (
                "risk is a shared 1 ms contact-physics observable; the pooled fixed splits "
                "meet the predeclared tail gates while per-instance training does not"
            ),
        },
        "artifacts": {
            "config": pointer(config_path),
            "runner": pointer(Path(__file__).resolve()),
            "episodes": pointer(manifest_path),
            "window_index": pointer(index_path),
            "source": config["source"],
        },
    }
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
