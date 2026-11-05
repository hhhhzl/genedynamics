#!/usr/bin/env python3
"""Independently rebuild and audit the pooled T2 substep-risk window index."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


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


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = json.loads((output / "report.json").read_text())
    config = json.loads(checked(report["artifacts"]["config"]).read_text())
    source_rows = [
        json.loads(line)
        for line in checked(config["source"]["progress"]).read_text().splitlines()
        if line
    ]
    episode_manifest = json.loads(checked(report["artifacts"]["episodes"]).read_text())
    episodes = episode_manifest["episodes"]
    methods = set(config["inclusion"]["methods"])
    expected_rows = sorted(
        (row for row in source_rows if row["method_key"] in methods),
        key=lambda row: row["record_id"],
    )
    if len(expected_rows) != len(episodes):
        raise ValueError("episode manifest length differs from source")
    horizon = int(config["inclusion"]["horizon_control_steps"])
    total_limit = float(config["risk_thresholds_N"]["total"])
    lateral_limit = float(config["risk_thresholds_N"]["lateral"])
    safe_stride = int(config["inclusion"]["fit_safe_window_stride"])
    rebuilt: dict[str, dict[str, list[np.ndarray]]] = {
        split: defaultdict(list) for split in ("fit", "calibration", "validation")
    }
    raw = Counter()
    high_counts = Counter()
    selected = Counter()
    episode_counts = Counter()
    slots: dict[str, set[int]] = defaultdict(set)
    episode_checks = []
    for index, (row, manifest_row) in enumerate(zip(expected_rows, episodes)):
        data = checked(row["data"])
        with np.load(data, allow_pickle=False) as archive:
            actions = archive["actions"]
            total = np.asarray(archive["trace.force_peak"], dtype=np.float32)
            lateral = np.asarray(archive["trace.lateral_force_peak"], dtype=np.float32)
            ticks = np.arange(max(0, len(actions) - horizon + 1), dtype=np.int32)
            horizon_ids = ticks[:, None] + np.arange(horizon)[None, :]
            high = ((total[horizon_ids].max(axis=1) >= total_limit)
                    | (lateral[horizon_ids].max(axis=1) >= lateral_limit))
        split = row["request"]["split"]
        keep = high | (ticks % safe_stride == 0) if split == "fit" else np.ones(len(ticks), bool)
        chosen = ticks[keep]
        rebuilt[split]["episode_index"].append(np.full(len(chosen), index, np.int32))
        rebuilt[split]["tick"].append(chosen)
        rebuilt[split]["high_risk"].append(high[keep].astype(np.bool_))
        raw[split] += len(ticks)
        high_counts[split] += int(high.sum())
        selected[split] += len(chosen)
        episode_counts[split] += 1
        slots[split].add(int(row["request"]["physical_initial_slot"]))
        episode_checks.append(
            manifest_row["episode_index"] == index
            and manifest_row["record_id"] == row["record_id"]
            and manifest_row["data"] == pointer(data)
            and manifest_row["windows"] == len(ticks)
            and manifest_row["high_risk_windows"] == int(high.sum())
            and manifest_row["selected_windows"] == len(chosen)
        )
    with np.load(checked(report["artifacts"]["window_index"]), allow_pickle=False) as archive:
        index_equal = all(
            np.array_equal(archive[f"{split}.{name}"], np.concatenate(values))
            for split, arrays in rebuilt.items() for name, values in arrays.items()
        ) and set(archive.files) == {
            f"{split}.{name}" for split, arrays in rebuilt.items() for name in arrays
        }
    expected_slots = {"fit": {0, 1, 2, 3}, "calibration": {4}, "validation": {5}}
    counts = {
        "episodes": dict(episode_counts),
        "raw_windows": dict(raw),
        "selected_windows": dict(selected),
        "high_risk_windows": dict(high_counts),
        "physical_initial_slots": {split: sorted(values) for split, values in slots.items()},
    }
    checks = {
        "all_source_hashes_match": all(
            sha(ROOT / row["data"]["path"]) == row["data"]["sha256"] for row in expected_rows
        ),
        "episode_manifest_exact": all(episode_checks),
        "window_index_exactly_rebuilt": index_equal,
        "counts_exactly_recomputed": counts == report["counts"],
        "physical_slots_disjoint_and_frozen": dict(slots) == expected_slots,
        "source_independent_audit_passed": json.loads(checked(
            config["source"]["independent_audit"]
        ).read_text())["passed"] is True,
        "pooled_report_passed": report["passed"] is True,
        "old_invalid_outcomes_unused": report["data_policy"]["old_invalid_outcomes_used"] is False,
        "confirmation_states_unopened": report["data_policy"]["confirmation_states_opened"] == 0,
    }
    audit = {
        "schema_version": "t2-pooled-substep-risk-dataset-independent-audit-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "checks": checks,
        "counts": counts,
        "artifacts": {
            "report": pointer(output / "report.json"),
            "episodes": pointer(output / "episodes.json"),
            "window_index": pointer(output / "window_index.npz"),
        },
    }
    write_json(output / "independent_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    raise SystemExit(0 if audit["passed"] else 1)


if __name__ == "__main__":
    main()
