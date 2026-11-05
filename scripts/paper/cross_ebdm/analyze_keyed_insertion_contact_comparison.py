#!/usr/bin/env python3
"""Build the paired keyed-insertion contact comparison from verified outcomes."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    ROOT / "results/_analysis/four_task_transfer/"
    "independent_main_table_formal_v2/verified_outcomes.json"
)
SOURCE_SHA256 = "b6c63ec5e0535877da9d4966f563964f84818334b1900433639d8afa3f0dd3f6"
DEFAULT_OUTPUT = (
    ROOT / "results/arm/keyed_insertion_recovery/analysis/"
    "paired_contact_comparison/report.json"
)
METHODS = (
    "cross_ebdm", "adaptsim", "asid", "response_mpc", "dial_transfer",
    "task_only",
)
STRONG_BASELINES = ("adaptsim", "asid", "response_mpc", "dial_transfer")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def instance_ordinal(value: str) -> int:
    match = re.search(r"(\d+)$", value)
    if match is None:
        raise ValueError(f"instance has no numeric identity: {value}")
    ordinal = int(match.group(1)) % 100
    if ordinal not in range(1, 9):
        raise ValueError(f"instance ordinal is outside 1..8: {value}")
    return ordinal


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty contact cell")
    count = len(rows)
    successes = sum(int(row["constrained_success"]) for row in rows)
    violations = sum(bool(row["constraint_violation"]) for row in rows)
    return {
        "episodes": count,
        "constrained_success_rate": successes / count,
        "constraint_violation_rate": violations / count,
        "failure_penalized_normalized_time": sum(
            float(row["capped_normalized_completion_time"]) for row in rows
        ) / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "-1"}:
        raise RuntimeError("paired contact analysis is CPU-only")
    if not SOURCE.is_file() or sha256(SOURCE) != SOURCE_SHA256:
        raise ValueError("verified formal outcomes changed")

    all_rows = json.loads(SOURCE.read_text())
    rows = [
        row for row in all_rows
        if row.get("task") == "T3" and row.get("method") in METHODS
    ]
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    indexed: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        method = str(row["method"])
        if (
            row.get("dispatch_status") != "executed"
            or row.get("target_physical_execution") != 1
            or row.get("evidence_audit_status") != "physical_evidence_verified"
        ):
            raise ValueError(f"{method} contains an unaudited physical outcome")
        ordinal = instance_ordinal(str(row["instance_id"]))
        key = (method, ordinal, str(row["request_id"]))
        if key in indexed:
            raise ValueError(f"duplicate paired outcome: {key}")
        indexed[key] = row
        by_method[method].append(row)
    if set(by_method) != set(METHODS) or any(
        len(by_method[method]) != 320 for method in METHODS
    ):
        raise ValueError("paired contact denominator must be 320 per method")

    pair_keys = {
        (instance_ordinal(str(row["instance_id"])), str(row["request_id"]))
        for row in by_method["cross_ebdm"]
    }
    for method in METHODS:
        method_keys = {
            (instance_ordinal(str(row["instance_id"])), str(row["request_id"]))
            for row in by_method[method]
        }
        if method_keys != pair_keys:
            raise ValueError(f"{method} does not use the common paired denominator")
    for ordinal, request_id in sorted(pair_keys):
        reset_hashes = {
            indexed[(method, ordinal, request_id)]["reset_physical_hash"]
            for method in METHODS
        }
        if len(reset_hashes) != 1:
            raise ValueError("physical reset hash differs within a paired request")

    summary = {method: metrics(by_method[method]) for method in METHODS}
    cells = []
    cell_keys = sorted({
        (str(row["mechanism_group"]), int(row["requested_mode"]))
        for row in by_method["cross_ebdm"]
    })
    for mechanism, mode in cell_keys:
        values = {}
        for method in METHODS:
            selected = [
                row for row in by_method[method]
                if row["mechanism_group"] == mechanism
                and int(row["requested_mode"]) == mode
            ]
            if len(selected) != 40:
                raise ValueError("each contact cell requires 40 outcomes per method")
            values[method] = metrics(selected)
        cells.append({
            "mechanism_group": mechanism,
            "requested_mode": mode,
            "methods": values,
        })

    paired_deltas = {}
    ours = "cross_ebdm"
    for baseline in (*STRONG_BASELINES, "task_only"):
        success_deltas = []
        violation_deltas = []
        time_deltas = []
        episode_wins = episode_ties = episode_losses = 0
        instance_success_deltas = {}
        for ordinal in range(1, 9):
            instance_rows = []
            for _, request_id in sorted(key for key in pair_keys if key[0] == ordinal):
                left = indexed[(ours, ordinal, request_id)]
                right = indexed[(baseline, ordinal, request_id)]
                success_delta = (
                    int(left["constrained_success"])
                    - int(right["constrained_success"])
                )
                success_deltas.append(success_delta)
                violation_deltas.append(
                    int(bool(left["constraint_violation"]))
                    - int(bool(right["constraint_violation"]))
                )
                time_deltas.append(
                    float(left["capped_normalized_completion_time"])
                    - float(right["capped_normalized_completion_time"])
                )
                episode_wins += success_delta > 0
                episode_ties += success_delta == 0
                episode_losses += success_delta < 0
                instance_rows.append(success_delta)
            instance_success_deltas[f"instance_{ordinal:02d}"] = (
                sum(instance_rows) / len(instance_rows)
            )
        paired_deltas[baseline] = {
            "ours_minus_baseline_constrained_success": (
                sum(success_deltas) / len(success_deltas)
            ),
            "ours_minus_baseline_constraint_violation": (
                sum(violation_deltas) / len(violation_deltas)
            ),
            "ours_minus_baseline_failure_penalized_normalized_time": (
                sum(time_deltas) / len(time_deltas)
            ),
            "episode_success_wins_ties_losses": {
                "wins": episode_wins,
                "ties": episode_ties,
                "losses": episode_losses,
            },
            "instance_block_success_deltas": instance_success_deltas,
            "nonnegative_instances": sum(
                value >= 0.0 for value in instance_success_deltas.values()
            ),
        }

    best_strong = max(
        STRONG_BASELINES,
        key=lambda method: summary[method]["constrained_success_rate"],
    )
    report = {
        "schema_version": "keyed-insertion-paired-contact-comparison-v1",
        "status": "complete_offline_paired_analysis",
        "task": "keyed_insertion_recovery",
        "legacy_task_alias_in_source": "T3",
        "scope": (
            "Read-only contact-focused decomposition of independently verified "
            "physical outcomes; no new physics, GPU calls, tuning, or evaluator changes."
        ),
        "design": {
            "methods": list(METHODS),
            "strong_baselines": list(STRONG_BASELINES),
            "independent_policy_instances_per_method": 8,
            "paired_requests_per_instance": 40,
            "paired_episodes_per_method": 320,
            "contact_cells": 8,
            "pairing_key": "instance_ordinal plus request_id",
            "physical_pairing_check": "identical reset_physical_hash",
        },
        "metrics": [
            "constrained_success_rate",
            "constraint_violation_rate",
            "failure_penalized_normalized_time",
        ],
        "summary": summary,
        "contact_cells": cells,
        "paired_deltas": paired_deltas,
        "claim_audit": {
            "best_strong_baseline": best_strong,
            "best_strong_baseline_success_rate": summary[best_strong][
                "constrained_success_rate"
            ],
            "ours_success_rate": summary[ours]["constrained_success_rate"],
            "ours_beats_best_strong_baseline_overall": (
                summary[ours]["constrained_success_rate"]
                > summary[best_strong]["constrained_success_rate"]
            ),
            "ours_beats_task_only_overall": (
                summary[ours]["constrained_success_rate"]
                > summary["task_only"]["constrained_success_rate"]
            ),
            "interpretation": (
                "The frozen formal actor does not support an overall strongest-method "
                "claim on keyed insertion. Cell-level gains may motivate a mechanism "
                "claim, but the aggregate gap to ASID and AdaptSim remains open."
            ),
        },
        "artifacts": {
            "verified_outcomes": pointer(SOURCE),
            "runner": pointer(Path(__file__).resolve()),
        },
        "new_physics_calls": 0,
        "gpu_calls": 0,
        "formal_outcomes_opened_for_tuning": False,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "sha256": sha256(output),
        "claim_audit": report["claim_audit"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
