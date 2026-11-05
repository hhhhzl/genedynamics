#!/usr/bin/env python3
"""Reproduce the static keyed-insertion overforce diagnosis from stored traces."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from genedynamics.tasks.transfer.keyed_insertion_diagnostics import (
    diagnose_episode,
    summarize_paired_diagnoses,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    ROOT / "results/_analysis/four_task_transfer/"
    "keyed_insertion_overforce_diagnosis_20261105/report.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def collect(relative: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows = []
    sources = []
    base = ROOT / "results/arm/keyed_insertion_recovery" / relative
    for instance_index in range(1, 9):
        instance_id = f"instance_{instance_index:02d}"
        episode_root = (
            base / "confirmation/independent_main_v2" / instance_id / "episodes"
        )
        result_paths = sorted(episode_root.glob("*/result.json"))
        if len(result_paths) != 40:
            raise ValueError(f"{relative}/{instance_id} must contain 40 episodes")
        for result_path in result_paths:
            evidence_path = result_path.with_name("evidence.json.gz")
            result = read_json(result_path)
            result["instance_id"] = instance_id
            row = diagnose_episode(result, read_gzip_json(evidence_path))
            rows.append(row)
            sources.append({
                "result": str(result_path.relative_to(ROOT)),
                "result_sha256": sha256(result_path),
                "evidence": str(evidence_path.relative_to(ROOT)),
                "evidence_sha256": sha256(evidence_path),
            })
    return rows, sources


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    ours, ours_sources = collect("main/cross_ebdm")
    task_only, task_sources = collect("baseline/task_only")
    report = {
        "schema_version": "keyed-insertion-overforce-diagnosis-v1",
        "status": "complete_offline_diagnosis",
        "scope": (
            "read-only analysis of independent-main-v2 physical traces; "
            "no new physics, GPU, tuning, or evaluator changes"
        ),
        "formal_summary": {
            "path": (
                "results/_analysis/four_task_transfer/"
                "independent_main_table_formal_v2/verified_outcomes.json"
            ),
            "sha256": sha256(
                ROOT / "results/_analysis/four_task_transfer/"
                "independent_main_table_formal_v2/verified_outcomes.json"
            ),
        },
        "summary": summarize_paired_diagnoses(ours, task_only),
        "interpretation": {
            "shared_execution_failure": (
                "158 of 185 ICDT overforce outcomes are paired task-only overforce "
                "outcomes; the first repair target is the shared early recovery path"
            ),
            "transport_specific_failure": (
                "27 regressions are all mode 0 and all follow at least one transport "
                "admission; only 8 are admitted on the terminal control, so admission "
                "must diagnose delayed force consequences rather than the last tick alone"
            ),
            "force_channel": (
                "160 of 185 overforce outcomes involve the lateral threshold"
            ),
            "recovery_event": (
                "all 185 terminate in phase 0 or 1 before retreating becomes true"
            ),
            "next_development_gate": (
                "qualify one shared early-contact/reference-unloading recovery, then "
                "qualify force-aware transport admission on held-out development requests"
            ),
        },
        "source_integrity": {
            "ours_episode_files": len(ours_sources),
            "task_only_episode_files": len(task_sources),
            "episode_sources": ours_sources + task_sources,
        },
        "gpu_calls": 0,
        "new_physics_calls": 0,
        "formal_outcomes_opened_for_tuning": False,
    }
    write_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "sha256": sha256(args.output),
        "summary": report["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
