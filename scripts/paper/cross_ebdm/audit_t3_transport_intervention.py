#!/usr/bin/env python3
"""Audit the first transport intervention against a matched task-only run.

This reads immutable episode evidence from a frozen T3 actor roster. It does
not reset a simulator or use a GPU, and is development diagnosis only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genedynamics.core.artifacts import atomic_json
from genedynamics.tasks.transfer.episode import load_episode_evidence
from genedynamics.tasks.transfer.method_evaluation import (
    audit_paired_evaluations,
    manifest_roster_config,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(row: dict) -> tuple[list, list]:
    evidence = row["evidence"]
    path = ROOT / evidence["path"]
    if _sha256(path) != evidence["sha256"]:
        raise ValueError(f"episode evidence SHA changed: {path}")
    records, _, _ = load_episode_evidence(path)
    if len(records) != 1:
        raise ValueError(f"expected one T3 episode in {path}")
    record = records[0]
    if len(record.traces) != len(record.control_diagnostics):
        raise ValueError(f"T3 control diagnostics missing: {path}")
    return record.traces, record.control_diagnostics


def audit(roster_path: Path) -> dict:
    roster_path = roster_path.resolve()
    roster = json.loads(roster_path.read_text())
    if roster["task"] != "T3" or not {"cross_ebdm", "task_only"} <= set(roster["methods"]):
        raise ValueError("requires a frozen T3 roster containing cross_ebdm and task_only")
    configs = {name: manifest_roster_config(roster_path, name)
               for name in ("cross_ebdm", "task_only")}
    paired = audit_paired_evaluations(configs["cross_ebdm"], configs["task_only"], write=False)
    if paired["status"] != "passed_development_only":
        raise ValueError("paired physical evidence has not passed independent audit")
    reports = {name: json.loads((ROOT / config["output"] / "report.json").read_text())
               for name, config in configs.items()}
    results = []
    for ours, task, pair in zip(
        reports["cross_ebdm"]["results"], reports["task_only"]["results"],
        paired["pairs"], strict=True,
    ):
        request_id = pair["request_id"]
        if ours["request"]["request_id"] != request_id or task["request"]["request_id"] != request_id:
            raise ValueError("paired request order changed")
        traces, diagnostics = _record(ours)
        task_traces, _ = _record(task)
        admissions = [int(row["transport_admission_controls"]) for row in diagnostics]
        if any(a < 0 or b < a or b - a > 1
               for a, b in zip([0, *admissions[:-1]], admissions)):
            raise ValueError(f"transport admission counter invalid: {request_id}")
        admission_indices = [i for i, (before, after) in enumerate(
            zip([0, *admissions[:-1]], admissions)
        ) if after > before]
        first_admission = admission_indices[0] if admission_indices else None
        first_difference = next((i for i, (a, b) in enumerate(zip(traces, task_traces))
                                 if not np.array_equal(a["issued_action"], b["issued_action"])), None)
        if first_admission is None and len(traces) != len(task_traces):
            raise ValueError(f"no-admission trajectories differ in length: {request_id}")
        if first_difference is not None and (
            first_admission is None or first_difference < first_admission
        ):
            raise ValueError(f"action differed before recorded transport admission: {request_id}")
        prefix = first_admission if first_admission is not None else len(traces)
        keys = ("issued_action", "tcp_position", "joint_position", "force_peak")
        exact_prefix = all(
            all(np.array_equal(a[key], b[key]) for key in keys)
            for a, b in zip(traces[:prefix], task_traces[:prefix], strict=True)
        )
        full_trace_prefix_exact = traces[:prefix] == task_traces[:prefix]
        if not exact_prefix or not full_trace_prefix_exact:
            raise ValueError(f"physical trajectory differed before transport admission: {request_id}")
        if first_admission is None and (
            first_difference is not None or ours["outcome"] != task["outcome"]
        ):
            raise ValueError(f"no-admission pair changed action or outcome: {request_id}")
        action_delta = None
        if first_difference is not None:
            action_delta = float(np.max(np.abs(
                np.asarray(traces[first_difference]["issued_action"], dtype=float)
                - np.asarray(task_traces[first_difference]["issued_action"], dtype=float)
            )))
        results.append({
            "request_id": request_id,
            "scene_cell": pair["scene_cell"],
            "reset_physical_hash": pair["reset_physical_hash"],
            "evidence_sha256": {
                "cross_ebdm": ours["evidence"]["sha256"],
                "task_only": task["evidence"]["sha256"],
            },
            "outcome": {"cross_ebdm": ours["outcome"], "task_only": task["outcome"]},
            "control_steps": {"cross_ebdm": len(traces), "task_only": len(task_traces)},
            "first_transport_admission_control_index": first_admission,
            "first_issued_action_difference_control_index": first_difference,
            "first_action_max_abs_difference": action_delta,
            "transport_admission_controls": admissions[-1] if admissions else 0,
            "transport_admission_control_indices": admission_indices,
            "first_admission_substep_force_peak_N": (
                float(traces[first_admission]["force_peak"]) if first_admission is not None else None
            ),
            "first_admission_substep_lateral_force_peak_N": (
                float(traces[first_admission]["lateral_force_peak"])
                if first_admission is not None else None
            ),
            "terminal_substep_force_peak_N": float(traces[-1]["force_peak"]),
            "terminal_substep_lateral_force_peak_N": float(traces[-1]["lateral_force_peak"]),
            "pre_admission_action_tcp_joint_force_exact": exact_prefix,
            "pre_admission_full_physical_trace_exact": full_trace_prefix_exact,
        })
    admitted = [row for row in results if row["first_transport_admission_control_index"] is not None]
    return {
        "version": "t3-first-transport-intervention-audit-v1",
        "status": "passed_development_only",
        "roster": {"path": str(roster_path.relative_to(ROOT)), "sha256": _sha256(roster_path)},
        "paired_report_sha256": paired["report_sha256"],
        "requests": len(results),
        "admitted_requests": len(admitted),
        "admitted_ours_failure_task_only_success": sum(
            row["outcome"]["cross_ebdm"] != "success"
            and row["outcome"]["task_only"] == "success" for row in admitted
        ),
        "no_admission_identical_trajectory_requests": len(results) - len(admitted),
        "pairs": results,
        "physical_calls": 0, "gpu_calls": 0,
        "formal_performance_evidence": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, required=True)
    args = parser.parse_args()
    path = args.roster if args.roster.is_absolute() else ROOT / args.roster
    report = audit(path)
    destination = (ROOT / "results/arm/keyed_insertion_recovery/validation" /
                   f"method_source_nominal_actor_screen_{json.loads(path.read_text())['requests'][0]['reset_seed']}" /
                   "first_intervention_audit.json")
    atomic_json(destination, report)
    print(json.dumps({"status": report["status"], "requests": report["requests"],
                      "admitted_requests": report["admitted_requests"],
                      "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
