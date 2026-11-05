#!/usr/bin/env python3
"""Independently re-score saved T3 same-state force-fork evidence on CPU."""
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
from genedynamics.tasks.transfer.episode import audit_episode_evidence, load_episode_evidence


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(pointer: dict[str, str]) -> Path:
    path = ROOT / pointer["path"]
    if sha(path) != pointer["sha256"]:
        raise ValueError(f"bound artifact SHA changed: {path}")
    return path


def audit(report_path: Path) -> dict:
    report = json.loads(report_path.read_text())
    if (report["version"] != "t3-paired-force-fork-preflight-v1"
            or report["status"] != "complete_development_preflight"
            or report["formal_performance_evidence"] is not False):
        raise ValueError("T3 fork report is not a completed development preflight")
    config = json.loads(checked(report["config"]).read_text())
    checked(report["runner"])
    checked(report["actor_roster"])
    checked(report["native_library"])
    if report["requests"] != config["requests"] or len(report["results"]) != len(report["requests"]):
        raise ValueError("T3 fork request list differs")
    rows = []
    charged = 0
    for request, result in zip(report["requests"], report["results"], strict=True):
        if result["request"] != request or result["status"] != "paired_fork_audited":
            raise ValueError("T3 fork request or status differs")
        if (result["same_state_public_restore"] is not True
                or result["first_action_exact_replay"] is not True
                or result["first_admission_control_index"] != result["prefix_control_steps"]
                or set(result["branches"]) != {"cross_ebdm", "task_only"}):
            raise ValueError("T3 same-state fork receipt differs")
        traces = {}
        requested_actions = {}
        branch_rows = {}
        for method, branch in result["branches"].items():
            path = checked(branch["evidence"])
            evidence_audit = audit_episode_evidence(path)
            if not evidence_audit["passed"]:
                raise ValueError(f"physical evidence failed independent scoring: {path}")
            records, _, evaluation = load_episode_evidence(path)
            if len(records) != 1 or len(records[0].traces) != len(records[0].control_diagnostics):
                raise ValueError(f"T3 fork trace and diagnostic counts differ: {path}")
            prefix = result["prefix_control_steps"]
            tail = records[0].traces[prefix:]
            if (len(tail) != branch["control_steps"]
                    or not 0 < len(tail) <= config["branch_horizon_controls"]):
                raise ValueError("T3 fork branch length differs")
            force = max(float(row["force_peak"]) for row in tail)
            lateral = max(float(row["lateral_force_peak"]) for row in tail)
            if (not np.isclose(force, branch["maximum_substep_force_N"], rtol=0., atol=0.)
                    or not np.isclose(lateral, branch["maximum_substep_lateral_force_N"],
                                      rtol=0., atol=0.)
                    or evaluation["episodes"][-1]["outcome"] != branch["evidence"]["outcome"]):
                raise ValueError("T3 fork substep force or outcome differs from physical trace")
            traces[method] = records[0].traces
            requested_actions[method] = records[0].requested_actions
            branch_rows[method] = {
                "evidence_sha256": branch["evidence"]["sha256"],
                "physical_control_steps": len(tail),
                "outcome": evaluation["episodes"][-1]["outcome"],
                "maximum_substep_force_N": force,
                "maximum_substep_lateral_force_N": lateral,
            }
        prefix = result["prefix_control_steps"]
        if traces["cross_ebdm"][:prefix] != traces["task_only"][:prefix]:
            raise ValueError("T3 fork branches do not share the full physical prefix")
        requested_a = np.asarray(requested_actions["cross_ebdm"][prefix], dtype=float)
        requested_b = np.asarray(requested_actions["task_only"][prefix], dtype=float)
        requested_difference = float(np.max(np.abs(requested_a - requested_b)))
        issued_a = np.asarray(traces["cross_ebdm"][prefix]["issued_action"], dtype=float)
        issued_b = np.asarray(traces["task_only"][prefix]["issued_action"], dtype=float)
        issued_difference = float(np.max(np.abs(issued_a - issued_b)))
        if not np.isclose(requested_difference, result["first_action_max_abs_difference"],
                          rtol=0., atol=1e-7):
            raise ValueError("T3 fork first policy-requested action difference differs")
        row_charge = prefix + sum(value["physical_control_steps"]
                                  for value in branch_rows.values())
        if row_charge != result["physical_control_steps_charged"]:
            raise ValueError("T3 fork physical control-step ledger differs")
        charged += row_charge
        rows.append({"request_id": request["request_id"],
                     "scene_cell": request["scene_cell"],
                     "exact_common_physical_prefix_controls": prefix,
                     "first_requested_action_max_abs_difference": requested_difference,
                     "first_issued_action_max_abs_difference": issued_difference,
                     "physical_control_steps_charged": row_charge,
                     "branches": branch_rows})
    if (charged != report["physical_control_steps_charged"]
            or charged > report["maximum_physical_control_steps"]):
        raise ValueError("T3 fork physical budget differs")
    return {
        "version": "t3-paired-force-fork-independent-audit-v1",
        "status": "passed_development_only",
        "report_sha256": sha(report_path),
        "config_sha256": report["config"]["sha256"],
        "audit_runner": {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                         "sha256": sha(Path(__file__).resolve())},
        "physical_evidence_verified": 2 * len(rows),
        "physical_control_steps_reconciled": charged,
        "rows": rows,
        "physical_calls": 0, "gpu_calls": 0,
        "formal_performance_evidence": False,
        "scope": "CPU-only trace/score/force/paired-prefix audit; not calibrated safety or task performance",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    path = args.report if args.report.is_absolute() else ROOT / args.report
    result = audit(path)
    output = path.parent / "independent_audit.json"
    atomic_json(output, result)
    print(json.dumps({"status": result["status"],
                      "physical_evidence_verified": result["physical_evidence_verified"],
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
