#!/usr/bin/env python3
"""Independently audit a complete four-task behavioral predeployment run."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from genedynamics.tasks.transfer.episode import audit_episode_evidence

from qualify_method_behavior import summarize, summarize_diagnostic
from validate_t4_completion import _factorial_effects


ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def execution_admission(
    plan: dict, report: dict, results: list[dict], evidence_audits: list[dict],
    *, integrity_passed: bool,
) -> dict:
    """Separate actor executability from retained task-performance failures."""
    by_row: dict[str, list[dict]] = {}
    for result in results:
        by_row.setdefault(result["row_id"], []).append(result)
    evidence_by_id = {row["record_id"]: row["passed"] for row in evidence_audits}
    rows = []
    for declared in report["rows"]:
        row = deepcopy(declared)
        if int(row.get("physical_jobs", 0)):
            members = by_row.get(row["row_id"], [])
            complete = len(members) == int(row["physical_jobs"])
            evidence_complete = complete and all(
                evidence_by_id.get(member["record_id"], False)
                for member in members
            )
            execution_errors = sum(
                member.get("outcome") == "reset_or_execution_error"
                for member in members
            )
            compatible = bool(
                integrity_passed and complete and evidence_complete
                and execution_errors == 0
            )
            row.update({
                "frozen_behavior_admission_status": row["admission_status"],
                "admission_status": (
                    "behavior_qualified" if compatible else "behavior_rejected"
                ),
                "final_for_current_protocol": complete,
                "execution_compatibility_passed": compatible,
                "task_performance_failures_retained": sum(
                    not bool(member.get("passed")) for member in members
                ),
                "retained_outcomes": dict(Counter(
                    member.get("outcome") for member in members
                )),
                "reset_or_execution_errors": execution_errors,
                "evidence_complete": evidence_complete,
            })
        rows.append(row)
    public = [
        row for row in rows
        if row["family"] == "main" and row["key"] == "public_geometry"
    ]
    tasks = {row["task"] for row in rows}
    public_task_gate = (
        len(public) == len(tasks)
        and {row["task"] for row in public} == tasks
        and all(row.get("execution_compatibility_passed") is True for row in public)
    )
    final = bool(
        integrity_passed and public_task_gate
        and all(row["final_for_current_protocol"] for row in rows)
        and not any(row["admission_status"] in {
            "training_artifact_required_after_shared_cell_repair",
            "current_protocol_artifact_unavailable",
            "behavior_execution_incomplete",
        } for row in rows)
    )
    return {
        "schema_version": "behavioral-execution-admission-v1",
        "status": "formal_evaluation_admitted" if final else "admission_failed",
        "passed": final,
        "formal_evaluation_allowed": final,
        "gpu_training_allowed": final,
        "confirmation_states_opened": 0,
        "old_invalid_outcomes_used": False,
        "failure_replacement": False,
        "admission_rule": (
            "Complete SHA-audited execution without reset or execution errors is "
            "admitted. Task timeout, task failure, and constraint outcomes remain "
            "recorded performance and do not become interface rejections. The common "
            "public-geometry sentinel must execute every frozen request with complete "
            "evidence and no runner error; its task failures remain performance."
        ),
        "common_task_public_sentinel_passed": public_task_gate,
        "counts": {
            "logical_rows": len(rows),
            "admission_status": dict(Counter(
                row["admission_status"] for row in rows
            )),
            "retained_task_performance_failures": sum(
                int(row.get("task_performance_failures_retained", 0))
                for row in rows
            ),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prior-integrity-audit", type=Path)
    parser.add_argument("--admission-output", type=Path)
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else ROOT / args.run
    output = args.output or run / "independent_audit.json"
    output = output if output.is_absolute() else ROOT / output
    plan = json.loads((run / "plan.json").read_text())
    report = json.loads((run / "report.json").read_text())
    factorial = plan.get("schema_version") == "t4-transport-completion-factorial-v1"
    progress_path = run / ("progress.json" if factorial else "progress.jsonl")
    results = (
        json.loads(progress_path.read_text())
        if factorial else [
            json.loads(line) for line in progress_path.read_text().splitlines()
            if line
        ]
    )
    planned_ids = [row["record_id"] for row in plan["jobs"]]
    result_ids = [row["record_id"] for row in results]
    prior_audit = None
    prior_audit_path = None
    prior_artifacts_match = False
    if args.prior_integrity_audit is not None:
        prior_audit_path = (
            args.prior_integrity_audit if args.prior_integrity_audit.is_absolute()
            else ROOT / args.prior_integrity_audit
        )
        prior_audit = json.loads(prior_audit_path.read_text())
        prior_artifacts = prior_audit.get("artifacts", {})
        expected_paths = {
            "plan": run / "plan.json", "progress": progress_path,
            "report": run / "report.json",
        }
        prior_artifacts_match = bool(
            prior_audit.get("passed") is True
            and prior_audit.get("integrity_passed") is True
            and prior_audit.get("counts", {}).get("planned") == len(planned_ids)
            and prior_audit.get("counts", {}).get("results") == len(result_ids)
            and prior_audit.get("counts", {}).get("evidence_reaudited") == len(results)
            and all(
                label in prior_artifacts
                and resolve(prior_artifacts[label]["path"]).resolve() == path.resolve()
                and path.is_file()
                and sha(path) == prior_artifacts[label]["sha256"]
                for label, path in expected_paths.items()
            )
        )
        evidence_audits = [
            {"record_id": row["record_id"], "passed": prior_artifacts_match}
            for row in results
        ]
    else:
        evidence_audits = []
        for row in results:
            pointer = row.get("evidence")
            if not isinstance(pointer, dict):
                evidence_audits.append({"record_id": row["record_id"], "passed": False})
                continue
            path = resolve(pointer["path"])
            valid_hash = path.is_file() and sha(path) == pointer["sha256"]
            audit = audit_episode_evidence(path) if valid_hash else {"passed": False}
            evidence_audits.append({
                "record_id": row["record_id"],
                "passed": bool(valid_hash and audit["passed"]),
            })
    diagnostic = "diagnostic_spec" in plan
    if factorial:
        effects = _factorial_effects(
            results, plan["decision"],
            matched_control_admitted=bool(
                plan["policy_sets"]["task_only"]["matched_training_admitted"]
            ),
            transport_composite_admitted=bool(
                plan["policy_sets"]["transport"]["composite_qualification"]
            ),
        )
        recomputed = {
            "counts": {
                "planned_episodes": len(plan["jobs"]),
                "completed_episodes": len(results),
                "outcomes": dict(Counter(row["outcome"] for row in results)),
                "constraint_violations": sum(
                    row["constraint_violation"] for row in results
                ),
                "physical_control_steps": sum(
                    int(row.get("accounting", {}).get("control_steps", 0))
                    for row in results
                ),
            },
            "effects": effects,
        }
    else:
        recomputed = (
            summarize_diagnostic(plan, results) if diagnostic
            else summarize(plan, results)
        )
    comparable_report = {key: report[key] for key in recomputed}
    integrity_checks = {
        "plan_and_results_have_unique_ids": (
            len(planned_ids) == len(set(planned_ids))
            and len(result_ids) == len(set(result_ids))
        ),
        "results_exactly_cover_plan": set(result_ids) == set(planned_ids),
        "all_episode_evidence_reaudited": (
            len(evidence_audits) == len(results)
            and all(row["passed"] for row in evidence_audits)
        ),
        "zero_reset_or_execution_errors": not any(
            row.get("outcome") == "reset_or_execution_error" for row in results
        ),
        "summary_recomputed_exactly": comparable_report == recomputed,
        "old_invalid_outcomes_unused": report.get(
            "old_invalid_outcomes_used", False,
        ) is False,
        "confirmation_states_unopened": report.get("confirmation_states_opened") == 0,
        "declared_artifact_hashes_match": (
            prior_artifacts_match
            if prior_audit is not None else all(
                sha(resolve(pointer["path"])) == pointer["sha256"]
                for pointer in report["artifacts"].values()
            )
        ),
        **({"prior_integrity_audit_input_hashes_match": prior_artifacts_match}
           if prior_audit is not None else {}),
    }
    release_gates = (
        {"diagnostic_integrity_only": True} if diagnostic else {
            "gpu_training_admitted": report.get("gpu_training_allowed") is True,
            "formal_evaluation_admitted": report.get("formal_evaluation_allowed") is True,
        }
    )
    integrity_passed = all(integrity_checks.values())
    gpu_training_gate_passed = bool(
        not diagnostic and integrity_passed
        and release_gates["gpu_training_admitted"]
    )
    formal_evaluation_gate_passed = bool(
        not diagnostic and gpu_training_gate_passed
        and release_gates["formal_evaluation_admitted"]
    )
    admission = None
    admission_path = args.admission_output or run / "execution_admission.json"
    admission_path = (
        admission_path if admission_path.is_absolute() else ROOT / admission_path
    )
    if not diagnostic:
        admission = execution_admission(
            plan, report, results, evidence_audits,
            integrity_passed=integrity_passed,
        )
        admission_path.write_text(
            json.dumps(admission, indent=2, sort_keys=True) + "\n"
        )
        release_gates["execution_admission"] = admission["passed"]
    execution_admission_gate_passed = bool(
        not diagnostic and admission is not None and admission["passed"]
    )
    payload = {
        "schema_version": "behavioral-predeployment-independent-audit-v2",
        "status": (
            "passed_diagnostic_integrity" if diagnostic and integrity_passed else
            "passed_execution_admission_gate" if execution_admission_gate_passed else
            "passed_formal_evaluation_gate" if formal_evaluation_gate_passed else
            "passed_gpu_training_gate" if gpu_training_gate_passed else
            "failed"
        ),
        "passed": (
            integrity_passed if diagnostic else execution_admission_gate_passed
        ),
        "diagnostic_only": diagnostic,
        "integrity_passed": integrity_passed,
        "gpu_training_gate_passed": gpu_training_gate_passed,
        "formal_evaluation_gate_passed": formal_evaluation_gate_passed,
        "execution_admission_gate_passed": execution_admission_gate_passed,
        "checks": integrity_checks,
        "release_gates": release_gates,
        "counts": {
            "planned": len(planned_ids),
            "results": len(result_ids),
            "evidence_reaudited": sum(row["passed"] for row in evidence_audits),
            "outcomes": dict(Counter(row.get("outcome") for row in results)),
        },
        "artifacts": {
            "plan": {"path": str((run / "plan.json").relative_to(ROOT)), "sha256": sha(run / "plan.json")},
            "progress": {"path": str(progress_path.relative_to(ROOT)), "sha256": sha(progress_path)},
            "report": {"path": str((run / "report.json").relative_to(ROOT)), "sha256": sha(run / "report.json")},
            **({
                "execution_admission": {
                    "path": str(admission_path.relative_to(ROOT)),
                    "sha256": sha(admission_path),
                },
            } if admission is not None else {}),
            **({
                "prior_integrity_audit": {
                    "path": str(prior_audit_path.relative_to(ROOT)),
                    "sha256": sha(prior_audit_path),
                },
            } if prior_audit_path is not None else {}),
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
