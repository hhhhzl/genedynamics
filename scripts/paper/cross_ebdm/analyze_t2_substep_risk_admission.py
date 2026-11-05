#!/usr/bin/env python3
"""Close out the sealed T2 online risk-admission validation on CPU.

This analysis never instantiates physics.  It verifies and reads the frozen
report, independent audit, progress log, and episode evidence, then diagnoses
whether the selector actually respected its learned risk bounds.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    ROOT
    / "results/arm/guided_contact_alignment/validation/substep_risk_admission"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/_protocol/four_task_transfer/diagnostics"
    / "t2_substep_risk_online_failure_revision_20261022.json"
)
RISK_SOURCE = (
    ROOT / "genedynamics/solvers/single/cross_ebdm/learning/substep_risk.py"
)
ACTOR_SOURCE = (
    ROOT
    / "genedynamics/tasks/transfer/controllers/cross_ebdm/guide_corridor_v2.py"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def checked(pointer_value: dict[str, str]) -> Path:
    path = ROOT / pointer_value["path"]
    if not path.is_file() or sha(path) != pointer_value["sha256"]:
        raise ValueError(f"artifact changed: {path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def trace_snapshot(record: dict[str, Any], tick: int, *, risk: bool) -> dict[str, Any]:
    trace = record["traces"][tick]
    diagnostic = record["control_diagnostics"][tick]
    base = diagnostic["base"] if risk else diagnostic
    value = {
        "tick": tick,
        "stage": base.get("stage"),
        "contact_ticks": base.get("contact_ticks"),
        "issued_twist": record["requested_actions"][tick][:6],
        "force_peak_N": trace["force_peak"],
        "lateral_force_peak_N": trace["lateral_force_peak"],
        "normal_force_at_control_boundary_N": trace["normal_force"],
        "public_reference_lead_m": trace["public_reference_lead"],
        "axial_support_at_control_boundary": trace["axial_support"],
        "backend_status": trace["backend_status"],
    }
    if risk:
        value.update(
            selected_fraction=diagnostic["last_risk_fraction"],
            selected_index=diagnostic["last_risk_index"],
            base_upper_bound_N=diagnostic["last_risk_upper_bounds_N"][-1],
        )
    return value


def load_evidence(row: dict[str, Any]) -> dict[str, Any]:
    path = checked(row["evidence"])
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        evidence = json.load(stream)
    if len(evidence["records"]) != 1:
        raise ValueError(f"one episode required: {path}")
    return evidence["records"][0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output if args.output.is_absolute() else ROOT / args.output

    report_path = source / "report.json"
    audit_path = source / "independent_audit.json"
    progress_path = source / "progress.jsonl"
    report = read_json(report_path)
    audit = read_json(audit_path)
    if report.get("status") != "failed" or report.get("passed") is not False:
        raise ValueError("the sealed online validation must remain a failed result")
    if not audit.get("passed") or audit.get("audited_result_status") != "failed":
        raise ValueError("a passing independent audit of the failed result is required")
    for name in ("plan", "progress", "report"):
        checked(audit["artifacts"][name])
    checked(report["artifacts"]["risk_qualification"])

    source_text = RISK_SOURCE.read_text()
    if "safe[..., -1] = True" not in source_text:
        raise ValueError("the diagnosed forced-safe selector is no longer present")

    rows = [json.loads(line) for line in progress_path.open()]
    if len(rows) != report["counts"]["completed"] or len(rows) != 264:
        raise ValueError("sealed progress does not contain exactly 264 results")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate result ids")

    soft_limits = (20.0, 8.0)
    risk_episodes: list[dict[str, Any]] = []
    evidence_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["variant"] not in {"risk_admitted", "sentinel"}:
            continue
        record = load_evidence(row)
        evidence_by_id[row["record_id"]] = record
        if row["variant"] != "risk_admitted":
            continue
        diagnostics = record["control_diagnostics"]
        if len(diagnostics) != row["control_steps"]:
            raise ValueError(f"diagnostic length mismatch: {row['record_id']}")
        fallback_ticks = []
        infeasible_base_ticks = []
        forced_infeasible_ticks = []
        for tick, diagnostic in enumerate(diagnostics):
            upper = diagnostic["last_risk_upper_bounds_N"]
            chosen = int(diagnostic["last_risk_index"])
            fallback = chosen == len(upper) - 1
            base_infeasible = any(
                value > limit for value, limit in zip(upper[-1], soft_limits)
            )
            if fallback:
                fallback_ticks.append(tick)
            if base_infeasible:
                infeasible_base_ticks.append(tick)
            if fallback and base_infeasible:
                forced_infeasible_ticks.append(tick)
        risk_episodes.append(
            {
                "record_id": row["record_id"],
                "method_key": row["method_key"],
                "instance": row["instance"],
                "outcome": row["outcome"],
                "success": row["success"],
                "control_steps": row["control_steps"],
                "fallback_ticks": len(fallback_ticks),
                "model_declared_base_infeasible_ticks": len(infeasible_base_ticks),
                "forced_infeasible_fallback_ticks": len(forced_infeasible_ticks),
                "first_fallback_tick": fallback_ticks[0] if fallback_ticks else None,
                "first_model_declared_base_infeasible_tick": (
                    infeasible_base_ticks[0] if infeasible_base_ticks else None
                ),
                "last_model_declared_base_infeasible_tick": (
                    infeasible_base_ticks[-1] if infeasible_base_ticks else None
                ),
            }
        )

    method_summaries = []
    for method in ("cross_ebdm", "task_only"):
        values = [row for row in risk_episodes if row["method_key"] == method]
        first_ticks = [
            row["first_model_declared_base_infeasible_tick"]
            for row in values
            if row["first_model_declared_base_infeasible_tick"] is not None
        ]
        method_summaries.append(
            {
                "method_key": method,
                "episodes": len(values),
                "successes": sum(row["success"] for row in values),
                "outcomes": dict(sorted(Counter(row["outcome"] for row in values).items())),
                "fallback_ticks": sum(row["fallback_ticks"] for row in values),
                "episodes_with_model_declared_base_infeasible": sum(
                    row["model_declared_base_infeasible_ticks"] > 0 for row in values
                ),
                "model_declared_base_infeasible_ticks": sum(
                    row["model_declared_base_infeasible_ticks"] for row in values
                ),
                "forced_infeasible_fallback_ticks": sum(
                    row["forced_infeasible_fallback_ticks"] for row in values
                ),
                "first_model_declared_base_infeasible_tick_range": [
                    min(first_ticks), max(first_ticks)
                ],
            }
        )

    task_failure_id = (
        "risk_admitted/task_only/instance_08/"
        "T2-RISK-ADMISSION/i03/m0"
    )
    matched_ours_id = (
        "risk_admitted/cross_ebdm/instance_08/"
        "T2-RISK-ADMISSION/i03/m0"
    )
    public_failure_id = "public_geometry/fixed/T2-RISK-ADMISSION/i00/m0"
    task_row = next(row for row in risk_episodes if row["record_id"] == task_failure_id)
    task_record = evidence_by_id[task_failure_id]
    ours_record = evidence_by_id[matched_ours_id]
    public_record = evidence_by_id[public_failure_id]
    task_peak = max(
        range(len(task_record["traces"])),
        key=lambda tick: task_record["traces"][tick]["force_peak"],
    )
    public_peak = max(
        range(len(public_record["traces"])),
        key=lambda tick: public_record["traces"][tick]["force_peak"],
    )
    ours_peak = max(
        range(len(ours_record["traces"])),
        key=lambda tick: ours_record["traces"][tick]["force_peak"],
    )

    result = {
        "schema_version": "t2-substep-risk-online-failure-revision-v1",
        "status": "revision_required_before_predeployment",
        "scope": (
            "CPU-only analysis of sealed development validation; no physics, "
            "training, confirmation state, or paper performance evidence."
        ),
        "device": "cpu",
        "confirmation_states_opened": 0,
        "old_invalid_outcomes_used": False,
        "sources": {
            "validation_report": pointer(report_path),
            "independent_audit": pointer(audit_path),
            "progress": pointer(progress_path),
            "risk_qualification": report["artifacts"]["risk_qualification"],
            "selector_source": pointer(RISK_SOURCE),
            "actor_source": pointer(ACTOR_SOURCE),
        },
        "sealed_validation": {
            "episodes": 264,
            "report_status": "failed",
            "independent_audit_passed": True,
            "cells": report["cells"],
        },
        "selector_diagnosis": {
            "soft_force_limits_N": list(soft_limits),
            "last_candidate_semantics": "exact public base command horizon",
            "hard_coded_last_candidate_safe": True,
            "violated_rule": (
                "The selector overwrites the model decision for the public-base "
                "candidate and executes it when every declared candidate exceeds "
                "the calibrated soft limits."
            ),
            "method_summaries": method_summaries,
            "interpretation": (
                "All 128 risk-admitted episodes entered states where the model "
                "declared even the public base infeasible. Most still succeeded, "
                "so the online candidate distribution also exposes severe "
                "over-rejection. The single failure is not a missed prediction: "
                "the unsafe prediction was deliberately overridden."
            ),
        },
        "matched_failure": {
            "task_only": {
                **task_row,
                "last_safe_tick": trace_snapshot(
                    task_record, task_peak - 1, risk=True
                ),
                "violating_tick": trace_snapshot(
                    task_record, task_peak, risk=True
                ),
                "first_model_declared_base_infeasible": trace_snapshot(
                    task_record,
                    task_row["first_model_declared_base_infeasible_tick"],
                    risk=True,
                ),
            },
            "same_instance_same_request_cross_ebdm": {
                "record_id": matched_ours_id,
                "outcome": "success",
                "control_steps": len(ours_record["traces"]),
                "peak_tick": trace_snapshot(ours_record, ours_peak, risk=True),
            },
            "causal_limit": (
                "A same-request success does not identify transport benefit because "
                "the learned checkpoints and closed-loop states differ."
            ),
        },
        "shared_public_failure": {
            "record_id": public_failure_id,
            "outcome": "overforce",
            "last_safe_tick": trace_snapshot(
                public_record, public_peak - 1, risk=False
            ),
            "violating_tick": trace_snapshot(public_record, public_peak, risk=False),
            "common_pattern_with_task_only_failure": {
                "task_phase": "mode-0 stage-2 guide traversal",
                "issued_x_velocity_mps": 0.0075,
                "one_control_tick_force_jump": True,
                "hard_limit_changed": False,
            },
            "decision": (
                "The fresh public sentinel failure reopens qualification of the "
                "shared mode-0 traversal executor. It does not justify changing "
                "the evaluator, force limit, deadline, mode, or reset distribution."
            ),
        },
        "next_revision": {
            "current_actor_manifest_eligible": False,
            "shared_executor": [
                "Diagnose and qualify a public mode-0 stage-2 reference brake or "
                "settling action using only declared public executor state.",
                "Apply any shared executor correction identically to every method "
                "and the public sentinel; it cannot count as a Cross-EBDM gain.",
                "Require a new disjoint public-sentinel cohort before learned-method "
                "qualification; preserve this failed cohort without replacement.",
            ],
            "risk_admission_v2": [
                "Remove the forced-safe assignment for the last candidate.",
                "Add an explicit finite emergency reference-brake horizon to the "
                "declared candidate set and send it through the same risk model.",
                "If no candidate satisfies the calibrated bounds, terminate with an "
                "explicit risk_set_empty outcome counted as failure; never execute a "
                "candidate by overriding its bound.",
                "Collect fit/calibration/validation blocks from the frozen deployment "
                "candidate generator, including interpolation and emergency horizons; "
                "recorded behavior-policy coverage alone is insufficient.",
                "Qualify both coverage and usefulness: episode-block coverage, high-risk "
                "recall, safe-window rejection, empty-set rate, progress, and online "
                "constraint violations on a disjoint closed-loop cohort.",
            ],
            "causal_comparison": [
                "Give the same qualified risk admission to ours and matched task-only.",
                "Keep external baselines on their declared native mechanisms plus the "
                "shared executor; do not relabel a baseline after adding the ours risk head.",
                "Add a no-substep-admission ablation only after v2 passes; do not reuse "
                "the existing anchoring-gate ablation for this question.",
            ],
            "forbidden_shortcuts": [
                "Do not lower or raise the frozen hard force limits to rescue a method.",
                "Do not interpret marginal conformal coverage as closed-loop invariance.",
                "Do not update actor bindings or start a paper roster from this result.",
            ],
        },
        "release_sequence": [
            "shared public-executor diagnosis and fresh sentinel qualification",
            "risk-admission-v2 zero-physics identity and empty-set tests",
            "candidate-distribution fit/calibration/held-out qualification",
            "fresh paired online ours/task-only qualification",
            "full zero-physics, package, device, and behavioral predeployment",
            "new disjoint four-task roster only after every preceding gate passes",
        ],
    }
    write_json(output, result)
    print(json.dumps({"output": str(output), "sha256": sha(output), "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
