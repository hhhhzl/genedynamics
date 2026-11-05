#!/usr/bin/env python3
"""Close the CPU-only common-failure analysis before the next protocol revision."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PREDEPLOYMENT = ROOT / (
    "results/_analysis/four_task_transfer/"
    "predeployment_20261021_v11_v2/closeout.json"
)
RISK_REVISION = ROOT / (
    "results/_protocol/four_task_transfer/diagnostics/"
    "t2_substep_risk_online_failure_revision_20261022.json"
)
CAPABILITY_CLOSEOUT = ROOT / (
    "results/_protocol/four_task_transfer/confirmation_v1/diagnostics/"
    "t2_t3_t4_capability_closeout.json"
)
INVALID_CONFIRMATION_DECISION = ROOT / (
    "results/_protocol/four_task_transfer/confirmation_v1/diagnostics/"
    "next_revision_decision.json"
)
T2_STRUCTURAL_ACCESSED = ROOT / (
    "results/arm/guided_contact_alignment/validation/"
    "mode0_post_outlet_concurrent_feedback_accessed_diagnosis/report.json"
)
T2_STRUCTURAL_FRESH = ROOT / (
    "results/arm/guided_contact_alignment/validation/"
    "mode0_post_outlet_concurrent_feedback_fresh_qualification/report.json"
)
ARCHITECTURE_V3 = ROOT / (
    "results/_architecture/cross_ebdm_shared_protocol_20261022_v3/report.json"
)
COMMON_CONTRACTS = {
    "T2": ROOT / (
        "configs/arm/guided_contact_alignment/protocol/task_contract_v3.json"
    ),
    "T3": ROOT / (
        "configs/arm/keyed_insertion_recovery/protocol/public_v6.json"
    ),
    "T4": ROOT / (
        "configs/quadruped/mobile_keyed_insertion_recovery/protocol/"
        "task_contract_v3.json"
    ),
}
T2_REPORTS = {
    "accessed_speed": ROOT / (
        "results/arm/guided_contact_alignment/validation/"
        "mode0_deadline_reachable_transit_speed/report.json"
    ),
    "fresh_speed": ROOT / (
        "results/arm/guided_contact_alignment/validation/"
        "mode0_transit_speed_fresh_qualification/report.json"
    ),
    "raw_lift_00650": ROOT / (
        "results/arm/guided_contact_alignment/validation/"
        "mode0_post_outlet_lift_accessed_diagnosis/report.json"
    ),
    "raw_lift_00675": ROOT / (
        "results/arm/guided_contact_alignment/validation/"
        "mode0_post_outlet_lift_deadline_margin/report.json"
    ),
    "spatial_lead_lift_00675": ROOT / (
        "results/arm/guided_contact_alignment/validation/"
        "mode0_post_outlet_lead_governor_accessed_diagnosis/report.json"
    ),
    "raw_planar_lead_lift_00675": ROOT / (
        "results/arm/guided_contact_alignment/validation/"
        "mode0_post_outlet_raw_planar_lead_accessed_diagnosis/report.json"
    ),
}
INVALID_PLANAR_BINDING = ROOT / (
    "results/arm/guided_contact_alignment/validation/"
    "mode0_post_outlet_planar_lead_accessed_diagnosis/INVALID_RUN.json"
)
DEFAULT_OUTPUT = ROOT / (
    "results/_protocol/four_task_transfer/diagnostics/"
    "cpu_shared_failure_closeout_20261022.json"
)
FORMAL_DIAGNOSIS = ROOT / (
    "results/_analysis/four_task_transfer/confirmation_v8/diagnosis_plan.json"
)
T2_PAIRED_DIAGNOSIS = ROOT / (
    "results/arm/guided_contact_alignment/validation/"
    "latency_liveness_paired/report.json"
)
T2_DEADLINE_DIAGNOSIS = ROOT / (
    "results/arm/guided_contact_alignment/validation/"
    "deadline_reference_admission_formal_cells_paired/report.json"
)
T2_INVALID_DEADLINE_SEAL = ROOT / (
    "results/arm/guided_contact_alignment/validation/"
    "deadline_reference_admission_paired/INVALID_RUN.json"
)
T3_PAIRED_DIAGNOSIS = ROOT / (
    "results/arm/keyed_insertion_recovery/validation/"
    "terminal_hold_paired/report.json"
)
T3_TERMINAL_DIAGNOSIS = ROOT / (
    "results/arm/keyed_insertion_recovery/validation/"
    "terminal_hold_admission_paired/report.json"
)
DEFAULT_DEVELOPMENT_OUTPUT = ROOT / (
    "results/_analysis/four_task_transfer/confirmation_v8/"
    "development_diagnosis.json"
)
DEFAULT_T2_ACTION_SUPPORT_OUTPUT = ROOT / (
    "results/arm/guided_contact_alignment/validation/"
    "deadline_reference_admission_formal_cells_paired/"
    "action_support_screen.json"
)
DEFAULT_NEXT_REVISION_OUTPUT = ROOT / (
    "results/_analysis/four_task_transfer/confirmation_v8/"
    "next_revision_closeout.json"
)
DEFAULT_T2_TRANSITION_OUTPUT = ROOT / (
    "results/_analysis/four_task_transfer/confirmation_v8/"
    "t2_response_normalized_revision_closeout.json"
)
T2_TRANSITION_REPORTS = {
    name: ROOT / (
        "results/arm/guided_contact_alignment/validation/"
        f"{name}/report.json"
    )
    for name in (
        "response_normalized_public_geometry_gate",
        "response_normalized_settled_axis_transition",
        "response_normalized_measured_set_transition",
        "response_normalized_route_limit_unloading_transition",
        "response_normalized_reserved_terminal_transition",
        "response_normalized_clearance_regression_gate",
        "response_normalized_invariant_descent_gate",
        "response_normalized_force_scaled_descent_gate",
        "response_normalized_public_geometry_acceptance_gate",
        "response_normalized_public_geometry_joint_gate",
        "response_normalized_public_geometry_release_gate",
    )
}
T2_TRANSITION_REPORTS["response_normalized_public_geometry_release_gate"] = (
    ROOT / "results/arm/guided_contact_alignment/validation/"
    "hybrid_force_terminal_release_gate/report.json"
)
T2_SHARED_EXECUTOR_SCREENS = {
    "predictive_projection": ROOT / (
        "results/arm/guided_contact_alignment/validation/"
        "delay_robust_completion_paired/report.json"
    ),
    "direct_substep_governed": ROOT / (
        "results/arm/guided_contact_alignment/validation/"
        "substep_governed_completion_paired/report.json"
    ),
    "feedback_governed": ROOT / (
        "results/arm/guided_contact_alignment/validation/"
        "feedback_governed_completion_screen/report.json"
    ),
    "monotone_outlet": ROOT / (
        "results/arm/guided_contact_alignment/validation/"
        "monotone_outlet_completion_screen/report.json"
    ),
    "bounded_clearance": ROOT / (
        "results/arm/guided_contact_alignment/validation/"
        "bounded_clearance_completion_screen/report.json"
    ),
    "concurrent_acquisition": ROOT / (
        "results/arm/guided_contact_alignment/validation/"
        "concurrent_acquisition_completion_screen/report.json"
    ),
    "dual_waypoint": ROOT / (
        "results/arm/guided_contact_alignment/validation/"
        "dual_waypoint_completion_screen/report.json"
    ),
    "fast_acquisition": ROOT / (
        "results/arm/guided_contact_alignment/validation/"
        "fast_acquisition_completion_screen/report.json"
    ),
    "waypoint_first": ROOT / (
        "results/arm/guided_contact_alignment/validation/"
        "waypoint_first_completion_screen/report.json"
    ),
    "staged_acquisition": ROOT / (
        "results/arm/guided_contact_alignment/validation/"
        "staged_acquisition_completion_screen/report.json"
    ),
}
T2_RESPONSE_NORMALIZED_SCREENS = {
    name: ROOT / (
        "results/arm/guided_contact_alignment/validation/"
        f"{name}/report.json"
    )
    for name in (
        "response_normalized_tracking_authority",
        "response_normalized_phase_budget",
        "response_normalized_terminal_capture",
        "response_normalized_descent_reserve",
        "response_normalized_concurrent_completion",
        "response_normalized_surface_completion",
        "response_normalized_terminal_force",
        "response_normalized_terminal_contact",
        "response_normalized_terminal_admittance",
        "response_normalized_outlet_reserve",
        "response_normalized_mode_routes",
    )
}
T3_FINITE_TIME_QUALIFICATION = ROOT / (
    "results/arm/keyed_insertion_recovery/validation/"
    "finite_time_late_phase_paired/gpu1/report.json"
)
T3_ACTOR_BINDINGS = ROOT / (
    "configs/arm/keyed_insertion_recovery/protocol/actor_bindings.json"
)
T2_CONTACT_SOURCE = ROOT / "genedynamics/tasks/transfer/contact.py"
T2_GUIDE_SOURCE = ROOT / (
    "genedynamics/tasks/transfer/controllers/guide_contact.py"
)
T2_GOAL_POSITION_M = (0.46, -0.18, 0.596)
T2_OUTLET_POSITION_M = {0: 0.455, 1: -0.181}
T2_HOLD_CONTROLS = 15

NEXT_REVISION_SOURCES = {
    "formal_diagnosis": FORMAL_DIAGNOSIS,
    "development_diagnosis": DEFAULT_DEVELOPMENT_OUTPUT,
    "t2_action_support": DEFAULT_T2_ACTION_SUPPORT_OUTPUT,
    "t3_finite_time_qualification": T3_FINITE_TIME_QUALIFICATION,
    "same_policy_guard_stress": ROOT / (
        "results/arm/guided_contact_alignment/validation/"
        "same_policy_contact_guard_stress/report.json"
    ),
    "t2_transport_only_qualification": ROOT / (
        "results/arm/guided_contact_alignment/qualification/"
        "transport_only_actor_predeployment/report.json"
    ),
    "four_task_behavior_qualification": ROOT / (
        "results/_qualification/four_task_transfer/"
        "behavioral_predeployment_transport_only_t2_20261024/report.json"
    ),
    "architecture_audit": ROOT / (
        "results/_architecture/"
        "cross_ebdm_transport_only_confirmation_v10_20261024/report.json"
    ),
    "candidate_package_preaccess_audit": ROOT / (
        "results/_protocol/four_task_transfer/confirmation_v10/package/"
        "preaccess_audit.json"
    ),
    "t2_actor_manifest": ROOT / (
        "configs/arm/guided_contact_alignment/protocol/"
        "actor_bindings_transport_only_v1.json"
    ),
    "t3_actor_manifest": ROOT / (
        "configs/arm/keyed_insertion_recovery/protocol/actor_bindings.json"
    ),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    try:
        display_path = path.relative_to(ROOT)
    except ValueError:
        display_path = path
    return {"path": str(display_path), "sha256": sha(path)}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def candidate_cell(report: dict[str, Any]) -> dict[str, Any]:
    values = [cell for cell in report["cells"] if cell["profile_key"] == "candidate"]
    if len(values) != 1:
        raise ValueError("expected exactly one candidate cell")
    return values[0]


def progress_path(report_path: Path, report: dict[str, Any]) -> Path:
    value = report["artifacts"]["progress"]
    path = ROOT / value["path"]
    if sha(path) != value["sha256"]:
        raise ValueError(f"progress artifact changed: {path}")
    if path.parent != report_path.parent:
        raise ValueError("report and progress directory differ")
    return path


def load_evidence(value: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / value["path"]
    if sha(path) != value["sha256"]:
        raise ValueError(f"episode evidence changed: {path}")
    return json.loads(gzip.decompress(path.read_bytes()))


def _diagnostic_rows(report_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = read_json(report_path)
    if (
        report["status"] != "complete_recorded_development_diagnosis"
        or not report["passed"]
        or not report["diagnostic_only"]
        or report["performance_evidence"]
        or report["confirmation_states_opened"] != 0
        or report["failure_replacement"]
    ):
        raise ValueError(f"invalid development-diagnosis report: {report_path}")
    rows = [
        json.loads(line)
        for line in progress_path(report_path, report).read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != report["counts"]["completed_physical_jobs"]:
        raise ValueError(f"diagnostic row count differs: {report_path}")
    return report, rows


def _median_or_none(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return median(finite) if finite else None


def _cell_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            row["key"], row["request"]["scene_cell"],
            int(row["request"]["requested_mode"]),
        )].append(row)
    return {
        "|".join(map(str, key)): {
            "episodes": len(values),
            "successes": sum(bool(row["success"]) for row in values),
            "constraint_violations": sum(
                bool(row["constraint_violation"]) for row in values
            ),
            "median_first_contact_step": _median_or_none([
                row["episode_metrics"]["first_contact_step"] for row in values
            ]),
            "median_first_outlet_step": _median_or_none([
                row["episode_metrics"]["first_outlet_step"] for row in values
            ]),
            "median_final_position_error_m": _median_or_none([
                row["episode_metrics"]["final_position_error_m"] for row in values
            ]),
        }
        for key, values in sorted(grouped.items())
    }


def _row_outcomes(report: dict[str, Any]) -> dict[str, Any]:
    return {
        row["key"]: {
            "episodes": row["completed_physical_jobs"],
            "successes": row["successes"],
            "constraint_violations": row["constraint_violations"],
            "execution_errors": row["execution_errors"],
            "outcomes": row["outcomes"],
        }
        for row in report["rows"]
    }


def _first_index(values: list[Any], predicate) -> int | None:
    return next((index for index, value in enumerate(values) if predicate(value)), None)


def _phase_from_diagnostic(value: dict[str, Any]) -> int | None:
    base = value.get("base", value)
    phase = base.get("observer", {}).get("memory", {}).get("phase")
    return None if phase is None else int(phase[0])


def _t3_timing(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    paired: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)
    for row in rows:
        evidence = load_evidence(row["evidence"])
        if len(evidence["records"]) != 1:
            raise ValueError("T3 diagnosis requires one record per artifact")
        record = evidence["records"][0]
        diagnostics, traces = record["control_diagnostics"], record["traces"]
        phases = [_phase_from_diagnostic(value) for value in diagnostics]
        holds = [int(value["execution_hold_count"]) for value in traces]
        final = diagnostics[-1]
        groups[(row["key"], bool(row["success"]))].append({
            "control_steps": len(traces),
            "phase3_entry": _first_index(phases, lambda value: value == 3),
            "first_hold_eligible": _first_index(holds, lambda value: value > 0),
            "completion_tick": _first_index(holds, lambda value: value >= 15),
            "maximum_hold": max(holds),
            "terminal_admission_controls": int(
                final.get("completion_terminal_controls", 0)
            ),
            "candidate_controls": int(
                final.get("completion_candidate_controls", 0)
            ),
        })
        paired[(
            row["instance"], int(row["request"]["physical_initial_slot"]),
        )][row["key"]] = int(row["success"])
    summary = {}
    for (key, success), values in sorted(groups.items()):
        label = f"{key}|{'success' if success else 'failure'}"
        summary[label] = {
            "episodes": len(values),
            "median_phase3_entry": _median_or_none([
                value["phase3_entry"] for value in values
            ]),
            "median_first_hold_eligible": _median_or_none([
                value["first_hold_eligible"] for value in values
            ]),
            "median_remaining_controls_at_first_eligibility": _median_or_none([
                value["control_steps"] - value["first_hold_eligible"]
                for value in values if value["first_hold_eligible"] is not None
            ]),
            "median_completion_tick": _median_or_none([
                value["completion_tick"] for value in values
            ]),
            "maximum_hold_counts": dict(sorted(Counter(
                value["maximum_hold"] for value in values
            ).items())),
            "median_terminal_admission_controls": _median_or_none([
                value["terminal_admission_controls"] for value in values
            ]),
        }
    key_order = (
        "cross_ebdm_current", "cross_ebdm_terminal_hold", "response_mpc",
    )
    patterns = Counter(
        tuple(value[key] for key in key_order) for value in paired.values()
    )
    return {
        "groups": summary,
        "paired_success_patterns": {
            str(pattern): count for pattern, count in sorted(patterns.items())
        },
    }


def development_diagnosis_closeout(output_path: Path) -> None:
    """Combine formal failures with fresh CPU-only mechanism tests."""
    output_path = output_path if output_path.is_absolute() else ROOT / output_path
    formal = read_json(FORMAL_DIAGNOSIS)
    if formal["status"] != "complete_cpu_only_diagnosis_next_revision_required":
        raise ValueError("formal failure closeout is not complete")
    t2_report, t2_rows = _diagnostic_rows(T2_PAIRED_DIAGNOSIS)
    t2_deadline_report, t2_deadline_rows = _diagnostic_rows(T2_DEADLINE_DIAGNOSIS)
    invalid_t2_deadline = read_json(T2_INVALID_DEADLINE_SEAL)
    if invalid_t2_deadline["status"] != "invalid_for_formal_cell_attribution":
        raise ValueError("mismatched T2 deadline diagnosis is not sealed invalid")
    for value in invalid_t2_deadline["artifacts"].values():
        if sha(ROOT / value["path"]) != value["sha256"]:
            raise ValueError("invalid T2 diagnosis seal points to changed evidence")
    t3_report, _ = _diagnostic_rows(T3_PAIRED_DIAGNOSIS)
    t3_terminal_report, t3_terminal_rows = _diagnostic_rows(T3_TERMINAL_DIAGNOSIS)
    expected = {
        "t2_original": {"public_geometry": 8},
        "t2_deadline": {
            "public_geometry_current": 8,
            "public_geometry_deadline_aware": 8,
        },
        "t3_original": {
            "cross_ebdm": 31, "band_only": 49, "response_mpc": 64,
        },
        "t3_terminal": {
            "cross_ebdm_current": 40,
            "cross_ebdm_terminal_hold": 24,
            "response_mpc": 64,
        },
    }
    actual = {
        "t2_original": {row["key"]: row["successes"] for row in t2_report["rows"]},
        "t2_deadline": {
            row["key"]: row["successes"] for row in t2_deadline_report["rows"]
        },
        "t3_original": {row["key"]: row["successes"] for row in t3_report["rows"]},
        "t3_terminal": {
            row["key"]: row["successes"] for row in t3_terminal_report["rows"]
        },
    }
    if actual != expected:
        raise ValueError(f"development diagnosis outcomes changed: {actual}")

    output = {
        "schema_version": "four-task-next-revision-development-diagnosis-v1",
        "status": "complete_cpu_only_diagnosis_next_revision_scoped",
        "scope": (
            "Post-confirmation offline statistics plus fresh development-only CPU "
            "mechanism tests. No confirmation state, replay, replacement, training, "
            "or paper performance evidence."
        ),
        "device": "cpu",
        "confirmation_states_opened": 0,
        "formal_failure_counts": {
            row["class"]: row["count"]
            for row in formal["three_physical_failure_classes"]
        },
        "T2_public_sentinel_reproduction": {
            "original_fresh_cells": _cell_summary(t2_rows),
            "deadline_candidate_rows": _row_outcomes(t2_deadline_report),
            "deadline_candidate_cells": _cell_summary(t2_deadline_rows),
            "finding": (
                "The public sentinel reproduces both T2 failures. The pre-outlet "
                "deadline-aware admission advances payload outlet crossing but leaves "
                "total success at 8/32, cannot reach the outlet in latency mode 0, "
                "does not complete latency mode 1 or payload mode 0, and introduces "
                "one payload-mode-0 overforce. It is rejected as a standalone protocol "
                "revision."
            ),
        },
        "T3_method_side_reproduction": {
            "prior_three_way_rows": _row_outcomes(t3_report),
            "terminal_candidate_rows": _row_outcomes(t3_terminal_report),
            "terminal_timing": _t3_timing(t3_terminal_rows),
            "finding": (
                "Cross-EBDM first satisfies the exact hold predicate too late on "
                "failed payload-mode-1 episodes. The pose-neighborhood fallback "
                "regresses 40/64 to 24/64 while response MPC remains 64/64, so it is "
                "rejected. The next method revision must improve finite-time terminal "
                "arrival margin rather than switch actions only after arrival."
            ),
        },
        "three_failure_class_decisions": [
            {
                "class": "strict_guide_contact_contract_failure",
                "attribution": "shared_T2_task_protocol",
                "next_revision": (
                    "Version the T2 task-owned controller with a latency-robust public "
                    "contact-acquisition stage. Use delayed public pose/wrench and issued "
                    "reference only; retain the frozen force/contact thresholds, deadline, "
                    "reset distribution and evaluator."
                ),
                "acceptance": (
                    "The public sentinel must pass every fresh latency/payload by mode "
                    "cell with exact strict-contact evidence and zero constraint violations "
                    "before the revision is inherited by every method."
                ),
            },
            {
                "class": "finite_time_progress_after_contact_gate",
                "attribution": "shared_T2_plus_method_specific_T3",
                "next_revision": (
                    "Extend the T2 task controller through post-outlet completion with an "
                    "explicit remaining-time reserve. For T3, replace terminal-neighborhood "
                    "fallback with a Cross-EBDM phase-3 progress objective/admission that "
                    "acts before the deadline margin is exhausted and does not mutate the "
                    "rejected candidate state before selection."
                ),
                "acceptance": (
                    "T2 public sentinel passes all four fresh cells. T3 is compared on the "
                    "same fresh resets against current Cross-EBDM, band-only and response "
                    "MPC; it must remove the late-arrival failures without regressing any "
                    "previously successful paired episode or violating a constraint."
                ),
            },
            {
                "class": "constraint_control_failure",
                "attribution": "single_T2_ungated_ablation_event",
                "next_revision": (
                    "Keep the shared speed and force constraints unchanged. Do not alter "
                    "the task protocol from one event; retain the selective gate as a "
                    "method component and run a fresh paired gated/ungated stress test."
                ),
                "acceptance": (
                    "Report the paired violation count and confidence interval; make no "
                    "general safety claim unless the effect reproduces across fresh cells."
                ),
            },
        ],
        "diagnostic_contracts": {
            "strict_guide_contact_contract_failure": {
                "scope": "shared T2 task protocol; never method-specific credit",
                "falsifiable_hypothesis": (
                    "Under latency shift, delayed public observations plus the current "
                    "reference/action support leave insufficient margin to acquire contact, "
                    "cross the outlet and finish within the frozen deadline."
                ),
                "cpu_screen": [
                    "Partition every public-sentinel trace into contact acquisition, stable dwell, outlet crossing, post-outlet completion and terminal hold.",
                    "At every failed stage, record delayed public pose/wrench, issued/accepted/applied reference, remaining path, remaining controls and the frozen lead/force/contact margins.",
                    "Estimate the observed positive-progress envelope from fresh public-sentinel traces; if it cannot meet the required remaining-time rate inside the frozen margins, stop scalar gain searches and treat the current action support as empirically insufficient.",
                ],
                "comparators": [
                    "current public geometry controller",
                    "a separately versioned shared acquisition-plus-completion controller, only if the CPU action-support screen is nonempty",
                ],
                "acceptance": (
                    "Fresh 8-episode cells for latency/payload by both modes must each "
                    "reach 8/8 strict success with zero constraint violations; stage "
                    "timings and exact contact evidence must be complete."
                ),
                "rejection": (
                    "Any failed cell, empty empirical progress set, or new force/speed "
                    "violation rejects the shared controller. The v8 contract remains "
                    "unchanged and the limitation is reported as benchmark action support."
                ),
            },
            "finite_time_progress_after_contact_gate": {
                "scope": (
                    "shared post-outlet completion for T2; Cross-EBDM method-side "
                    "terminal arrival for T3; no T4 change"
                ),
                "falsifiable_hypothesis": (
                    "T3 failures arise because the learned candidate reaches the exact "
                    "terminal set after the 15-control hold reserve is already exhausted, "
                    "rather than because the terminal hold predicate is wrong."
                ),
                "cpu_screen": [
                    "Use only public phase, goal pose, common action and remaining controls to compute the per-state progress needed to enter the terminal set before the hold reserve.",
                    "Penalize the worst finite-time progress deficits during phase 3 and contract the candidate residual to the public common action inside the exact terminal set.",
                    "Perform action selection before any stateful candidate update; rejected candidates must be observationally and state-transition equivalent to the common path.",
                ],
                "comparators": [
                    "current Cross-EBDM",
                    "band-only",
                    "response MPC",
                ],
                "acceptance": (
                    "On identical fresh payload-mode-1 resets, the revision must preserve "
                    "every paired current success, add no constraint violation, beat "
                    "band-only, and match the 64/64 response-MPC reference before binding."
                ),
                "rejection": (
                    "A late terminal switch, success regression, terminal residual without "
                    "arrival-margin improvement, or use of evaluator/reward/private labels "
                    "rejects the method revision. T4 completion projection stays frozen."
                ),
            },
            "constraint_control_failure": {
                "scope": "single T2 ungated-ablation event under frozen shared limits",
                "falsifiable_hypothesis": (
                    "The selective gate reduces violations relative to the matched ungated "
                    "actor without changing the shared executor or constraint thresholds."
                ),
                "cpu_screen": [
                    "Use fresh paired resets and align the final safe and first violating 1 ms physics substeps.",
                    "Record candidate, issued, accepted and applied commands, predicted and measured force/speed margins, and the selector decision.",
                    "Report paired event counts and intervals; do not replace failures or replay confirmation states.",
                ],
                "comparators": ["gated-full", "ungated-full"],
                "acceptance": (
                    "A reproducible paired reduction with unchanged shared constraints may "
                    "support a narrow constraint-filter claim."
                ),
                "rejection": (
                    "A non-reproduced single event supports no general safety claim and "
                    "triggers no shared protocol change."
                ),
            },
        },
        "next_version_decision": {
            "cross_ebdm_method_revision": {
                "task": "T3 keyed insertion recovery",
                "change": (
                    "Add a phase-3 finite-time terminal-progress term to the existing "
                    "gated transport objective. Its target is the larger of common-action "
                    "progress and the public progress rate required to enter the exact "
                    "terminal set before reserving 15 hold controls."
                ),
                "implementation_scope": [
                    "genedynamics/solvers/single/cross_ebdm/learning/gated_transport.py",
                    "configs/arm/keyed_insertion_recovery/training",
                    "scripts/paper/cross_ebdm",
                ],
                "inputs": [
                    "public recovery phase",
                    "public current and goal pose",
                    "public common action",
                    "public elapsed/remaining controls",
                    "frozen public response model and existing task proxy",
                ],
                "forbidden_inputs": [
                    "evaluator success",
                    "reward",
                    "hidden mechanism label",
                    "confirmation reset or state",
                ],
                "status": "approved_for_cpu_unit_and_preflight_work; not actor-bound",
            },
            "shared_T2_revision": {
                "change": (
                    "Do not bind the rejected deadline-aware candidate. First run the CPU "
                    "action-support screen; only a nonempty result permits a new, shared, "
                    "versioned acquisition-plus-completion controller."
                ),
                "status": "separate task-contract work; blocks a new formal T2 roster",
            },
            "T4": {
                "change": "none",
                "status": "freeze the completion-projected final actor",
            },
            "confirmation": (
                "Do not generate a new roster until the separate T2 shared gate, T3 method "
                "gate, constraint stress test and architecture/package/device audits pass."
            ),
        },
        "theory_revision": [
            "Separate constraint preservation from finite-time reachability and terminal dwell.",
            "Condition the liveness margin on delayed observation, contact-acquisition time, remaining path length, remaining deadline and required hold duration.",
            "Define candidate admission before stateful controller mutation and require positive certified progress toward the terminal set in addition to safety.",
            "Limit current claims to T4 completion projection and observed constraint filtering; T1--T3 do not yet establish a task-performance advantage from transport.",
        ],
        "release_order": [
            "finish CPU unit and preflight checks for the T3 finite-time terminal-progress objective",
            "run the T2 action-support screen and qualify a new shared acquisition-plus-completion controller only if the screen is nonempty",
            "train and qualify the T3 finite-time terminal-progress revision against current, band-only and response MPC",
            "run the fresh gated-versus-ungated T2 constraint stress test",
            "rerun all affected behavioral, architecture, package and device gates",
            "generate a new seed-disjoint confirmation roster only after every affected row passes",
        ],
        "sources": {
            "analysis_script": pointer(Path(__file__).resolve()),
            "formal_diagnosis": pointer(FORMAL_DIAGNOSIS),
            "t2_public_reproduction": pointer(T2_PAIRED_DIAGNOSIS),
            "t2_deadline_candidate": pointer(T2_DEADLINE_DIAGNOSIS),
            "t2_mismatched_deadline_invalid_seal": pointer(T2_INVALID_DEADLINE_SEAL),
            "t3_three_way_reproduction": pointer(T3_PAIRED_DIAGNOSIS),
            "t3_terminal_candidate": pointer(T3_TERMINAL_DIAGNOSIS),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_path)
    print(json.dumps({
        "output": pointer(output_path), "status": output["status"],
        "formal_failure_counts": output["formal_failure_counts"],
        "T2_deadline_candidate": actual["t2_deadline"],
        "T3_terminal_candidate": actual["t3_terminal"],
    }, indent=2, sort_keys=True))


def _quantile_or_none(values: list[float], probability: float) -> float | None:
    values = sorted(float(value) for value in values if math.isfinite(value))
    if not values:
        return None
    location = (len(values) - 1) * probability
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    fraction = location - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(value)]
    return {
        "count": len(finite),
        "minimum": min(finite) if finite else None,
        "median": _quantile_or_none(finite, 0.5),
        "p95": _quantile_or_none(finite, 0.95),
        "maximum": max(finite) if finite else None,
    }


def _first_true_run_end(values: list[bool], length: int) -> int | None:
    run = 0
    for index, value in enumerate(values):
        run = run + 1 if value else 0
        if run >= length:
            return index
    return None


def _position_error_m(trace: dict[str, Any]) -> float:
    position = trace["tcp_position"]
    return math.sqrt(sum(
        (float(position[index]) - T2_GOAL_POSITION_M[index]) ** 2
        for index in range(3)
    ))


def _t2_episode_action_support(row: dict[str, Any]) -> dict[str, Any]:
    evidence = load_evidence(row["evidence"])
    if len(evidence["records"]) != 1:
        raise ValueError("T2 action-support screen requires one record per artifact")
    record = evidence["records"][0]
    traces = record["traces"]
    diagnostics = record["control_diagnostics"]
    if not traces or len(traces) != len(diagnostics):
        raise ValueError("T2 action-support traces and diagnostics differ")
    config = evidence["task_spec"]["config"]
    if (
        evidence["task_spec"]["alias"] != "T2"
        or float(config["control_dt"]) != 0.02
        or float(config["timeout_seconds"]) != 20.0
    ):
        raise ValueError("T2 action-support source contract changed")

    mode = int(row["request"]["requested_mode"])
    first_contact = row["episode_metrics"]["first_contact_step"]
    first_outlet = row["episode_metrics"]["first_outlet_step"]
    loaded = [
        bool(trace["axial_support"])
        and float(trace["normal_force"]) > max(
            0.5, float(config["contact_threshold"]),
        )
        for trace in traces
    ]
    stable_dwell_end = _first_true_run_end(loaded, 10)
    first_terminal_entry = _first_index(
        [int(trace["execution_hold_count"]) for trace in traces],
        lambda value: value > 0,
    )
    completion = _first_index(
        [int(trace["execution_hold_count"]) for trace in traces],
        lambda value: value >= T2_HOLD_CONTROLS,
    )
    if row["constraint_violation"]:
        failed_stage = "constraint_control"
    elif first_contact is None:
        failed_stage = "contact_acquisition"
    elif stable_dwell_end is None:
        failed_stage = "stable_dwell"
    elif first_outlet is None:
        failed_stage = "outlet_crossing"
    elif first_terminal_entry is None:
        failed_stage = "post_outlet_completion"
    elif completion is None:
        failed_stage = "terminal_hold"
    else:
        failed_stage = None

    pre_outlet_samples = []
    deadline_diagnostics = all(
        "required_speed_mps" in value and "achieved_speed_mps" in value
        for value in diagnostics
    )
    if deadline_diagnostics and stable_dwell_end is not None:
        end = len(traces) if first_outlet is None else min(
            len(traces), int(first_outlet) + 1,
        )
        for tick in range(stable_dwell_end, end):
            trace, diagnostic = traces[tick], diagnostics[tick]
            axis_lead = (
                float(trace["public_reference_position"][mode])
                - float(trace["tcp_position"][mode])
            )
            within_hard_margins = bool(
                float(trace["force_peak"]) <= float(config["force_limit"])
                and float(trace["lateral_force_peak"])
                <= float(config["lateral_force_limit"])
                and abs(axis_lead) <= float(diagnostic["lead_limit_m"]) + 1e-9
            )
            required = float(diagnostic["required_speed_mps"])
            achieved = float(diagnostic["achieved_speed_mps"])
            selected = float(diagnostic["last_selected_speed_mps"])
            if within_hard_margins and required > 0.0:
                pre_outlet_samples.append({
                    "required_speed_mps": required,
                    "achieved_speed_mps": achieved,
                    "selected_speed_mps": selected,
                    "achieved_required_ratio": achieved / required,
                    "selected_meets_required": selected + 1e-6 >= required,
                    "force_utilization": float(
                        diagnostic["last_force_utilization"]
                    ),
                    "axis_reference_lead_m": axis_lead,
                })

    post_outlet = None
    if first_outlet is not None:
        outlet_tick = int(first_outlet)
        errors = [_position_error_m(trace) for trace in traces]
        remaining_controls = max(
            1, 1000 - T2_HOLD_CONTROLS - outlet_tick,
        )
        required_rate = max(
            0.0,
            errors[outlet_tick] - float(config["position_tolerance"]),
        ) / (remaining_controls * float(config["control_dt"]))
        elapsed = max(
            float(config["control_dt"]),
            (len(traces) - 1 - outlet_tick) * float(config["control_dt"]),
        )
        achieved_rate = (errors[outlet_tick] - errors[-1]) / elapsed
        post_outlet = {
            "remaining_controls_after_hold_reserve": remaining_controls,
            "position_error_at_outlet_m": errors[outlet_tick],
            "final_position_error_m": errors[-1],
            "best_position_error_m": min(errors[outlet_tick:]),
            "required_net_progress_rate_mps": required_rate,
            "achieved_net_progress_rate_mps": achieved_rate,
            "achieved_required_ratio": (
                achieved_rate / required_rate if required_rate > 0.0 else None
            ),
        }

    return {
        "key": row["key"],
        "scene_cell": row["request"]["scene_cell"],
        "requested_mode": mode,
        "physical_initial_slot": int(
            row["request"]["physical_initial_slot"]
        ),
        "reset_seed": int(row["request"]["reset_seed"]),
        "outcome": row["outcome"],
        "success": bool(row["success"]),
        "constraint_violation": bool(row["constraint_violation"]),
        "failed_stage": failed_stage,
        "control_steps": len(traces),
        "stage_ticks": {
            "first_contact": first_contact,
            "stable_dwell_end": stable_dwell_end,
            "first_outlet": first_outlet,
            "first_terminal_entry": first_terminal_entry,
            "completion": completion,
        },
        "margins": {
            "maximum_force_peak_N": max(float(x["force_peak"]) for x in traces),
            "maximum_lateral_force_peak_N": max(
                float(x["lateral_force_peak"]) for x in traces
            ),
            "minimum_joint_margin_rad": min(
                float(x["minimum_joint_margin_rad"]) for x in traces
            ),
        },
        "pre_outlet_samples": pre_outlet_samples,
        "post_outlet": post_outlet,
        "evidence": row["evidence"],
    }


def _t2_action_support_cell(
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    ticks = {
        name: [
            episode["stage_ticks"][name]
            for episode in episodes
            if episode["stage_ticks"][name] is not None
        ]
        for name in (
            "first_contact", "stable_dwell_end", "first_outlet",
            "first_terminal_entry", "completion",
        )
    }
    samples = [
        sample for episode in episodes
        for sample in episode["pre_outlet_samples"]
    ]
    post = [
        episode["post_outlet"] for episode in episodes
        if episode["post_outlet"] is not None
    ]
    ratios = [sample["achieved_required_ratio"] for sample in samples]
    post_ratios = [
        value["achieved_required_ratio"] for value in post
        if value["achieved_required_ratio"] is not None
    ]
    count = len(episodes)
    violations = sum(x["constraint_violation"] for x in episodes)
    pre_supported = None if not samples else bool(
        len(ticks["first_outlet"]) == count
        and violations == 0
        and _quantile_or_none(ratios, 0.5) >= 0.9
    )
    post_supported = bool(
        post_ratios
        and len(ticks["first_terminal_entry"]) == count
        and len(ticks["completion"]) == count
        and violations == 0
        and min(post_ratios) >= 1.0
    )
    return {
        "episodes": count,
        "successes": sum(x["success"] for x in episodes),
        "constraint_violations": violations,
        "outcomes": dict(sorted(Counter(
            x["outcome"] for x in episodes
        ).items())),
        "failed_stages": dict(sorted(Counter(
            x["failed_stage"] or "none" for x in episodes
        ).items())),
        "stage_reach_counts": {
            name: len(values) for name, values in ticks.items()
        },
        "stage_tick_distributions": {
            name: _distribution(values) for name, values in ticks.items()
        },
        "hard_margin_observations": {
            "force_peak_N": _distribution([
                x["margins"]["maximum_force_peak_N"] for x in episodes
            ]),
            "lateral_force_peak_N": _distribution([
                x["margins"]["maximum_lateral_force_peak_N"]
                for x in episodes
            ]),
            "minimum_joint_margin_rad": _distribution([
                x["margins"]["minimum_joint_margin_rad"] for x in episodes
            ]),
        },
        "pre_outlet_progress_envelope": {
            "hard_margin_controls_with_positive_requirement": len(samples),
            "required_speed_mps": _distribution([
                x["required_speed_mps"] for x in samples
            ]),
            "achieved_speed_mps": _distribution([
                x["achieved_speed_mps"] for x in samples
            ]),
            "selected_speed_mps": _distribution([
                x["selected_speed_mps"] for x in samples
            ]),
            "achieved_required_ratio": _distribution(ratios),
            "selected_meets_required_controls": sum(
                x["selected_meets_required"] for x in samples
            ),
            "maximum_force_utilization": max(
                (x["force_utilization"] for x in samples), default=None,
            ),
            "maximum_absolute_axis_reference_lead_m": max(
                (abs(x["axis_reference_lead_m"]) for x in samples),
                default=None,
            ),
            "supported": pre_supported,
        },
        "post_outlet_progress_envelope": {
            "episodes_reaching_outlet": len(post),
            "required_net_progress_rate_mps": _distribution([
                x["required_net_progress_rate_mps"] for x in post
            ]),
            "achieved_net_progress_rate_mps": _distribution([
                x["achieved_net_progress_rate_mps"] for x in post
            ]),
            "achieved_required_ratio": _distribution(post_ratios),
            "best_position_error_m": _distribution([
                x["best_position_error_m"] for x in post
            ]),
            "supported": post_supported,
        },
        "empirical_action_support_nonempty": (
            None if pre_supported is None
            else bool(pre_supported and post_supported)
        ),
        "episode_stage_rows": [
            {
                "physical_initial_slot": value["physical_initial_slot"],
                "reset_seed": value["reset_seed"],
                "outcome": value["outcome"],
                "failed_stage": value["failed_stage"],
                "control_steps": value["control_steps"],
                "stage_ticks": value["stage_ticks"],
                "margins": value["margins"],
                "post_outlet": value["post_outlet"],
                "evidence": value["evidence"],
            }
            for value in sorted(
                episodes,
                key=lambda item: item["physical_initial_slot"],
            )
        ],
    }


def t2_action_support_screen(output_path: Path) -> None:
    """Close the T2 support branch using frozen fresh public-sentinel traces."""
    output_path = output_path if output_path.is_absolute() else ROOT / output_path
    report, rows = _diagnostic_rows(T2_DEADLINE_DIAGNOSIS)
    if len(rows) != 64:
        raise ValueError("T2 action-support screen requires all 64 paired rows")
    episodes = [_t2_episode_action_support(row) for row in rows]
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[(
            episode["key"], episode["scene_cell"],
            episode["requested_mode"],
        )].append(episode)
    if set(grouped) != {
        (key, cell, mode)
        for key in ("public_geometry_current", "public_geometry_deadline_aware")
        for cell in ("latency_gain_shift", "payload_shift")
        for mode in (0, 1)
    } or any(len(value) != 8 for value in grouped.values()):
        raise ValueError("T2 action-support paired cell inventory changed")
    cells = {
        "|".join(map(str, key)): _t2_action_support_cell(value)
        for key, value in sorted(grouped.items())
    }
    candidate_cells = {
        key: value for key, value in cells.items()
        if key.startswith("public_geometry_deadline_aware|")
    }
    nonempty = all(
        value["empirical_action_support_nonempty"]
        for value in candidate_cells.values()
    )
    if nonempty:
        raise ValueError(
            "T2 action support unexpectedly became nonempty; review before revision"
        )

    t3 = read_json(T3_FINITE_TIME_QUALIFICATION)
    t3_rows = {row["key"]: row for row in t3["rows"]}
    expected_t3 = {
        "cross_ebdm_current": (33, 0),
        "cross_ebdm_finite_time": (64, 0),
        "band_only": (48, 0),
        "response_mpc": (57, 0),
    }
    actual_t3 = {
        key: (t3_rows[key]["successes"], t3_rows[key]["constraint_violations"])
        for key in expected_t3
    }
    if actual_t3 != expected_t3:
        raise ValueError(f"T3 finite-time qualification changed: {actual_t3}")
    bindings = read_json(T3_ACTOR_BINDINGS)
    bound = bindings["main"]["cross_ebdm"]
    if (
        bound["method_identity"] != "ours_gated_v5_late_phase"
        or bound["status"] != "instances_ready"
        or bound["qualification_report"]["sha256"]
        != sha(T3_FINITE_TIME_QUALIFICATION)
    ):
        raise ValueError("qualified T3 finite-time actor is not bound")

    output = {
        "schema_version": "t2-public-action-support-offline-screen-v1",
        "status": "complete_empirical_action_support_empty",
        "scope": (
            "CPU-only post hoc screen of frozen fresh public-sentinel evidence. "
            "No simulator step, GPU call, training, confirmation-state access, "
            "failure replacement, threshold change, or paper performance evidence."
        ),
        "device": "cpu",
        "gpu_calls": 0,
        "physics_steps_executed": 0,
        "confirmation_states_opened": 0,
        "failure_replacement": False,
        "performance_evidence": False,
        "source_counts": {
            "paired_episodes_reaudited": len(episodes),
            "recorded_control_steps_reaudited": sum(
                episode["control_steps"] for episode in episodes
            ),
            "candidate_cells": len(candidate_cells),
            "episodes_per_cell": 8,
        },
        "stage_definition": {
            "contact_acquisition": "reset through first exact physics-rate loaded contact",
            "stable_dwell": (
                "first loaded contact through the end of ten consecutive "
                "loaded control endpoints, matching the public 0.2 s dwell"
            ),
            "outlet_crossing": (
                "stable dwell through the first mode-specific exact physics-rate "
                "outlet crossing"
            ),
            "post_outlet_completion": (
                "outlet crossing through first entry into the frozen terminal set"
            ),
            "terminal_hold": (
                "terminal-set entry through the frozen 15-control hold"
            ),
        },
        "screen_rule": {
            "pre_outlet": (
                "All eight episodes must cross the outlet without a constraint "
                "violation and the median achieved/required rate over positive-rate "
                "hard-margin controls must be at least 0.9."
            ),
            "post_outlet": (
                "All eight episodes must enter and complete the terminal hold without "
                "a constraint violation, and every observed net/required post-outlet "
                "rate ratio must be at least 1.0."
            ),
            "shared_revision": (
                "Every latency/payload by mode cell must have nonempty pre- and "
                "post-outlet empirical support before a new shared controller is allowed."
            ),
        },
        "observability": {
            "available": [
                "exact physics-rate contact and outlet evidence",
                "issued and applied actions and references",
                "recorded physical pose, force, lateral force and joint margins",
                "deadline wrapper's delayed-observation-derived achieved speed, required speed, force utilization and lead limit",
            ],
            "unavailable": [
                "raw delayed policy observation at every control",
                "raw delayed wrench before the wrapper's utilization reduction",
            ],
            "interpretation_limit": (
                "The result establishes empirical insufficiency of the exercised "
                "public action/reference support. It is not a mathematical "
                "impossibility proof for every controller in the continuous action space."
            ),
        },
        "cells": cells,
        "decision": {
            "screen_nonempty": False,
            "new_shared_controller_permitted": False,
            "rejected_candidate": "deadline_aware_v1",
            "finding": (
                "Only payload-shift mode 1 has nonempty empirical support. Latency "
                "mode 0 cannot realize its required pre-outlet rate within the recorded "
                "lead and hard constraints; latency mode 1 crosses the outlet but its "
                "post-outlet rate is below the remaining-time requirement; payload "
                "mode 0 crosses in seven of eight candidate episodes and then moves "
                "away from the terminal position, with the eighth episode overforce."
            ),
            "current_contract": (
                "Keep task_contract_v3 and the v8 result unchanged. Stop scalar lead, "
                "gain and speed searches under this support."
            ),
            "if_t2_is_redesigned": (
                "Create a new task contract before any new roster. Change the task-owned "
                "reference/action support itself: a delay-aware contact-acquisition "
                "primitive must expose enough pre-outlet progress, and the shared "
                "post-outlet primitive must provide monotone goal-directed progress. "
                "Keep the force/speed limits, strict evaluator, reset cells and deadline "
                "fixed, then require 8/8 in every cell for the public sentinel before "
                "all methods inherit it."
            ),
        },
        "three_failure_class_closeout": {
            "strict_guide_contact_contract_failure": {
                "count_in_v8": 1133,
                "diagnosis": "shared T2 latency/action-support insufficiency",
                "status": "closed_for_current_contract; no controller revision admitted",
            },
            "finite_time_progress_after_contact_gate": {
                "count_in_v8": 2192,
                "T2": "unresolved shared post-outlet support; no method-specific credit",
                "T3": {
                    "status": "resolved_and_bound_for_predeployment",
                    "current_successes": 33,
                    "finite_time_successes": 64,
                    "band_only_successes": 48,
                    "response_mpc_successes": 57,
                    "episodes_per_row": 64,
                    "constraint_violations": 0,
                    "method_identity": "ours_gated_v5_late_phase",
                },
                "T4": "unchanged completion-projected actor remains frozen",
            },
            "constraint_control_failure": {
                "count_in_v8": 1,
                "status": "open_fresh_paired_stress_required",
                "next_test": {
                    "comparators": ["gated_full", "ungated_full"],
                    "cell": "T2 payload_shift, xarm7_fixed_base, requested_mode=1",
                    "fresh_design": (
                        "Eight actor instances by eight unused reset slots, paired by "
                        "physical reset: 64 episodes per comparator and 128 total."
                    ),
                    "record": [
                        "candidate, issued, accepted and applied commands",
                        "selector decision and predicted force/speed margins",
                        "last safe and first violating 1 ms substeps",
                        "paired violation counts, discordant pairs and exact interval",
                    ],
                    "accept": (
                        "Only a reproducible paired reduction under unchanged shared "
                        "limits supports a narrow constraint-filter claim."
                    ),
                    "reject": (
                        "No reproduced event means no safety claim and no shared protocol change."
                    ),
                },
            },
        },
        "next_version": {
            "code_changes": [
                "Retain the bound T3 ours_gated_v5_late_phase actor; do not tune it further.",
                "Do not add or bind another T2 shared controller under task_contract_v3.",
                "Add only the fresh paired T2 gated/ungated diagnostic path; it cannot alter protocol thresholds or actor bindings.",
                "Keep the T4 completion projection unchanged.",
            ],
            "theory_changes": [
                "State safety invariance and finite-time reachability as separate properties.",
                "Condition any liveness claim on a nonempty progress-feasible action set after delay, force, lead and hold-reserve constraints.",
                "Use the T2 empty-support result as a boundary condition of the theorem, not as method performance evidence.",
                "Use T3 only to support the finite-time terminal-progress mechanism; it does not by itself prove a general transport advantage.",
                "Restrict gating language to a constraint-filter hypothesis until the fresh paired stress reproduces the effect.",
            ],
            "next_executable_step": "fresh paired T2 gated-versus-ungated constraint stress",
            "new_confirmation_roster_allowed": False,
        },
        "sources": {
            "analysis_script": pointer(Path(__file__).resolve()),
            "development_diagnosis": pointer(DEFAULT_DEVELOPMENT_OUTPUT),
            "paired_source_report": pointer(T2_DEADLINE_DIAGNOSIS),
            "paired_source_progress": report["artifacts"]["progress"],
            "t2_task_contract": pointer(COMMON_CONTRACTS["T2"]),
            "t2_contact_source": pointer(T2_CONTACT_SOURCE),
            "t2_guide_source": pointer(T2_GUIDE_SOURCE),
            "t3_finite_time_qualification": pointer(T3_FINITE_TIME_QUALIFICATION),
            "t3_actor_bindings": pointer(T3_ACTOR_BINDINGS),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_path)
    print(json.dumps({
        "output": pointer(output_path),
        "status": output["status"],
        "candidate_cell_support": {
            key: value["empirical_action_support_nonempty"]
            for key, value in candidate_cells.items()
        },
        "new_shared_controller_permitted": False,
        "next_executable_step": output["next_version"]["next_executable_step"],
    }, indent=2, sort_keys=True))


def timeout_diagnostics(report_path: Path, report: dict[str, Any]) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in progress_path(report_path, report).read_text().splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row["profile_key"] == "candidate"]
    timeouts = [row for row in rows if row["outcome"] == "timeout"]
    summaries = []
    for row in sorted(timeouts, key=lambda item: item["request"]["reset_seed"]):
        evidence = load_evidence(row["evidence"])
        if len(evidence["records"]) != 1:
            raise ValueError("expected one episode per evidence artifact")
        record = evidence["records"][0]
        diagnostics = record["control_diagnostics"]
        traces = record["traces"]
        transitions = {}
        for global_tick, value in enumerate(diagnostics):
            candidate = value.get("candidate")
            if candidate is None:
                continue
            stage = int(candidate["stage"])
            local_tick = int(candidate["tick"])
            if local_tick >= 0 and str(stage) not in transitions:
                transitions[str(stage)] = {
                    "global_tick": global_tick,
                    "local_tick": local_tick,
                }
        final_diagnostic = diagnostics[-1].get("candidate", diagnostics[-1])
        final_trace = traces[-1]
        summaries.append({
            "reset_seed": row["request"]["reset_seed"],
            "requested_mode": row["request"]["requested_mode"],
            "control_steps": row["control_steps"],
            "stage_transitions": transitions,
            "terminal_stage": final_diagnostic.get("stage"),
            "terminal_tcp_position_m": final_trace["tcp_position"],
            "terminal_reference_position_m": final_trace["public_reference_position"],
            "terminal_reference_lead_m": final_trace["public_reference_lead"],
            "terminal_normal_force_N": final_trace["normal_force"],
        })
    stage_ticks: dict[str, list[int]] = {}
    for row in summaries:
        for stage, transition in row["stage_transitions"].items():
            stage_ticks.setdefault(stage, []).append(transition["global_tick"])
    return {
        "timeouts": len(summaries),
        "requested_mode_counts": dict(sorted(Counter(
            row["requested_mode"] for row in summaries
        ).items())),
        "global_stage_tick_ranges": {
            stage: {
                "minimum": min(values),
                "median": median(values),
                "maximum": max(values),
            }
            for stage, values in sorted(stage_ticks.items())
        },
        "episodes": summaries,
    }


def _formal_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "by_task": dict(sorted(Counter(row["task"] for row in rows).items())),
        "by_family_method": dict(sorted(Counter(
            f'{row["family"]}/{row["method"]}' for row in rows
        ).items())),
        "by_frozen_cell": {
            "|".join(map(str, key)): value
            for key, value in sorted(Counter((
                row["mechanism_group"], row["body_support_cell"],
                row["requested_mode"],
            ) for row in rows).items())
        },
    }


def _candidate_screen_key(rows: list[dict[str, Any]]) -> str:
    keys = sorted({row["key"] for row in rows})
    candidates = [key for key in keys if key != "public_geometry_current"]
    if len(candidates) != 1 or "public_geometry_current" not in keys:
        raise ValueError(f"unexpected public-sentinel screen rows: {keys}")
    return candidates[0]


def _candidate_episode_response(row: dict[str, Any]) -> dict[str, Any]:
    evidence = load_evidence(row["evidence"])
    records = evidence["records"]
    if len(records) != 1:
        raise ValueError("shared-executor screen must contain one episode per artifact")
    record = records[0]
    traces = record["traces"]
    if not traces:
        raise ValueError("shared-executor screen has no physical trace")
    config = evidence["task_spec"]["config"]
    control_dt = float(config["control_dt"])
    axis = int(row["request"]["requested_mode"])
    if axis not in (0, 1):
        raise ValueError("T2 response screen requires mode 0 or mode 1")

    start = float(traces[0]["tcp_position"][axis])
    terminal = float(traces[-1]["tcp_position"][axis])
    applied = [float(trace["applied_action"][axis]) for trace in traces]
    leads = [
        abs(float(trace["applied_reference_position"][axis])
            - float(trace["tcp_position"][axis]))
        for trace in traces
    ]
    measured_rates = [
        (float(after["tcp_position"][axis])
         - float(before["tcp_position"][axis])) / control_dt
        for before, after in zip(traces, traces[1:])
    ]
    positive_command_path = sum(max(value, 0.) * control_dt for value in applied)
    progress = terminal - start
    return {
        "axis": axis,
        "controls": len(traces),
        "net_axis_progress_m": progress,
        "positive_applied_command_path_m": positive_command_path,
        "net_to_positive_command_ratio": (
            progress / positive_command_path if positive_command_path > 0 else None
        ),
        "maximum_applied_axis_command_mps": max(applied),
        "fraction_controls_at_80mmps_command_ceiling": (
            sum(value >= .0795 for value in applied) / len(applied)
        ),
        "median_measured_axis_rate_mps": _median_or_none(measured_rates),
        "maximum_reference_tracking_lag_m": max(leads),
        "terminal_reference_tracking_lag_m": leads[-1],
        "minimum_joint_margin_rad": min(
            float(trace["minimum_joint_margin_rad"]) for trace in traces
        ),
        "maximum_joint_speed_radps": max(
            float(trace["joint_speed_peak"]) for trace in traces
        ),
    }


def _response_distribution(
    values: list[dict[str, Any]], field: str,
) -> dict[str, float | int | None]:
    return _distribution([
        float(value[field]) for value in values if value[field] is not None
    ])


def _t2_shared_executor_screen_closeout() -> dict[str, Any]:
    screens: dict[str, Any] = {}
    total_rows = total_candidate_rows = 0
    total_candidate_successes = total_candidate_violations = 0
    for screen_name, report_path in T2_SHARED_EXECUTOR_SCREENS.items():
        report, rows = _diagnostic_rows(report_path)
        candidate_key = _candidate_screen_key(rows)
        candidate_rows = [row for row in rows if row["key"] == candidate_key]
        response_rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in candidate_rows:
            response_rows[(
                row["request"]["scene_cell"],
                int(row["request"]["requested_mode"]),
            )].append(_candidate_episode_response(row))

        cells = {}
        for (scene_cell, mode), values in sorted(response_rows.items()):
            matched = [
                row for row in candidate_rows
                if row["request"]["scene_cell"] == scene_cell
                and int(row["request"]["requested_mode"]) == mode
            ]
            cells[f"{scene_cell}|{mode}"] = {
                "episodes": len(matched),
                "successes": sum(bool(row["success"]) for row in matched),
                "constraint_violations": sum(
                    bool(row["constraint_violation"]) for row in matched
                ),
                "outcomes": dict(sorted(Counter(
                    row["outcome"] for row in matched
                ).items())),
                "median_first_contact_step": _median_or_none([
                    row["episode_metrics"]["first_contact_step"]
                    for row in matched
                ]),
                "median_first_outlet_step": _median_or_none([
                    row["episode_metrics"]["first_outlet_step"]
                    for row in matched
                ]),
                "median_final_position_error_m": _median_or_none([
                    row["episode_metrics"]["final_position_error_m"]
                    for row in matched
                ]),
                "response": {
                    field: _response_distribution(values, field)
                    for field in (
                        "net_axis_progress_m",
                        "positive_applied_command_path_m",
                        "net_to_positive_command_ratio",
                        "maximum_applied_axis_command_mps",
                        "fraction_controls_at_80mmps_command_ceiling",
                        "median_measured_axis_rate_mps",
                        "maximum_reference_tracking_lag_m",
                        "terminal_reference_tracking_lag_m",
                        "minimum_joint_margin_rad",
                        "maximum_joint_speed_radps",
                    )
                },
            }

        successes = sum(bool(row["success"]) for row in candidate_rows)
        violations = sum(bool(row["constraint_violation"]) for row in candidate_rows)
        screens[screen_name] = {
            "candidate_key": candidate_key,
            "all_physical_episodes": len(rows),
            "candidate_episodes": len(candidate_rows),
            "candidate_successes": successes,
            "candidate_constraint_violations": violations,
            "cells": cells,
            "report": pointer(report_path),
        }
        total_rows += len(rows)
        total_candidate_rows += len(candidate_rows)
        total_candidate_successes += successes
        total_candidate_violations += violations

    totals = {
        "all_physical_episodes": total_rows,
        "candidate_episodes": total_candidate_rows,
        "candidate_successes": total_candidate_successes,
        "candidate_constraint_violations": total_candidate_violations,
        "execution_errors": 0,
    }
    expected = {
        "all_physical_episodes": 256,
        "candidate_episodes": 128,
        "candidate_successes": 38,
        "candidate_constraint_violations": 23,
        "execution_errors": 0,
    }
    if totals != expected:
        raise ValueError(f"T2 shared-executor screen totals changed: {totals}")
    return {"totals": totals, "screens": screens}


def _response_normalized_episode(row: dict[str, Any]) -> dict[str, Any]:
    evidence = load_evidence(row["evidence"])
    records = evidence["records"]
    if len(records) != 1:
        raise ValueError("response-normalized screen requires one record per artifact")
    record = records[0]
    traces = record["traces"]
    diagnostics = record["control_diagnostics"]
    if not traces or len(traces) != len(diagnostics):
        raise ValueError("response-normalized trace and diagnostics differ")
    final_trace = traces[-1]
    final_diagnostic = diagnostics[-1]
    holds = [int(trace["execution_hold_count"]) for trace in traces]
    completion_stages = [
        int(value.get("completion", {}).get("stage", 0))
        for value in diagnostics
    ]
    final_tcp = [float(value) for value in final_trace["tcp_position"]]
    final_reference = [
        float(value) for value in final_trace["applied_reference_position"]
    ]
    return {
        "success": bool(row["success"]),
        "constraint_violation": bool(row["constraint_violation"]),
        "outcome": row["outcome"],
        "controls": len(traces),
        "first_contact_step": row["episode_metrics"]["first_contact_step"],
        "first_outlet_step": row["episode_metrics"]["first_outlet_step"],
        "strict_contact_success": bool(
            row["episode_metrics"]["strict_contact_success"]
        ),
        "final_position_error_m": float(
            row["episode_metrics"]["final_position_error_m"]
        ),
        "terminal_set_margin_m": (
            float(evidence["task_spec"]["config"]["position_tolerance"])
            - float(row["episode_metrics"]["final_position_error_m"])
        ),
        "force_peak_N": float(row["force_peak_N"]),
        "lateral_force_peak_N": float(row["lateral_force_peak_N"]),
        "maximum_hold_controls": max(holds),
        "final_hold_controls": holds[-1],
        "first_hold_eligible_step": _first_index(holds, lambda value: value > 0),
        "completion_descent_step": _first_index(
            completion_stages, lambda value: value >= 1,
        ),
        "final_tcp_position_m": final_tcp,
        "final_reference_position_m": final_reference,
        "final_tcp_reference_distance_m": float(math.sqrt(sum(
            (tcp - reference) ** 2
            for tcp, reference in zip(final_tcp, final_reference)
        ))),
        "final_authority": float(final_diagnostic.get("authority", 0.)),
        "final_required_speed_mps": float(
            final_diagnostic.get("required_speed_mps", 0.)
        ),
        "final_tracking_lag_m": float(
            final_diagnostic.get("tracking_lag_m", 0.)
        ),
        "progress_deficit_controls": int(
            final_diagnostic.get("progress_deficit_controls", 0)
        ),
    }


def _response_normalized_screen_closeout() -> dict[str, Any]:
    screens: dict[str, Any] = {}
    totals = {
        "all_physical_episodes": 0,
        "candidate_episodes": 0,
        "candidate_successes": 0,
        "candidate_constraint_violations": 0,
        "execution_errors": 0,
    }
    for screen_name, report_path in T2_RESPONSE_NORMALIZED_SCREENS.items():
        report, rows = _diagnostic_rows(report_path)
        candidate_key = _candidate_screen_key(rows)
        candidate_rows = [row for row in rows if row["key"] == candidate_key]
        by_mode: dict[str, Any] = {}
        for mode in (0, 1):
            episodes = [
                _response_normalized_episode(row)
                for row in candidate_rows
                if int(row["request"]["requested_mode"]) == mode
            ]
            if len(episodes) != 2:
                raise ValueError(
                    f"response-normalized screen {screen_name} mode {mode} "
                    "does not contain two fresh episodes"
                )
            by_mode[str(mode)] = {
                "episodes": len(episodes),
                "successes": sum(value["success"] for value in episodes),
                "strict_contact_successes": sum(
                    value["strict_contact_success"] for value in episodes
                ),
                "constraint_violations": sum(
                    value["constraint_violation"] for value in episodes
                ),
                "outcomes": dict(sorted(Counter(
                    value["outcome"] for value in episodes
                ).items())),
                "completion_controls": _distribution([
                    value["controls"] for value in episodes
                ]),
                "first_contact_step": _distribution([
                    value["first_contact_step"] for value in episodes
                    if value["first_contact_step"] is not None
                ]),
                "first_outlet_step": _distribution([
                    value["first_outlet_step"] for value in episodes
                    if value["first_outlet_step"] is not None
                ]),
                "final_position_error_m": _distribution([
                    value["final_position_error_m"] for value in episodes
                ]),
                "terminal_set_margin_m": _distribution([
                    value["terminal_set_margin_m"] for value in episodes
                ]),
                "force_peak_N": _distribution([
                    value["force_peak_N"] for value in episodes
                ]),
                "lateral_force_peak_N": _distribution([
                    value["lateral_force_peak_N"] for value in episodes
                ]),
                "maximum_hold_controls": _distribution([
                    value["maximum_hold_controls"] for value in episodes
                ]),
                "completion_descent_step": _distribution([
                    value["completion_descent_step"] for value in episodes
                    if value["completion_descent_step"] is not None
                ]),
                "final_tcp_reference_distance_m": _distribution([
                    value["final_tcp_reference_distance_m"] for value in episodes
                ]),
                "final_authority": _distribution([
                    value["final_authority"] for value in episodes
                ]),
                "final_required_speed_mps": _distribution([
                    value["final_required_speed_mps"] for value in episodes
                ]),
                "episodes_detail": episodes,
            }
        candidate_successes = sum(bool(row["success"]) for row in candidate_rows)
        candidate_violations = sum(
            bool(row["constraint_violation"]) for row in candidate_rows
        )
        screens[screen_name] = {
            "candidate_key": candidate_key,
            "all_physical_episodes": len(rows),
            "candidate_episodes": len(candidate_rows),
            "candidate_successes": candidate_successes,
            "candidate_constraint_violations": candidate_violations,
            "modes": by_mode,
            "report": pointer(report_path),
        }
        totals["all_physical_episodes"] += len(rows)
        totals["candidate_episodes"] += len(candidate_rows)
        totals["candidate_successes"] += candidate_successes
        totals["candidate_constraint_violations"] += candidate_violations
        totals["execution_errors"] += int(
            report["counts"]["execution_errors"]
        )

    expected = {
        "all_physical_episodes": 88,
        "candidate_episodes": 44,
        "candidate_successes": 8,
        "candidate_constraint_violations": 9,
        "execution_errors": 0,
    }
    if totals != expected:
        raise ValueError(f"response-normalized screen totals changed: {totals}")
    final = screens["response_normalized_mode_routes"]
    if (
        final["candidate_successes"] != 3
        or final["candidate_constraint_violations"] != 0
        or final["modes"]["0"]["successes"] != 2
        or final["modes"]["1"]["successes"] != 1
        or final["modes"]["1"]["strict_contact_successes"] != 2
    ):
        raise ValueError("final response-normalized mode-route result changed")
    failed = [
        value for value in final["modes"]["1"]["episodes_detail"]
        if not value["success"]
    ]
    if len(failed) != 1:
        raise ValueError("expected one final mode-1 finite-time failure")
    failed = failed[0]
    if (
        failed["constraint_violation"]
        or not failed["strict_contact_success"]
        or failed["outcome"] != "timeout"
        or failed["maximum_hold_controls"] != 0
        or not -0.0002 < failed["terminal_set_margin_m"] < 0.
    ):
        raise ValueError("final mode-1 failure mechanism changed")
    return {
        "totals": totals,
        "screens": screens,
        "final_candidate": {
            "result": "3/4 success, zero constraint violations",
            "mode_0": "2/2 strict success",
            "mode_1": "1/2 success; both episodes satisfy strict contact",
            "remaining_failure": failed,
            "qualification_status": "rejected_before_full_public_gate",
        },
    }


def next_revision_closeout(output_path: Path) -> None:
    """Consolidate the frozen formal failures and completed development diagnoses."""

    sources = {
        key: read_json(path) for key, path in NEXT_REVISION_SOURCES.items()
    }
    formal = sources["formal_diagnosis"]
    action_support = sources["t2_action_support"]
    t3 = sources["t3_finite_time_qualification"]
    guard = sources["same_policy_guard_stress"]
    t2_qualification = sources["t2_transport_only_qualification"]
    four_task_qualification = sources["four_task_behavior_qualification"]
    architecture = sources["architecture_audit"]
    package_audit = sources["candidate_package_preaccess_audit"]
    t2_manifest = sources["t2_actor_manifest"]
    t3_manifest = sources["t3_actor_manifest"]
    executor_screens = _t2_shared_executor_screen_closeout()
    response_normalized = _response_normalized_screen_closeout()

    formal_counts = {
        row["class"]: row["count"]
        for row in formal["three_physical_failure_classes"]
    }
    expected_counts = {
        "strict_guide_contact_contract_failure": 1133,
        "finite_time_progress_after_contact_gate": 2192,
        "constraint_control_failure": 1,
    }
    if formal_counts != expected_counts:
        raise ValueError(f"formal failure counts changed: {formal_counts}")
    if formal["counts"] != {
        "scheduled_records": 21200,
        "physical_executions": 15360,
        "predeployment_penalty_slots": 5840,
        "physical_failures": 3370,
        "raw_timeouts": 3325,
        "infrastructure_retained_failures": 44,
    }:
        raise ValueError("formal confirmation counts changed")

    support_decision = action_support["decision"]
    if (
        action_support["status"] != "complete_empirical_action_support_empty"
        or support_decision["screen_nonempty"]
        or support_decision["new_shared_controller_permitted"]
    ):
        raise ValueError("T2 action-support decision changed")

    if not t3["passed"] or not all(t3["qualification_checks"].values()):
        raise ValueError("T3 finite-time qualification is not fully passed")
    t3_rows = {row["key"]: row for row in t3["rows"]}
    expected_t3 = {
        "cross_ebdm_current": (33, 0),
        "cross_ebdm_finite_time": (64, 0),
        "band_only": (48, 0),
        "response_mpc": (57, 0),
    }
    for key, expected in expected_t3.items():
        row = t3_rows[key]
        actual = (row["successes"], row["constraint_violations"])
        if actual != expected or row["physical_jobs"] != 64:
            raise ValueError(f"unexpected T3 qualification row {key}: {actual}")

    stress = guard["constraint_stress_checks"]
    if (
        not guard["passed"]
        or not stress["comparator_semantics_valid"]
        or stress["guard_reduction_claim_supported"]
        or stress["reproduced_measured_event"]
        or stress["paired_force_or_speed_events"] != {
            "both": 0, "gated_only": 0, "ungated_only": 0, "neither": 64
        }
    ):
        raise ValueError("same-policy guard stress decision changed")

    t2_rows = {row["key"]: row for row in t2_qualification["rows"]}
    for key in ("cross_ebdm", "task_only"):
        row = t2_rows[key]
        if (
            row["admission_status"] != "behavior_qualified"
            or row["behavior_successes"] != 32
            or row["constraint_violations"] != 0
        ):
            raise ValueError(f"T2 transport-only qualification changed: {key}")
    public_sentinel = t2_rows["public_geometry"]
    if public_sentinel["physical_jobs"] != 4:
        raise ValueError("T2 qualification sentinel denominator changed")
    if (
        not t2_qualification["passed"]
        or not four_task_qualification["passed"]
        or not four_task_qualification["formal_evaluation_allowed"]
    ):
        raise ValueError("current behavior qualification is not admitted")
    if architecture["status"] != "passed" or architecture["physics_steps"] != 0:
        raise ValueError("current architecture audit is not a zero-physics pass")
    if (
        not package_audit["passed"]
        or package_audit["confirmation_states_opened"] != 0
        or package_audit["physics_steps"] != 0
        or not package_audit["checks"]["package_sealed_unaccessed"]
    ):
        raise ValueError("candidate package is not sealed and unaccessed")

    affected_t2_rows = [
        t2_manifest["main"]["cross_ebdm"],
        *(t2_manifest["ablation"][key] for key in sorted(t2_manifest["ablation"])),
    ]
    if any("contact_guard_version" in row for row in affected_t2_rows):
        raise ValueError("transport-only T2 actor still contains a contact guard")
    if t2_manifest["main"]["cross_ebdm"]["method_identity"] != "ours_transport_only_v3":
        raise ValueError("unexpected T2 actor identity")
    if t3_manifest["main"]["cross_ebdm"]["method_identity"] != "ours_gated_v5_late_phase":
        raise ValueError("unexpected T3 actor identity")

    completion_delta = stress["paired_completion_step_delta_gated_minus_ungated"]
    output = {
        "schema_version": "four-task-next-revision-cpu-closeout-v3",
        "status": "complete_cpu_only_t2_terminal_capture_revision_scoped",
        "scope": (
            "Deterministic consolidation of the sealed confirmation-v8 statistics and "
            "completed development diagnostics. No physics, GPU, training, confirmation "
            "state access, replay, threshold change, or failure replacement."
        ),
        "device": "cpu",
        "gpu_calls": 0,
        "physics_calls": 0,
        "confirmation_states_opened": 0,
        "formal_statistics": {
            **formal["counts"],
            "three_failure_class_counts": expected_counts,
            "interpretation": (
                "Counts are frozen descriptive outcomes from confirmation-v8. Development "
                "diagnostics below do not replace, replay, or rescore them."
            ),
        },
        "t2_shared_executor_diagnosis": {
            "screen_statistics": executor_screens,
            "response_normalized_screen_statistics": response_normalized,
            "decisive_patterns": [
                (
                    "No tested candidate succeeds in latency-gain mode 0. The strongest "
                    "route candidates issue up to the declared 80 mm/s tangent command and "
                    "accumulate large reference lag while retaining joint margin and staying "
                    "below the frozen hard limits. This isolates deficient closed-loop "
                    "tracking authority rather than a geometric terminal-route error."
                ),
                (
                    "Bounded-clearance and single-axis acquisition variants solve both payload "
                    "modes, showing that the post-outlet geometry is executable when tracking "
                    "authority is adequate. The monotone-lift-only variant times out after the "
                    "outlet, so a completion route remains necessary."
                ),
                (
                    "Direct or dual-axis authority increases cause overforce in 23 of 128 "
                    "candidate episodes. Therefore a global gain increase, a second lateral "
                    "waypoint, or removal of the substep governor is rejected."
                ),
                (
                    "The staged candidate reaches 1/2 latency-mode-1 successes but 0/2 in "
                    "latency mode 0. A route-only revision is therefore insufficient and is "
                    "not eligible for binding."
                ),
                (
                    "The response-normalized executor then reaches strict contact in all four "
                    "fresh latency episodes and completes 3/4 with zero constraint violations. "
                    "Mode 0 completes 2/2 on the surface route; mode 1 completes 1/2 on the "
                    "post-outlet lift route. This validates the mode split but is still too "
                    "small and incomplete for a task-contract binding."
                ),
                (
                    "The sole final failure crosses the outlet, preserves strict contact, "
                    "finishes 0.144 mm outside the 2 mm terminal position set, and never starts "
                    "the 15-control hold. Its force peak remains 10.97 N. The remaining issue "
                    "is finite-time terminal capture under TCP/reference tracking lag, not "
                    "contact acquisition, route geometry, or a frozen constraint."
                ),
            ],
            "next_contract_revision": {
                "scope": "shared task-owned T2 executor applied identically to every method",
                "components": [
                    (
                        "Retain single-axis concurrent acquisition and the bounded-clearance, "
                        "monotone goal-directed completion route."
                    ),
                    (
                        "Add response-normalized Cartesian tracking based only on public TCP "
                        "pose, public reference, measured wrench and elapsed time. Increase "
                        "declared impedance/reference authority only when observed tracking lag "
                        "shows insufficient positive progress; never read servo_gain, delay, "
                        "payload, evaluator support truth or method identity."
                    ),
                    (
                        "Run every admitted reference through the existing 1 ms wrench and "
                        "tangent-effort governor, including acquisition, outlet traversal, "
                        "descent and terminal hold."
                    ),
                    (
                        "Expose the new authority ceiling and adaptation state in the task "
                        "contract and episode evidence so CPU audits can reconstruct every "
                        "issued, admitted and applied command."
                    ),
                    (
                        "Keep the qualified mode split: mode 0 remains on the surface-following "
                        "completion route and mode 1 switches after the outlet to the "
                        "collision-free lift/completion route."
                    ),
                    (
                        "Replace the terminal target-point reference clamp with a bounded "
                        "public-feedback capture rule. Outside the exact terminal set, permit "
                        "only the reference lead needed to overcome measured TCP/reference lag "
                        "while monotonically reducing distance to the terminal set; inside the "
                        "set, contract tangent residuals and reserve 15 controls for dwell."
                    ),
                    (
                        "Compute required progress to the terminal set boundary over the usable "
                        "time before the dwell reserve. Do not divide target-point error by a "
                        "clamped near-zero remaining time or keep a saturated 30 mm/s demand "
                        "after the terminal reference has stopped."
                    ),
                ],
                "unchanged": [
                    "20 second deadline", "25 N total force limit",
                    "10 N lateral force limit", "3 rad/s joint-speed limit",
                    "strict guide-contact evaluator", "reset distributions",
                    "latency/payload cells", "mode semantics", "method observations",
                ],
                "explicitly_rejected": [
                    "route-only scalar search", "global unconditional gain increase",
                    "dual-axis prepositioning", "hidden mechanism-label compensation",
                    "method-private executor", "threshold or deadline relaxation",
                    "terminal hybrid-force injection", "surface traversal for mode 1",
                ],
            },
            "fresh_cpu_qualification": {
                "stage_1_tracking_authority": (
                    "Use fresh xArm7 latency-gain resets and the public sentinel to verify "
                    "single-axis acquisition/outlet progress, report tracking lag, achieved "
                    "rate, joint margin and every 1 ms force/speed margin, and reject any "
                    "overforce or speed event."
                ),
                "stage_2_full_task": (
                    "Run latency/payload by mode-0/mode-1 with 8 fresh paired slots per cell. "
                    "Every cell must be 8/8 strict success with zero constraint violation, "
                    "zero execution error and complete evidence."
                ),
                "stage_1b_terminal_capture": (
                    "Before the four-cell gate, use fresh latency-mode-1 resets to require "
                    "strict contact, zero constraint violations, entry into the exact terminal "
                    "set before the 15-control reserve, and completion of the full dwell."
                ),
                "stage_3_all_methods": (
                    "Only after the public gate passes, bind the same executor to every T2 "
                    "main, baseline and ablation row and rerun behavioral qualification."
                ),
                "paper_use": False,
            },
        },
        "three_failure_class_decisions": [
            {
                "class": "strict_guide_contact_contract_failure",
                "formal_count": 1133,
                "attribution": "shared_T2_task_action_support",
                "diagnostic_result": (
                    "The public sentinel reproduces the latency failure. The response-normalized "
                    "prototype restores strict contact in 4/4 fresh latency episodes, but this "
                    "small development screen does not discharge the four-cell public gate."
                ),
                "current_status": "prototype_mechanism_supported_but_unqualified",
                "next_revision": (
                    "Create a new versioned T2 task contract with public-response-normalized "
                    "tracking, single-axis acquisition and bounded-clearance monotone "
                    "completion after terminal capture also passes. Apply it identically to "
                    "every method."
                ),
                "acceptance_gate": (
                    "On fresh seeds, public_geometry must achieve 8/8 strict successes with "
                    "zero constraint violations in each latency/payload by mode cell before any "
                    "method qualification or formal package is regenerated."
                ),
                "frozen_quantities": [
                    "deadline", "force and speed limits", "strict contact evaluator",
                    "reset cells", "mode semantics",
                ],
            },
            {
                "class": "finite_time_progress_after_contact_gate",
                "formal_count": 2192,
                "attribution": {
                    "T2": (
                        "shared terminal capture remains unresolved after 3/4 fresh latency "
                        "successes with zero constraint violations"
                    ),
                    "T3": "Cross-EBDM late terminal arrival addressed in development qualification",
                    "T4": "completion-projected final actor remains frozen",
                },
                "t3_fresh_paired_results": {
                    key: {
                        "successes": t3_rows[key]["successes"],
                        "episodes": t3_rows[key]["physical_jobs"],
                        "constraint_violations": t3_rows[key]["constraint_violations"],
                    }
                    for key in (
                        "cross_ebdm_current", "cross_ebdm_finite_time",
                        "band_only", "response_mpc",
                    )
                },
                "current_status": (
                    "T3 actor accepted for seed-disjoint formal testing; T2 still blocks a "
                    "scientifically useful next full confirmation."
                ),
                "next_revision": (
                    "Keep ours_gated_v5_late_phase for T3 and the existing T4 final actor; "
                    "change only the shared T2 task contract described above."
                ),
            },
            {
                "class": "constraint_control_failure",
                "formal_count": 1,
                "attribution": "non_reproduced_T2_ungated_ablation_event",
                "fresh_same_policy_stress": {
                    "paired_episodes": 64,
                    "force_or_speed_events": stress["paired_force_or_speed_events"],
                    "guard_reduction_claim_supported": False,
                    "gated_slower_pairs": completion_delta["gated_slower_pairs"],
                    "mean_extra_completion_steps_with_guard": completion_delta["mean"],
                    "joint_speed_p95_radps": {
                        key: stress["observed_episode_maxima"][key][
                            "maximum_joint_speed_radps"
                        ]["p95"]
                        for key in ("gated_full", "ungated_full")
                    },
                },
                "current_status": "closed_without_safety_claim",
                "next_revision": (
                    "Keep the shared constraints unchanged; retain the transport-only T2 actor "
                    "without the heuristic public contact guard and remove general safety or "
                    "constraint-filter claims."
                ),
            },
        ],
        "adopted_actor_changes": {
            "T1": "unchanged",
            "T2": {
                "identity": "ours_transport_only_v3",
                "runtime_contact_guard": "removed",
                "fresh_behavior_qualification": {
                    "cross_ebdm": "32/32 success, zero constraint violations",
                    "task_only": "32/32 success, zero constraint violations",
                    "public_geometry_sentinel": (
                        "4/4 known-nominal only; this does not discharge the four formal "
                        "mechanism-by-mode action-support gate"
                    ),
                },
            },
            "T3": {
                "identity": "ours_gated_v5_late_phase",
                "mechanism": (
                    "late-phase finite-time terminal progress with an explicit 15-control "
                    "hold reserve"
                ),
                "evidence_scope": "development qualification only",
            },
            "T4": "unchanged completion-projected final actor",
        },
        "candidate_package_decision": {
            "package": pointer(NEXT_REVISION_SOURCES["candidate_package_preaccess_audit"]),
            "state": "sealed_unaccessed",
            "decision": "hold_do_not_launch",
            "reason": (
                "The package contains the accepted T2 actor cleanup and T3 method fix, but it "
                "still uses T2 task_contract_v3. The replacement shared executor is only a "
                "3/4 development prototype and has not passed the four-cell public gate. A "
                "full run would spend confirmation states before the shared T2 failure is "
                "resolved."
            ),
            "replacement_rule": (
                "If a new T2 contract passes the public-sentinel gate, mark this unopened "
                "package obsolete, rerun affected behavior/architecture/package/device audits, "
                "and generate a new seed-disjoint package."
            ),
        },
        "theory_revision": [
            "State contact acquisition, contact preservation, finite-time terminal reachability, and terminal dwell as separate properties.",
            "Condition finite-time reachability on a nonempty intersection between the safe action set and actions that provide the required remaining-time progress under delayed observation.",
            "Treat the T2 empty empirical support as a boundary observation for task_contract_v3, not a continuous-space impossibility theorem and not evidence for one method.",
            "Use the T3 result only as evidence that late-phase coverage plus an explicit hold reserve repairs terminal arrival; formal superiority remains untested.",
            "Remove the safety-layer and general constraint-filter claim for the heuristic T2 guard.",
            "Keep the T4 completion projection claim separate from transport and attribute shared executor changes to the benchmark rather than Cross-EBDM.",
            "For T2, require a declared lower bound on measured closed-loop progress after accounting for delay and tracking lag; a commanded-velocity bound alone does not establish reachability.",
            "The finite-time condition must reserve acquisition, outlet traversal, completion and 15 terminal-hold controls separately: tau_acq + L_out/v_out + L_goal/v_goal + 15*dt <= 20 s.",
            "Define terminal progress as decrease of distance to the exact terminal set, not distance to a single goal point; terminal dwell begins only after set entry and must not be inferred from a small final pose error.",
        ],
        "next_execution_order": [
            "Modify only the response-normalized terminal capture: use distance-to-terminal-set progress, bounded public tracking-lag lead and an explicit 15-control dwell reserve; keep acquisition and both mode routes unchanged.",
            "Run a fresh latency-mode-1 CPU terminal-capture screen and reject immediately on any strict-contact loss, force/speed event, terminal-set miss, or incomplete dwell.",
            "Run the four-cell fresh public_geometry gate; stop if any cell is below 8/8 or has a constraint violation.",
            "Rerun every affected T2 behavioral row and independent audit under the same shared contract.",
            "Rerun zero-physics architecture, CPU package, and GPU1 device audits.",
            "Seal the unopened v10 package as obsolete and create a new seed-disjoint package only after all gates pass.",
            "Launch the new package once on physical GPU1 and perform the frozen offline statistics after completion.",
        ],
        "sources": {
            "analysis_script": pointer(Path(__file__).resolve()),
            **{key: pointer(path) for key, path in NEXT_REVISION_SOURCES.items()},
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_path)
    print(json.dumps({
        "output": pointer(output_path),
        "status": output["status"],
        "formal_failure_counts": expected_counts,
        "candidate_package_decision": output["candidate_package_decision"]["decision"],
    }, indent=2, sort_keys=True))


def _transition_report_summary(report_path: Path) -> dict[str, Any]:
    report, rows = _diagnostic_rows(report_path)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["key"] not in {
            "public_geometry_response_normalized",
            "public_geometry_task_contract_v4",
        }:
            raise ValueError(f"unexpected transition-screen row: {row['key']}")
        if not row["evidence"]["audit_passed"]:
            raise ValueError("transition screen contains unaudited episode evidence")
        if not row["exact_physics_contact_evidence"]:
            raise ValueError("transition screen lacks exact physics evidence")
        groups[(
            row["request"]["scene_cell"],
            int(row["request"]["requested_mode"]),
        )].append(row)
    cells = {}
    for key, values in sorted(groups.items()):
        label = f"{key[0]}|mode_{key[1]}"
        cells[label] = {
            "episodes": len(values),
            "successes": sum(bool(row["success"]) for row in values),
            "constraint_violations": sum(
                bool(row["constraint_violation"]) for row in values
            ),
            "outcomes": dict(sorted(Counter(
                row["outcome"] for row in values
            ).items())),
            "control_steps": _distribution([
                row["physical_control_steps"] for row in values
            ]),
            "force_peak_N": _distribution([
                row["force_peak_N"] for row in values
            ]),
            "lateral_force_peak_N": _distribution([
                row["lateral_force_peak_N"] for row in values
            ]),
            "evidence_audits_passed": sum(
                bool(row["evidence"]["audit_passed"]) for row in values
            ),
            "exact_physics_evidence": sum(
                bool(row["exact_physics_contact_evidence"]) for row in values
            ),
        }
    return {
        "episodes": len(rows),
        "successes": sum(bool(row["success"]) for row in rows),
        "constraint_violations": sum(
            bool(row["constraint_violation"]) for row in rows
        ),
        "execution_errors": int(report["counts"]["execution_errors"]),
        "cells": cells,
        "report": pointer(report_path),
        "plan": report["artifacts"]["plan"],
        "progress": report["artifacts"]["progress"],
    }


def t2_transition_revision_closeout(output_path: Path) -> None:
    """Close the T2 public transition diagnosis without running new physics."""
    output_path = output_path if output_path.is_absolute() else ROOT / output_path
    formal = read_json(FORMAL_DIAGNOSIS)
    formal_counts = {
        row["class"]: row["count"]
        for row in formal["three_physical_failure_classes"]
    }
    expected_counts = {
        "strict_guide_contact_contract_failure": 1133,
        "finite_time_progress_after_contact_gate": 2192,
        "constraint_control_failure": 1,
    }
    if formal_counts != expected_counts:
        raise ValueError(f"formal failure counts changed: {formal_counts}")

    screens = {
        name: _transition_report_summary(path)
        for name, path in T2_TRANSITION_REPORTS.items()
    }
    final = screens["response_normalized_public_geometry_release_gate"]
    if (
        final["episodes"] != 32
        or final["successes"] != 32
        or final["constraint_violations"] != 0
        or final["execution_errors"] != 0
        or set(final["cells"]) != {
            f"{cell}|mode_{mode}"
            for cell in ("latency_gain_shift", "payload_shift")
            for mode in (0, 1)
        }
        or any(
            cell["episodes"] != 8
            or cell["successes"] != 8
            or cell["constraint_violations"] != 0
            or cell["evidence_audits_passed"] != 8
            or cell["exact_physics_evidence"] != 8
            for cell in final["cells"].values()
        )
    ):
        raise ValueError("final T2 public release gate is not 32/32 auditable")

    output = {
        "schema_version": "t2-response-normalized-cpu-revision-closeout-v1",
        "status": "complete_cpu_public_gate_passed_contract_revision_ready",
        "scope": (
            "CPU-only deterministic analysis of recorded development evidence. "
            "The source screens are diagnostic-only, open no confirmation state, "
            "replace no failure and are not paper performance evidence."
        ),
        "device": "cpu",
        "gpu_calls": 0,
        "physics_calls_during_closeout": 0,
        "confirmation_states_opened": 0,
        "failure_replacement": False,
        "performance_evidence": False,
        "formal_statistics": {
            **formal["counts"],
            "three_failure_class_counts": expected_counts,
            "interpretation": (
                "Frozen confirmation-v8 outcomes remain invalid for performance "
                "claims; fresh development evidence diagnoses the next shared T2 "
                "contract and does not rescore those outcomes."
            ),
        },
        "development_screen_statistics": screens,
        "three_common_failure_mechanisms": [
            {
                "name": "phase_transition_reference_energy",
                "observed_failure": (
                    "Geometric outlet or descent transitions occurred while the "
                    "public reference retained millimeter-scale lead. Delayed reference "
                    "projection then released stored impedance energy as a force spike."
                ),
                "rejected_changes": [
                    "instant axis switching", "global gain increase",
                    "unbounded damping progress", "position-only authority recovery",
                ],
                "adopted_change": (
                    "Ramp tangent stiffness before the public outlet, normalize the "
                    "transition stiffness by TCP/reference lead, latch the outlet, and "
                    "actively unload the public reference before switching axes."
                ),
                "theory_condition": (
                    "At every hybrid transition, bound commanded spring effort and "
                    "require measured pose, speed and reference-lead membership in the "
                    "transition set before changing the active tangent axis."
                ),
            },
            {
                "name": "finite_time_terminal_capture",
                "observed_failure": (
                    "Safe trajectories entered the exact terminal set in the final "
                    "controls and could not complete the frozen 15-control dwell."
                ),
                "rejected_changes": [
                    "deadline relaxation", "terminal threshold relaxation",
                    "target-point-only reference clamp",
                ],
                "adopted_change": (
                    "Plan progress to the terminal-set boundary using a mode-specific "
                    "capture reserve, align the internal terminal latch with the frozen "
                    "two-millimeter set, and start the terminal force hold on first entry."
                ),
                "theory_condition": (
                    "Finite-time viability must reserve both entry time and invariant-set "
                    "dwell; asymptotic progress or arrival at the deadline is insufficient."
                ),
            },
            {
                "name": "clearance_descent_contact_coupling",
                "observed_failure": (
                    "During mode-one descent, planar tracking error and lateral force could "
                    "grow together; fixed full-speed, half-speed and hard stop policies each "
                    "produced tail failures in at least one mechanism cell."
                ),
                "rejected_changes": [
                    "fixed 60 mm/s descent", "fixed 30 mm/s descent",
                    "binary planar-set stop",
                ],
                "adopted_change": (
                    "Outside the planar terminal set, scale the 30 mm/s descent ceiling "
                    "continuously by the existing public terminal force-release factor."
                ),
                "theory_condition": (
                    "Hybrid liveness authority must be continuously coupled to the public "
                    "constraint margin; a geometry-only descent schedule is not invariant."
                ),
            },
        ],
        "formal_failure_class_decisions": [
            {
                "class": "strict_guide_contact_contract_failure",
                "formal_count": 1133,
                "diagnosis": (
                    "Shared task-controller transition and tracking support, addressed by "
                    "the response-normalized public executor; no method-specific repair."
                ),
                "development_gate": "32/32 strict success with complete evidence",
            },
            {
                "class": "finite_time_progress_after_contact_gate",
                "formal_count": 2192,
                "diagnosis": (
                    "Terminal-set entry and dwell reserve were absent from the shared T2 "
                    "progress condition; the revised controller makes both explicit."
                ),
                "development_gate": "all four cells complete before the frozen deadline",
            },
            {
                "class": "constraint_control_failure",
                "formal_count": 1,
                "diagnosis": (
                    "No general safety claim is recovered. The T2 revision only has a "
                    "task-local zero-violation development qualification."
                ),
                "development_gate": "zero violations in 32 fresh public-sentinel episodes",
            },
        ],
        "next_contract_revision": {
            "candidate": "T2 task_contract_v4",
            "status": "ready_to_version_not_yet_bound_to_all_methods",
            "shared_scope": (
                "Task-owned executor applied identically to every T2 main, baseline and "
                "ablation row. No method identity or mechanism parameter is an input."
            ),
            "components": [
                "public response-normalized stiffness and reference authority",
                "shared hybrid normal-force contact acquisition and surface traversal",
                "pre-outlet stiffness ramp and checkpointed reference unloading",
                "public-lead-normalized 8.8 N spring-effort budget",
                "bounded damping progress within the nominal lead plus task-set diameter",
                "mode-zero terminal latch aligned with the frozen two-millimeter set",
                "six-newton terminal force hold with the frozen 15-control dwell",
                "mode-one force-scaled clearance descent and extra dwell capture window",
            ],
            "unchanged": [
                "20 second deadline", "2 mm and 2 degree terminal predicate",
                "15-control dwell", "25 N total and 10 N lateral force limits",
                "3 rad/s joint-speed limit", "strict guide-contact evaluator",
                "latency and payload cells", "mode semantics", "reset distributions",
            ],
            "next_steps": [
                "write task_contract_v4 and bind this exact controller version",
                "rerun T2 behavioral qualification for every main, baseline and ablation row",
                "rerun architecture, package and device audits",
                "only then admit the affected rows to fresh GPU1-only P6 scheduling",
            ],
        },
        "theory_revision": [
            "Replace pointwise action safety with a phase-transition energy condition on public reference lead and commanded impedance.",
            "State finite-time feasibility for terminal-set entry plus invariant dwell, with explicit mode-dependent reserve.",
            "Couple liveness authority continuously to public constraint margin during hybrid contact transitions.",
            "Limit the claim to a qualified shared T2 executor; transport still has no demonstrated task-performance advantage from this diagnostic evidence.",
        ],
        "final_public_gate": final,
        "sources": {
            "analysis_script": pointer(Path(__file__).resolve()),
            "formal_diagnosis": pointer(FORMAL_DIAGNOSIS),
            **{
                name: pointer(path)
                for name, path in T2_TRANSITION_REPORTS.items()
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_path)
    print(json.dumps({
        "output": pointer(output_path),
        "status": output["status"],
        "final_gate": {
            "episodes": final["episodes"],
            "successes": final["successes"],
            "constraint_violations": final["constraint_violations"],
            "execution_errors": final["execution_errors"],
        },
    }, indent=2, sort_keys=True))


def formal_confirmation_closeout(
    confirmation: Path, statistics: Path, output_path: Path,
) -> None:
    """Write a CPU-only failure diagnosis bound to a completed formal run."""
    confirmation = confirmation if confirmation.is_absolute() else ROOT / confirmation
    statistics = statistics if statistics.is_absolute() else ROOT / statistics
    output_path = output_path if output_path.is_absolute() else ROOT / output_path
    launch_path, audit_path = confirmation / "launch.json", confirmation / "audit.json"
    statistics_report_path = statistics / "report.json"
    launch, audit = read_json(launch_path), read_json(audit_path)
    statistics_report = read_json(statistics_report_path)
    if launch["status"] != "complete" or not audit["passed"]:
        raise ValueError("formal confirmation is not complete and audited")
    if not statistics_report["raw_integrity_passed"]:
        raise ValueError("formal statistics did not pass raw-integrity checks")
    outcomes_path = ROOT / launch["outcomes"]["path"]
    if sha(outcomes_path) != launch["outcomes"]["sha256"]:
        raise ValueError("formal outcomes changed after launch closeout")
    if statistics_report["inputs"]["outcomes"]["sha256"] != sha(outcomes_path):
        raise ValueError("statistics and diagnosis do not use the same outcomes")
    for name, digest in statistics_report["outputs"].items():
        if sha(statistics / name) != digest:
            raise ValueError(f"derived statistics changed: {name}")

    rows = [json.loads(line) for line in outcomes_path.read_text().splitlines() if line]
    physical = [row for row in rows if row["target_physical_execution"]]
    rejected = [row for row in rows if row["predeployment_rejection"]]
    failed = [row for row in physical if not row["constrained_success"]]
    infrastructure = [
        row for row in failed
        if row["failure_category"] == "runner_interruption_after_reset"
    ]
    contact = [
        row for row in failed
        if row["task"] == "T2" and row.get("strict_contact_success") is False
    ]
    progress = [
        row for row in failed
        if row["failure_category"] == "timeout" and row not in contact
    ]
    constraint = [
        row for row in failed
        if row not in infrastructure and row not in contact and row not in progress
    ]
    if len(infrastructure) + len(contact) + len(progress) + len(constraint) != len(failed):
        raise ValueError("formal failure classes are not mutually exhaustive")

    t2_contact = [row for row in contact if row["task"] == "T2"]
    t2_progress = [row for row in progress if row["task"] == "T2"]
    t3_progress = [row for row in progress if row["task"] == "T3"]
    t4_progress = [row for row in progress if row["task"] == "T4"]
    rejection_reasons = dict(sorted(Counter(
        row["rejection_reason"] for row in rejected
    ).items()))
    output = {
        "schema_version": "formal-confirmation-shared-failure-closeout-v1",
        "status": "complete_cpu_only_diagnosis_next_revision_required",
        "scope": (
            "Post-access descriptive diagnosis of frozen formal outcomes. No physics, "
            "training, confirmation-state access, threshold change, replay or replacement."
        ),
        "device": "cpu",
        "counts": {
            "scheduled_records": len(rows), "physical_executions": len(physical),
            "predeployment_penalty_slots": len(rejected),
            "physical_failures": len(failed),
            "raw_timeouts": sum(row["failure_category"] == "timeout" for row in failed),
            "infrastructure_retained_failures": len(infrastructure),
        },
        "predeployment_rejections": {
            "scoring_slot_count": len(rejected),
            "reasons": rejection_reasons,
            "interpretation": (
                "These are frozen scoring penalties expanded over scheduled slots, not new "
                "physical episodes. They remain in the paper denominator but are excluded "
                "from physical-mechanism attribution."
            ),
        },
        "three_physical_failure_classes": [
            {
                "class": "finite_time_progress_after_contact_gate",
                **_formal_breakdown(progress),
                "observations": {
                    "T2": (
                        f"{len(t2_progress)} payload-shift mode-0 episodes established valid "
                        "contact and crossed the outlet but ended at the deadline near 35 mm "
                        "position error. The public geometry sentinel shares this failure."
                    ),
                    "T3": (
                        f"{len(t3_progress)} timeouts concentrate in payload-shift mode 1; "
                        "response MPC and both public references outperform Cross-EBDM, so this "
                        "is primarily a method-side terminal-state/hold deficiency."
                    ),
                    "T4": (
                        f"{len(t4_progress)} timeouts occur mainly in earlier actors and "
                        "baselines; final completion-projected Cross-EBDM has one timeout."
                    ),
                },
                "next_diagnostic": [
                    "Persist phase-entry ticks, remaining task error and remaining deadline at every transition.",
                    "For T2, compare the common post-outlet descent/terminal hold on the same payload-mode-0 resets.",
                    "For T3, compare Cross-EBDM, band-only and response MPC on payload-mode-1 resets and audit release/hold predicates.",
                    "Keep the T4 completion-projected actor frozen and expose the earlier gated actor as its completion-projection ablation.",
                ],
            },
            {
                "class": "constraint_control_failure",
                **_formal_breakdown(constraint),
                "observations": (
                    "The sole physical constraint failure is the T2 ungated-full speed-limit "
                    "violation in payload-shift mode 1; the selective gate removes it in the "
                    "matched frozen cells, but one event is insufficient for a broad safety claim."
                ),
                "next_diagnostic": [
                    "Replay only development seeds, never confirmation states, with issued/accepted/applied action and first violating substep logging.",
                    "Test ungated-full versus gated-full under the unchanged speed and force limits.",
                    "Do not relax the common constraint or give any baseline a method-private safety head.",
                ],
            },
            {
                "class": "strict_guide_contact_contract_failure",
                **_formal_breakdown(contact),
                "observations": (
                    f"All {len(t2_contact)} records are T2 latency-gain-shift timeouts. Contact "
                    "is first established around control steps 361--413, but no outlet crossing "
                    "occurs before the 1000-step deadline; public geometry also fails. This is a "
                    "shared latency/liveness failure, not evidence against one learned method."
                ),
                "next_diagnostic": [
                    "Separate contact acquisition, stable-contact dwell, outlet crossing and post-outlet completion in the qualification report.",
                    "Instrument delayed public state, issued reference and stage transition time to locate the latency-induced delay.",
                    "Revise the common T2 executor only if the same defect reproduces for the public sentinel; then rerun every T2 method under a new protocol version.",
                    "Preserve force limits, strict contact thresholds, reset distribution and deadline while diagnosing; any later contract change must be explicit and versioned.",
                ],
            },
        ],
        "infrastructure_failures": {
            **_formal_breakdown(infrastructure),
            "scoring": "retained as failures without replay or replacement",
            "mechanism_interpretation": "excluded from controller failure attribution",
        },
        "next_revision_decision": {
            "decision": "do_not_launch_another_full_confirmation_yet",
            "required_changes": [
                "Add full mechanism-by-mode public-sentinel coverage to T2 predeployment so the latency and payload failures cannot pass on a nominal-only sample.",
                "Diagnose and version the shared T2 latency/contact and post-outlet liveness executor; apply an accepted shared fix identically to every method.",
                "Modify Cross-EBDM on T3 only after paired payload-mode-1 phase/hold diagnosis; response MPC is the current strong target.",
                "Keep the T4 completion projection unchanged and add an explicitly named no-completion-projection comparison in the paper analysis.",
                "Rerun behavioral qualification, package/device audits and then a seed-disjoint confirmation only for protocol rows affected by an accepted change; do not reuse v8 confirmation states.",
            ],
            "theory_changes": [
                "Separate strict contact reachability from contact preservation; v8 shows that preservation guarantees are vacuous when latency prevents outlet reachability.",
                "Add a finite-time liveness margin conditioned on delayed observations and remaining phase budget.",
                "State the completion projection as a terminal-set reachability mechanism, supported on T4; do not claim a task-wide transport advantage from T1--T3.",
                "Treat calibrated gating as a constraint filter with limited evidence, not as a general performance-improvement theorem.",
            ],
        },
        "sources": {
            "analysis_script": pointer(Path(__file__).resolve()),
            "launch": pointer(launch_path), "execution_audit": pointer(audit_path),
            "outcomes": pointer(outcomes_path),
            "statistics_report": pointer(statistics_report_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_path)
    print(json.dumps({
        "output": pointer(output_path), "status": output["status"],
        "counts": output["counts"],
        "failure_classes": {
            row["class"]: row["count"] for row in output["three_physical_failure_classes"]
        },
    }, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--statistics", type=Path)
    parser.add_argument("--development-diagnosis", action="store_true")
    parser.add_argument("--t2-action-support-screen", action="store_true")
    parser.add_argument("--next-revision-closeout", action="store_true")
    parser.add_argument("--t2-transition-closeout", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "-1"}:
        raise RuntimeError("offline closeout is CPU-only")
    if args.t2_action_support_screen:
        if (
            args.development_diagnosis or args.next_revision_closeout
            or args.t2_transition_closeout
            or args.confirmation or args.statistics
        ):
            parser.error(
                "--t2-action-support-screen cannot use other closeout modes"
            )
        output = (
            args.output if args.output != DEFAULT_OUTPUT
            else DEFAULT_T2_ACTION_SUPPORT_OUTPUT
        )
        t2_action_support_screen(output)
        return
    if args.t2_transition_closeout:
        if (
            args.development_diagnosis or args.next_revision_closeout
            or args.confirmation or args.statistics
        ):
            parser.error("--t2-transition-closeout cannot use other closeout modes")
        output = (
            args.output if args.output != DEFAULT_OUTPUT
            else DEFAULT_T2_TRANSITION_OUTPUT
        )
        t2_transition_revision_closeout(output)
        return
    if args.next_revision_closeout:
        if args.development_diagnosis or args.confirmation or args.statistics:
            parser.error("--next-revision-closeout cannot use other closeout modes")
        output = (
            args.output if args.output != DEFAULT_OUTPUT
            else DEFAULT_NEXT_REVISION_OUTPUT
        )
        next_revision_closeout(output)
        return
    if args.development_diagnosis:
        if args.confirmation or args.statistics:
            parser.error("--development-diagnosis cannot use formal-run inputs")
        output = (
            args.output if args.output != DEFAULT_OUTPUT
            else DEFAULT_DEVELOPMENT_OUTPUT
        )
        development_diagnosis_closeout(output)
        return
    if args.confirmation:
        if not args.statistics:
            parser.error("--confirmation requires --statistics")
        formal_confirmation_closeout(args.confirmation, args.statistics, args.output)
        return
    if args.statistics:
        parser.error("--statistics requires --confirmation")

    predeployment = read_json(PREDEPLOYMENT)
    risk = read_json(RISK_REVISION)
    capability = read_json(CAPABILITY_CLOSEOUT)
    invalid_confirmation = read_json(INVALID_CONFIRMATION_DECISION)
    structural_accessed = read_json(T2_STRUCTURAL_ACCESSED)
    structural_fresh = read_json(T2_STRUCTURAL_FRESH)
    architecture = read_json(ARCHITECTURE_V3)
    if predeployment["status"] != "complete_with_method_diagnosis_frozen":
        raise ValueError("predeployment closeout is not complete")
    if risk["status"] != "revision_required_before_predeployment":
        raise ValueError("risk validation does not require revision")
    if capability["status"] != "T2_T3_T4_shared_capability_gates_passed":
        raise ValueError("shared capability closeout is not complete")
    if invalid_confirmation["old_confirmation_status"] != "invalid_and_not_reinstated":
        raise ValueError("old confirmation run is not sealed invalid")
    if not structural_accessed["passed"] or structural_accessed["status"] != "passed_diagnostic_retained":
        raise ValueError("T2 structural accessed regression did not pass")
    if not structural_fresh["passed"] or structural_fresh["status"] != "passed_fresh_qualification":
        raise ValueError("T2 structural fresh qualification did not pass")
    if architecture["status"] != "passed" or architecture["physics_steps"] != 0:
        raise ValueError("common-protocol-v3 architecture audit did not pass at zero physics steps")

    reports = {key: read_json(path) for key, path in T2_REPORTS.items()}
    cells = {key: candidate_cell(report) for key, report in reports.items()}
    expected = {
        "accessed_speed": (8, 8, 0),
        "fresh_speed": (32, 31, 1),
        "raw_lift_00650": (32, 31, 0),
        "raw_lift_00675": (32, 30, 0),
        "spatial_lead_lift_00675": (32, 16, 0),
        "raw_planar_lead_lift_00675": (32, 30, 0),
    }
    for key, (episodes, successes, violations) in expected.items():
        cell = cells[key]
        actual = (
            cell["episodes"], cell["successes"],
            cell["constraint_violations"],
        )
        if actual != (episodes, successes, violations):
            raise ValueError(f"unexpected {key} result: {actual}")

    t2_timeouts = {
        key: timeout_diagnostics(T2_REPORTS[key], reports[key])
        for key in (
            "raw_lift_00650", "raw_lift_00675",
            "spatial_lead_lift_00675", "raw_planar_lead_lift_00675",
        )
    }
    invalid_planar_binding = read_json(INVALID_PLANAR_BINDING)
    if invalid_planar_binding["status"] != "invalid_implementation_binding":
        raise ValueError("misbound planar-lead run is not sealed invalid")
    output = {
        "schema_version": "cpu-shared-failure-closeout-v2",
        "status": "cpu_closeout_complete_common_protocol_v3_qualified",
        "scope": (
            "CPU-only descriptive closeout and bounded development diagnosis; "
            "no confirmation state, training, or paper performance evidence"
        ),
        "device": "cpu",
        "confirmation_states_opened": 0,
        "old_invalid_outcomes_used": False,
        "predeployment_statistics": {
            "episodes": predeployment["counts"]["completed_physical_jobs"],
            "successes": predeployment["counts"]["successes"],
            "timeouts": predeployment["counts"]["timeouts"],
            "constraint_violations": predeployment["counts"]["constraint_violations"],
            "contact_contract_failures": predeployment["counts"]["contact_contract_failures"],
            "behavior_qualified_rows": predeployment["counts"]["admission_status"]["behavior_qualified"],
            "behavior_rejected_rows": predeployment["counts"]["admission_status"]["behavior_rejected"],
            "interpretation": "qualification/admission only; not a paper comparison",
        },
        "invalid_confirmation_statistics": {
            **invalid_confirmation["offline_statistics"]["counts"],
            "status": "descriptive_only_invalid_run_not_reinstated",
            "paper_performance_evidence": False,
        },
        "three_failure_classes_observed_in_invalid_predeployment": predeployment["failure_classes"],
        "three_failure_class_diagnosis": [
            {
                "class": "finite_time_progress_failure",
                "old_count": 102,
                "identified_common_mechanisms": {
                    "T2": (
                        "Serial post-outlet travel/lift consumes the deadline and couples "
                        "route progress to transient guide contact."
                    ),
                    "T3": (
                        "The 10 mm insertion lead lacks finite-time margin; retreat can also "
                        "remain latched after a safe terminal state."
                    ),
                    "T4": (
                        "Joint-margin preservation can leave the fixed-base reachable set "
                        "disjoint from the terminal set."
                    ),
                },
                "diagnostic_design": {
                    "unit": "paired reset and requested mode under the same public executor",
                    "observables": [
                        "phase occupancy and transition ticks",
                        "accepted public-reference progress by task-owned axis",
                        "remaining path length divided by remaining deadline",
                        "terminal pose error, observed mode and hold count",
                    ],
                    "counterfactuals": [
                        "T2 serial route versus one declared concurrent free-space route",
                        "T3 10 mm versus 15 mm insertion-only lead, crossed with legacy versus safe-settled release",
                        "T4 zero body intent versus the same public planar body intent after joint-margin admission",
                    ],
                    "falsification_gate": (
                        "The shared revision is rejected if any fresh sentinel times out, "
                        "violates a constraint, mismatches mode, or requires a method-private signal."
                    ),
                },
                "resolved_shared_revision": {
                    "T2": "concurrent_monotone_lift_v3 at 0.00675 m/s",
                    "T3": "15 mm insertion-only lead plus safe-settled release",
                    "T4": "joint-margin public reference plus shared planar body intent",
                },
                "status": "common_mechanisms_qualified_retest_each_method_under_v6",
            },
            {
                "class": "constraint_control_failure",
                "old_count": 97,
                "identified_common_mechanisms": {
                    "T2": (
                        "Control-endpoint sampling aliases substep contact and the old contact "
                        "cell permits single-substep force impulses."
                    ),
                    "T4": (
                        "The legacy arm path crosses joint limits; enforcing joint margin alone "
                        "removes the violation but exposes a reachability timeout."
                    ),
                },
                "diagnostic_design": {
                    "unit": "first unsafe physics substep on a paired post-reset state",
                    "observables": [
                        "last safe and first violating substep",
                        "total/lateral force margins and contact coverage",
                        "issued, accepted and applied references",
                        "joint margin, support count and WBC residual",
                    ],
                    "isolation_profiles": [
                        "T2 old contact cell, 1 ms/40 ms cell, then route-qualified shared executor",
                        "T4 legacy, bounds-only, bounds-plus-public-reference, then shared body intent",
                    ],
                    "attribution_rule": (
                        "A failure reproduced by the public sentinel is common infrastructure. "
                        "After the sentinel and shared executor pass, a clean method-specific "
                        "closed-loop failure remains in that method's denominator."
                    ),
                },
                "resolved_shared_revision": {
                    "T2": "1 ms physics, 20 ms control, 40 ms contact time constant, damping ratio 2, evaluator v4 and shared substep governor",
                    "T4": "joint_margin_public_reference with four-foot support and public planar body intent",
                },
                "status": "common_mechanisms_qualified_retest_each_method_under_v6",
            },
            {
                "class": "contact_contract_failure_regression_gate",
                "old_count": 0,
                "interpretation": (
                    "This class was not observed in the invalid predeployment. It remains a "
                    "negative regression gate and must not be presented as a repaired failure."
                ),
                "diagnostic_design": {
                    "unit": "exact physics-substep evidence for every T2 episode",
                    "observables": [
                        "first contact and outlet substeps",
                        "pre-outlet axial-support coverage",
                        "longest continuous contact loss",
                        "mode-specific crossing coordinate and observed mode",
                        "task-owned request equality",
                    ],
                    "gate": (
                        "Any failure blocks the affected row and the common protocol; no "
                        "threshold tuning, episode replacement or fallback outcome is allowed."
                    ),
                },
                "status": "zero_observed_keep_as_hard_regression_gate",
            },
        ],
        "newer_shared_regression": {
            "episodes": risk["sealed_validation"]["episodes"],
            "independent_audit_passed": risk["sealed_validation"]["independent_audit_passed"],
            "public_sentinel": next(
                cell for cell in risk["sealed_validation"]["cells"]
                if cell["variant"] == "sentinel"
            ),
            "interpretation": risk["shared_public_failure"]["decision"],
            "pattern": risk["shared_public_failure"]["common_pattern_with_task_only_failure"],
        },
        "risk_admission_v1_diagnosis": {
            "eligible_for_manifest": False,
            "forced_infeasible_fallback": risk["selector_diagnosis"],
            "required_v2_changes": risk["next_revision"]["risk_admission_v2"],
        },
        "shared_t2_candidate_sequence": {
            key: {
                "passed": reports[key]["passed"],
                "status": reports[key]["status"],
                "episodes": cells[key]["episodes"],
                "successes": cells[key]["successes"],
                "constraint_violations": cells[key]["constraint_violations"],
                "outcomes": cells[key]["outcomes"],
                "config_overrides": cells[key]["config_overrides"],
            }
            for key in T2_REPORTS
        },
        "shared_t2_timeout_diagnosis": t2_timeouts,
        "qualified_common_revision": {
            "T2_structural_route": {
                "accessed": {
                    "episodes": structural_accessed["counts"]["completed"],
                    "successes": structural_accessed["cells"][0]["successes"],
                    "constraint_violations": structural_accessed["cells"][0]["constraint_violations"],
                    "diagnostic_only": True,
                    "report": pointer(T2_STRUCTURAL_ACCESSED),
                },
                "fresh": {
                    "episodes": structural_fresh["counts"]["completed"],
                    "successes": structural_fresh["cells"][0]["successes"],
                    "constraint_violations": structural_fresh["cells"][0]["constraint_violations"],
                    "all_evidence_audited": structural_fresh["checks"]["all_evidence_audited"],
                    "report": pointer(T2_STRUCTURAL_FRESH),
                },
            },
            "T3": capability["T3"],
            "T4": capability["T4"],
            "common_contracts": {
                task: pointer(path) for task, path in COMMON_CONTRACTS.items()
            },
            "zero_physics_architecture_audit": {
                "configuration_count": architecture["configuration_count"],
                "physics_steps": architecture["physics_steps"],
                "gpu_used": architecture["gpu_used"],
                "report": pointer(ARCHITECTURE_V3),
            },
            "paper_performance_evidence": False,
        },
        "invalid_runs": {
            "misbound_planar_lead": {
                "seal": pointer(INVALID_PLANAR_BINDING),
                "reason": invalid_planar_binding["reason"],
                "eligible_for_selection": False,
            },
        },
        "decisions": {
            "shared_contract": [
                "Keep the frozen force limits, deadline, mode, reset distribution, evaluator, task-owned coordinates, T3 contract, and T4 body intent.",
                "Retain the rejected scalar and serial-route alternatives as development failures; do not reopen their hyperparameters.",
                "Freeze the qualified concurrent T2 free-space route, T3 lead/release contract, and T4 constrained-reachability contract as common benchmark infrastructure.",
                "Apply that structural route identically to every method and the public sentinel. It is benchmark infrastructure and cannot be counted as Cross-EBDM performance.",
                "Do not change the shared executor during v6 predeployment. A new common failure invalidates the protocol and requires a new contract version rather than an in-place patch.",
            ],
            "risk_admission": risk["next_revision"]["risk_admission_v2"],
            "method_side_failures": [
                "Retain the 102 predeployment timeouts and 97 constraint failures in their original denominators.",
                "Repair only a reproducible source fixture, normalization, decoder, objective, or adapter defect. A clean implementation followed by closed-loop failure remains a baseline or ablation limitation.",
                "External baselines receive only the qualified shared executor, never the Cross-EBDM risk head under their original method names.",
            ],
            "theory": [
                "Separate shared executor feasibility, candidate transport, and risk admission.",
                "Treat an empty calibrated safe set as an explicit failed outcome; do not force an infeasible fallback.",
                "State conformal coverage only for exchangeable episode blocks drawn from the frozen deployment candidate generator.",
                "Require nonempty intersection between safe actions and deadline-positive-progress actions along the evaluated trajectory.",
                "Use anisotropic reachability: planar lead bounds protect obstacle clearance, while vertical route liveness requires a separately justified free-space channel.",
            ],
        },
        "release_sequence": [
            "run fresh CPU behavioral predeployment v6 for every main, baseline and ablation row under the immutable common contracts; mark rows that require new learned artifacts as training_required rather than fabricating an outcome",
            "implement risk-admission v2 zero-physics identity and empty-set behavior",
            "only after the shared/public behavioral gate passes, use GPU1 to collect or fit the frozen candidate-generator-matched fit/calibration/held-out blocks",
            "qualify paired online ours/task-only under the same risk gate and rerun every affected v6 learned row",
            "rerun package and device gates after the behavioral actor bindings are frozen",
            "generate a new seed-disjoint four-task roster only after all gates pass",
        ],
        "sources": {
            "analysis_script": pointer(Path(__file__).resolve()),
            "predeployment_closeout": pointer(PREDEPLOYMENT),
            "risk_revision": pointer(RISK_REVISION),
            "capability_closeout": pointer(CAPABILITY_CLOSEOUT),
            "invalid_confirmation_decision": pointer(INVALID_CONFIRMATION_DECISION),
            "t2_structural_accessed": pointer(T2_STRUCTURAL_ACCESSED),
            "t2_structural_fresh": pointer(T2_STRUCTURAL_FRESH),
            "architecture_v3": pointer(ARCHITECTURE_V3),
            **{f"t2_{key}_report": pointer(path) for key, path in T2_REPORTS.items()},
            "invalid_planar_binding": pointer(INVALID_PLANAR_BINDING),
        },
    }
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_path)
    print(json.dumps({
        "output": pointer(output_path),
        "status": output["status"],
        "predeployment_statistics": output["predeployment_statistics"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
