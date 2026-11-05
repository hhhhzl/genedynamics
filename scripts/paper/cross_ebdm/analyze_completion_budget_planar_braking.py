#!/usr/bin/env python3
"""Close the T2 planar-braking exact-request gate from recorded CPU evidence.

The analysis reads completed episode artifacts only.  It does not construct an
environment, execute physics, train an actor, use a GPU, or open confirmation
states.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CURRENT = ROOT / (
    "results/arm/guided_contact_alignment/validation/"
    "completion_budget_planar_braking_exact_request_gate"
)
DEFAULT_REFERENCE = ROOT / (
    "results/arm/guided_contact_alignment/validation/"
    "completion_budget_capture_band_full_comparable_replay"
)
DEFAULT_OUTPUT = ROOT / (
    "results/_analysis/four_task_transfer/"
    "t2_completion_budget_planar_braking_closeout_20261024"
)

CURRENT_TO_REFERENCE_ROW = {
    "T2/main/public_geometry_planar_braking_reference": (
        "T2/main/public_geometry_binding_reference"
    ),
    "T2/main/cross_ebdm_planar_braking": (
        "T2/main/cross_ebdm_capture_band_v12_full"
    ),
    "T2/main/task_only_planar_braking": (
        "T2/main/task_only_main_capture_band_v12_full"
    ),
    "T2/main/response_mpc_planar_braking": (
        "T2/main/response_mpc_capture_band_v12_full"
    ),
    "T2/ablation/matching_no_band_planar_braking": (
        "T2/ablation/matching_no_band_capture_band_v12_full"
    ),
    "T2/ablation/band_only_planar_braking": (
        "T2/ablation/band_only_capture_band_v12_full"
    ),
    "T2/ablation/ungated_full_planar_braking": (
        "T2/ablation/ungated_full_capture_band_v12_full"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def planar_norm(value: list[float]) -> float:
    return math.hypot(float(value[0]), float(value[1]))


def summarize(values: list[float | int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "minimum": min(numeric),
        "median": statistics.median(numeric),
        "mean": statistics.mean(numeric),
        "maximum": max(numeric),
    }


def request_identity(row: dict[str, Any], row_id: str) -> tuple[Any, ...]:
    request = row["request"]
    return (
        row_id,
        row["instance"],
        request["scene_cell"],
        int(request["physical_initial_slot"]),
        int(request["requested_mode"]),
        int(request["reset_seed"]),
    )


def load_record(row: dict[str, Any]) -> dict[str, Any]:
    pointer = row["evidence"]
    path = ROOT / pointer["path"]
    if sha256(path) != pointer["sha256"]:
        raise ValueError(f"episode evidence hash differs: {row['record_id']}")
    evidence = json.loads(gzip.decompress(path.read_bytes()))
    records = evidence.get("records", [])
    if len(records) != 1:
        raise ValueError(f"episode evidence cardinality differs: {row['record_id']}")
    return records[0]


def common_state(diagnostic: dict[str, Any]) -> dict[str, Any]:
    state = diagnostic.get("common", diagnostic.get("base", {}))
    if not isinstance(state, dict):
        raise ValueError("common controller diagnostics are missing")
    return state


def first_hard_violation(
    traces: list[dict[str, Any]],
) -> tuple[int, int, float, float] | None:
    for tick, trace in enumerate(traces):
        forces = trace["force_peak_substeps"]
        lateral = trace["lateral_force_substeps"]
        if len(forces) != len(lateral):
            raise ValueError("force substep arrays differ")
        for substep, (force, side) in enumerate(zip(forces, lateral)):
            if float(force) > 25.0 or float(side) > 10.0:
                return tick, substep, float(force), float(side)
    return None


def maximum_consecutive(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def episode_profile(
    current: dict[str, Any], reference: dict[str, Any],
) -> dict[str, Any]:
    record = load_record(current)
    traces = record.get("traces", [])
    diagnostics = record.get("control_diagnostics", [])
    if not traces or len(traces) != len(diagnostics):
        raise ValueError(f"incomplete evidence: {current['record_id']}")
    states = [common_state(item) for item in diagnostics]
    stage2 = next((
        tick for tick, state in enumerate(states)
        if int((state.get("completion") or {}).get("stage", -1)) >= 2
    ), None)
    cap = next((
        tick for tick, state in enumerate(states)
        if state.get("terminal_planar_speed_cap_active") is True
    ), None)
    if stage2 != cap and int(current["request"]["requested_mode"]) == 1:
        raise ValueError(f"stage/cap entry differs: {current['record_id']}")
    violation = first_hard_violation(traces)
    active_distance = [
        float(state.get("terminal_planar_distance_m", -1.0))
        for state in states
    ]
    active_distance = [value for value in active_distance if value >= 0.0]
    in_planar_set = [value == 0.0 for value in active_distance]
    profile: dict[str, Any] = {
        "record_id": current["record_id"],
        "row_id": current["row_id"],
        "method": current["method"],
        "instance": current["instance"],
        "scene_cell": current["request"]["scene_cell"],
        "physical_initial_slot": int(
            current["request"]["physical_initial_slot"]
        ),
        "requested_mode": int(current["request"]["requested_mode"]),
        "reset_seed": int(current["request"]["reset_seed"]),
        "v12_outcome": reference["outcome"],
        "v13_outcome": current["outcome"],
        "transition": f"{reference['outcome']}->{current['outcome']}",
        "control_steps": len(traces),
        "stage2_entry_control": stage2,
        "planar_cap_entry_control": cap,
        "terminal_planar_distance_final_m": (
            active_distance[-1] if active_distance else None
        ),
        "terminal_planar_distance_minimum_m": (
            min(active_distance) if active_distance else None
        ),
        "controls_observed_in_planar_set": sum(in_planar_set),
        "longest_consecutive_controls_in_planar_set": maximum_consecutive(
            in_planar_set
        ),
        "terminal_pose_hold_controls_final": int(
            states[-1].get("terminal_pose_hold_controls", 0)
        ),
        "terminal_pose_hold_controls_maximum": max(
            int(state.get("terminal_pose_hold_controls", 0))
            for state in states
        ),
        "force_peak_N": float(current["force_peak_N"]),
        "lateral_force_peak_N": float(current["lateral_force_peak_N"]),
        "evidence_sha256": current["evidence"]["sha256"],
    }
    if stage2 is not None:
        entry_trace = traces[stage2]
        profile.update({
            "stage2_public_reference_lead_norm_m": planar_norm(
                entry_trace["public_reference_lead"]
            ),
            "stage2_issued_planar_speed_mps": planar_norm(
                entry_trace["issued_action"]
            ),
            "stage2_applied_planar_speed_mps": planar_norm(
                entry_trace["applied_action"]
            ),
            "stage2_force_peak_N": float(entry_trace["force_peak"]),
            "stage2_lateral_force_peak_N": float(
                entry_trace["lateral_force_peak"]
            ),
            "maximum_issued_planar_speed_after_stage2_mps": max(
                planar_norm(trace["issued_action"])
                for trace in traces[stage2:]
            ),
            "maximum_applied_planar_speed_after_stage2_mps": max(
                planar_norm(trace["applied_action"])
                for trace in traces[stage2:]
            ),
        })
    else:
        for key in (
            "stage2_public_reference_lead_norm_m",
            "stage2_issued_planar_speed_mps",
            "stage2_applied_planar_speed_mps",
            "stage2_force_peak_N",
            "stage2_lateral_force_peak_N",
            "maximum_issued_planar_speed_after_stage2_mps",
            "maximum_applied_planar_speed_after_stage2_mps",
        ):
            profile[key] = None
    if violation is None:
        profile.update({
            "violation_control": None,
            "violation_substep": None,
            "controls_from_stage2_to_violation": None,
            "pre_violation_force_peak_N": None,
            "pre_violation_lateral_force_peak_N": None,
            "violation_total_force_N": None,
            "violation_lateral_force_N": None,
            "violation_applied_planar_speed_mps": None,
        })
    else:
        tick, substep, force, lateral = violation
        previous = traces[max(0, tick - 1)]
        profile.update({
            "violation_control": tick,
            "violation_substep": substep,
            "controls_from_stage2_to_violation": (
                None if stage2 is None else tick - stage2
            ),
            "pre_violation_force_peak_N": float(previous["force_peak"]),
            "pre_violation_lateral_force_peak_N": float(
                previous["lateral_force_peak"]
            ),
            "violation_total_force_N": force,
            "violation_lateral_force_N": lateral,
            "violation_applied_planar_speed_mps": planar_norm(
                traces[tick]["applied_action"]
            ),
        })
    return profile


def select(
    profiles: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool],
) -> list[dict[str, Any]]:
    result = [profile for profile in profiles if predicate(profile)]
    if not result:
        raise ValueError("expected a non-empty failure class")
    return result


def audit_run(directory: Path, expected: int) -> tuple[dict[str, Any], dict[str, Any]]:
    report = read_json(directory / "report.json")
    audit = read_json(directory / "independent_audit.json")
    required = {
        "all_episode_evidence_reaudited",
        "confirmation_states_unopened",
        "declared_artifact_hashes_match",
        "old_invalid_outcomes_unused",
        "plan_and_results_have_unique_ids",
        "results_exactly_cover_plan",
        "summary_recomputed_exactly",
        "zero_reset_or_execution_errors",
    }
    if not audit.get("integrity_passed"):
        raise ValueError(f"independent audit failed: {directory}")
    if not all(audit["checks"].get(key) for key in required):
        raise ValueError(f"independent audit checks incomplete: {directory}")
    if audit["counts"]["results"] != expected:
        raise ValueError(f"unexpected audited result count: {directory}")
    if report["counts"]["completed_physical_jobs"] != expected:
        raise ValueError(f"unexpected report result count: {directory}")
    if report["counts"]["execution_errors"] != 0:
        raise ValueError(f"execution errors present: {directory}")
    return report, audit


def build(
    current_directory: Path,
    reference_directory: Path,
    output_directory: Path,
    *,
    seal_rejection: bool,
) -> dict[str, Any]:
    current_report, current_audit = audit_run(current_directory, 104)
    reference_report, reference_audit = audit_run(reference_directory, 584)
    current_rows = read_jsonl(current_directory / "progress.jsonl")
    reference_rows = read_jsonl(reference_directory / "progress.jsonl")
    if len(current_rows) != 104 or len(reference_rows) != 584:
        raise ValueError("progress cardinality differs")

    reference_index = {
        request_identity(row, row["row_id"]): row for row in reference_rows
    }
    if len(reference_index) != len(reference_rows):
        raise ValueError("reference request identities are not unique")
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in current_rows:
        reference_row_id = CURRENT_TO_REFERENCE_ROW[row["row_id"]]
        identity = request_identity(row, reference_row_id)
        if identity not in reference_index:
            raise ValueError(f"unmatched exact request: {row['record_id']}")
        pairs.append((row, reference_index[identity]))
    if len(pairs) != 104:
        raise ValueError("exact-request pairing is incomplete")

    profiles = [episode_profile(current, reference) for current, reference in pairs]
    wrapped = [profile for profile in profiles if profile["method"] != "public_geometry"]
    failures = [profile for profile in profiles if profile["v13_outcome"] != "success"]
    timeout = select(
        profiles,
        lambda profile: profile["requested_mode"] == 1
        and profile["v13_outcome"] == "timeout",
    )
    mode1_overforce = select(
        profiles,
        lambda profile: profile["requested_mode"] == 1
        and profile["v13_outcome"] == "overforce",
    )
    mode0_overforce = select(
        profiles,
        lambda profile: profile["requested_mode"] == 0
        and profile["v13_outcome"] == "overforce",
    )
    if (len(failures), len(timeout), len(mode1_overforce), len(mode0_overforce)) != (
        49, 40, 8, 1
    ):
        raise ValueError("failure-class counts differ")
    if any(profile["v13_outcome"] == "success" for profile in wrapped if profile["requested_mode"] == 1):
        raise ValueError("a wrapped mode-1 request unexpectedly succeeded")
    if any(profile["v13_outcome"] != "success" for profile in wrapped if profile["requested_mode"] == 0 and profile not in mode0_overforce):
        raise ValueError("an unexpected wrapped mode-0 failure is present")

    transitions = Counter(
        (profile["v12_outcome"], profile["v13_outcome"])
        for profile in profiles
    )
    expected_transitions = Counter({
        ("success", "success"): 55,
        ("success", "timeout"): 39,
        ("success", "overforce"): 2,
        ("overforce", "overforce"): 7,
        ("overforce", "timeout"): 1,
    })
    if transitions != expected_transitions:
        raise ValueError(f"paired transitions differ: {transitions}")

    row_mode = defaultdict(Counter)
    for profile in profiles:
        row_mode[(profile["row_id"], profile["requested_mode"])][
            profile["v13_outcome"]
        ] += 1
    row_table = []
    for (row_id, mode), outcomes in sorted(row_mode.items()):
        row_table.append({
            "row_id": row_id,
            "requested_mode": mode,
            "episodes": sum(outcomes.values()),
            "success": outcomes["success"],
            "timeout": outcomes["timeout"],
            "overforce": outcomes["overforce"],
        })

    timeout_final_distance = [
        float(profile["terminal_planar_distance_final_m"])
        for profile in timeout
    ]
    timeout_set_visits = [
        int(profile["controls_observed_in_planar_set"])
        for profile in timeout
    ]
    overforce_stage_lead = [
        float(profile["stage2_public_reference_lead_norm_m"])
        for profile in mode1_overforce
    ]
    overforce_delta = [
        int(profile["controls_from_stage2_to_violation"])
        for profile in mode1_overforce
    ]
    if set(overforce_delta) != {16}:
        raise ValueError("mode-1 impact timing is no longer exact")
    if any(float(profile["stage2_force_peak_N"]) != 0.0 for profile in mode1_overforce):
        raise ValueError("mode-1 impact class no longer enters stage 2 unloaded")
    if any(value <= 0 for value in timeout_set_visits):
        raise ValueError("a timeout never entered the planar task set")

    output_directory.mkdir(parents=True, exist_ok=True)
    paired_rows = [{
        key: profile[key] for key in (
            "record_id", "row_id", "method", "instance", "scene_cell",
            "physical_initial_slot", "requested_mode", "reset_seed",
            "v12_outcome", "v13_outcome", "transition", "evidence_sha256",
        )
    } for profile in profiles]
    write_csv(output_directory / "paired_outcomes.csv", paired_rows)
    write_csv(output_directory / "failure_profiles.csv", failures)
    write_csv(output_directory / "row_mode_outcomes.csv", row_table)

    closeout = {
        "schema_version": "t2-completion-budget-planar-braking-closeout-v1",
        "status": "rejected_after_complete_exact_request_gate",
        "candidate": "public-T2-completion-budget-planar-braking-v13",
        "scope": "CPU-only analysis of immutable v12/v13 episode evidence",
        "diagnostic_only": True,
        "performance_claim": False,
        "confirmation_states_opened": 0,
        "integrity": {
            "current_independent_audit_passed": current_audit["integrity_passed"],
            "reference_independent_audit_passed": reference_audit["integrity_passed"],
            "exact_requests_paired": len(pairs),
            "execution_errors": current_report["counts"]["execution_errors"],
        },
        "counts": {
            "planned": 104,
            "completed": 104,
            "success": 55,
            "timeout": 40,
            "overforce": 9,
            "failures": 49,
            "wrapped_mode_1_episodes": sum(
                profile["requested_mode"] == 1 for profile in wrapped
            ),
            "wrapped_mode_1_successes": sum(
                profile["requested_mode"] == 1
                and profile["v13_outcome"] == "success"
                for profile in wrapped
            ),
            "wrapped_mode_0_successes": sum(
                profile["requested_mode"] == 0
                and profile["v13_outcome"] == "success"
                for profile in wrapped
            ),
            "public_geometry_successes": sum(
                profile["method"] == "public_geometry"
                and profile["v13_outcome"] == "success"
                for profile in profiles
            ),
        },
        "paired_transitions_v12_to_v13": {
            f"{before}_to_{after}": count
            for (before, after), count in sorted(transitions.items())
        },
        "row_mode_outcomes": row_table,
        "three_failure_classes": [
            {
                "class": "mode1_finite_time_liveness_regression",
                "logical_failures": len(timeout),
                "unique_physical_traces": len({
                    profile["evidence_sha256"] for profile in timeout
                }),
                "paired_v12_outcomes": dict(sorted(Counter(
                    profile["v12_outcome"] for profile in timeout
                ).items())),
                "final_planar_distance_outside_tolerance_m": summarize(
                    timeout_final_distance
                ),
                "controls_observed_inside_planar_set": summarize(
                    timeout_set_visits
                ),
                "finding": (
                    "Every wrapped mode-1 method times out in at least one selected request. "
                    "All 40 timeouts enter the planar task set, later leave it, finish outside "
                    "the 2 mm tolerance, and accumulate no terminal hold.  Thirty-nine are "
                    "v12 success-to-timeout regressions.  A permanent monotone goal projection "
                    "removes the bounded reference overshoot used by the low-gain closed loop "
                    "to recover terminal position within the frozen deadline."
                ),
            },
            {
                "class": "mode1_descent_recontact_overforce",
                "logical_failures": len(mode1_overforce),
                "unique_physical_traces": len({
                    profile["evidence_sha256"] for profile in mode1_overforce
                }),
                "paired_v12_outcomes": dict(sorted(Counter(
                    profile["v12_outcome"] for profile in mode1_overforce
                ).items())),
                "stage2_reference_lead_norm_m": summarize(overforce_stage_lead),
                "controls_from_stage2_to_violation": summarize(overforce_delta),
                "finding": (
                    "All eight failures are latency-gain mode 1.  Stage 2 starts with zero "
                    "measured force and zero issued/applied planar action, yet the first hard "
                    "violation occurs exactly 16 controls later.  Six failures persist from "
                    "v12 and two are new.  The 10 mm/s cap therefore does not discharge the "
                    "8.20--8.95 mm public reference lead before descent/recontact; permanent "
                    "post-entry clipping is neither a braking phase nor a safety certificate."
                ),
            },
            {
                "class": "mode0_ungated_preoutlet_collision",
                "logical_failures": len(mode0_overforce),
                "unique_physical_traces": 1,
                "paired_v12_outcomes": {"overforce": 1},
                "finding": (
                    "The declared ungated-full payload mode-0 request fails before completion "
                    "stage 2 and remains overforce in v12 and v13.  A mode-1 shared completion "
                    "revision cannot repair it; it remains an ablation result in its denominator."
                ),
            },
        ],
        "decision": (
            "Reject v13 and do not run the 584-episode full roster.  The gate preserves mode 0 "
            "but fails every wrapped mode-1 episode, so the shared interface is unusable."
        ),
    }
    closeout_path = output_directory / "closeout.json"
    write_json(closeout_path, closeout)

    diagnosis = {
        "schema_version": "t2-pre-descent-reference-settling-plan-v1",
        "status": "next_revision_confirmed_not_implemented",
        "rejected_candidate": "public-T2-completion-budget-planar-braking-v13",
        "next_candidate": "public-T2-completion-budget-pre-descent-settling-v14",
        "causal_conclusion": (
            "v13 couples two different obligations into a permanent terminal action clamp. "
            "It is too late to remove public reference lead before descent and too persistent "
            "to preserve finite-time terminal recovery.  The next version must separate a "
            "finite pre-descent settling barrier from the unchanged v12 completion feedback."
        ),
        "single_mechanism_change": {
            "owner": "genedynamics.tasks.transfer.controllers.guide_contact",
            "scope": "shared task controller; mode-1 transition into completion descent",
            "change": (
                "Start from v12, not v13.  When completion first requests stage-2 descent, "
                "temporarily hold vertical descent and freeze planar reference motion.  Admit "
                "descent only after public planar reference lead is within the frozen 2 mm "
                "position tolerance and public planar TCP speed is within the existing 4 mm/s "
                "completion settling limit for four consecutive controls, covering the declared "
                "three-control maximum delay plus one observation.  After admission, restore "
                "the v12 planar completion action exactly; do not retain v13's permanent "
                "monotone projection or 10 mm/s full-vector cap."
            ),
            "new_learned_parameters": 0,
            "actor_or_solver_change": False,
            "gain_search": False,
            "hidden_mechanism_state": False,
        },
        "why_this_addresses_each_class": [
            {
                "class": "mode1_finite_time_liveness_regression",
                "effect": (
                    "Removing the permanent v13 projection restores the v12 terminal feedback "
                    "and bounded overshoot after the finite settling barrier."
                ),
            },
            {
                "class": "mode1_descent_recontact_overforce",
                "effect": (
                    "The barrier discharges observable reference lead and planar speed while "
                    "vertical clearance is held, before descent can create recontact."
                ),
            },
            {
                "class": "mode0_ungated_preoutlet_collision",
                "effect": (
                    "No change by design; the declared ungated mode-0 failure is preserved."
                ),
            },
        ],
        "must_preserve": [
            "v12 actions before the mode-1 completion-descent transition",
            "all mode-0 actions and the declared ungated mode-0 outcome",
            "45 mm/s far-field completion request and 10 mm capture latch",
            "20 s deadline, 2 mm/2 degree terminal set and 15-control dwell",
            "25 N total and 10 N lateral force limits",
            "actors, solver outputs, reset roster, evaluator and confirmation-state rules",
        ],
        "new_evidence_fields": [
            "pre_descent_settling_active",
            "pre_descent_settling_controls",
            "pre_descent_settled_controls",
            "pre_descent_reference_lead_norm_m",
            "pre_descent_planar_speed_mps",
        ],
        "bounded_validation": [
            {
                "order": 1,
                "device": "cpu",
                "work": (
                    "Unit-test exact v12 identity outside the finite settling interval, state "
                    "checkpoint/restore, four-control causal admission, and mode-0 identity."
                ),
                "gate": "No actor, solver, evaluator, threshold, or mode-0 behavior changes.",
            },
            {
                "order": 2,
                "device": "gpu1",
                "work": (
                    "Run the same 104 exact requests so v14 is paired directly with v12/v13."
                ),
                "gate": (
                    "Public 8/8; all 48 wrapped mode-1 requests succeed; all 47 prior wrapped "
                    "mode-0 successes remain success; the one declared ungated mode-0 overforce "
                    "is preserved; zero execution or evidence errors."
                ),
            },
            {
                "order": 3,
                "device": "gpu1",
                "work": "Only after order 2 passes, replay the exact 584-job comparable roster.",
                "gate": (
                    "Public 8/8, every non-ungated comparable episode succeeds, only the "
                    "declared ungated mode-0 collision remains, and independent audit passes."
                ),
            },
            {
                "order": 4,
                "device": "gpu1",
                "work": (
                    "Run seed-disjoint public/Cross release and fresh full predeployment before "
                    "binding a new shared task contract."
                ),
                "gate": (
                    "Architecture, package, device and independent evidence audits all pass."
                ),
            },
        ],
        "stop_rule": (
            "Any new failure, any loss of a v12 success, or any missing evidence rejects the "
            "candidate.  Do not scan gains, settling windows, or lead thresholds after seeing "
            "the gate outcomes."
        ),
        "theory_revision": [
            "Separate pre-contact dissipativity from post-contact finite-time reachability.",
            "A pointwise action cap does not bound stored reference energy in a delayed closed loop.",
            "A terminal controller needs a causal phase barrier whose state includes reference lead, TCP speed, delay-covering persistence and dwell reserve.",
            "The shared controller is benchmark infrastructure and cannot be claimed as Cross-EBDM's learned contribution.",
            "Transport performance advantage remains unproven until seed-disjoint main-table evidence exists.",
        ],
    }
    diagnosis_path = output_directory / "diagnosis_plan.json"
    write_json(diagnosis_path, diagnosis)

    manifest_files = {
        path.name: sha256(path) for path in (
            closeout_path,
            diagnosis_path,
            output_directory / "paired_outcomes.csv",
            output_directory / "failure_profiles.csv",
            output_directory / "row_mode_outcomes.csv",
        )
    }
    manifest = {
        "schema_version": "derived-artifact-manifest-v1",
        "source_artifacts": {
            "current_plan": {
                "path": relative(current_directory / "plan.json"),
                "sha256": sha256(current_directory / "plan.json"),
            },
            "current_progress": {
                "path": relative(current_directory / "progress.jsonl"),
                "sha256": sha256(current_directory / "progress.jsonl"),
            },
            "current_report": {
                "path": relative(current_directory / "report.json"),
                "sha256": sha256(current_directory / "report.json"),
            },
            "current_independent_audit": {
                "path": relative(current_directory / "independent_audit.json"),
                "sha256": sha256(current_directory / "independent_audit.json"),
            },
            "reference_progress": {
                "path": relative(reference_directory / "progress.jsonl"),
                "sha256": sha256(reference_directory / "progress.jsonl"),
            },
            "reference_report": {
                "path": relative(reference_directory / "report.json"),
                "sha256": sha256(reference_directory / "report.json"),
            },
            "reference_independent_audit": {
                "path": relative(reference_directory / "independent_audit.json"),
                "sha256": sha256(reference_directory / "independent_audit.json"),
            },
        },
        "files": dict(sorted(manifest_files.items())),
    }
    manifest_path = output_directory / "manifest.json"
    write_json(manifest_path, manifest)

    rejection_path = current_directory / "REJECTED_CANDIDATE.json"
    if seal_rejection:
        rejection = {
            "schema_version": "t2-terminal-interface-rejection-v3",
            "status": "rejected_after_complete_exact_request_gate",
            "candidate": "public-T2-completion-budget-planar-braking-v13",
            "diagnostic_only": True,
            "performance_claim": False,
            "confirmation_states_opened": 0,
            "integrity_audit_passed": True,
            "counts": {
                "planned": 104,
                "completed": 104,
                "success": 55,
                "timeout": 40,
                "overforce": 9,
                "constraint_violations": 9,
                "execution_errors": 0,
                "logical_failures": 49,
                "unique_failed_physical_traces": 49,
            },
            "device": current_report["device"],
            "artifacts": {
                name: {"path": relative(path), "sha256": sha256(path)}
                for name, path in {
                    "plan.json": current_directory / "plan.json",
                    "progress.jsonl": current_directory / "progress.jsonl",
                    "report.json": current_directory / "report.json",
                    "independent_audit.json": (
                        current_directory / "independent_audit.json"
                    ),
                    "offline_closeout.json": closeout_path,
                    "diagnosis_plan.json": diagnosis_path,
                }.items()
            },
            "failures": [{
                "record_id": profile["record_id"],
                "outcome": profile["v13_outcome"],
                "evidence_sha256": profile["evidence_sha256"],
            } for profile in failures],
            "reason": (
                "The complete 104-request gate fails all 48 wrapped mode-1 requests: 40 "
                "timeouts and eight overforce outcomes.  Relative to v12 it introduces 39 "
                "success-to-timeout and two success-to-overforce regressions."
            ),
            "finding": (
                "Permanent stage-2 monotone planar clipping removes terminal liveness without "
                "dissipating pre-descent reference lead.  The next isolated candidate must use "
                "a finite public-state settling barrier and then restore v12 completion feedback."
            ),
            "gate_required": (
                "Public 8/8, all wrapped mode-1 episodes successful, all prior mode-0 successes "
                "preserved, only the declared ungated mode-0 collision retained, and zero "
                "execution/evidence errors."
            ),
        }
        write_json(rejection_path, rejection)

    return {
        "status": closeout["status"],
        "output": relative(output_directory),
        "closeout_sha256": sha256(closeout_path),
        "diagnosis_plan_sha256": sha256(diagnosis_path),
        "manifest_sha256": sha256(manifest_path),
        "rejection_sha256": sha256(rejection_path) if seal_rejection else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seal-rejection", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(
        args.current.resolve(),
        args.reference.resolve(),
        args.output.resolve(),
        seal_rejection=args.seal_rejection,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
