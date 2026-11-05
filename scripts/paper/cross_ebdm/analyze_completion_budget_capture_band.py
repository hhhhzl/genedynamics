#!/usr/bin/env python3
"""Close the T2 completion-budget capture-band full replay on CPU.

This entrypoint reads immutable episode evidence.  It does not construct an
environment, run physics, train a model, or open a confirmation state.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CURRENT = ROOT / (
    "results/arm/guided_contact_alignment/validation/"
    "completion_budget_capture_band_full_comparable_replay"
)
DEFAULT_REFERENCE = ROOT / (
    "results/_qualification/four_task_transfer/"
    "t2_task_contract_v6_handoff_viability_predeployment_fresh_restart_20261024"
)
DEFAULT_OUTPUT = ROOT / (
    "results/_analysis/four_task_transfer/"
    "t2_completion_budget_capture_band_full_replay_closeout_20261024"
)

CURRENT_TO_REFERENCE_ROW = {
    "cross_ebdm_capture_band_v12_full": "T2/main/cross_ebdm",
    "task_only_main_capture_band_v12_full": "T2/main/task_only",
    "response_mpc_capture_band_v12_full": "T2/main/response_mpc",
    "frozen_warm_capture_band_v12_full": "T2/ablation/frozen_warm",
    "task_only_ablation_capture_band_v12_full": "T2/ablation/task_only",
    "matching_no_band_capture_band_v12_full": "T2/ablation/matching_no_band",
    "band_only_capture_band_v12_full": "T2/ablation/band_only",
    "ungated_full_capture_band_v12_full": "T2/ablation/ungated_full",
    "gated_full_capture_band_v12_full": "T2/ablation/gated_full",
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


def vector_norm(value: list[float]) -> float:
    return math.sqrt(sum(float(item) ** 2 for item in value))


def planar_norm(value: list[float]) -> float:
    return math.hypot(float(value[0]), float(value[1]))


def subtract(left: list[float], right: list[float]) -> list[float]:
    return [float(a) - float(b) for a, b in zip(left, right)]


def wilson(successes: int, episodes: int) -> dict[str, float] | None:
    if episodes == 0:
        return None
    z = 1.959963984540054
    p = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (p + z * z / (2.0 * episodes)) / denominator
    half = z * math.sqrt(
        p * (1.0 - p) / episodes + z * z / (4.0 * episodes * episodes)
    ) / denominator
    return {"lower": center - half, "upper": center + half}


def request_identity(row: dict[str, Any], semantic_row: str) -> tuple[Any, ...]:
    request = row["request"]
    parts = request["request_id"].split("/")
    return (
        semantic_row,
        row["instance"],
        request["scene_cell"],
        parts[-2],
        parts[-1],
        int(request["requested_mode"]),
        int(request["reset_seed"]),
    )


def common_state(diagnostic: dict[str, Any]) -> dict[str, Any]:
    state = diagnostic.get("common", diagnostic.get("base", {}))
    if not isinstance(state, dict):
        raise ValueError("common controller diagnostics are missing")
    return state


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


def best_action_delay(
    traces: list[dict[str, Any]], start: int, stop: int, maximum: int = 6,
) -> dict[str, Any]:
    scores: list[tuple[float, int, int]] = []
    for lag in range(maximum + 1):
        squared = 0.0
        count = 0
        for tick in range(max(start, lag), stop + 1):
            issued = traces[tick - lag]["issued_action"][:3]
            applied = traces[tick]["applied_action"][:3]
            squared += sum((float(a) - float(b)) ** 2 for a, b in zip(issued, applied))
            count += 3
        if count:
            scores.append((math.sqrt(squared / count), lag, count))
    error, lag, count = min(scores)
    return {
        "best_lag_controls": lag,
        "translation_rmse_mps": error,
        "scalar_samples": count,
        "diagnostic_only": True,
    }


def failure_profile(row: dict[str, Any]) -> dict[str, Any]:
    record = load_record(row)
    traces = record.get("traces", [])
    diagnostics = record.get("control_diagnostics", [])
    if not traces or len(traces) != len(diagnostics):
        raise ValueError(f"incomplete episode trace: {row['record_id']}")

    stage2 = next((
        tick for tick, diagnostic in enumerate(diagnostics)
        if int((common_state(diagnostic).get("completion") or {}).get("stage", -1)) >= 2
    ), None)
    violation = None
    for tick, trace in enumerate(traces):
        forces = trace["force_peak_substeps"]
        lateral = trace["lateral_force_substeps"]
        if len(forces) != len(lateral):
            raise ValueError(f"substep arrays differ: {row['record_id']}")
        for substep, (force, side) in enumerate(zip(forces, lateral)):
            if float(force) > 25.0 or float(side) > 10.0:
                violation = (tick, substep, float(force), float(side))
                break
        if violation is not None:
            break
    if violation is None:
        raise ValueError(f"overforce has no violating substep: {row['record_id']}")

    tick, substep, force, lateral = violation
    diagnostic = diagnostics[tick]
    state = common_state(diagnostic)
    completion = state.get("completion") or {}
    route = state.get("route") or {}
    trace = traces[tick]
    entry = None
    if stage2 is not None:
        entry_trace = traces[stage2]
        entry_state = common_state(diagnostics[stage2])
        goal_xy = [0.46, -0.18]
        tcp_error = subtract(entry_trace["tcp_position"][:2], goal_xy)
        reference_error = subtract(
            entry_trace["public_reference_position"][:2], goal_xy,
        )
        lead = [float(value) for value in entry_trace["public_reference_lead"][:2]]
        entry = {
            "control_tick": stage2,
            "remaining_deadline_controls": 1000 - stage2,
            "tcp_planar_error_m": tcp_error,
            "tcp_planar_error_norm_m": planar_norm(tcp_error),
            "public_reference_planar_error_m": reference_error,
            "public_reference_planar_error_norm_m": planar_norm(reference_error),
            "public_reference_planar_lead_m": lead,
            "public_reference_planar_lead_norm_m": planar_norm(lead),
            "tcp_reference_goal_dot_m2": sum(
                left * right for left, right in zip(tcp_error, reference_error)
            ),
            "terminal_distance_m": float(entry_state["terminal_distance_m"]),
            "legacy_six_mm_cap_active": bool(
                float(entry_state["terminal_distance_m"]) < 0.006
            ),
        }

    first_outlet = row["episode_metrics"]["first_outlet_step"]
    pre_tick = max(0, tick - 1)
    issued_planar = [
        planar_norm(item["issued_action"][:2]) for item in traces[stage2 or 0:tick + 1]
    ]
    applied_planar = [
        planar_norm(item["applied_action"][:2]) for item in traces[stage2 or 0:tick + 1]
    ]
    return {
        "record_id": row["record_id"],
        "row_key": row["key"],
        "instance": row["instance"],
        "scene_cell": row["request"]["scene_cell"],
        "request_suffix": "/".join(row["request"]["request_id"].split("/")[-2:]),
        "requested_mode": int(row["request"]["requested_mode"]),
        "reset_seed": int(row["request"]["reset_seed"]),
        "evidence_sha256": row["evidence"]["sha256"],
        "first_outlet_control": first_outlet,
        "stage2_entry": entry,
        "first_violation": {
            "control_tick": tick,
            "physics_substep": substep,
            "is_final_recorded_control": tick == len(traces) - 1,
            "controls_after_stage2_entry": None if stage2 is None else tick - stage2,
            "controls_after_outlet": None if first_outlet is None else tick - first_outlet,
            "total_force_N": force,
            "lateral_force_N": lateral,
            "previous_control_total_force_N": float(traces[pre_tick]["force_peak"]),
            "previous_control_lateral_force_N": float(
                traces[pre_tick]["lateral_force_peak"]
            ),
            "selector_code": int(diagnostic["last_selector_code"]),
            "candidate_authority_fraction": float(
                diagnostic["candidate_authority_fraction"]
            ),
            "terminal_ownership_active": bool(
                diagnostic["terminal_ownership_active"]
            ),
            "controller_stage": int(state["stage"]),
            "route_stage": int(route.get("stage", -1)),
            "completion_stage": int(completion.get("stage", -1)),
            "capture_latched": bool(state.get("clearance_capture_latched", False)),
            "terminal_distance_m": float(state["terminal_distance_m"]),
            "force_utilization_observation": float(state["force_utilization"]),
            "issued_translation_mps": [float(value) for value in trace["issued_action"][:3]],
            "applied_translation_mps": [float(value) for value in trace["applied_action"][:3]],
            "public_reference_lead_m": [
                float(value) for value in trace["public_reference_lead"]
            ],
        },
        "terminal_window": {
            "maximum_issued_planar_speed_mps": max(issued_planar),
            "maximum_applied_planar_speed_mps": max(applied_planar),
            "applied_to_issued_delay_fit": best_action_delay(
                traces, stage2 or 0, tick,
            ),
        },
    }


def row_statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["key"]].append(row)
    statistics = []
    for key, values in sorted(grouped.items()):
        successes = sum(bool(row["success"]) for row in values)
        statistics.append({
            "key": key,
            "family": values[0]["family"],
            "episodes": len(values),
            "successes": successes,
            "success_rate": successes / len(values),
            "wilson95": wilson(successes, len(values)),
            "constraint_violations": sum(
                bool(row["constraint_violation"]) for row in values
            ),
            "outcomes": dict(sorted(Counter(
                row["outcome"] for row in values
            ).items())),
            "unique_evidence_traces": len({
                row["evidence"]["sha256"] for row in values
            }),
        })
    return statistics


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build(
    current: Path, reference: Path, output: Path, *, seal_rejection: bool,
) -> dict[str, Any]:
    current_report = read_json(current / "report.json")
    current_audit = read_json(current / "independent_audit.json")
    current_rows = read_jsonl(current / "progress.jsonl")
    reference_rows = read_jsonl(reference / "progress.jsonl")
    if not (
        current_report.get("status") == "complete_recorded_development_diagnosis"
        and current_report.get("diagnostic_only") is True
        and current_report.get("performance_evidence") is False
        and current_audit.get("status") == "passed_diagnostic_integrity"
        and current_audit.get("integrity_passed") is True
        and current_audit.get("counts", {}).get("results") == 584
        and current_audit.get("counts", {}).get("outcomes")
        == {"overforce": 16, "success": 568}
        and len(current_rows) == 584
    ):
        raise ValueError("capture-band replay or independent audit signature changed")

    comparable = [row for row in current_rows if row["key"] in CURRENT_TO_REFERENCE_ROW]
    public = [row for row in current_rows if row["key"] == "public_geometry_binding_reference"]
    if not (len(comparable) == 576 and len(public) == 8 and all(row["success"] for row in public)):
        raise ValueError("current comparable/public coverage changed")

    reference_semantic_rows = set(CURRENT_TO_REFERENCE_ROW.values())
    reference_comparable = [
        row for row in reference_rows if row.get("row_id") in reference_semantic_rows
    ]
    old = {
        request_identity(row, row["row_id"]): row for row in reference_comparable
    }
    new = {
        request_identity(row, CURRENT_TO_REFERENCE_ROW[row["key"]]): row
        for row in comparable
    }
    if len(old) != 576 or set(old) != set(new):
        raise ValueError("v14 and capture-band request rosters differ")
    transitions = Counter((old[key]["outcome"], new[key]["outcome"]) for key in old)
    expected_transitions = Counter({
        ("success", "success"): 550,
        ("success", "overforce"): 15,
        ("timeout", "success"): 8,
        ("overforce", "success"): 2,
        ("overforce", "overforce"): 1,
    })
    if transitions != expected_transitions:
        raise ValueError(f"paired transition signature changed: {transitions}")

    failures = [row for row in comparable if row["outcome"] != "success"]
    mode1 = [row for row in failures if int(row["request"]["requested_mode"]) == 1]
    mode0 = [row for row in failures if int(row["request"]["requested_mode"]) == 0]
    if not (
        len(failures) == 16
        and len(mode1) == 15
        and len({row["evidence"]["sha256"] for row in mode1}) == 7
        and len(mode0) == 1
        and mode0[0]["key"] == "ungated_full_capture_band_v12_full"
    ):
        raise ValueError("capture-band failure membership changed")

    profiles_by_hash: dict[str, dict[str, Any]] = {}
    logical_profiles = []
    for row in failures:
        evidence_hash = row["evidence"]["sha256"]
        profile = profiles_by_hash.get(evidence_hash)
        if profile is None:
            profile = failure_profile(row)
            profiles_by_hash[evidence_hash] = profile
        logical_profiles.append({
            "record_id": row["record_id"],
            "row_key": row["key"],
            "evidence_sha256": evidence_hash,
            "physical_profile": profile,
        })

    terminal_profiles = [profiles_by_hash[row["evidence"]["sha256"]] for row in mode1]
    if not all(
        profile["first_violation"]["selector_code"] == 2
        and profile["first_violation"]["candidate_authority_fraction"] == 0.0
        and profile["first_violation"]["terminal_ownership_active"] is True
        and profile["first_violation"]["completion_stage"] == 2
        and profile["first_violation"]["is_final_recorded_control"] is True
        and profile["first_violation"]["previous_control_total_force_N"] == 0.0
        and profile["stage2_entry"]["legacy_six_mm_cap_active"] is False
        for profile in terminal_profiles
    ):
        raise ValueError("mode-1 terminal-impact signature changed")
    ungated_profile = profiles_by_hash[mode0[0]["evidence"]["sha256"]]
    if not (
        ungated_profile["first_outlet_control"] is None
        and ungated_profile["first_violation"]["selector_code"] == 0
        and ungated_profile["first_violation"]["candidate_authority_fraction"] == 1.0
        and ungated_profile["first_violation"]["completion_stage"] == -1
    ):
        raise ValueError("mode-0 ungated collision signature changed")

    stats = row_statistics(current_rows)
    output.mkdir(parents=True, exist_ok=True)
    stats_csv = output / "row_statistics.csv"
    write_csv(stats_csv, [
        "key", "family", "episodes", "successes", "success_rate",
        "constraint_violations", "outcomes", "unique_evidence_traces",
    ], [{
        **{key: row[key] for key in (
            "key", "family", "episodes", "successes", "success_rate",
            "constraint_violations", "unique_evidence_traces",
        )},
        "outcomes": json.dumps(row["outcomes"], sort_keys=True),
    } for row in stats])

    transition_rows = [{
        "reference_outcome": source,
        "capture_band_outcome": target,
        "episodes": count,
    } for (source, target), count in sorted(transitions.items())]
    transition_csv = output / "paired_transition_statistics.csv"
    write_csv(
        transition_csv,
        ["reference_outcome", "capture_band_outcome", "episodes"],
        transition_rows,
    )

    unique_profile_list = sorted(
        profiles_by_hash.values(), key=lambda value: value["record_id"],
    )
    profiles_path = output / "failure_profiles.json"
    write_json(profiles_path, {
        "schema_version": "t2-capture-band-failure-profiles-v1",
        "logical_failures": len(failures),
        "unique_physical_failures": len(unique_profile_list),
        "profiles": unique_profile_list,
    })

    source_artifacts = {}
    for name, path in {
        "current_plan": current / "plan.json",
        "current_progress": current / "progress.jsonl",
        "current_report": current / "report.json",
        "current_independent_audit": current / "independent_audit.json",
        "reference_progress": reference / "progress.jsonl",
        "analysis_script": Path(__file__),
    }.items():
        source_artifacts[name] = {"path": relative(path), "sha256": sha256(path)}

    closeout = {
        "schema_version": "t2-completion-budget-capture-band-closeout-v1",
        "status": "rejected_full_replay_next_revision_specified",
        "device": "cpu_only_offline_analysis",
        "physical_or_training_steps": 0,
        "confirmation_states_opened": 0,
        "performance_evidence": False,
        "source_artifacts": source_artifacts,
        "integrity": {
            "independent_audit_passed": True,
            "episodes_reaudited": 584,
            "planned_and_completed": 584,
            "execution_errors": 0,
            "public_sentinel": "8/8 success",
            "comparable_roster_exactly_matches_v14": True,
        },
        "counts": {
            "episodes": 584,
            "successes": 568,
            "overforce": 16,
            "comparable_episodes": 576,
            "comparable_successes": 560,
            "comparable_logical_failures": 16,
            "comparable_unique_failed_physical_traces": 8,
        },
        "row_statistics": stats,
        "paired_v14_to_capture_band_transitions": {
            f"{source}->{target}": count
            for (source, target), count in sorted(transitions.items())
        },
        "three_failure_classes": [
            {
                "class": "finite_time_completion_budget_shortfall",
                "v14_logical_failures": 8,
                "v14_unique_physical_traces": 5,
                "capture_band_logical_failures": 0,
                "finding": (
                    "All eight exact v14 timeouts become success.  The 10 mm capture band "
                    "therefore repairs the recorded finite-time completion debt, but cannot "
                    "be accepted because it regresses other previously successful requests."
                ),
            },
            {
                "class": "terminal_recontact_planar_reference_energy",
                "logical_failures": 15,
                "unique_physical_traces": 7,
                "paired_transition": "success->overforce",
                "finding": (
                    "Every event is mode 1, occurs after exact-common selector code 2 with "
                    "zero candidate authority and completion stage 2, and first violates a "
                    "hard limit in the final recorded control after the previous control had "
                    "zero force.  The existing near-terminal speed cap is keyed to 3D terminal "
                    "distance, which is 27--37 mm during clearance descent, so it is inactive "
                    "while delayed planar reference energy drives the recontact impulse."
                ),
            },
            {
                "class": "mode0_preoutlet_ungated_collision",
                "logical_failures": 1,
                "unique_physical_traces": 1,
                "paired_transition": "overforce->overforce",
                "finding": (
                    "The intended ungated ablation still exceeds the total-force limit before "
                    "outlet with selector code 0 and full candidate authority.  It is outside "
                    "the mode-1 shared completion revision and must remain in its denominator."
                ),
            },
        ],
        "decision": (
            "Reject public-T2-completion-budget-capture-band-v12 for release.  Preserve its "
            "liveness change and diagnose the next version as a task-owned mode-1 terminal "
            "planar braking envelope; do not retrain actors or change limits/deadline."
        ),
    }
    closeout_path = output / "closeout.json"
    write_json(closeout_path, closeout)

    diagnosis = {
        "schema_version": "t2-terminal-planar-braking-diagnosis-plan-v1",
        "status": "next_revision_confirmed_not_implemented",
        "candidate_interface": "public-T2-completion-budget-planar-braking-v13",
        "causal_hypothesis": (
            "The v12 capture band restores stage-1 liveness.  Its new failures arise later: "
            "the public completion controller uses full 3D terminal distance to decide whether "
            "to apply a 10 mm/s terminal cap, so vertical clearance keeps that cap disabled "
            "during descent.  Planar common actions can then reach roughly 30 mm/s and survive "
            "the delayed executor long enough to create a lateral recontact impulse."
        ),
        "single_code_change": {
            "owner": "genedynamics.tasks.transfer.controllers.guide_contact",
            "scope": "shared task controller; mode-1 completion stage 2 only",
            "change": (
                "Retain the v12 45 mm/s far-field requirement and 10 mm capture latch.  At "
                "completion stage 2, compute planar terminal distance separately from vertical "
                "clearance and apply the existing 10 mm/s terminal limit to the complete x/y "
                "common action vector.  Clamp both planar references monotonically toward the "
                "public goal; keep vertical completion, force release and dwell unchanged."
            ),
            "new_learned_parameters": 0,
            "new_threshold_search": False,
            "actor_or_solver_change": False,
        },
        "must_preserve": [
            "all eight v14 finite-time failures repaired by v12",
            "all 550 requests that are success in both v14 and v12",
            "the ten repaired v14 shared failures",
            "public geometry 8/8",
            "mode-0 behavior, including the declared ungated ablation failure",
            "20 s deadline, 2 mm/2 degree terminal set, 15-control dwell",
            "25 N total and 10 N lateral force limits",
            "method actors, task-owned coordinates, reset roster and evaluator",
        ],
        "required_new_evidence_fields": [
            "terminal_planar_distance_m",
            "terminal_planar_speed_cap_active",
            "issued/applied planar action norm",
            "public planar reference lead and goal error",
        ],
        "bounded_validation": [
            {
                "order": 1,
                "device": "cpu",
                "work": (
                    "Unit and recorded-state screen: prove v13 is identical to v12 outside "
                    "mode-1 completion stage 2; verify the planar cap triggers on all seven "
                    "unique terminal-impact traces."
                ),
                "gate": "No actor calls, coordinates, thresholds or mode-0 actions change.",
            },
            {
                "order": 2,
                "device": "gpu1",
                "work": (
                    "Exact-request gate containing all 15 logical terminal regressions, the "
                    "overlapping preserved-success neighbors, all ten repaired v14 shared "
                    "failures, the persistent ungated mode-0 request and eight public sentinels."
                ),
                "gate": (
                    "Every mode-1 and public episode succeeds; every prior v12 success is "
                    "preserved; the ungated mode-0 request remains the sole declared failure; "
                    "zero execution/evidence errors."
                ),
            },
            {
                "order": 3,
                "device": "gpu1",
                "work": "Replay the exact 584-job full comparable roster only after order 2 passes.",
                "gate": (
                    "Public 8/8 and all 576 non-ungated comparable episodes succeed; the one "
                    "ungated pre-outlet collision is preserved exactly; no new violation."
                ),
            },
            {
                "order": 4,
                "device": "gpu1",
                "work": (
                    "Run a seed-disjoint public/Cross release gate and then fresh full behavior "
                    "predeployment; only afterward update the shared task contract."
                ),
                "gate": "Independent evidence, architecture, CPU package and GPU1 device audits pass.",
            },
        ],
        "fallback_if_bounded_gate_fails": (
            "Do not scan gains.  Add a public planar-reference settling substage before descent "
            "and test it as a separate version, because the current single-state entry metrics "
            "overlap preserved successes and are not a valid classifier by themselves."
        ),
        "theory_revision": [
            "Exact-common authority is not a safety certificate when delayed reference energy remains in the executor.",
            "The augmented terminal state must include planar reference lead, TCP twist, phase, delayed action debt and dwell reserve.",
            "Finite-time reachability, contact continuity and force safety remain separate conditional claims.",
            "This is shared benchmark infrastructure and cannot be counted as a Cross-EBDM contribution.",
            "Transport task-performance advantage remains unproven until a fresh seed-disjoint main-table comparison.",
        ],
    }
    diagnosis_path = output / "diagnosis_plan.json"
    write_json(diagnosis_path, diagnosis)

    manifest_files = {
        path.name: sha256(path) for path in (
            closeout_path, diagnosis_path, profiles_path, stats_csv, transition_csv,
        )
    }
    manifest = {
        "schema_version": "derived-artifact-manifest-v1",
        "files": dict(sorted(manifest_files.items())),
    }
    write_json(output / "manifest.json", manifest)

    if seal_rejection:
        rejection = {
            "schema_version": "t2-terminal-interface-rejection-v3",
            "status": "rejected_after_complete_full_comparable_replay",
            "candidate": "public-T2-completion-budget-capture-band-v12",
            "diagnostic_only": True,
            "performance_claim": False,
            "confirmation_states_opened": 0,
            "integrity_audit_passed": True,
            "counts": {
                "planned": 584,
                "completed": 584,
                "success": 568,
                "overforce": 16,
                "constraint_violations": 16,
                "execution_errors": 0,
                "logical_failures": 16,
                "unique_failed_physical_traces": 8,
            },
            "device": current_report["device"],
            "artifacts": {
                name: {"path": relative(path), "sha256": sha256(path)}
                for name, path in {
                    "plan.json": current / "plan.json",
                    "progress.jsonl": current / "progress.jsonl",
                    "report.json": current / "report.json",
                    "independent_audit.json": current / "independent_audit.json",
                    "offline_closeout.json": closeout_path,
                    "diagnosis_plan.json": diagnosis_path,
                }.items()
            },
            "failures": [{
                "record_id": row["record_id"],
                "outcome": row["outcome"],
                "evidence_sha256": row["evidence"]["sha256"],
            } for row in failures],
            "reason": (
                "The bounded gate passed 80/80, but the complete exact roster regresses 15 "
                "previously successful mode-1 requests to terminal-stage overforce and retains "
                "the declared ungated mode-0 collision.  Full-roster preservation therefore fails."
            ),
            "finding": (
                "All new mode-1 failures occur after exact-common handoff in completion stage 2. "
                "The next bounded version must cap the complete planar common action by planar "
                "terminal distance during descent while preserving v12 liveness."
            ),
            "gate_required": (
                "Public 8/8, all non-ungated comparable episodes successful, only the declared "
                "ungated mode-0 failure preserved, and zero execution/evidence errors."
            ),
        }
        write_json(current / "REJECTED_CANDIDATE.json", rejection)

    return {
        "output": relative(output),
        "manifest_sha256": sha256(output / "manifest.json"),
        "closeout_sha256": sha256(closeout_path),
        "diagnosis_plan_sha256": sha256(diagnosis_path),
        "rejection_sha256": (
            sha256(current / "REJECTED_CANDIDATE.json") if seal_rejection else None
        ),
        "status": closeout["status"],
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
    print(json.dumps(build(
        args.current.resolve(), args.reference.resolve(), args.output.resolve(),
        seal_rejection=args.seal_rejection,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
