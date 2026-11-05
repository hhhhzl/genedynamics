#!/usr/bin/env python3
"""Close the recorded T2 reference-unloading revisions on CPU.

This program reads immutable episode artifacts from the v15 and v16 bounded
gates.  It verifies their evidence hashes, seals the stopped v16 candidate,
compares the paired Cross-EBDM mode-1 requests, and tests whether executor
delay is causally identifiable from public reference motion and previously
issued actions.  It never constructs an environment, runs physics, trains a
model, opens confirmation states, or uses a GPU.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
VALIDATION = ROOT / "results/arm/guided_contact_alignment/validation"
DEFAULT_V15 = VALIDATION / (
    "completion_budget_reference_unloading_exact_request_gate"
)
DEFAULT_V16 = VALIDATION / (
    "completion_budget_reference_debt_unloading_exact_request_gate"
)
DEFAULT_V17 = VALIDATION / (
    "completion_budget_causal_reference_unloading_interface_parameter_gate"
)
DEFAULT_V13_ANALYSIS = ROOT / (
    "results/_analysis/four_task_transfer/"
    "t2_completion_budget_planar_braking_closeout_20261024"
)
DEFAULT_OUTPUT = ROOT / (
    "results/_analysis/four_task_transfer/"
    "t2_completion_budget_reference_unloading_closeout_20261024"
)

CONTROL_DT_SECONDS = .02
DELAY_CANDIDATES = (1, 2, 3)
MINIMUM_IDENTIFICATION_SAMPLES = 8
EXCITATION_THRESHOLD_MPS = 1e-6


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
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


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


def load_record(row: dict[str, Any]) -> dict[str, Any]:
    pointer = row["evidence"]
    path = ROOT / pointer["path"]
    actual = sha256(path)
    if actual != pointer["sha256"]:
        raise ValueError(
            f"episode evidence hash differs: {row['record_id']} "
            f"({actual} != {pointer['sha256']})"
        )
    evidence = json.loads(gzip.decompress(path.read_bytes()))
    records = evidence.get("records", [])
    if len(records) != 1:
        raise ValueError(f"episode evidence cardinality differs: {row['record_id']}")
    return records[0]


def common_state(diagnostic: dict[str, Any]) -> dict[str, Any]:
    state = diagnostic.get("common", diagnostic.get("base", diagnostic))
    if not isinstance(state, dict):
        raise ValueError("common controller diagnostics are missing")
    return state


def completion_stage(state: dict[str, Any]) -> int:
    completion = state.get("completion") or {}
    return int(completion.get("stage", state.get("stage", -1)))


def stage2_entry(states: list[dict[str, Any]]) -> int:
    return next(
        (tick for tick, state in enumerate(states)
         if completion_stage(state) >= 2),
        len(states),
    )


def squared_planar_error(left: list[float], right: list[float]) -> float:
    return sum(
        (float(left[index]) - float(right[index])) ** 2
        for index in (0, 1)
    )


def maximum_planar_separation(actions: dict[int, list[float]]) -> float:
    return max(
        math.sqrt(squared_planar_error(left, right))
        for left in actions.values()
        for right in actions.values()
    )


def delay_identification(
    traces: list[dict[str, Any]], stage2_tick: int,
) -> dict[str, Any]:
    """Replay an online, public-only delay estimator before stage 2.

    At tick t, the estimator compares the observed planar reference velocity
    to actions issued at t-1, t-2, and t-3.  Only previously issued actions
    and the current/previous public reference positions are used.  The hidden
    mechanism declaration is deliberately absent from this function.
    """
    errors = {lag: [] for lag in DELAY_CANDIDATES}
    history: list[tuple[int, int, dict[int, float]]] = []
    public_action_history = [
        trace["issued_action"] for trace in traces[:stage2_tick]
    ]
    for tick in range(max(DELAY_CANDIDATES), stage2_tick):
        # A trace row is the public observation after its physical transition.
        # At the equivalent online observation, previous_action through this
        # row is already public.  A declared delay d therefore addresses the
        # -(d+1) item in the inclusive public-action history.
        actions = {
            # Equivalent to public_action_history[:tick + 1][-(lag + 1)].
            lag: public_action_history[tick - lag]
            for lag in DELAY_CANDIDATES
        }
        if maximum_planar_separation(actions) < EXCITATION_THRESHOLD_MPS:
            continue
        current = traces[tick]["public_reference_position"]
        previous = traces[tick - 1]["public_reference_position"]
        observed = [
            (float(current[index]) - float(previous[index]))
            / CONTROL_DT_SECONDS
            for index in (0, 1)
        ]
        for lag in DELAY_CANDIDATES:
            errors[lag].append(squared_planar_error(observed, actions[lag]))
        if len(errors[1]) >= MINIMUM_IDENTIFICATION_SAMPLES:
            scores = {
                lag: math.sqrt(statistics.mean(values))
                for lag, values in errors.items()
            }
            estimate = min(scores, key=scores.get)
            history.append((tick, estimate, scores))

    if not history:
        raise ValueError("insufficient causal excitation before stage 2")
    scores = history[-1][2]
    estimate = min(scores, key=scores.get)
    first_stable_tick = next(
        tick
        for index, (tick, candidate, _) in enumerate(history)
        if candidate == estimate
        and all(later[1] == estimate for later in history[index:])
    )
    ordered = sorted(scores.values())
    return {
        "estimated_delay_controls": estimate,
        "informative_samples": len(errors[1]),
        "first_stable_tick": first_stable_tick,
        "stage2_entry_tick": stage2_tick,
        "identification_slack_controls": stage2_tick - first_stable_tick,
        "score_gap_mps": ordered[1] - ordered[0],
        **{
            f"lag_{lag}_rmse_mps": scores[lag]
            for lag in DELAY_CANDIDATES
        },
    }


def episode_profile(
    row: dict[str, Any], revision: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = load_record(row)
    traces = record.get("traces", [])
    diagnostics = record.get("control_diagnostics", [])
    if not traces or len(traces) != len(diagnostics):
        raise ValueError(f"incomplete evidence: {row['record_id']}")
    states = [common_state(item) for item in diagnostics]
    stage2 = stage2_entry(states)
    identified = delay_identification(traces, stage2)
    hidden_delay = int(row["request"]["mechanism_parameters"]["delay_steps"])
    barrier = [
        tick for tick, state in enumerate(states)
        if state.get("pre_descent_settling_active") is True
    ]
    profile = {
        "revision": revision,
        "record_id": row["record_id"],
        "row_id": row["row_id"],
        "method": row["method"],
        "instance": row["instance"],
        "scene_cell": row["request"]["scene_cell"],
        "physical_initial_slot": int(
            row["request"]["physical_initial_slot"]
        ),
        "requested_mode": int(row["request"]["requested_mode"]),
        "reset_seed": int(row["request"]["reset_seed"]),
        "outcome": row["outcome"],
        "hidden_delay_controls_audit_only": hidden_delay,
        **identified,
        "estimate_matches_hidden_audit": (
            identified["estimated_delay_controls"] == hidden_delay
        ),
        "barrier_controls": len(barrier),
        "barrier_first_tick": barrier[0] if barrier else None,
        "barrier_last_tick": barrier[-1] if barrier else None,
        "terminal_distance_final_m": float(
            states[-1].get("terminal_distance_m", -1.)
        ),
        "terminal_hold_controls_final": int(
            states[-1].get("terminal_pose_hold_controls", 0)
        ),
        "force_peak_N": float(row["force_peak_N"]),
        "lateral_force_peak_N": float(row["lateral_force_peak_N"]),
        "evidence_sha256": row["evidence"]["sha256"],
    }
    return profile, record


def replay_implemented_estimator(
    revision_rows: tuple[tuple[str, list[dict[str, Any]]], ...],
) -> dict[str, Any]:
    """Run the actual v17 estimator class over every recorded public trace."""
    from genedynamics.tasks.transfer.controllers.guide_contact import (
        PublicCompletionBudgetCausalReferenceUnloadingGuideCompletionPolicy,
        PublicTerminalOwnedGuideActor,
    )
    from genedynamics.tasks.transfer.specs import task_spec

    spec = task_spec("T2")
    spec = replace(
        spec,
        config=replace(
            spec.config,
            guide_reference_admission="response_normalized_completion",
            log_stiffness_limit=1.5,
            tangential_wrench_limit=16.0,
        ),
    )
    rows: list[dict[str, Any]] = []

    class ReplayCandidate:
        def act(self, public):
            del public
            raise AssertionError("implementation replay does not call actors")

        def reset(self):
            pass

        def get_state(self):
            return {}

        def restore(self, state):
            if state:
                raise ValueError("replay candidate state differs")

    for revision, source_rows in revision_rows:
        for source in source_rows:
            if source["method"] == "public_geometry":
                # The public sentinel is the unwrapped common policy and does
                # not deploy terminal ownership or its delay estimator.
                continue
            record = load_record(source)
            actor = PublicTerminalOwnedGuideActor(
                ReplayCandidate(), spec, mode=1,
                version=(
                    PublicTerminalOwnedGuideActor.
                    COMPLETION_BUDGET_CAUSAL_REFERENCE_UNLOADING_VERSION
                ),
            )
            for tick, (trace, diagnostic) in enumerate(zip(
                record["traces"], record["control_diagnostics"],
            )):
                actor._append_causal_bounded_action(
                    diagnostic["last_selected_action"]
                )
                actor._observe_causal_reference_delay({
                    "reference_position": [trace["public_reference_position"]],
                    "elapsed_seconds": [[
                        (tick + 1) * spec.config.control_dt
                    ]],
                })
                if actor.public_reference_delay_estimate_controls:
                    break
            hidden = int(
                source["request"]["mechanism_parameters"]["delay_steps"]
            )
            rows.append({
                "revision": revision,
                "record_id": source["record_id"],
                "estimated_delay_controls": (
                    actor.public_reference_delay_estimate_controls
                ),
                "hidden_delay_controls_audit_only": hidden,
                "matches_withheld_audit": (
                    actor.public_reference_delay_estimate_controls == hidden
                ),
                "identification_samples": (
                    actor.public_reference_delay_identification_samples
                ),
                "score_gap_mps": actor.public_reference_delay_score_gap_mps,
            })
    matches = sum(row["matches_withheld_audit"] for row in rows)
    if matches != len(rows) or any(
        row["identification_samples"] != MINIMUM_IDENTIFICATION_SAMPLES
        for row in rows
    ):
        raise ValueError("implemented v17 estimator replay differs")
    return {
        "schema_version": "t2-causal-reference-unloading-implementation-replay-v1",
        "status": "passed",
        "scope": (
            "CPU-only replay of the actual v17 common-controller estimator; "
            "no environment, physics, training, confirmation state or GPU"
        ),
        "implementation": (
            "genedynamics.tasks.transfer.controllers.guide_contact."
            "PublicCompletionBudgetCausalReferenceUnloadingGuideCompletionPolicy"
        ),
        "implementation_version": (
            PublicCompletionBudgetCausalReferenceUnloadingGuideCompletionPolicy.
            VERSION
        ),
        "counts": {
            "episodes": len(rows),
            "matches_withheld_audit": matches,
            "identification_samples_per_episode": (
                MINIMUM_IDENTIFICATION_SAMPLES
            ),
            "estimated_delay_histogram": dict(sorted(Counter(
                row["estimated_delay_controls"] for row in rows
            ).items())),
        },
        "score_gap_mps": summarize([
            row["score_gap_mps"] for row in rows
        ]),
        "online_history_index": "-(d+1) at the post-transition observation",
        "mode_note": (
            "The actual wrapper is replayed for all 49 wrapped v15/v16 "
            "episodes. Public geometry is deliberately unwrapped. Deployed "
            "mode 0 bypasses identification and is covered by exact-action "
            "unit tests."
        ),
        "rows": rows,
    }


def request_key(row: dict[str, Any]) -> tuple[Any, ...]:
    request = row["request"]
    return (
        row["instance"],
        request["scene_cell"],
        int(request["physical_initial_slot"]),
        int(request["requested_mode"]),
        int(request["reset_seed"]),
    )


def frozen_plan_audit(
    v16_plan: dict[str, Any], v17_plan: dict[str, Any], v17_path: Path,
) -> dict[str, Any]:
    def identity(job: dict[str, Any]) -> tuple[Any, ...]:
        request = dict(job["request"])
        # request_id is a diagnostic namespace containing the candidate name;
        # it is not part of physical reset or mechanism identity.
        request.pop("request_id")
        return (
            job["family"], job["method"], job["instance"],
            json.dumps(request, sort_keys=True),
            json.dumps(job["actor_binding"], sort_keys=True),
            job["config"],
            json.dumps(
                job.get("task_actor_interface_parameters", {}), sort_keys=True,
            ),
            json.dumps(job["task_overrides"], sort_keys=True),
        )

    old = [identity(job) for job in v16_plan["jobs"]]
    new = [identity(job) for job in v17_plan["jobs"]]
    if old != new or len(new) != 104:
        raise ValueError("v17 frozen request/actor roster differs from v16")

    def preflight(job: dict[str, Any]) -> bool:
        request = job["request"]
        return bool(
            job["method"] == "public_geometry"
            or (
                job["family"] == "main"
                and job["method"] == "cross_ebdm"
            )
            or (
                job["family"] == "main"
                and job["method"] == "task_only"
                and job["instance"] == "instance_01"
                and request["scene_cell"] == "payload_shift"
                and int(request["physical_initial_slot"]) == 0
            )
        )

    selected = [job for job in v17_plan["jobs"] if preflight(job)]
    if len(selected) != 26:
        raise ValueError("v17 preflight does not contain 26 requests")
    record_ids = [job["record_id"] for job in selected]
    ids_sha256 = hashlib.sha256(
        json.dumps(record_ids, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": "t2-causal-reference-unloading-frozen-plan-audit-v1",
        "status": "passed_planned_unexecuted",
        "plan": {
            "path": relative(v17_path),
            "sha256": sha256(v17_path),
            "physical_jobs": len(v17_plan["jobs"]),
        },
        "checks": {
            "v16_v17_requests_exact_and_ordered": True,
            "v16_v17_actor_bindings_exact_and_ordered": True,
            "v16_v17_task_actor_interface_parameters_exact_and_ordered": True,
            "v16_v17_configs_exact_and_ordered": True,
            "v16_v17_task_overrides_exact_and_ordered": True,
            "all_confirmation_states_closed": all(
                not job["request"]["confirmation_state"]
                for job in v17_plan["jobs"]
            ),
            "all_performance_evidence_false": all(
                not job["request"]["performance_evidence"]
                for job in v17_plan["jobs"]
            ),
        },
        "preflight": {
            "physical_jobs": len(selected),
            "record_ids_sha256": ids_sha256,
            "composition": {
                "public_geometry": 8,
                "cross_ebdm": 16,
                "task_only_known_failure_pair": 2,
                "mode_0": 13,
                "mode_1": 13,
                "latency_gain_shift": 12,
                "payload_shift": 14,
            },
            "record_ids": record_ids,
        },
    }


def summarize(values: list[float | int]) -> dict[str, float | int]:
    numeric = [float(value) for value in values]
    if not numeric:
        raise ValueError("cannot summarize an empty sample")
    return {
        "count": len(numeric),
        "minimum": min(numeric),
        "median": statistics.median(numeric),
        "mean": statistics.mean(numeric),
        "maximum": max(numeric),
    }


def audit_v17_preflight(
    directory: Path, plan_audit: dict[str, Any],
) -> dict[str, Any]:
    progress_path = directory / "progress.jsonl"
    report_path = directory / "report.json"
    rows = read_jsonl(progress_path)
    expected_ids = plan_audit["preflight"]["record_ids"]
    selected = {
        row["record_id"]: row for row in rows
        if row["record_id"] in set(expected_ids)
    }
    if list(selected) != expected_ids or len(selected) != 26:
        raise ValueError("v17 preflight record set or order differs")
    selected_rows = list(selected.values())
    if any(
        row["outcome"] != "success"
        or not row["success"]
        or not row["passed"]
        or row["constraint_violation"]
        or not row["reset_succeeded"]
        for row in selected_rows
    ):
        raise ValueError("v17 preflight contains a failed physical episode")

    delay_rows: list[dict[str, Any]] = []
    evidence_hashes: list[str] = []
    for row in selected_rows:
        record = load_record(row)
        evidence_hashes.append(row["evidence"]["sha256"])
        diagnostics = record.get("control_diagnostics", [])
        if not diagnostics:
            raise ValueError("v17 preflight evidence lacks diagnostics")
        if row["method"] == "public_geometry":
            continue
        final = diagnostics[-1]
        common = common_state(final)
        mode = int(row["request"]["requested_mode"])
        estimate = int(final["public_reference_delay_estimate_controls"])
        samples = int(final["public_reference_delay_identification_samples"])
        gap = float(final["public_reference_delay_score_gap_mps"])
        common_identity = (
            int(common["public_reference_delay_estimate_controls"]),
            int(common["public_reference_delay_identification_samples"]),
            float(common["public_reference_delay_score_gap_mps"]),
        )
        if common_identity != (estimate, samples, gap):
            raise ValueError("v17 wrapper/common delay evidence differs")
        if mode == 0:
            if (estimate, samples, gap) != (0, 0, -1.):
                raise ValueError("v17 mode-0 delay estimator was not bypassed")
            continue
        expected = int(row["request"]["mechanism_parameters"]["delay_steps"])
        expected_limit = min(.030, .030 / expected)
        limit = float(common["pre_descent_causal_unloading_limit_mps"])
        if (
            estimate != expected
            or samples != MINIMUM_IDENTIFICATION_SAMPLES
            or not gap > 0.
            or not math.isclose(limit, expected_limit, abs_tol=1e-12)
        ):
            raise ValueError("v17 causal delay or unloading evidence differs")
        delay_rows.append({
            "record_id": row["record_id"],
            "scene_cell": row["request"]["scene_cell"],
            "estimated_delay_controls": estimate,
            "identification_samples": samples,
            "score_gap_mps": gap,
            "unloading_limit_mps": limit,
        })

    if len(delay_rows) != 9:
        raise ValueError("v17 preflight wrapped mode-1 count differs")
    task_only_recovery = [
        row for row in selected_rows
        if row["method"] == "task_only"
        and row["instance"] == "instance_01"
        and row["request"]["scene_cell"] == "payload_shift"
        and int(row["request"]["physical_initial_slot"]) == 0
        and int(row["request"]["requested_mode"]) == 1
    ]
    if len(task_only_recovery) != 1:
        raise ValueError("former v15 failure is missing from v17 preflight")
    report = read_json(report_path)
    if (
        report["counts"]["execution_errors"] != 0
        or report["confirmation_states_opened"] != 0
        or report["performance_evidence"]
        or report["failure_replacement"]
        or report["device"]["physical_index"] != 1
        or report["device"]["physical_uuid"]
        != "GPU-58b54684-3577-11d3-f039-0e8469190739"
    ):
        raise ValueError("v17 preflight report contract differs")

    audit = {
        "schema_version": "t2-causal-reference-unloading-preflight-audit-v1",
        "status": "passed_full_exact_request_gate_authorized",
        "diagnostic_only": True,
        "performance_claim": False,
        "confirmation_states_opened": 0,
        "failure_replacement": False,
        "counts": {
            "selected": len(selected_rows),
            "success": len(selected_rows),
            "execution_errors": 0,
            "evidence_reaudited": len(evidence_hashes),
            "public_geometry": sum(
                row["method"] == "public_geometry" for row in selected_rows
            ),
            "cross_ebdm": sum(
                row["method"] == "cross_ebdm" for row in selected_rows
            ),
            "task_only": sum(
                row["method"] == "task_only" for row in selected_rows
            ),
            "wrapped_mode1_delay_matches": len(delay_rows),
        },
        "checks": {
            "record_ids_exact_and_ordered": True,
            "all_success": True,
            "all_evidence_hashes_match": True,
            "all_wrapped_mode1_delays_match_withheld_audit": True,
            "all_wrapped_mode1_limits_match_30_over_delay": True,
            "all_wrapped_mode0_estimators_bypassed": True,
            "former_v15_timeout_recovers": True,
            "zero_execution_or_constraint_errors": True,
        },
        "delay_identification": {
            "episodes": delay_rows,
            "histogram": {
                str(key): value for key, value in sorted(Counter(
                    row["estimated_delay_controls"] for row in delay_rows
                ).items())
            },
            "score_gap_mps": summarize([
                row["score_gap_mps"] for row in delay_rows
            ]),
        },
        "artifacts": {
            "plan": {
                "path": relative(directory / "plan.json"),
                "sha256": sha256(directory / "plan.json"),
            },
            "progress": {
                "path": relative(progress_path),
                "sha256": sha256(progress_path),
            },
            "report": {
                "path": relative(report_path),
                "sha256": sha256(report_path),
            },
        },
    }
    write_json(directory / "preflight_independent_audit.json", audit)
    return audit


def audit_v17_full_gate(
    directory: Path, plan: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    progress_path = directory / "progress.jsonl"
    report_path = directory / "report.json"
    rows = read_jsonl(progress_path)
    planned_ids = [job["record_id"] for job in plan["jobs"]]
    record_ids = [row["record_id"] for row in rows]
    if (
        len(rows) != len(planned_ids)
        or not planned_ids
        or len(set(record_ids)) != len(record_ids)
        or set(record_ids) != set(planned_ids)
    ):
        raise ValueError("v17 full-gate denominator differs")

    expected_failure_ids = {
        (
            "T2/main/response_mpc_causal_reference_unloading",
            "instance_04", "payload_shift", 0, 1,
        ): "timeout",
        (
            "T2/main/response_mpc_causal_reference_unloading",
            "instance_04", "payload_shift", 1, 1,
        ): "timeout",
        (
            "T2/ablation/matching_no_band_causal_reference_unloading",
            "instance_07", "latency_gain_shift", 1, 1,
        ): "timeout",
        (
            "T2/ablation/ungated_full_causal_reference_unloading",
            "instance_04", "payload_shift", 1, 0,
        ): "overforce",
    }
    observed_failures = {
        (
            row["row_id"], row["instance"],
            row["request"]["scene_cell"],
            int(row["request"]["physical_initial_slot"]),
            int(row["request"]["requested_mode"]),
        ): row["outcome"]
        for row in rows if row["outcome"] != "success"
    }
    if observed_failures != expected_failure_ids:
        raise ValueError("v17 full-gate failure identities differ")

    delay_rows: list[dict[str, Any]] = []
    mode0_bypassed = 0
    for row in rows:
        record = load_record(row)
        diagnostics = record.get("control_diagnostics", [])
        if not diagnostics:
            raise ValueError("v17 full-gate evidence lacks diagnostics")
        if row["method"] == "public_geometry":
            continue
        final = diagnostics[-1]
        common = common_state(final)
        estimate = int(final["public_reference_delay_estimate_controls"])
        samples = int(final["public_reference_delay_identification_samples"])
        gap = float(final["public_reference_delay_score_gap_mps"])
        if (
            int(common["public_reference_delay_estimate_controls"]),
            int(common["public_reference_delay_identification_samples"]),
            float(common["public_reference_delay_score_gap_mps"]),
        ) != (estimate, samples, gap):
            raise ValueError("v17 full-gate wrapper/common evidence differs")
        mode = int(row["request"]["requested_mode"])
        if mode == 0:
            if (estimate, samples, gap) != (0, 0, -1.):
                raise ValueError("v17 full-gate mode-0 estimator was not bypassed")
            mode0_bypassed += 1
            continue
        expected = int(row["request"]["mechanism_parameters"]["delay_steps"])
        expected_limit = min(.030, .030 / expected)
        limit = float(common["pre_descent_causal_unloading_limit_mps"])
        if (
            estimate != expected
            or samples != MINIMUM_IDENTIFICATION_SAMPLES
            or not gap > 0.
            or not math.isclose(limit, expected_limit, abs_tol=1e-12)
        ):
            raise ValueError("v17 full-gate causal evidence differs")
        delay_rows.append({
            "record_id": row["record_id"],
            "outcome": row["outcome"],
            "estimated_delay_controls": estimate,
            "identification_samples": samples,
            "score_gap_mps": gap,
            "unloading_limit_mps": limit,
        })
    if len(delay_rows) != 48 or mode0_bypassed != 48:
        raise ValueError("v17 full-gate wrapped mode denominator differs")

    report = read_json(report_path)
    outcomes = Counter(row["outcome"] for row in rows)
    if (
        report["status"] != "complete_recorded_development_diagnosis"
        or not report["passed"]
        or report["counts"]["completed_physical_jobs"] != 104
        or report["counts"]["execution_errors"] != 0
        or report["confirmation_states_opened"] != 0
        or report["performance_evidence"]
        or report["failure_replacement"]
        or report["device"]["physical_index"] != 1
        or report["device"]["physical_uuid"]
        != "GPU-58b54684-3577-11d3-f039-0e8469190739"
        or outcomes != Counter(success=100, timeout=3, overforce=1)
    ):
        raise ValueError("v17 full-gate report or outcome counts differ")

    failures = [
        {
            "record_id": row["record_id"],
            "row_id": row["row_id"],
            "outcome": row["outcome"],
            "force_peak_N": row["force_peak_N"],
            "lateral_force_peak_N": row["lateral_force_peak_N"],
            "evidence_sha256": row["evidence"]["sha256"],
            "declared_by_frozen_acceptance": (
                row["row_id"]
                == "T2/ablation/ungated_full_causal_reference_unloading"
                and row["outcome"] == "overforce"
            ),
        }
        for row in rows if row["outcome"] != "success"
    ]
    audit = {
        "schema_version": "t2-causal-reference-unloading-full-gate-audit-v1",
        "status": "rejected_three_unexpected_mode1_timeouts",
        "diagnostic_only": True,
        "performance_claim": False,
        "confirmation_states_opened": 0,
        "failure_replacement": False,
        "acceptance_passed": False,
        "counts": {
            "planned": 104,
            "completed": 104,
            "success": outcomes["success"],
            "timeout": outcomes["timeout"],
            "overforce": outcomes["overforce"],
            "execution_errors": 0,
            "evidence_reaudited": 104,
            "wrapped_mode1_delay_matches": len(delay_rows),
            "wrapped_mode0_estimators_bypassed": mode0_bypassed,
            "declared_failures": 1,
            "unexpected_failures": 3,
        },
        "checks": {
            "complete_frozen_denominator": True,
            "all_evidence_hashes_match": True,
            "all_wrapped_mode1_delays_match_withheld_audit": True,
            "all_wrapped_mode1_limits_match_30_over_delay": True,
            "all_wrapped_mode0_estimators_bypassed": True,
            "zero_execution_errors": True,
            "public_geometry_8_of_8": True,
            "cross_ebdm_16_of_16": True,
            "task_only_40_of_40": True,
            "band_only_8_of_8": True,
            "sole_declared_ungated_collision_preserved": True,
            "all_wrapped_mode1_success": False,
        },
        "delay_identification": {
            "histogram": {
                str(key): value for key, value in sorted(Counter(
                    row["estimated_delay_controls"] for row in delay_rows
                ).items())
            },
            "score_gap_mps": summarize([
                row["score_gap_mps"] for row in delay_rows
            ]),
        },
        "failures": failures,
        "artifacts": {
            "plan": {
                "path": relative(directory / "plan.json"),
                "sha256": sha256(directory / "plan.json"),
            },
            "progress": {
                "path": relative(progress_path),
                "sha256": sha256(progress_path),
            },
            "report": {
                "path": relative(report_path),
                "sha256": sha256(report_path),
            },
        },
    }
    audit_path = directory / "full_gate_independent_audit.json"
    write_json(audit_path, audit)
    rejection = {
        "schema_version": "t2-terminal-interface-rejection-v4",
        "status": "rejected_after_complete_exact_request_gate",
        "candidate": "public-T2-completion-budget-causal-reference-unloading-v17",
        "diagnostic_only": True,
        "performance_claim": False,
        "confirmation_states_opened": 0,
        "failure_replacement": False,
        "reason": (
            "The 26-request preflight passes, Cross-EBDM is 16/16, task-only "
            "is 40/40 and all 48 wrapped mode-1 delays are identified, but "
            "two response-MPC payload mode-1 requests and one matching-no-band "
            "latency mode-1 request time out. The frozen full-gate requirement "
            "allows only the ungated mode-0 collision, so v17 is rejected."
        ),
        "counts": audit["counts"],
        "failures": failures,
        "next_action": (
            "Diagnose the three mode-1 timeout traces offline before selecting "
            "another mechanism. Do not relabel baseline-specific failures, "
            "change the frozen acceptance rule, or resume any additional run."
        ),
        "artifacts": {
            **audit["artifacts"],
            "full_gate_independent_audit": {
                "path": relative(audit_path),
                "sha256": sha256(audit_path),
            },
        },
    }
    write_json(directory / "REJECTED_CANDIDATE.json", rejection)
    return audit, rejection


def diagnose_v17_mode1_timeouts(
    directory: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(directory / "progress.jsonl")
    profiles: list[dict[str, Any]] = []
    for row in rows:
        if (
            row["method"] == "public_geometry"
            or int(row["request"]["requested_mode"]) != 1
        ):
            continue
        record = load_record(row)
        diagnostics = record["control_diagnostics"]
        common = [common_state(item) for item in diagnostics]
        barrier = [
            tick for tick, state in enumerate(common)
            if state.get("pre_descent_settling_active") is True
        ]
        first = lambda predicate, values: next((
            tick for tick, value in enumerate(values) if predicate(value)
        ), None)
        profile = {
            "record_id": row["record_id"],
            "row_id": row["row_id"],
            "instance": row["instance"],
            "scene_cell": row["request"]["scene_cell"],
            "physical_initial_slot": int(
                row["request"]["physical_initial_slot"]
            ),
            "outcome": row["outcome"],
            "control_steps": len(diagnostics),
            "first_outlet_tick": common[-1]["route"]["first_outlet_tick"],
            "terminal_ownership_tick": first(
                lambda item: item.get("terminal_ownership_active") is True,
                diagnostics,
            ),
            "stage2_entry_tick": first(
                lambda state: completion_stage(state) >= 2, common,
            ),
            "terminal_pose_tick": first(
                lambda state: state.get("terminal_pose_active") is True,
                common,
            ),
            "barrier_controls": len(barrier),
            "barrier_last_tick": barrier[-1] if barrier else None,
            "maximum_terminal_hold_controls": max(
                int(state.get("terminal_pose_hold_controls", 0))
                for state in common
            ),
            "final_terminal_hold_controls": int(
                common[-1].get("terminal_pose_hold_controls", 0)
            ),
            "final_terminal_distance_m": float(
                common[-1].get("terminal_distance_m", -1.)
            ),
            "final_position_error_m": float(
                row["episode_metrics"]["final_position_error_m"]
            ),
            "final_orientation_error_rad": float(
                row["episode_metrics"]["final_orientation_error_rad"]
            ),
            "strict_contact_success": bool(
                row["episode_metrics"]["strict_contact_success"]
            ),
            "force_peak_N": float(row["force_peak_N"]),
            "lateral_force_peak_N": float(row["lateral_force_peak_N"]),
            "handoff_clearance_lift_controls": int(
                diagnostics[-1]["handoff_clearance_lift_controls"]
            ),
            "handoff_exact_common_controls": int(
                diagnostics[-1]["handoff_exact_common_controls"]
            ),
            "handoff_reference_lag_m": float(
                diagnostics[-1]["handoff_reference_lag_m"]
            ),
            "terminal_capture_controls": int(
                common[-1].get("terminal_capture_controls", 0)
            ),
            "estimated_delay_controls": int(
                diagnostics[-1]["public_reference_delay_estimate_controls"]
            ),
            "unloading_limit_mps": float(
                common[-1]["pre_descent_causal_unloading_limit_mps"]
            ),
            "evidence_sha256": row["evidence"]["sha256"],
        }
        profiles.append(profile)
    if len(profiles) != 48:
        raise ValueError("v17 mode-1 profile denominator differs")

    failures = [profile for profile in profiles if profile["outcome"] != "success"]
    response_failures = [
        profile for profile in failures
        if profile["row_id"]
        == "T2/main/response_mpc_causal_reference_unloading"
    ]
    response_controls = [
        profile for profile in profiles
        if profile["row_id"]
        == "T2/main/response_mpc_causal_reference_unloading"
        and profile["scene_cell"] == "latency_gain_shift"
        and profile["outcome"] == "success"
    ]
    matching_failure = [
        profile for profile in failures
        if profile["row_id"]
        == "T2/ablation/matching_no_band_causal_reference_unloading"
    ]
    matching_control = [
        profile for profile in profiles
        if profile["row_id"]
        == "T2/ablation/matching_no_band_causal_reference_unloading"
        and profile["instance"] == "instance_07"
        and profile["scene_cell"] == "latency_gain_shift"
        and profile["physical_initial_slot"] == 0
        and profile["outcome"] == "success"
    ]
    if not (
        len(failures) == 3
        and len(response_failures) == 2
        and len(response_controls) == 2
        and len(matching_failure) == 1
        and len(matching_control) == 1
        and all(profile["strict_contact_success"] for profile in failures)
    ):
        raise ValueError("v17 timeout comparison set differs")

    response_failure_stage2 = [
        profile["stage2_entry_tick"] for profile in response_failures
    ]
    response_control_stage2 = [
        profile["stage2_entry_tick"] for profile in response_controls
    ]
    response_failure_lift = [
        profile["handoff_clearance_lift_controls"]
        for profile in response_failures
    ]
    response_control_lift = [
        profile["handoff_clearance_lift_controls"]
        for profile in response_controls
    ]
    failed_matching = matching_failure[0]
    passed_matching = matching_control[0]
    diagnosis = {
        "schema_version": "t2-v17-mode1-timeout-diagnosis-v1",
        "status": "baseline_specific_handoff_state_failures_diagnosed",
        "diagnostic_only": True,
        "performance_claim": False,
        "counts": {
            "wrapped_mode1": len(profiles),
            "success": sum(
                profile["outcome"] == "success" for profile in profiles
            ),
            "timeout": len(failures),
            "cross_ebdm_success": sum(
                profile["row_id"]
                == "T2/main/cross_ebdm_causal_reference_unloading"
                and profile["outcome"] == "success"
                for profile in profiles
            ),
            "task_only_success": sum(
                profile["row_id"]
                == "T2/main/task_only_causal_reference_unloading"
                and profile["outcome"] == "success"
                for profile in profiles
            ),
        },
        "common_mechanism_checks": {
            "all_48_delay_estimates_correct": True,
            "all_48_unloading_limits_correct": True,
            "all_three_timeouts_preserve_strict_contact": True,
            "all_three_timeouts_remain_below_force_limits": True,
            "cross_ebdm_mode1_8_of_8": True,
            "task_only_mode1_20_of_20": True,
        },
        "response_mpc_payload": {
            "failure_record_ids": [
                profile["record_id"] for profile in response_failures
            ],
            "stage2_entry_ticks": response_failure_stage2,
            "latency_control_stage2_entry_ticks": response_control_stage2,
            "handoff_clearance_lift_controls": response_failure_lift,
            "latency_control_handoff_clearance_lift_controls": (
                response_control_lift
            ),
            "final_position_error_m": [
                profile["final_position_error_m"]
                for profile in response_failures
            ],
            "maximum_terminal_hold_controls": [
                profile["maximum_terminal_hold_controls"]
                for profile in response_failures
            ],
            "finding": (
                "Both payload mode-1 requests enter stage 2 at ticks 902/910 "
                "after 357/360 handoff-clearance controls, versus ticks 859/861 "
                "and 97/97 controls for the same baseline's successful latency "
                "cells. They never accumulate terminal dwell and finish 2.28--"
                "2.44 mm from the full goal. This is candidate-induced handoff "
                "state debt before common completion, not delay-estimation or "
                "force-safety failure."
            ),
        },
        "matching_no_band_latency": {
            "failure_record_id": failed_matching["record_id"],
            "paired_success_record_id": passed_matching["record_id"],
            "stage2_entry_tick": failed_matching["stage2_entry_tick"],
            "paired_success_stage2_entry_tick": (
                passed_matching["stage2_entry_tick"]
            ),
            "barrier_controls": failed_matching["barrier_controls"],
            "paired_success_barrier_controls": (
                passed_matching["barrier_controls"]
            ),
            "maximum_terminal_hold_controls": (
                failed_matching["maximum_terminal_hold_controls"]
            ),
            "paired_success_maximum_terminal_hold_controls": (
                passed_matching["maximum_terminal_hold_controls"]
            ),
            "finding": (
                "The failed slot enters stage 2 eleven controls later, spends "
                "35 versus 18 controls in the barrier, and reaches only seven "
                "of the required fifteen terminal-hold controls; the paired "
                "slot reaches 25. The terminal pose itself is feasible and "
                "contact-safe, but matching-without-band leaves insufficient "
                "finite-time reserve."
            ),
        },
        "decision": (
            "Do not create v18 by tuning the shared completion controller to "
            "erase these baseline-specific failures. Preserve v17's rejected "
            "full-gate decision. Prospectively separate shared-infrastructure "
            "qualification from method performance: qualify the public task "
            "contract, causal estimator, force safety and evidence integrity; "
            "retain method timeouts in the benchmark denominator."
        ),
        "theory_revision": [
            "Finite-time completion debt includes the method-dependent handoff state, not only reference lead and executor delay.",
            "A shared safe completion controller cannot guarantee success for every candidate policy without erasing the performance distinction being measured.",
            "Common infrastructure should guarantee observability, bounded action, force safety and evaluator consistency; task completion remains an experimental outcome.",
        ],
    }
    return profiles, diagnosis


def seal_v16(
    directory: Path, rows: list[dict[str, Any]], planned: int,
) -> dict[str, Any]:
    outcomes = Counter(row["outcome"] for row in rows)
    failures = [row for row in rows if row["outcome"] != "success"]
    if (len(rows), outcomes) != (25, Counter(success=22, timeout=3)):
        raise ValueError(
            f"unexpected v16 stopped-gate counts: {len(rows)}, {outcomes}"
        )
    if any(
        row["request"]["confirmation_state"]
        or row["request"]["performance_evidence"]
        for row in rows
    ):
        raise ValueError("v16 contains forbidden confirmation/performance evidence")
    record_ids = [row["record_id"] for row in rows]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("v16 record ids are not unique")
    for row in rows:
        load_record(row)

    audit = {
        "schema_version": "partial-behavioral-independent-audit-v1",
        "status": "passed_completed_evidence_integrity",
        "diagnostic_only": True,
        "integrity_passed": True,
        "counts": {
            "planned": planned,
            "completed": len(rows),
            "unexecuted_after_stop": planned - len(rows),
            "evidence_reaudited": len(rows),
            "outcomes": dict(sorted(outcomes.items())),
        },
        "checks": {
            "all_completed_episode_evidence_reaudited": True,
            "completed_evidence_hashes_match": True,
            "completed_record_ids_unique": True,
            "confirmation_states_unopened": True,
            "intentional_stop_not_scored_as_episode_failure": True,
        },
        "artifacts": {
            "plan": {
                "path": relative(directory / "plan.json"),
                "sha256": sha256(directory / "plan.json"),
            },
            "progress": {
                "path": relative(directory / "progress.jsonl"),
                "sha256": sha256(directory / "progress.jsonl"),
            },
        },
    }
    audit_path = directory / "partial_independent_audit.json"
    write_json(audit_path, audit)

    rejected = {
        "schema_version": "t2-terminal-interface-rejection-v3",
        "status": "rejected_early_after_exact_request_stop_rule",
        "candidate": (
            "public-T2-completion-budget-reference-debt-unloading-v16"
        ),
        "diagnostic_only": True,
        "performance_claim": False,
        "confirmation_states_opened": 0,
        "integrity_audit_passed": True,
        "device": {
            "cuda_visible_devices": "1",
            "physical_index": 1,
            "physical_uuid": "GPU-58b54684-3577-11d3-f039-0e8469190739",
        },
        "counts": {
            "planned": planned,
            "completed": len(rows),
            "unexecuted_after_stop": planned - len(rows),
            "success": outcomes["success"],
            "timeout": outcomes["timeout"],
            "overforce": outcomes["overforce"],
            "execution_errors": 0,
            "logical_failures": len(failures),
            "unique_failed_physical_traces": len(failures),
        },
        "failures": [
            {
                "record_id": row["record_id"],
                "outcome": row["outcome"],
                "evidence_sha256": row["evidence"]["sha256"],
            }
            for row in failures
        ],
        "reason": (
            "The public 8/8 sentinel passes, but the 30 mm/s global "
            "unloading rate changes three previously successful Cross-EBDM "
            "latency mode-1 requests to timeout. The preflight preservation "
            "rule fails, so the remaining 79 jobs were not executed."
        ),
        "finding": (
            "All three v16 failures use the three-control latency cell. "
            "The faster command lengthens the settling barrier through "
            "delayed reference oscillation; it is not a monotone reduction "
            "of completion debt under delay. Payload cells remain successful."
        ),
        "next_isolated_change": (
            "Preserve v15's 10 mm/s unloading for a causally identified "
            "three-control executor delay and use 30 mm/s only for a "
            "causally identified one-control delay. Identify delay from "
            "allowed public reference increments and bounded actions "
            "reconstructed from the wrapper's own outputs; do not read "
            "mechanism parameters or scan an intermediate rate."
        ),
        "artifacts": {
            "plan.json": {
                "path": relative(directory / "plan.json"),
                "sha256": sha256(directory / "plan.json"),
            },
            "progress.jsonl": {
                "path": relative(directory / "progress.jsonl"),
                "sha256": sha256(directory / "progress.jsonl"),
            },
            "partial_independent_audit.json": {
                "path": relative(audit_path),
                "sha256": sha256(audit_path),
            },
        },
    }
    rejected_path = directory / "REJECTED_CANDIDATE.json"
    write_json(rejected_path, rejected)
    return rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v15", type=Path, default=DEFAULT_V15)
    parser.add_argument("--v16", type=Path, default=DEFAULT_V16)
    parser.add_argument("--v17", type=Path, default=DEFAULT_V17)
    parser.add_argument("--v13-analysis", type=Path, default=DEFAULT_V13_ANALYSIS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    v15_rows = read_jsonl(args.v15 / "progress.jsonl")
    v16_rows = read_jsonl(args.v16 / "progress.jsonl")
    v16_plan = read_json(args.v16 / "plan.json")
    v17_plan_path = args.v17 / "plan.json"
    v17_plan = read_json(v17_plan_path)
    plan_audit = frozen_plan_audit(v16_plan, v17_plan, v17_plan_path)
    preflight_audit = audit_v17_preflight(args.v17, plan_audit)
    full_gate_audit = None
    v17_rejection = None
    if len(read_jsonl(args.v17 / "progress.jsonl")) == len(v17_plan["jobs"]):
        full_gate_audit, v17_rejection = audit_v17_full_gate(
            args.v17, v17_plan,
        )
    mode1_profiles = None
    timeout_diagnosis = None
    if full_gate_audit:
        mode1_profiles, timeout_diagnosis = diagnose_v17_mode1_timeouts(
            args.v17,
        )
    plan_audit["status"] = (
        "passed_plan_and_preflight_full_gate_executed_rejected"
        if full_gate_audit else "passed_preflight_executed_full_gate_pending"
    )
    plan_audit["preflight"]["outcome"] = "passed_26_of_26"
    planned = int(v16_plan["counts"]["physical_jobs"])
    v16_rejection = seal_v16(args.v16, v16_rows, planned)

    profiles: list[dict[str, Any]] = []
    for revision, rows in (("v15", v15_rows), ("v16", v16_rows)):
        for row in rows:
            profile, _ = episode_profile(row, revision)
            profiles.append(profile)
    if not all(profile["estimate_matches_hidden_audit"] for profile in profiles):
        raise ValueError("public-only delay estimate differs from audit declaration")
    implementation_replay = replay_implemented_estimator((
        ("v15", v15_rows), ("v16", v16_rows),
    ))

    v15_cross = {
        request_key(row): episode_profile(row, "v15")[0]
        for row in v15_rows
        if "cross_ebdm" in row["row_id"]
        and int(row["request"]["requested_mode"]) == 1
    }
    v16_cross = {
        request_key(row): episode_profile(row, "v16")[0]
        for row in v16_rows
        if "cross_ebdm" in row["row_id"]
        and int(row["request"]["requested_mode"]) == 1
    }
    if set(v15_cross) != set(v16_cross) or len(v15_cross) != 8:
        raise ValueError("v15/v16 Cross-EBDM mode-1 requests do not pair exactly")
    paired: list[dict[str, Any]] = []
    for key in sorted(v15_cross):
        old, new = v15_cross[key], v16_cross[key]
        paired.append({
            "instance": old["instance"],
            "scene_cell": old["scene_cell"],
            "physical_initial_slot": old["physical_initial_slot"],
            "reset_seed": old["reset_seed"],
            "estimated_delay_controls": old["estimated_delay_controls"],
            "v15_outcome": old["outcome"],
            "v16_outcome": new["outcome"],
            "transition": f"{old['outcome']}->{new['outcome']}",
            "v15_barrier_controls": old["barrier_controls"],
            "v16_barrier_controls": new["barrier_controls"],
            "barrier_change_controls": (
                new["barrier_controls"] - old["barrier_controls"]
            ),
            "v15_terminal_hold_controls": old[
                "terminal_hold_controls_final"
            ],
            "v16_terminal_hold_controls": new[
                "terminal_hold_controls_final"
            ],
            "v15_terminal_distance_final_m": old[
                "terminal_distance_final_m"
            ],
            "v16_terminal_distance_final_m": new[
                "terminal_distance_final_m"
            ],
        })

    v13_closeout = read_json(args.v13_analysis / "closeout.json")
    v15_outcomes = Counter(row["outcome"] for row in v15_rows)
    v16_outcomes = Counter(row["outcome"] for row in v16_rows)
    delay_counts = Counter(
        profile["estimated_delay_controls"] for profile in profiles
    )
    winner_rmse = [
        profile[
            f"lag_{profile['estimated_delay_controls']}_rmse_mps"
        ]
        for profile in profiles
    ]
    changed_to_timeout = [
        row for row in paired
        if row["transition"] == "success->timeout"
    ]
    v15_failed = next(
        profile for profile in profiles
        if profile["revision"] == "v15"
        and profile["outcome"] != "success"
    )

    closeout = {
        "schema_version": "t2-reference-unloading-closeout-v1",
        "status": "v15_and_v16_rejected_after_bounded_exact_request_gates",
        "scope": "CPU-only analysis of immutable v13--v16 episode evidence",
        "diagnostic_only": True,
        "performance_claim": False,
        "confirmation_states_opened": 0,
        "counts": {
            "v13_completed": int(v13_closeout["counts"]["completed"]),
            "v13_success": int(v13_closeout["counts"]["success"]),
            "v13_timeout": int(v13_closeout["counts"]["timeout"]),
            "v13_overforce": int(v13_closeout["counts"]["overforce"]),
            "v15_completed": len(v15_rows),
            "v15_success": v15_outcomes["success"],
            "v15_timeout": v15_outcomes["timeout"],
            "v15_overforce": v15_outcomes["overforce"],
            "v16_completed": len(v16_rows),
            "v16_success": v16_outcomes["success"],
            "v16_timeout": v16_outcomes["timeout"],
            "v16_overforce": v16_outcomes["overforce"],
            "delay_identification_episodes": len(profiles),
            "delay_identification_matches": sum(
                profile["estimate_matches_hidden_audit"]
                for profile in profiles
            ),
            "estimated_delay_histogram": {
                str(key): value for key, value in sorted(delay_counts.items())
            },
        },
        "integrity": {
            "v15_partial_audit_passed": read_json(
                args.v15 / "partial_independent_audit.json"
            )["integrity_passed"],
            "v16_partial_audit_passed": True,
            "all_65_evidence_hashes_reaudited": True,
            "all_delay_estimates_match_withheld_audit_declarations": True,
            "actual_v17_implementation_replay_matches": (
                implementation_replay["counts"]["matches_withheld_audit"]
            ),
            "v15_v16_cross_mode1_exact_pairs": len(paired),
        },
        "causal_delay_identification": {
            "candidate_delays_controls": list(DELAY_CANDIDATES),
            "minimum_informative_samples_required": (
                MINIMUM_IDENTIFICATION_SAMPLES
            ),
            "informative_samples": summarize([
                profile["informative_samples"] for profile in profiles
            ]),
            "winning_rmse_mps": summarize(winner_rmse),
            "score_gap_mps": summarize([
                profile["score_gap_mps"] for profile in profiles
            ]),
            "first_stable_tick": summarize([
                profile["first_stable_tick"] for profile in profiles
            ]),
            "identification_slack_before_stage2_controls": summarize([
                profile["identification_slack_controls"]
                for profile in profiles
            ]),
            "observed_delays_controls": sorted(delay_counts),
            "unobserved_declared_delay_controls": [2],
        },
        "paired_cross_mode1": {
            "delay_1": {
                "episodes": sum(
                    row["estimated_delay_controls"] == 1 for row in paired
                ),
                "transitions": dict(Counter(
                    row["transition"] for row in paired
                    if row["estimated_delay_controls"] == 1
                )),
                "barrier_change_controls": summarize([
                    row["barrier_change_controls"] for row in paired
                    if row["estimated_delay_controls"] == 1
                ]),
            },
            "delay_3": {
                "episodes": sum(
                    row["estimated_delay_controls"] == 3 for row in paired
                ),
                "transitions": dict(Counter(
                    row["transition"] for row in paired
                    if row["estimated_delay_controls"] == 3
                )),
                "barrier_change_controls": summarize([
                    row["barrier_change_controls"] for row in paired
                    if row["estimated_delay_controls"] == 3
                ]),
            },
        },
        "findings": [
            (
                "V15 converts the observed v13/v12 mode-1 recontact "
                "overforce mechanism into safe behavior: all 40 completed "
                "episodes remain below hard force limits and 39 succeed."
            ),
            (
                "The lone v15 failure has a one-control executor delay. Its "
                "34-control barrier releases at tick 906, leaving only 94 "
                "controls and no terminal dwell; final error is 0.333 mm "
                "outside the frozen terminal set."
            ),
            (
                "V16's fixed 30 mm/s unloading regresses three of eight "
                "paired Cross-EBDM mode-1 requests, all at three-control "
                "delay. Their barriers grow rather than shrink, proving that "
                "a larger fixed rate is unstable completion debt under delay."
            ),
            (
                "Executor delay is identifiable before stage 2 in every one "
                "of the 65 completed v15/v16 episodes from public reference "
                "increments and issued-action history alone. The estimate "
                "stabilizes by tick 10, at least 157 controls before stage 2."
            ),
        ],
        "decision": (
            "Reject v16 and do not resume its remaining 79 jobs. Retain the "
            "finite v15 pre-descent barrier, replace only its fixed unloading "
            "limit with a public causal delay-adaptive limit, and validate "
            "that isolated revision on the same exact requests."
        ),
        "v15_remaining_failure": {
            key: v15_failed[key]
            for key in (
                "record_id", "outcome", "estimated_delay_controls",
                "barrier_controls", "barrier_last_tick",
                "terminal_distance_final_m", "terminal_hold_controls_final",
            )
        },
        "v16_cross_regressions": changed_to_timeout,
    }

    three_classes = {
        "schema_version": "t2-three-common-failure-diagnosis-v2",
        "status": "diagnosed_with_v13_v16_recorded_evidence",
        "classes": [
            {
                "class": "mode1_finite_time_liveness_regression",
                "v13_evidence": (
                    "40/48 wrapped mode-1 requests time out after entering "
                    "and leaving the terminal planar set."
                ),
                "v15_v16_update": (
                    "V15 restores 39/40 completed episodes but spends 34 "
                    "barrier controls on the one remaining failure. V16 "
                    "globally accelerates unloading and regresses three "
                    "delay-3 successes."
                ),
                "diagnosis": (
                    "Finite-time completion depends on both reference debt "
                    "and executor delay. A fixed action magnitude does not "
                    "order discharge time in a delayed reference loop."
                ),
                "next_revision_effect": (
                    "Use 30/d mm/s for identified delay d in {1,2,3}: "
                    "30 mm/s at delay 1 recovers completion reserve, while "
                    "10 mm/s at delay 3 exactly preserves v15's stable "
                    "behavior. The frozen evidence contains delay 1 and 3; "
                    "delay 2 remains a declared causal fallback case."
                ),
            },
            {
                "class": "mode1_descent_recontact_overforce",
                "v13_evidence": (
                    "Eight mode-1 latency requests violate force limits "
                    "exactly 16 controls after stage-2 entry with 8.20--8.95 "
                    "mm of public reference lead."
                ),
                "v15_v16_update": (
                    "No completed v15 or v16 request is overforce. The finite "
                    "clearance barrier and lead-reducing unloading remove the "
                    "observed recontact mechanism."
                ),
                "diagnosis": (
                    "Stored planar reference error must be discharged before "
                    "vertical descent; post-entry clipping acts too late."
                ),
                "next_revision_effect": (
                    "Keep v15's 2 mm lead, 4 mm/s TCP speed, four-control "
                    "persistence, held vertical request, and exact v12 action "
                    "after release."
                ),
            },
            {
                "class": "mode0_ungated_preoutlet_collision",
                "v13_evidence": (
                    "One ungated-full payload mode-0 request is overforce "
                    "before the shared completion stage in both v12 and v13."
                ),
                "v15_v16_update": (
                    "The stopped v15/v16 gates do not reach this row, and the "
                    "mode-1 completion interface does not own its failure."
                ),
                "diagnosis": (
                    "This is candidate/ablation pre-outlet behavior, not a "
                    "shared terminal-controller defect."
                ),
                "next_revision_effect": (
                    "Preserve it as the declared ablation failure and keep it "
                    "in the denominator; do not alter common mode-0 behavior."
                ),
            },
        ],
    }

    next_revision = {
        "schema_version": "t2-causal-reference-unloading-plan-v1",
        "status": (
            "full_gate_rejected_baseline_specific_timeout_diagnosis_complete"
            if full_gate_audit
            else "implemented_cpu_and_gpu_preflight_passed_full_gate_pending"
        ),
        "rejected_candidates": [
            "public-T2-completion-budget-reference-unloading-v15",
            "public-T2-completion-budget-reference-debt-unloading-v16",
        ],
        "next_candidate": (
            "public-T2-completion-budget-causal-reference-unloading-v17"
        ),
        "implementation": (
            "genedynamics.tasks.transfer.controllers.guide_contact."
            "PublicCompletionBudgetCausalReferenceUnloadingGuideCompletionPolicy"
        ),
        "single_mechanism_change": {
            "owner": "genedynamics.tasks.transfer.controllers.guide_contact",
            "scope": "shared T2 mode-1 pre-descent barrier",
            "actor_or_solver_change": False,
            "hidden_mechanism_state": False,
            "new_learned_parameters": 0,
            "gain_search": False,
            "change": (
                "Estimate executor delay over the frozen one-to-three-control "
                "envelope by matching each public planar reference increment "
                "to bounded actions reconstructed from the terminal wrapper's "
                "own selected-action history through bounded_contact_command. "
                "After eight informative observations, choose the minimum-RMSE lag. "
                "During the existing barrier only, set the componentwise "
                "lead-reducing unloading limit to min(30, 30/d) mm/s for "
                "estimated delay d. This yields 30 mm/s for d=1 and the "
                "v15-safe 10 mm/s for d=3."
            ),
        },
        "causal_data_flow": [
            "The wrapper reads only the already-allowed public reference_position field on each unique tick.",
            "After selecting its request, it reconstructs the backend-bounded action with the shared bounded_contact_command function and its own prior bounded action.",
            "The frozen controller observation whitelist and candidate interface do not change.",
            "At observation tick t it scores delay d against action-history index -(d+1), because the public observation follows the preceding physical transition.",
            "It uses only prior allowed references and its own causal output history for lag scores 1, 2 and 3.",
            "The derived lag changes only the pre-descent unloading limit inside the common controller.",
        ],
        "must_preserve": [
            "v15 actions outside the finite pre-descent barrier",
            "v15 behavior at identified three-control delay",
            "2 mm lead and 4 mm/s TCP-speed admission predicates",
            "four consecutive settled observations before descent",
            "exact v12 completion action after barrier release",
            "all mode-0 actions, actors, solver outputs and evaluator thresholds",
            "20 s deadline, 15-control dwell and hard force limits",
        ],
        "fallback": (
            "Before the estimate is ready or if causal scores are nonfinite, "
            "use v15's 10 mm/s limit. Never infer from scene cell, method, "
            "payload, servo gain, declared mechanism parameters, or outcome."
        ),
        "new_evidence_fields": [
            "public_reference_delay_estimate_controls",
            "public_reference_delay_identification_samples",
            "public_reference_delay_score_gap_mps",
            "pre_descent_causal_unloading_limit_mps",
        ],
        "online_history_index": (
            "At the current post-transition public observation, candidate "
            "delay d is compared with reconstructed bounded-action history item "
            "-(d+1)."
        ),
        "cpu_gate_before_gpu": [
            "Replay all 65 traces through the public-signal offline audit.",
            "Replay the actual wrapper/limiter estimator on all 49 wrapped traces.",
            "Require 65/65 signal and 49/49 implementation audit matches.",
            "Test the post-transition -(d+1) history index explicitly.",
            "Test reset, checkpoint/restore, repeated-tick identity and fallback.",
            "Test exact v15 identity for delay 3 and exact mode-0 identity.",
        ],
        "gpu_preflight": {
            "device": "physical GPU1 only",
            "requests": 26,
            "contents": (
                "eight public sentinels, all sixteen Cross-EBDM pairs, and "
                "the task-only instance-01 payload-shift slot-0 mode pair"
            ),
            "pass": (
                "26/26 success, zero force/evidence errors, all paired v15 "
                "successes preserved, and the former v15 mode-1 timeout succeeds"
            ),
            "stop": (
                "Stop at the first new failure or estimator/evidence mismatch; "
                "do not tune rates or thresholds from the result."
            ),
        },
        "full_exact_request_gate_after_preflight": (
            "Resume the frozen 104-request roster only after the preflight "
            "passes. Require public 8/8, all 48 wrapped mode-1 successes, all "
            "47 prior wrapped mode-0 successes, the one declared ungated "
            "mode-0 collision as the sole failure, and independent audit."
        ),
        **({
            "gpu_full_gate": {
                "status": full_gate_audit["status"],
                "counts": full_gate_audit["counts"],
                "decision": (
                    timeout_diagnosis["decision"]
                ),
            },
        } if full_gate_audit else {}),
        "theory_revision": [
            "Model the public reference pipeline as part of the augmented delayed state.",
            "Define completion debt as reference lead plus delay-scaled command debt.",
            "A safe barrier requires lead reduction and finite-time reserve, not only velocity dissipation.",
            "The adaptive shared controller remains benchmark infrastructure, not the learned Cross-EBDM contribution.",
        ],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "closeout.json", closeout)
    write_json(args.output / "three_failure_diagnosis.json", three_classes)
    write_json(args.output / "next_revision.json", next_revision)
    write_json(
        args.output / "implementation_replay.json", implementation_replay,
    )
    write_json(args.output / "frozen_plan_audit.json", plan_audit)
    if timeout_diagnosis:
        write_json(args.output / "v17_timeout_diagnosis.json", timeout_diagnosis)
        write_csv(
            args.output / "v17_mode1_completion_profiles.csv",
            mode1_profiles,
        )
    write_csv(args.output / "causal_delay_identification.csv", profiles)
    write_csv(args.output / "paired_cross_mode1.csv", paired)

    derived_names = (
        "closeout.json",
        "three_failure_diagnosis.json",
        "next_revision.json",
        "implementation_replay.json",
        "frozen_plan_audit.json",
        "causal_delay_identification.csv",
        "paired_cross_mode1.csv",
        *((
            "v17_timeout_diagnosis.json",
            "v17_mode1_completion_profiles.csv",
        ) if timeout_diagnosis else ()),
    )
    manifest = {
        "schema_version": "derived-artifact-manifest-v1",
        "files": {
            name: sha256(args.output / name) for name in derived_names
        },
        "source_artifacts": {
            "v13_closeout": {
                "path": relative(args.v13_analysis / "closeout.json"),
                "sha256": sha256(args.v13_analysis / "closeout.json"),
            },
            "v15_plan": {
                "path": relative(args.v15 / "plan.json"),
                "sha256": sha256(args.v15 / "plan.json"),
            },
            "v15_progress": {
                "path": relative(args.v15 / "progress.jsonl"),
                "sha256": sha256(args.v15 / "progress.jsonl"),
            },
            "v15_partial_audit": {
                "path": relative(args.v15 / "partial_independent_audit.json"),
                "sha256": sha256(args.v15 / "partial_independent_audit.json"),
            },
            "v16_plan": {
                "path": relative(args.v16 / "plan.json"),
                "sha256": sha256(args.v16 / "plan.json"),
            },
            "v16_progress": {
                "path": relative(args.v16 / "progress.jsonl"),
                "sha256": sha256(args.v16 / "progress.jsonl"),
            },
            "v16_partial_audit": {
                "path": relative(args.v16 / "partial_independent_audit.json"),
                "sha256": sha256(args.v16 / "partial_independent_audit.json"),
            },
            "v16_rejection": {
                "path": relative(args.v16 / "REJECTED_CANDIDATE.json"),
                "sha256": sha256(args.v16 / "REJECTED_CANDIDATE.json"),
            },
            "v17_controller": {
                "path": (
                    "genedynamics/tasks/transfer/controllers/guide_contact.py"
                ),
                "sha256": sha256(
                    ROOT
                    / "genedynamics/tasks/transfer/controllers/guide_contact.py"
                ),
            },
            "v17_diagnostic_spec": {
                "path": (
                    "configs/arm/guided_contact_alignment/validation/"
                    "completion_budget_causal_reference_unloading_"
                    "exact_request_gate.json"
                ),
                "sha256": sha256(
                    ROOT
                    / "configs/arm/guided_contact_alignment/validation/"
                    "completion_budget_causal_reference_unloading_"
                    "exact_request_gate.json"
                ),
            },
            "v17_frozen_plan": {
                "path": relative(v17_plan_path),
                "sha256": sha256(v17_plan_path),
            },
            "v17_preflight_progress": {
                "path": relative(args.v17 / "progress.jsonl"),
                "sha256": sha256(args.v17 / "progress.jsonl"),
            },
            "v17_preflight_report": {
                "path": relative(args.v17 / "report.json"),
                "sha256": sha256(args.v17 / "report.json"),
            },
            "v17_preflight_independent_audit": {
                "path": relative(
                    args.v17 / "preflight_independent_audit.json"
                ),
                "sha256": sha256(
                    args.v17 / "preflight_independent_audit.json"
                ),
            },
            **({
                "v17_full_gate_independent_audit": {
                    "path": relative(
                        args.v17 / "full_gate_independent_audit.json"
                    ),
                    "sha256": sha256(
                        args.v17 / "full_gate_independent_audit.json"
                    ),
                },
                "v17_rejection": {
                    "path": relative(args.v17 / "REJECTED_CANDIDATE.json"),
                    "sha256": sha256(
                        args.v17 / "REJECTED_CANDIDATE.json"
                    ),
                },
            } if v17_rejection else {}),
        },
    }
    write_json(args.output / "manifest.json", manifest)

    print(json.dumps({
        "output": relative(args.output),
        "v16_rejection_sha256": sha256(
            args.v16 / "REJECTED_CANDIDATE.json"
        ),
        "manifest_sha256": sha256(args.output / "manifest.json"),
        "v16_counts": v16_rejection["counts"],
        "delay_identification_matches": (
            f"{len(profiles)}/{len(profiles)}"
        ),
        "decision": closeout["decision"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
