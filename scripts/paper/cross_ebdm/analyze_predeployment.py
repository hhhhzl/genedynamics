#!/usr/bin/env python3
"""Close a sealed predeployment run and freeze method-side diagnosis work.

This entrypoint is CPU-only.  It reads the qualification journal and its
independent audit, computes descriptive admission statistics, and writes a
diagnosis plan.  It never opens confirmation states or runs task physics.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def source_contract_checks() -> dict[str, Any]:
    paths = {
        "arm_actor": ROOT / "genedynamics/tasks/transfer/controllers/cross_ebdm/arm_dense.py",
        "dial_actor": ROOT / "genedynamics/tasks/transfer/controllers/dial.py",
        "ordinary_mpc": ROOT / "genedynamics/solvers/single/response_mpc/arm.py",
    }
    text = {name: path.read_text() for name, path in paths.items()}
    arm_source_prior = text["arm_actor"].split(
        "class PublicArmTaskSourcePriorActor", 1
    )[1]
    dial_arm = text["dial_actor"].split("class PublicArmTaskDIALMPCActor", 1)[1].split(
        "class PublicPlannedObstacleDIALMPCActor", 1
    )[0]
    checks = {
        "learned_dense_preserves_T2_task_owned_coordinates": (
            "T2_TASK_OWNED_COORDINATES" in text["arm_actor"].split(
                "class PublicArmTaskResponseMPCActor", 1
            )[0]
        ),
        "ordinary_response_mpc_projects_T2_task_owned_coordinates": (
            "project_task_owned_requests" in text["ordinary_mpc"]
            and "T2_TASK_OWNED_COORDINATES" in text["ordinary_mpc"]
        ),
        "dial_projects_T2_task_owned_coordinates": (
            "project_task_owned_requests" in dial_arm
            or "T2_TASK_OWNED_COORDINATES" in dial_arm
        ),
        "public_source_prior_projects_T2_task_owned_coordinates": (
            "project_task_owned_requests" in arm_source_prior
            or "T2_TASK_OWNED_COORDINATES" in arm_source_prior
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "artifacts": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "interpretation": (
            "A false DIAL or public-source-prior check is an adapter/interface defect "
            "under the frozen T2 task-owned request contract, not evidence that the "
            "underlying baseline mechanism fails."
        ),
    }


PAIRED_T2_ROWS = {
    ("main", "cross_ebdm"): "cross_ebdm_clearance_progress",
    ("main", "task_only"): "task_only_clearance_progress",
    ("main", "response_mpc"): "response_mpc_clearance_progress",
    ("ablation", "frozen_warm"): "frozen_warm_clearance_progress",
    ("ablation", "task_only"): "ablation_task_only_clearance_progress",
    ("ablation", "matching_no_band"): "matching_no_band_clearance_progress",
    ("ablation", "band_only"): "band_only_clearance_progress",
    ("ablation", "ungated_full"): "ungated_full_clearance_progress",
    ("ablation", "gated_full"): "gated_full_clearance_progress",
}


def _episode_record(row: dict[str, Any]) -> dict[str, Any]:
    pointer = row["evidence"]
    path = ROOT / pointer["path"]
    if not path.is_file() or sha256(path) != pointer["sha256"]:
        raise ValueError(f"missing or changed episode evidence: {row['record_id']}")
    payload = json.loads(gzip.decompress(path.read_bytes()))
    records = payload.get("records", [])
    if len(records) != 1:
        raise ValueError(f"episode evidence cardinality differs: {row['record_id']}")
    return records[0]


def _paired_key(row: dict[str, Any], *, candidate: bool) -> tuple[Any, ...]:
    family = row["family"]
    key = row["key"]
    if candidate:
        reverse = {
            (family_name, candidate_key): baseline_key
            for (family_name, baseline_key), candidate_key in PAIRED_T2_ROWS.items()
        }
        try:
            key = reverse[(family, key)]
        except KeyError as error:
            raise ValueError(f"unexpected candidate row: {row['row_id']}") from error
    request = row["request"]
    return (
        family, key, row["instance"], request["scene_cell"],
        int(request["physical_initial_slot"]), int(request["requested_mode"]),
    )


def _phase_evidence(row: dict[str, Any], *, candidate: bool) -> dict[str, Any]:
    record = _episode_record(row)
    diagnostics = record.get("control_diagnostics", [])
    traces = record.get("traces", [])
    if not diagnostics or len(diagnostics) != len(traces):
        raise ValueError(f"incomplete control evidence: {row['record_id']}")

    def base(value: dict[str, Any]) -> dict[str, Any]:
        return value.get("common", {}) if candidate else value.get("base", {})

    bases = [base(value) for value in diagnostics]

    def first_index(predicate) -> int | None:
        return next((index for index, value in enumerate(bases) if predicate(value)), None)

    completion_ticks = {
        str(stage): first_index(
            lambda value, stage=stage: int(
                value.get("completion", {}).get("stage", -1)
            ) >= stage
        )
        for stage in (0, 1, 2)
    }
    first_hold = next((
        index for index, value in enumerate(traces)
        if int(value.get("execution_hold_count", 0)) >= 1
    ), None)
    hold_15 = next((
        index for index, value in enumerate(traces)
        if int(value.get("execution_hold_count", 0)) >= 15
    ), None)
    final_base = bases[-1]
    result = {
        "control_steps": len(traces),
        "first_contact_step": row["episode_metrics"]["first_contact_step"],
        "first_outlet_step": row["episode_metrics"]["first_outlet_step"],
        "completion_stage_ticks": completion_ticks,
        "terminal_capture_tick": first_index(
            lambda value: value.get("terminal_pose_active") is True
        ),
        "first_terminal_hold_tick": first_hold,
        "hold_15_tick": hold_15,
        "final_position_error_m": row["episode_metrics"]["final_position_error_m"],
        "final_orientation_error_rad": row["episode_metrics"][
            "final_orientation_error_rad"
        ],
        "force_peak_N": row["force_peak_N"],
        "lateral_force_peak_N": row["lateral_force_peak_N"],
        "final_terminal_distance_m": final_base.get("terminal_distance_m"),
        "final_terminal_hold_controls": final_base.get(
            "terminal_pose_hold_controls"
        ),
    }
    if candidate:
        final = diagnostics[-1]
        result["candidate_authority"] = {
            "final_fraction": final.get("candidate_authority_fraction"),
            "ownership_controls": final.get("terminal_ownership_controls"),
            "ownership_activations": final.get("terminal_ownership_activations"),
        }
    return result


def paired_terminal_authority_closeout(
    qualification: Path, candidate_directory: Path, output: Path,
) -> None:
    """Close the exact v12-to-v3 CPU counterfactual without new physics."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "-1"}:
        raise RuntimeError("paired closeout requires CUDA_VISIBLE_DEVICES='' or -1")
    qualification = qualification.resolve()
    candidate_directory = candidate_directory.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"derived output already exists: {output}")

    source_files = {
        "qualification_plan": qualification / "plan.json",
        "qualification_progress": qualification / "progress.jsonl",
        "qualification_report": qualification / "report.json",
        "qualification_audit": qualification / "independent_audit.json",
        "candidate_plan": candidate_directory / "plan.json",
        "candidate_progress": candidate_directory / "progress.jsonl",
        "candidate_report": candidate_directory / "report.json",
        "candidate_audit": candidate_directory / "independent_audit.json",
    }
    for path in source_files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    qualification_report = json.loads(source_files["qualification_report"].read_text())
    qualification_audit = json.loads(source_files["qualification_audit"].read_text())
    candidate_report = json.loads(source_files["candidate_report"].read_text())
    candidate_audit = json.loads(source_files["candidate_audit"].read_text())
    if not qualification_report.get("passed") or not qualification_audit.get("passed"):
        raise ValueError("v12 qualification or independent audit did not pass")
    if (
        not candidate_report.get("passed")
        or not candidate_report.get("diagnostic_only")
        or not candidate_audit.get("integrity_passed")
    ):
        raise ValueError("candidate diagnostic or independent audit did not pass")

    baseline_rows = [
        json.loads(line)
        for line in source_files["qualification_progress"].read_text().splitlines()
    ]
    candidate_rows = [
        json.loads(line)
        for line in source_files["candidate_progress"].read_text().splitlines()
    ]
    baseline_paired = {
        _paired_key(row, candidate=False): row
        for row in baseline_rows
        if (row["family"], row["key"]) in PAIRED_T2_ROWS
    }
    candidate_key_names = set(PAIRED_T2_ROWS.values())
    candidate_paired = {
        _paired_key(row, candidate=True): row
        for row in candidate_rows
        if row["key"] in candidate_key_names
    }
    if len(baseline_paired) != 576 or set(baseline_paired) != set(candidate_paired):
        raise ValueError("exact learned/model row coverage differs from 576 paired episodes")
    candidate_sentinel = [
        row for row in candidate_rows if row["key"] == "public_geometry_current"
    ]
    if len(candidate_sentinel) != 8 or not all(row["success"] for row in candidate_sentinel):
        raise ValueError("candidate public sentinel is not 8/8")

    transitions: Counter[tuple[str, str]] = Counter()
    paired_details = []
    old_terminal_cluster = []
    for key in sorted(baseline_paired):
        baseline = baseline_paired[key]
        candidate = candidate_paired[key]
        for field in ("reset_seed", "mechanism_parameters", "robot"):
            if baseline["request"][field] != candidate["request"][field]:
                raise ValueError(f"paired request changed {field}: {key}")
        transition = (baseline["outcome"], candidate["outcome"])
        transitions[transition] += 1
        if baseline["outcome"] != "success" or candidate["outcome"] != "success":
            detail = {
                "pair": {
                    "family": key[0], "key": key[1], "instance": key[2],
                    "scene_cell": key[3], "physical_initial_slot": key[4],
                    "requested_mode": key[5],
                },
                "transition": f"{transition[0]}->{transition[1]}",
                "baseline_record_id": baseline["record_id"],
                "candidate_record_id": candidate["record_id"],
                "baseline": _phase_evidence(baseline, candidate=False),
                "candidate": _phase_evidence(candidate, candidate=True),
            }
            paired_details.append(detail)
            if (
                baseline["outcome"] in {"overforce", "timeout"}
                and int(baseline["request"]["requested_mode"]) == 1
            ):
                old_terminal_cluster.append(detail)

    expected_transitions = Counter({
        ("success", "success"): 551,
        ("overforce", "success"): 18,
        ("timeout", "success"): 2,
        ("timeout", "timeout"): 4,
        ("success", "timeout"): 1,
    })
    if transitions != expected_transitions:
        raise ValueError(f"paired transition inventory changed: {transitions}")
    if len(old_terminal_cluster) != 24:
        raise ValueError("v12 terminal-signature cluster no longer has 24 episodes")

    cluster_rows = {
        "terminal_authority": [
            row for row in baseline_rows
            if (row["family"], row["key"]) in PAIRED_T2_ROWS
            and row["outcome"] != "success"
        ],
        "fada": [row for row in baseline_rows if row["key"] == "fada"],
        "optimizer_and_prior": [
            row for row in baseline_rows
            if row["key"] in {"dial_transfer", "public_source_prior"}
        ],
    }
    if {name: len(rows) for name, rows in cluster_rows.items()} != {
        "terminal_authority": 24, "fada": 64, "optimizer_and_prior": 72,
    }:
        raise ValueError("three common-failure cluster inventory changed")

    def outcome_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        return dict(sorted(Counter(row["outcome"] for row in rows).items()))

    terminal_subtypes = {
        "unsafe_clearance_descent_residual": [
            row for row in old_terminal_cluster
            if row["transition"] == "overforce->success"
        ],
        "terminal_completion_stagnation_recoverable": [
            row for row in old_terminal_cluster
            if row["transition"] == "timeout->success"
        ],
        "upstream_budget_exhaustion": [
            row for row in old_terminal_cluster
            if row["transition"] == "timeout->timeout"
        ],
    }
    if {name: len(rows) for name, rows in terminal_subtypes.items()} != {
        "unsafe_clearance_descent_residual": 18,
        "terminal_completion_stagnation_recoverable": 2,
        "upstream_budget_exhaustion": 4,
    }:
        raise ValueError("terminal cluster causal split changed")
    regression = next(
        row for row in paired_details if row["transition"] == "success->timeout"
    )

    early_rejections = {}
    for name in (
        "terminal_completion_ownership_exact_replay",
        "post_outlet_completion_ownership_exact_replay",
    ):
        path = candidate_directory.parent / name / "REJECTED_CANDIDATE.json"
        value = json.loads(path.read_text())
        early_rejections[name] = {
            "candidate": value["candidate"],
            "status": value["status"],
            "reason": value["reason"],
            "completed_before_early_rejection": value["counts"][
                "completed_before_early_rejection"
            ],
            "outcomes": value["counts"]["outcomes"],
            "artifact": {
                "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
            },
        }

    artifacts = {
        name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
        for name, path in source_files.items()
    }
    artifacts["analysis_script"] = {
        "path": str(Path(__file__).resolve().relative_to(ROOT)),
        "sha256": sha256(Path(__file__).resolve()),
    }
    closeout = {
        "schema_version": "t2-terminal-authority-paired-cpu-closeout-v1",
        "status": "complete_candidate_rejected_next_revision_scoped",
        "scope": (
            "CPU-only analysis of exact recorded v12 and diagnostic v3 episodes; "
            "no physics, training, confirmation state, or paper performance claim"
        ),
        "device": "cpu",
        "gpu_calls": 0,
        "physics_calls_during_closeout": 0,
        "confirmation_states_opened": 0,
        "performance_evidence": False,
        "artifacts": artifacts,
        "counts": {
            "v12_physical_episodes": len(baseline_rows),
            "candidate_physical_episodes": len(candidate_rows),
            "paired_method_episodes": len(baseline_paired),
            "candidate_public_sentinel_successes": sum(
                bool(row["success"]) for row in candidate_sentinel
            ),
            "candidate_public_sentinel_episodes": len(candidate_sentinel),
            "paired_transitions": {
                f"{before}->{after}": count
                for (before, after), count in sorted(transitions.items())
            },
        },
        "candidate_sequence": {
            **early_rejections,
            "clearance_progress_authority_exact_replay": {
                "candidate": "public-T2-clearance-progress-authority-v3",
                "status": "rejected_after_complete_exact_replay",
                "reason": (
                    "It removes all 18 paired v12 overforces and recovers two timeouts, "
                    "but regresses one paired v12 success to timeout."
                ),
                "report_outcomes": candidate_audit["counts"]["outcomes"],
            },
        },
        "three_common_failure_clusters": {
            "learned_or_optimized_terminal_signature": {
                "episodes": 24,
                "outcomes": outcome_counts(cluster_rows["terminal_authority"]),
                "causal_split_after_exact_v3_counterfactual": {
                    name: len(rows) for name, rows in terminal_subtypes.items()
                },
                "decision": (
                    "The shared terminal signature is not a single root cause. The v3 "
                    "counterfactual separates unsafe residual coupling, recoverable late "
                    "stagnation, and upstream deadline-budget exhaustion."
                ),
            },
            "fada_closed_loop_failure": {
                "episodes": 64,
                "outcomes": outcome_counts(cluster_rows["fada"]),
                "episodes_without_outlet": sum(
                    row["episode_metrics"]["first_outlet_step"] is None
                    for row in cluster_rows["fada"]
                ),
                "force_peak_range_N": [
                    min(row["force_peak_N"] for row in cluster_rows["fada"]),
                    max(row["force_peak_N"] for row in cluster_rows["fada"]),
                ],
            },
            "dial_and_direct_prior_stagnation": {
                "episodes": 72,
                "outcomes": outcome_counts(cluster_rows["optimizer_and_prior"]),
                "dial": {
                    "episodes": 64,
                    "episodes_without_outlet": 64,
                    "force_peak_range_N": [
                        min(row["force_peak_N"] for row in cluster_rows["optimizer_and_prior"]
                            if row["key"] == "dial_transfer"),
                        max(row["force_peak_N"] for row in cluster_rows["optimizer_and_prior"]
                            if row["key"] == "dial_transfer"),
                    ],
                },
                "public_source_prior": {
                    "episodes": 8,
                    "episodes_reaching_outlet": 8,
                },
            },
        },
        "terminal_cluster_details": {
            name: rows for name, rows in terminal_subtypes.items()
        },
        "candidate_regression": regression,
        "paired_non_success_or_discordant": paired_details,
    }

    diagnosis = {
        "schema_version": "t2-three-common-failure-diagnosis-plan-v1",
        "status": "cpu_diagnosis_complete_next_revision_frozen_for_test",
        "scope": "bounded CPU development diagnosis; no formal performance evidence",
        "decisions": [
            {
                "cluster": "learned_or_optimized_terminal_signature",
                "finding": (
                    "V1 hands off too late, V2 hard-hands off too early, and V3 safely "
                    "attenuates clearance residuals but reaches zero residual authority at "
                    "stage-2 descent in the one paired success that regresses. This localizes "
                    "the next counterfactual; it does not yet prove which residual component "
                    "is causally necessary."
                ),
                "next_test": (
                    "Keep the v3 distance envelope through clearance traverse. During "
                    "stage-2 descent, keep every task-owned coordinate equal to the public "
                    "common request and admit only T2 transport-coordinate residuals "
                    "{0,1,3,4,5} at fractions {1,.5,.25,0}. A fraction is admissible only "
                    "when its public one-step translational and rotational terminal-set "
                    "progress is no worse than common; multiply it by the existing public "
                    "terminal force-release factor and return exact common inside the exact "
                    "terminal set."
                ),
                "rejection_gate": (
                    "Reject on any v12 success->failure transition, any overforce, any "
                    "guide-contact failure, any change to task-owned coordinates, or any "
                    "candidate that does not call both candidate and common every tick."
                ),
                "acceptance_gate": (
                    "Exact 576-pair replay must keep all 552 prior paired successes, retain "
                    "Cross-EBDM 64/64, produce zero overforce, and preserve the 8/8 public "
                    "sentinel. Persistent upstream-budget timeouts may remain and must be "
                    "reported as method limitations rather than repaired outcomes."
                ),
            },
            {
                "cluster": "fada_closed_loop_failure",
                "finding": (
                    "FADA has 60 overforces and four timeouts; 49/64 episodes never reach "
                    "the outlet. Outcome labels do not distinguish source/decoder defects "
                    "from a clean closed-loop method limitation."
                ),
                "next_test": (
                    "For every frozen instance, verify source fixture reconstruction, "
                    "normalization round-trip, decoder frame/unit/Box semantics and "
                    "teacher-forced source error. Then align the first target closed-loop "
                    "divergence with the last safe and first unsafe 1 ms substeps."
                ),
                "gate": (
                    "Repair only a reproducible fixture, normalization, decoder or adapter "
                    "defect. A correctly decoded collision or stagnation remains FADA's T2 "
                    "behavior and stays in its denominator."
                ),
            },
            {
                "cluster": "dial_and_direct_prior_stagnation",
                "finding": (
                    "All 64 DIAL episodes time out before outlet with low force peaks, while "
                    "all eight direct-prior episodes reach outlet and still time out. The "
                    "two baselines therefore need separate optimizer and prior audits."
                ),
                "next_test": (
                    "For DIAL, record objective decrease, projected-gradient norm, bound "
                    "activity, candidate-to-accepted displacement and outlet progress per "
                    "remaining second. For direct prior, verify source frame/mode/unit "
                    "semantics and compare issued, accepted and applied progress with the "
                    "paired public common request."
                ),
                "gate": (
                    "Repair only objective, projection, unit or source-schema plumbing. "
                    "Internally consistent no-outlet or post-outlet stagnation under the "
                    "frozen deadline remains the baseline result."
                ),
            },
        ],
        "next_revision": {
            "candidate_version": "public-T2-clearance-terminal-set-admission-v4",
            "implementation_scope": (
                "Extend the existing PublicTerminalOwnedGuideActor in "
                "genedynamics/tasks/transfer/controllers/guide_contact.py and its existing "
                "scripts/paper/cross_ebdm diagnostic path; do not add a solver, learning "
                "package, integration layer, or parallel entrypoint."
            ),
            "unchanged": [
                "controller-v50 and route-v4",
                "20 s deadline, 2 mm/2 degree terminal set and 15-control dwell",
                "25 N total-force, 10 N lateral-force and 3 rad/s joint-speed limits",
                "reset cells, mode semantics, evaluator and 1 ms backend governor",
                "mode-0 behavior and mode-1 pre-outlet/acquisition behavior",
            ],
            "training": (
                "No retraining and no GPU work before the exact CPU paired gate passes. "
                "The four persistent timeouts start too late for a terminal authority "
                "change and must not drive transport retraining."
            ),
            "release": (
                "A passing exact replay is still diagnostic. Bind a new shared task-contract "
                "version only after a fresh-seed full T2 behavior qualification and independent "
                "audit; then run zero-physics architecture, CPU package and GPU1 device audits."
            ),
        },
        "theory_revision": [
            "Replace a binary finite authority interval with a state-dependent admissible residual set on transport coordinates.",
            "Define the terminal regime by exact terminal-set progress and invariant dwell, not by a route-stage label alone.",
            "Require measured closed-loop progress budgets before terminal capture; terminal projection cannot repair upstream deadline exhaustion.",
            "Keep shared executor feasibility, method residual benefit, force safety and finite-time completion as separate claims.",
            "The v3 counterfactual is compatibility evidence only and provides no transport-performance advantage claim.",
        ],
    }

    output.mkdir(parents=True, exist_ok=False)
    (output / "closeout.json").write_text(
        json.dumps(closeout, indent=2, sort_keys=True) + "\n"
    )
    (output / "diagnosis_plan.json").write_text(
        json.dumps(diagnosis, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "schema_version": "derived-artifact-manifest-v1",
        "files": {
            path.name: sha256(path)
            for path in sorted(output.iterdir()) if path.name != "manifest.json"
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    try:
        output_label = str(output.relative_to(ROOT))
    except ValueError:
        output_label = str(output)
    print(json.dumps({
        "output": output_label,
        "status": closeout["status"],
        "counts": closeout["counts"],
        "next_revision": diagnosis["next_revision"]["candidate_version"],
    }, indent=2, sort_keys=True))


def terminal_set_admission_closeout(
    qualification: Path,
    clearance_directory: Path,
    candidate_directory: Path,
    output: Path,
) -> None:
    """Seal the rejected v4 early stop and scope the next CPU counterfactual."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "-1"}:
        raise RuntimeError("terminal admission closeout requires CPU-only visibility")
    qualification = qualification.resolve()
    clearance_directory = clearance_directory.resolve()
    candidate_directory = candidate_directory.resolve()
    output = output.resolve()
    rejection_path = candidate_directory / "REJECTED_CANDIDATE.json"
    if output.exists() or rejection_path.exists():
        raise FileExistsError("terminal admission closeout is already sealed")

    source_files = {
        "qualification_progress": qualification / "progress.jsonl",
        "clearance_progress": clearance_directory / "progress.jsonl",
        "candidate_plan": candidate_directory / "plan.json",
        "candidate_progress": candidate_directory / "progress.jsonl",
    }
    for path in source_files.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    def journal(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text().splitlines()]

    baseline_rows = journal(source_files["qualification_progress"])
    clearance_rows = journal(source_files["clearance_progress"])
    candidate_journal = journal(source_files["candidate_progress"])
    candidate_rows = [row for row in candidate_journal if row.get("evidence")]
    interrupted_rows = [row for row in candidate_journal if not row.get("evidence")]
    if len(candidate_rows) != 5 or len(interrupted_rows) != 2:
        raise ValueError(
            "v4 journal no longer contains five completed and two interrupted rows"
        )
    if any(row.get("outcome") != "reset_or_execution_error" for row in interrupted_rows):
        raise ValueError("unexpected evidence-free row in v4 early-stop journal")
    if any(
        row.get("request", {}).get("performance_evidence")
        or not row.get("evidence", {}).get("audit_passed")
        for row in candidate_rows
    ):
        raise ValueError("v4 journal is not audited diagnostic-only evidence")
    for row in candidate_rows:
        pointer = row["evidence"]
        path = ROOT / pointer["path"]
        if not path.is_file() or sha256(path) != pointer["sha256"]:
            raise ValueError(f"missing or changed v4 evidence: {row['record_id']}")

    pair = {
        "family": "ablation",
        "instance": "instance_04",
        "scene_cell": "latency_gain_shift",
        "physical_initial_slot": 1,
        "requested_mode": 1,
    }

    def matching(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
        selected = [
            row for row in rows
            if row["family"] == pair["family"]
            and row["key"] == key
            and row["instance"] == pair["instance"]
            and row["request"]["scene_cell"] == pair["scene_cell"]
            and int(row["request"]["physical_initial_slot"])
            == pair["physical_initial_slot"]
            and int(row["request"]["requested_mode"])
            == pair["requested_mode"]
        ]
        if len(selected) != 1:
            raise ValueError(f"expected one exact paired row for {key}, got {len(selected)}")
        return selected[0]

    baseline = matching(baseline_rows, "ungated_full")
    clearance = matching(clearance_rows, "ungated_full_clearance_progress")
    candidate = matching(candidate_rows, "ungated_full_terminal_set_admission")
    for field in ("reset_seed", "mechanism_parameters", "robot"):
        if not (
            baseline["request"][field]
            == clearance["request"][field]
            == candidate["request"][field]
        ):
            raise ValueError(f"exact reset pairing changed: {field}")
    if [baseline["outcome"], clearance["outcome"], candidate["outcome"]] != [
        "success", "timeout", "timeout",
    ]:
        raise ValueError("v12/v3/v4 regression outcomes changed")

    record = _episode_record(candidate)
    diagnostics = record["control_diagnostics"]
    traces = record["traces"]
    if len(diagnostics) != 1000 or len(traces) != 1000:
        raise ValueError("v4 regression is no longer a 1000-control timeout")

    from genedynamics.core.control.transfer import orientation_error
    from genedynamics.tasks.transfer.specs import task_spec

    spec = task_spec("T2")
    goal = np.asarray(spec.config.goal, dtype=float)
    goal_rotation = np.diag([1.0, -1.0, -1.0])
    dt = float(spec.config.control_dt)
    position_tolerance = float(spec.config.position_tolerance)
    orientation_tolerance = float(spec.config.orientation_tolerance)
    dwell_controls = 15
    transport = [0, 1, 3, 4, 5]
    terminal_rows = []
    for tick, (trace, diagnostic) in enumerate(zip(traces, diagnostics)):
        if (
            int(diagnostic.get("common", {}).get("stage", -1)) != 3
            or int(diagnostic.get("last_selector_code", -1)) not in {2, 4}
        ):
            continue
        position_error = goal - np.asarray(trace["tcp_position"], dtype=float)
        rotation_error = orientation_error(
            np, np.asarray(trace["tcp_rotation"], dtype=float), goal_rotation,
        )
        current_position = max(
            0.0, float(np.linalg.norm(position_error)) - position_tolerance,
        )
        current_rotation = max(
            0.0, float(np.linalg.norm(rotation_error)) - orientation_tolerance,
        )
        common = np.asarray(diagnostic["last_common_action"], dtype=float)[0]
        raw_candidate = np.asarray(
            diagnostic["last_candidate_action"], dtype=float,
        )[0]
        utilization = float(diagnostic["common"]["force_utilization"])
        force_scale = float(np.clip((0.55 - utilization) / 0.20, 0.0, 1.0))
        trial = common.copy()
        trial[transport] += force_scale * (raw_candidate[transport] - common[transport])

        def predicted(action: np.ndarray) -> tuple[float, float]:
            return (
                max(
                    0.0,
                    float(np.linalg.norm(position_error - dt * action[:3]))
                    - position_tolerance,
                ),
                max(
                    0.0,
                    float(np.linalg.norm(rotation_error - dt * action[3:6]))
                    - orientation_tolerance,
                ),
            )

        common_position, common_rotation = predicted(common)
        trial_position, trial_rotation = predicted(trial)
        usable = max(1, 1000 - tick - dwell_controls)
        required_position = current_position / usable
        required_rotation = current_rotation / usable
        terminal_rows.append({
            "tick": tick,
            "selector_code": int(diagnostic["last_selector_code"]),
            "force_scale": force_scale,
            "current_position_set_distance_m": current_position,
            "common_predicted_position_set_distance_m": common_position,
            "trial_predicted_position_set_distance_m": trial_position,
            "relative_position_disadvantage_m": trial_position - common_position,
            "deadline_viable": bool(
                trial_position
                <= max(0.0, current_position - required_position) + 1e-12
                and trial_rotation
                <= max(0.0, current_rotation - required_rotation) + 1e-12
            ),
        })
    if len(terminal_rows) != 86:
        raise ValueError("v4 terminal-admission interval changed")
    for row in terminal_rows:
        next_position = np.asarray(
            traces[min(row["tick"] + 1, len(traces) - 1)]["tcp_position"],
            dtype=float,
        )
        next_distance = max(
            0.0, float(np.linalg.norm(goal - next_position)) - position_tolerance,
        )
        row["measured_next_position_progress_m"] = (
            row["current_position_set_distance_m"] - next_distance
        )

    intervals = []
    for row in terminal_rows:
        if (
            not intervals
            or intervals[-1]["selector_code"] != row["selector_code"]
            or intervals[-1]["end_tick"] + 1 != row["tick"]
        ):
            intervals.append({
                "start_tick": row["tick"], "end_tick": row["tick"],
                "selector_code": row["selector_code"],
            })
        else:
            intervals[-1]["end_tick"] = row["tick"]
    fallback = [row for row in terminal_rows if row["selector_code"] == 2]
    admitted = [row for row in terminal_rows if row["selector_code"] == 4]
    if len(fallback) != 30 or len(admitted) != 56:
        raise ValueError("v4 selector inventory changed")

    outcomes = dict(sorted(Counter(row["outcome"] for row in candidate_rows).items()))
    rejection = {
        "schema_version": "rejected-development-candidate-v1",
        "candidate": "public-T2-clearance-terminal-set-admission-v4",
        "status": "rejected_early_preserved",
        "scope": "CPU-only exact-reset development diagnosis; no performance evidence",
        "reason": (
            "The first known v12 success that v3 regressed still times out. The v4 "
            "relative one-step dominance test rejects 30 force-safe transport residuals "
            "even though all 30 satisfy the absolute remaining-time terminal budget."
        ),
        "counts": {
            "planned_physical_jobs": int(json.loads(
                source_files["candidate_plan"].read_text()
            )["counts"]["physical_jobs"]),
            "completed_before_early_rejection": len(candidate_rows),
            "outcomes": outcomes,
            "interrupted_journal_rows": len(interrupted_rows),
        },
        "regression_record_id": candidate["record_id"],
        "process_exit": 143,
        "failure_replacement": False,
        "interrupted_record_ids": [row["record_id"] for row in interrupted_rows],
    }
    rejection_path.write_text(json.dumps(rejection, indent=2, sort_keys=True) + "\n")

    closeout = {
        "schema_version": "t2-terminal-set-admission-cpu-closeout-v1",
        "status": "complete_candidate_rejected_next_counterfactual_scoped",
        "scope": (
            "CPU-only analysis of recorded exact-reset v12, v3 and v4 evidence; "
            "no new physics, training, GPU call, confirmation state or performance claim"
        ),
        "device": "cpu",
        "gpu_calls": 0,
        "physics_calls_during_closeout": 0,
        "candidate_early_stop": rejection,
        "exact_regression": {
            "pair": pair,
            "reset_seed": candidate["request"]["reset_seed"],
            "v12": _phase_evidence(baseline, candidate=False),
            "v3": _phase_evidence(clearance, candidate=True),
            "v4": _phase_evidence(candidate, candidate=True),
        },
        "terminal_selector_diagnosis": {
            "terminal_controls": len(terminal_rows),
            "transport_residual_admitted_controls": len(admitted),
            "common_fallback_controls": len(fallback),
            "fallbacks_whose_force_scaled_transport_residual_met_absolute_deadline_budget": sum(
                row["deadline_viable"] for row in fallback
            ),
            "admissions_meeting_absolute_deadline_budget": sum(
                row["deadline_viable"] for row in admitted
            ),
            "fallback_relative_position_disadvantage_range_m": [
                min(row["relative_position_disadvantage_m"] for row in fallback),
                max(row["relative_position_disadvantage_m"] for row in fallback),
            ],
            "selector_intervals": intervals,
            "final_position_excess_over_set_m": max(
                0.0,
                float(candidate["episode_metrics"]["final_position_error_m"])
                - position_tolerance,
            ),
            "remaining_dwell_controls_at_timeout": 0,
            "interpretation": (
                "Relative command-space dominance against common is too conservative "
                "and is not a closed-loop terminal viability certificate. It discards "
                "force-safe transport residuals based on micrometer-scale one-step "
                "differences while ignoring reference lag, coupled dynamics and dwell."
            ),
        },
        "artifacts": {
            **{
                name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                for name, path in source_files.items()
            },
            "rejection": {
                "path": str(rejection_path.relative_to(ROOT)),
                "sha256": sha256(rejection_path),
            },
            "analysis_script": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": sha256(Path(__file__).resolve()),
            },
        },
    }
    diagnosis = {
        "schema_version": "t2-three-common-failure-diagnosis-plan-v2",
        "status": "cpu_diagnosis_complete_next_counterfactual_frozen",
        "three_failure_clusters": [
            {
                "cluster": "learned_or_optimized_terminal_signature",
                "finding": (
                    "V3 removes all 18 paired overforces but regresses one success; v4 "
                    "recovers 1.419 mm of its v3 terminal error and misses the 2 mm set "
                    "by only 24.2 micrometers, while its relative dominance rule rejects "
                    "30/30 transport residuals that satisfy the absolute deadline budget."
                ),
                "next_test": (
                    "Keep the v3 clearance-distance envelope. In stage-2 descent and "
                    "terminal dwell, retain only transport-coordinate residuals "
                    "{0,1,3,4,5}, scaled by the existing public force-release factor; "
                    "keep task-owned coordinates {2,6,7,8,9,10,11,12} exactly common. "
                    "Remove the v4 no-worse-than-common blend search."
                ),
                "preflight": (
                    "Replay the exact regression seed and one representative prior "
                    "overforce seed. Reject immediately unless the former succeeds and "
                    "the latter remains force-safe."
                ),
            },
            {
                "cluster": "fada_closed_loop_failure",
                "finding": "60 overforces, four timeouts, and 49/64 episodes without outlet.",
                "next_test": (
                    "Offline-check every frozen source fixture, normalization inverse, "
                    "decoder frame/unit/Box semantics and teacher-forced source error; "
                    "then align the final safe and first unsafe recorded 1 ms substeps."
                ),
                "repair_gate": (
                    "Repair only a reproduced fixture, normalization, decoder or adapter "
                    "defect. A correctly decoded collision or stagnation remains FADA's result."
                ),
            },
            {
                "cluster": "dial_and_direct_prior_stagnation",
                "finding": (
                    "DIAL times out before outlet in 64/64; direct source prior reaches "
                    "outlet in 8/8 and still times out, so they are distinct mechanisms."
                ),
                "next_test": (
                    "For DIAL, audit objective decrease, projected-gradient norm, active "
                    "bounds, accepted displacement and outlet progress per remaining "
                    "second. For direct prior, audit source frame/mode/unit semantics and "
                    "issued-versus-applied progress against the paired public controller."
                ),
                "repair_gate": (
                    "Repair only objective, projection, unit or source-schema plumbing; "
                    "an internally consistent stagnation remains in the denominator."
                ),
            },
        ],
        "next_revision": {
            "candidate_version": "public-T2-clearance-terminal-transport-v5",
            "implementation": (
                "Modify only PublicTerminalOwnedGuideActor in "
                "genedynamics/tasks/transfer/controllers/guide_contact.py, expose its "
                "existing episode diagnostics, and invoke it through the existing "
                "scripts/paper/cross_ebdm entrypoints."
            ),
            "unchanged": [
                "controller-v50, route-v4 and v3 clearance envelope",
                "20 s deadline, 2 mm/2 degree terminal set and 15-control dwell",
                "25 N total-force, 10 N lateral-force and 3 rad/s joint-speed limits",
                "reset cells, modes, evaluator, 1 ms backend and task-owned coordinates",
            ],
            "full_cpu_gate": (
                "Only after the two-seed preflight passes, run the exact 576 method-pair "
                "plus 8 public-sentinel replay. Preserve all 552 v12 paired successes, "
                "Cross-EBDM 64/64, zero overforce and sentinel 8/8. Keep the four upstream "
                "budget timeouts as limitations."
            ),
            "gpu_gate": (
                "No GPU work or retraining before the exact CPU gate passes. A passing "
                "diagnostic then receives fresh-seed T2 qualification and independent "
                "audit before GPU1 package/device checks."
            ),
        },
        "theory_revision": [
            "Replace relative one-step dominance over common with coordinate-partitioned residual authority plus an absolute finite-time closed-loop progress obligation.",
            "Treat command-space one-step prediction as a diagnostic proxy, not an invariant-set or reachability certificate.",
            "Include reference lag, coupled execution and the dwell reserve in the terminal viability state.",
            "Separate task-owned safety coordinates from method-owned transport coordinates throughout descent and dwell.",
            "Keep upstream deadline exhaustion outside the terminal residual claim.",
        ],
    }

    output.mkdir(parents=True, exist_ok=False)
    for name, payload in (("closeout.json", closeout), ("diagnosis_plan.json", diagnosis)):
        (output / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "derived-artifact-manifest-v1",
        "files": {
            path.name: sha256(path)
            for path in sorted(output.iterdir()) if path.name != "manifest.json"
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    try:
        output_label = str(output.relative_to(ROOT))
    except ValueError:
        output_label = str(output)
    print(json.dumps({
        "output": output_label,
        "status": closeout["status"],
        "candidate": diagnosis["next_revision"]["candidate_version"],
        "fallback_deadline_viable": (
            closeout["terminal_selector_diagnosis"]
            ["fallbacks_whose_force_scaled_transport_residual_met_absolute_deadline_budget"]
        ),
    }, indent=2, sort_keys=True))


V13_COMPARABLE_ROWS = {
    "T2/main/cross_ebdm",
    "T2/main/task_only",
    "T2/main/response_mpc",
    "T2/ablation/frozen_warm",
    "T2/ablation/task_only",
    "T2/ablation/matching_no_band",
    "T2/ablation/band_only",
    "T2/ablation/ungated_full",
    "T2/ablation/gated_full",
}


def _wrapper_common(diagnostic: dict[str, Any]) -> dict[str, Any]:
    """Return the public common-controller state from wrapped or plain evidence."""
    return diagnostic.get("common", diagnostic.get("base", {}))


def _first_index(values: list[Any], predicate) -> int | None:
    return next((index for index, value in enumerate(values) if predicate(value)), None)


def _norm(value: Any) -> float | None:
    if value is None:
        return None
    return float(np.linalg.norm(np.asarray(value, dtype=float)))


def _v13_trace_profile(
    row: dict[str, Any], *, deadline_controls: int,
    force_limit_N: float, lateral_force_limit_N: float,
) -> dict[str, Any]:
    record = _episode_record(row)
    diagnostics = record.get("control_diagnostics", [])
    traces = record.get("traces", [])
    if not diagnostics or len(diagnostics) != len(traces):
        raise ValueError(f"incomplete v13 trace: {row['record_id']}")

    selector_codes = [int(value.get("last_selector_code", -1)) for value in diagnostics]
    first_blend = _first_index(selector_codes, lambda value: value == 3)
    first_common = _first_index(selector_codes, lambda value: value == 2)
    common_states = [_wrapper_common(value) for value in diagnostics]

    first_violation = None
    for tick, trace in enumerate(traces):
        forces = trace.get("force_peak_substeps", [])
        laterals = trace.get("lateral_force_substeps", [])
        if len(forces) != len(laterals):
            raise ValueError(f"substep evidence differs: {row['record_id']}")
        for substep, (force, lateral) in enumerate(zip(forces, laterals)):
            if float(force) > force_limit_N or float(lateral) > lateral_force_limit_N:
                first_violation = {
                    "control_tick": tick,
                    "physics_substep": substep,
                    "force_N": float(force),
                    "lateral_force_N": float(lateral),
                    "selector_code": selector_codes[tick],
                    "relative_to_exact_common_controls": (
                        tick - first_common if first_common is not None else None
                    ),
                }
                break
        if first_violation is not None:
            break

    entry = None
    if first_common is not None:
        state = common_states[first_common]
        trace = traces[first_common]
        entry = {
            "control_tick": first_common,
            "deadline_reserve_controls": deadline_controls - first_common,
            "observed_post_handoff_controls": len(traces) - first_common,
            "terminal_set_excess_m": state.get("terminal_distance_m"),
            "tracking_lag_m": state.get("tracking_lag_m"),
            "public_reference_lead_m": _norm(trace.get("public_reference_lead")),
            "force_N": float(trace.get("force_peak", 0.)),
            "lateral_force_N": float(trace.get("lateral_force_peak", 0.)),
        }

    completion_stage_ticks = {
        str(stage): _first_index(
            common_states,
            lambda value, stage=stage: int(
                (value.get("completion") or {}).get("stage", -1)
            ) >= stage,
        )
        for stage in (1, 2)
    }
    final = common_states[-1]
    return {
        "record_id": row["record_id"],
        "row_id": row["row_id"],
        "instance": row["instance"],
        "request_id": row["request"]["request_id"],
        "requested_mode": int(row["request"]["requested_mode"]),
        "outcome": row["outcome"],
        "success": bool(row["success"]),
        "evidence_sha256": row["evidence"]["sha256"],
        "control_steps": len(traces),
        "first_contact_step": row["episode_metrics"]["first_contact_step"],
        "first_outlet_step": row["episode_metrics"]["first_outlet_step"],
        "first_clearance_blend_control": first_blend,
        "first_exact_common_control": first_common,
        "completion_stage_ticks": completion_stage_ticks,
        "selector_counts": dict(sorted(Counter(selector_codes).items())),
        "exact_common_entry": entry,
        "first_force_violation": first_violation,
        "final_position_error_m": row["episode_metrics"]["final_position_error_m"],
        "final_orientation_error_rad": row["episode_metrics"][
            "final_orientation_error_rad"
        ],
        "final_terminal_set_excess_m": final.get("terminal_distance_m"),
        "final_terminal_hold_controls": final.get("terminal_pose_hold_controls"),
        "force_peak_N": float(row["force_peak_N"]),
        "lateral_force_peak_N": float(row["lateral_force_peak_N"]),
    }


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "q05": float(np.quantile(array, .05)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, .95)),
        "maximum": float(np.max(array)),
    }


def v13_shared_failure_attribution(
    outcomes: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Attribute the three shared mode-1 failures from sealed v13 evidence."""
    contract_path = ROOT / (
        "configs/arm/guided_contact_alignment/protocol/task_contract_v5.json"
    )
    contract = json.loads(contract_path.read_text())
    shared = contract["shared_execution"]
    deadline_controls = round(
        float(shared["timeout_seconds"]) / float(shared["control_dt_seconds"])
    )
    profiles = [
        _v13_trace_profile(
            row,
            deadline_controls=deadline_controls,
            force_limit_N=float(shared["force_limit_N"]),
            lateral_force_limit_N=float(shared["lateral_force_limit_N"]),
        )
        for row in outcomes if row["row_id"] in V13_COMPARABLE_ROWS
    ]
    if len(profiles) != 576:
        raise ValueError("v13 comparable learned/model coverage is not 576 episodes")
    by_identity = {
        (profile["row_id"], profile["instance"], profile["request_id"]): profile
        for profile in profiles
    }
    mode1 = [profile for profile in profiles if profile["requested_mode"] == 1]
    mode1_failures = [profile for profile in mode1 if not profile["success"]]
    mode0_failures = [
        profile for profile in profiles
        if profile["requested_mode"] == 0 and not profile["success"]
    ]

    convergence_tail = [
        profile for profile in mode1_failures
        if profile["row_id"] in {
            "T2/main/cross_ebdm", "T2/ablation/gated_full",
        }
    ]
    late_handoff = [
        profile for profile in mode1_failures
        if profile["row_id"] == "T2/main/response_mpc"
    ]
    delayed_impact = [
        profile for profile in mode1_failures
        if profile["row_id"] in {
            "T2/ablation/matching_no_band", "T2/ablation/band_only",
        }
    ]
    if (
        len(convergence_tail), len(late_handoff), len(delayed_impact),
        len(mode0_failures),
    ) != (2, 2, 2, 1):
        raise ValueError("v13 shared-failure membership changed")
    if len({profile["evidence_sha256"] for profile in convergence_tail}) != 1:
        raise ValueError("Cross-EBDM and gated-full no longer share exact evidence")
    if any(
        profile["first_force_violation"] is None
        or profile["first_force_violation"]["selector_code"] != 2
        for profile in delayed_impact
    ):
        raise ValueError("v13 delayed impacts are not after exact-common handoff")
    ungated = mode0_failures[0]
    if not (
        ungated["row_id"] == "T2/ablation/ungated_full"
        and ungated["first_outlet_step"] is None
        and ungated["first_force_violation"] is not None
        and ungated["first_force_violation"]["selector_code"] == 0
    ):
        raise ValueError("v13 mode-0 ungated failure attribution changed")

    paired = []
    for failure in mode1_failures + mode0_failures:
        comparator = by_identity[(
            "T2/main/task_only", failure["instance"], failure["request_id"],
        )]
        if not comparator["success"]:
            raise ValueError("paired task-only comparator is not successful")
        paired.append({"failure": failure, "task_only_comparator": comparator})

    successful_mode1 = [profile for profile in mode1 if profile["success"]]
    exact_success = [
        profile for profile in successful_mode1
        if profile["exact_common_entry"] is not None
    ]
    if len(exact_success) != 282:
        raise ValueError("v13 successful mode-1 handoff coverage changed")
    envelope = {
        "episodes": len(exact_success),
        "unique_evidence_traces": len({
            profile["evidence_sha256"] for profile in exact_success
        }),
        "first_exact_common_control": _quantiles([
            float(profile["first_exact_common_control"])
            for profile in exact_success
        ]),
        "deadline_reserve_controls": _quantiles([
            float(profile["exact_common_entry"]["deadline_reserve_controls"])
            for profile in exact_success
        ]),
        "observed_post_handoff_controls": _quantiles([
            float(profile["exact_common_entry"]["observed_post_handoff_controls"])
            for profile in exact_success
        ]),
        "entry_tracking_lag_m": _quantiles([
            float(profile["exact_common_entry"]["tracking_lag_m"])
            for profile in exact_success
        ]),
        "entry_public_reference_lead_m": _quantiles([
            float(profile["exact_common_entry"]["public_reference_lead_m"])
            for profile in exact_success
        ]),
    }

    attribution = {
        "schema_version": "t2-v13-shared-failure-attribution-v1",
        "scope": "CPU-only descriptive diagnosis; no counterfactual outcome claim",
        "task_contract": {
            "path": str(contract_path.relative_to(ROOT)),
            "sha256": sha256(contract_path),
            "deadline_controls": deadline_controls,
            "force_limit_N": shared["force_limit_N"],
            "lateral_force_limit_N": shared["lateral_force_limit_N"],
        },
        "coverage": {
            "comparable_episodes": len(profiles),
            "mode1_episodes": len(mode1),
            "mode1_successes": len(successful_mode1),
            "mode1_failed_rows": len(mode1_failures),
            "mode1_unique_failed_traces": len({
                profile["evidence_sha256"] for profile in mode1_failures
            }),
        },
        "successful_mode1_handoff_envelope": envelope,
        "three_shared_mode1_failure_classes": [
            {
                "class": "terminal_convergence_tail",
                "logical_rows": len(convergence_tail),
                "unique_evidence_traces": 1,
                "members": convergence_tail,
                "finding": (
                    "Cross-EBDM and gated-full are the same physical trace. Exact common "
                    "authority starts at control 899 with 101 controls left; the episode ends "
                    "0.303 mm outside the 2 mm terminal set and never starts the dwell."
                ),
            },
            {
                "class": "late_handoff_deadline_exhaustion",
                "logical_rows": len(late_handoff),
                "unique_evidence_traces": len(late_handoff),
                "members": late_handoff,
                "finding": (
                    "Response MPC reaches exact common authority only at controls 957 and 963, "
                    "leaving 43 and 37 controls; final position errors remain 8.094 and "
                    "15.383 mm."
                ),
            },
            {
                "class": "delayed_impact_after_exact_common_handoff",
                "logical_rows": len(delayed_impact),
                "unique_evidence_traces": len(delayed_impact),
                "members": delayed_impact,
                "finding": (
                    "Matching-no-band and band-only enter exact common authority with 7.94 and "
                    "8.96 mm tracking lag, then violate the 10 N lateral limit 13 and 15 "
                    "controls later. Exact terminal actions do not erase delayed state debt."
                ),
            },
        ],
        "ablation_specific_nonshared_failure": {
            "class": "mode0_preoutlet_unguarded_collision",
            "member": ungated,
            "finding": (
                "Ungated-full violates the 25 N total-force limit before outlet while selector "
                "code 0 retains full candidate authority. Main Cross-EBDM and gated-full pass "
                "the same request, so this remains the intended gating ablation result."
            ),
        },
        "paired_failure_profiles": paired,
        "causal_limits": [
            "Recorded trajectories identify when the released interface becomes insufficient; they do not prove a new action would succeed.",
            "Thresholds chosen from this rejected qualification are development parameters and require a fresh seed-disjoint gate.",
            "Cross-EBDM and gated-full count as two logical rows but one physical evidence trace.",
        ],
    }
    return attribution, profiles


def v14_rejected_predeployment_closeout(
    *, qualification: Path, output: Path, plan: dict[str, Any],
    report: dict[str, Any], audit: dict[str, Any],
    outcomes: list[dict[str, Any]], files: dict[str, Path],
) -> None:
    """Close the complete v14 run whose public sentinel failed before physics."""
    if report.get("schema_version") != "four-task-behavioral-predeployment-v14":
        raise ValueError("rejected closeout requires v14")
    infrastructure = [
        row for row in outcomes
        if row.get("outcome") == "reset_or_execution_error"
    ]
    evidence_rows = [row for row in outcomes if row not in infrastructure]
    expected_audit_checks = {
        key: value for key, value in audit.get("checks", {}).items()
        if key not in {
            "all_episode_evidence_reaudited", "zero_reset_or_execution_errors",
        }
    }
    if not (
        len(outcomes) == len(plan["jobs"]) == 720
        and report["counts"]["completed_physical_jobs"] == 720
        and len(infrastructure) == 8
        and all(row["row_id"] == "T2/main/public_geometry" for row in infrastructure)
        and all(row.get("accounting", {}).get("physics_steps") == 0 for row in infrastructure)
        and all(
            row.get("error") == "public geometry cannot use terminal actor parameters"
            for row in infrastructure
        )
        and len(evidence_rows) == 712
        and all(row.get("evidence", {}).get("audit_passed") for row in evidence_rows)
        and audit.get("counts", {}).get("evidence_reaudited") == 712
        and all(expected_audit_checks.values())
        and audit.get("checks", {}).get("all_episode_evidence_reaudited") is False
        and audit.get("checks", {}).get("zero_reset_or_execution_errors") is False
    ):
        raise ValueError("v14 rejected-run integrity signature changed")

    sentinel_validation_path = ROOT / (
        "results/arm/guided_contact_alignment/validation/"
        "public_sentinel_terminal_binding_regression/independent_audit.json"
    )
    sentinel_validation = json.loads(sentinel_validation_path.read_text())
    if not (
        sentinel_validation.get("passed") is True
        and sentinel_validation.get("diagnostic_only") is True
        and sentinel_validation.get("integrity_passed") is True
        and sentinel_validation.get("counts", {}).get("planned") == 8
        and sentinel_validation.get("counts", {}).get("results") == 8
        and sentinel_validation.get("counts", {}).get("outcomes") == {"success": 8}
    ):
        raise ValueError("public sentinel binding regression did not pass 8/8")

    contract_path = ROOT / (
        "configs/arm/guided_contact_alignment/protocol/task_contract_v6.json"
    )
    contract = json.loads(contract_path.read_text())
    shared = contract["shared_execution"]
    from genedynamics.tasks.transfer.specs import task_spec
    goal_position = np.asarray(task_spec("T2").config.goal, dtype=float)
    if not np.array_equal(goal_position, np.asarray([.46, -.18, .596])):
        raise ValueError("T2 public goal changed")
    deadline_controls = round(
        float(shared["timeout_seconds"]) / float(shared["control_dt_seconds"])
    )
    comparable = [row for row in outcomes if row["row_id"] in V13_COMPARABLE_ROWS]
    failures = [row for row in comparable if not row["success"]]
    profiles = [
        _v13_trace_profile(
            row, deadline_controls=deadline_controls,
            force_limit_N=float(shared["force_limit_N"]),
            lateral_force_limit_N=float(shared["lateral_force_limit_N"]),
        )
        for row in failures
    ]
    settling = [
        row for row in profiles
        if row["requested_mode"] == 1 and row["outcome"] == "timeout"
    ]
    post_handoff_force = [
        row for row in profiles
        if row["requested_mode"] == 1 and row["outcome"] == "overforce"
    ]
    preoutlet_ungated = [
        row for row in profiles
        if row["requested_mode"] == 0 and row["outcome"] == "overforce"
    ]
    if (
        len(comparable), len(profiles), len(settling),
        len(post_handoff_force), len(preoutlet_ungated),
    ) != (576, 11, 8, 2, 1):
        raise ValueError("v14 comparable failure membership changed")
    if (
        len({row["evidence_sha256"] for row in settling}) != 5
        or len({row["evidence_sha256"] for row in post_handoff_force}) != 1
        or len({row["evidence_sha256"] for row in preoutlet_ungated}) != 1
    ):
        raise ValueError("v14 unique failure-trace counts changed")
    if any(
        row["first_force_violation"] is None
        or row["first_force_violation"]["selector_code"] != 2
        for row in post_handoff_force
    ):
        raise ValueError("v14 task-only impact is not after exact-common handoff")
    if not (
        preoutlet_ungated[0]["first_outlet_step"] is None
        and preoutlet_ungated[0]["first_force_violation"] is not None
        and preoutlet_ungated[0]["first_force_violation"]["selector_code"] == 0
    ):
        raise ValueError("v14 ungated failure is no longer pre-outlet")

    by_identity = {
        (row["row_id"], row["instance"], row["request"]["request_id"]): row
        for row in comparable
    }
    representatives = {}
    for row, profile in zip(failures, profiles):
        representatives.setdefault(profile["evidence_sha256"], (row, profile))
    v9_parameters = {
        "completion_reserve_controls": 150,
        "reserve_taper_controls": 100,
        "reference_lag_full_authority_m": .008,
        "reference_lag_zero_authority_m": .012,
        "maximum_authority_decay_per_control": .05,
        "completion_progress_floor_mps": .020,
        "post_traverse_reserve_controls": 100,
        "completion_budget_taper_controls": 60,
        "force_speed_full_authority_mps": .010,
        "force_speed_zero_authority_mps": .025,
    }

    def v9_trigger(row: dict[str, Any]) -> dict[str, Any]:
        record = _episode_record(row)
        diagnostics = record["control_diagnostics"]
        traces = record["traces"]
        cap = 1.0
        first_partial = None
        first_exact = None
        nonzero_controls = 0
        trigger = None
        for tick, (diagnostic, trace) in enumerate(zip(diagnostics, traces)):
            if int(diagnostic.get("handoff_viability_controls", 0)) <= 0:
                continue
            common = _wrapper_common(diagnostic)
            stage = int((common.get("completion") or {}).get("stage", -1))
            remaining = deadline_controls - tick
            reference_lag = _norm(trace.get("public_reference_lead")) or 0.0
            progress_scale = float(diagnostic.get("handoff_progress_scale", 1.0))
            deadline_scale = float(np.clip(
                (
                    remaining - v9_parameters["completion_reserve_controls"]
                ) / v9_parameters["reserve_taper_controls"],
                0.0, 1.0,
            ))
            lag_scale = float(np.clip(
                (
                    v9_parameters["reference_lag_zero_authority_m"]
                    - reference_lag
                ) / (
                    v9_parameters["reference_lag_zero_authority_m"]
                    - v9_parameters["reference_lag_full_authority_m"]
                ),
                0.0, 1.0,
            ))
            budget_controls = 0
            planar_remaining = 0.0
            budget_scale = 1.0
            speed_scale = 1.0
            progress_speed = max(
                0.0, float(common.get("achieved_speed_mps", 0.0)),
            )
            if stage == 1:
                position = np.asarray(trace["tcp_position"], dtype=float)[:2]
                goal = goal_position[:2]
                planar_remaining = max(
                    0.0, float(np.linalg.norm(goal - position))
                    - float(shared["terminal_position_tolerance_m"]),
                )
                traverse = math.ceil(
                    planar_remaining
                    / (
                        v9_parameters["completion_progress_floor_mps"]
                        * float(shared["control_dt_seconds"])
                    )
                )
                budget_controls = (
                    traverse + v9_parameters["post_traverse_reserve_controls"]
                )
                budget_scale = float(np.clip(
                    (remaining - budget_controls)
                    / v9_parameters["completion_budget_taper_controls"],
                    0.0, 1.0,
                ))
                speed_scale = float(np.clip(
                    (
                        v9_parameters["force_speed_zero_authority_mps"]
                        - progress_speed
                    ) / (
                        v9_parameters["force_speed_zero_authority_mps"]
                        - v9_parameters["force_speed_full_authority_mps"]
                    ),
                    0.0, 1.0,
                ))
            elif stage >= 2:
                budget_scale = 0.0
            target = min(
                deadline_scale, lag_scale, progress_scale,
                budget_scale, speed_scale,
            )
            cap = min(
                cap,
                max(
                    target,
                    cap - v9_parameters["maximum_authority_decay_per_control"],
                ),
                progress_scale,
            )
            if cap > 1e-7:
                nonzero_controls += 1
            if first_partial is None and cap < 1.0 - 1e-7:
                first_partial = tick
            if first_exact is None and cap <= 1e-7:
                cap = 0.0
                first_exact = tick
                trigger = {
                    "completion_stage": stage,
                    "remaining_controls": remaining,
                    "completion_budget_controls": budget_controls,
                    "planar_remaining_m": planar_remaining,
                    "progress_speed_mps": progress_speed,
                    "reference_lag_m": reference_lag,
                }
        old_exact = next((
            tick for tick, diagnostic in enumerate(diagnostics)
            if int(diagnostic.get("last_selector_code", -1)) == 2
        ), None)
        return {
            "record_id": row["record_id"],
            "outcome": row["outcome"],
            "success": bool(row["success"]),
            "v7_first_exact_common_control": old_exact,
            "v9_first_partial_control": first_partial,
            "v9_first_exact_common_control": first_exact,
            "v9_earlier_exact_common_controls": (
                None if old_exact is None or first_exact is None
                else old_exact - first_exact
            ),
            "v9_nonzero_authority_controls": nonzero_controls,
            "v9_trigger": trigger,
        }

    paired = []
    trigger_pairs = []
    for _, (row, profile) in sorted(representatives.items()):
        comparator = by_identity[(
            "T2/ablation/frozen_warm", row["instance"],
            row["request"]["request_id"],
        )]
        if not comparator["success"]:
            raise ValueError("v14 frozen-warm comparator is not successful")
        paired.append({
            "failure": profile,
            "frozen_warm_comparator": _v13_trace_profile(
                comparator, deadline_controls=deadline_controls,
                force_limit_N=float(shared["force_limit_N"]),
                lateral_force_limit_N=float(shared["lateral_force_limit_N"]),
            ),
        })
        trigger_pairs.append({
            "failure": v9_trigger(row),
            "frozen_warm_comparator": v9_trigger(comparator),
        })
    completion_trigger_rows = [
        pair["failure"] for pair in trigger_pairs
        if pair["failure"]["outcome"] == "timeout"
    ]
    delayed_force_trigger_rows = [
        pair["failure"] for pair in trigger_pairs
        if pair["failure"]["outcome"] == "overforce"
        and pair["failure"]["v7_first_exact_common_control"] is not None
    ]
    applicable_comparators = [
        pair["frozen_warm_comparator"] for pair in trigger_pairs
        if pair["failure"]["v7_first_exact_common_control"] is not None
    ]
    if not (
        len(completion_trigger_rows) == 5
        and all(
            row["v9_earlier_exact_common_controls"] >= 50
            for row in completion_trigger_rows
        )
        and len(delayed_force_trigger_rows) == 1
        and delayed_force_trigger_rows[0]["v9_earlier_exact_common_controls"] >= 100
        and all(row["v9_nonzero_authority_controls"] > 0 for row in applicable_comparators)
    ):
        raise ValueError("v9 CPU trigger-coverage gate failed")
    trigger_screen = {
        "schema_version": "t2-completion-budget-v9-trigger-screen-v1",
        "scope": "recorded-trajectory trigger coverage only; no counterfactual outcome claim",
        "candidate_interface": "public-T2-completion-budget-viability-v9",
        "parameters": v9_parameters,
        "passed": True,
        "checks": {
            "five_completion_failures_trigger_at_least_50_controls_earlier": True,
            "delayed_force_failure_triggers_at_least_100_controls_earlier": True,
            "matched_successes_retain_nonzero_candidate_authority": True,
            "mode0_ungated_failure_is_outside_v9_scope": True,
        },
        "pairs": trigger_pairs,
        "causal_limit": (
            "The screen evaluates only when the candidate would lose authority on recorded v14 "
            "states. A GPU physics replay is required to learn the resulting trajectory."
        ),
    }

    row_statistics = []
    for sealed in report["rows"]:
        selected = [row for row in outcomes if row["row_id"] == sealed["row_id"]]
        episodes = len(selected)
        successes = sum(bool(row.get("success")) for row in selected)
        row_statistics.append({
            "row_id": sealed["row_id"], "task": sealed["task"],
            "family": sealed["family"], "key": sealed["key"],
            "method_identity": sealed["method_identity"],
            "admission_status": sealed["admission_status"],
            "episodes": episodes, "successes": successes,
            "success_rate": successes / episodes if episodes else None,
            "constraint_violations": sum(
                bool(row.get("constraint_violation")) for row in selected
            ),
            "outcomes": dict(sorted(Counter(
                row.get("outcome") for row in selected
            ).items())),
        })

    attribution = {
        "schema_version": "t2-v14-rejected-development-attribution-v1",
        "scope": "CPU-only diagnosis of 712 individually audited episodes; no qualification or performance claim",
        "coverage": {
            "planned_jobs": 720, "completed_jobs": 720,
            "individually_audited_episode_evidence": 712,
            "public_sentinel_binding_errors_before_physics": 8,
            "comparable_method_and_ablation_episodes": 576,
            "failed_logical_episodes": 11,
            "unique_failed_physical_traces": 7,
        },
        "failure_classes": [
            {
                "class": "finite_time_completion_budget_shortfall",
                "logical_failures": len(settling),
                "unique_physical_traces": 5,
                "members": settling,
                "finding": (
                    "All members are mode-1 timeouts without force violations. V7 reaches exact "
                    "common authority with 150--181 controls left, but the common completion "
                    "stage and terminal dwell still do not finish by control 1000."
                ),
            },
            {
                "class": "delayed_force_debt_after_exact_common_handoff",
                "logical_failures": len(post_handoff_force),
                "unique_physical_traces": 1,
                "members": post_handoff_force,
                "finding": (
                    "The duplicated task-only trace violates the lateral limit 39 controls after "
                    "exact-common handoff. Exact common commands do not erase delayed reference, "
                    "velocity or contact-energy debt."
                ),
            },
            {
                "class": "mode0_preoutlet_unguarded_collision",
                "logical_failures": 1,
                "unique_physical_traces": 1,
                "members": preoutlet_ungated,
                "finding": (
                    "The ungated ablation violates force before outlet under full candidate "
                    "authority; the gated path passes the matched request. Preserve this failure."
                ),
            },
        ],
        "paired_unique_failures_against_frozen_warm": paired,
        "v9_cpu_trigger_screen": trigger_screen,
        "resolved_v13_failure_classes": {
            "matching_no_band": "64/64; prior delayed-impact failure removed",
            "band_only": "64/64; prior delayed-impact failure removed",
            "frozen_warm": "64/64",
        },
        "causal_limits": [
            "These recorded trajectories identify insufficient public-state conditions; they do not prove a candidate revision succeeds.",
            "The public sentinel binding failure invalidates whole-run qualification even though 712 episode files pass their own audits.",
            "Logical duplicates sharing an evidence hash count once as physical evidence.",
        ],
    }

    diagnosis = {
        "schema_version": "t2-v14-next-interface-diagnosis-plan-v1",
        "status": "bounded_v9_development_required",
        "qualification_decision": (
            "Reject v14 for P6. Repair and replay the eight public sentinels, then develop v9; "
            "Cross-EBDM and gated-full are 62/64 and therefore fail the paper gate."
        ),
        "confirmed_implementation_repair": {
            "scope": "task controller binding only",
            "change": (
                "Return public_geometry before inheriting method terminal-interface parameters; "
                "the sentinel remains an unwrapped public task reference."
            ),
            "solver_or_learning_change": False,
            "gpu1_validation": {
                "status": "passed_8_of_8_with_independent_integrity_audit",
                "path": str(sentinel_validation_path.relative_to(ROOT)),
                "sha256": sha256(sentinel_validation_path),
            },
        },
        "next_revision": {
            "candidate_interface": "public-T2-completion-budget-viability-v9",
            "principle": (
                "Replace the scalar deadline/lag cap with a conservative public-state completion "
                "budget. Candidate authority may only decrease after outlet and must reach zero "
                "before the lower bound on remaining traverse, descent, settling and dwell equals "
                "the remaining deadline. Add a force-debt condition over public reference lag, TCP "
                "twist and measured wrench."
            ),
            "decision_rule": (
                "At every post-outlet control, compute B_complete as remaining public geometric "
                "progress divided by a conservative filtered progress rate, plus declared descent, "
                "reference-settling and 15-control dwell budgets. Compute B_force from public "
                "reference lag, closing speed and wrench margin. Let slack equal remaining controls "
                "minus max(B_complete, B_force), map only positive slack to residual transport "
                "authority, and take a monotone nonincreasing envelope. Completion stage 2 or "
                "nonpositive slack selects exact common authority."
            ),
            "public_state": [
                "remaining deadline controls and terminal dwell debt",
                "common completion stage, stage residence and geometric progress",
                "finite-difference progress rate and TCP twist",
                "public reference/TCP lag and issued-reference debt",
                "measured total/lateral wrench margin",
            ],
            "must_preserve": [
                "matching-no-band 64/64 and band-only 64/64",
                "frozen-warm 64/64",
                "mode-0 gating and the ungated ablation definition",
                "task-owned coordinates, 20 s deadline and force limits",
            ],
            "must_not_do": [
                "Do not merely increase completion_reserve_controls.",
                "Do not condition on method identity, hidden delay, payload or servo gain.",
                "Do not repair FADA, DIAL or public-source-prior behavior through the shared wrapper.",
            ],
        },
        "bounded_validation": [
            {
                "order": 1, "device": "cpu",
                "work": "Screen the v9 trigger on the seven unique failed traces and their seven frozen-warm comparators.",
                "gate": "Cover all completion/force debts while preserving a nonempty authority interval on successful comparators; this is trigger coverage only.",
            },
            {
                "order": 2, "device": "gpu1",
                "work": "Replay the eight public sentinels after the binding repair.",
                "gate": "Require 8/8 strict success, zero execution errors and eight audited evidence files before testing v9.",
                "status": "completed_passed",
            },
            {
                "order": 3, "device": "gpu1",
                "work": "Run the seven unique v14 failures, matched frozen-warm comparators and fixed public sentinels under v9.",
                "gate": "Reject on any preserved-success regression, timeout, overforce, evidence gap or execution error.",
            },
            {
                "order": 4, "device": "gpu1",
                "work": "Only after the bounded gate, replay all 576 comparable rows and then use fresh seed-disjoint release/predeployment gates.",
                "gate": "Require Cross-EBDM 64/64, gated-full 64/64, public sentinel 8/8 and zero new violations before P6.",
            },
        ],
        "theory_revision": [
            "Handoff viability is an augmented-state finite-time reach-avoid condition, not a terminal pose or action-switch condition.",
            "The sufficient state must include completion-time debt and delayed force debt; exact common authority alone is insufficient.",
            "The authority schedule must be causal, method-agnostic and monotone after outlet.",
            "Baseline mechanism failures remain results once their source and interface contracts pass; the shared task wrapper must not make them competitive.",
            "Transport benefit remains unproven until a seed-disjoint performance comparison beats task-only and strong baselines.",
        ],
        "attribution": attribution,
    }
    artifacts = {
        name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
        for name, path in files.items()
    }
    artifacts["analysis_script"] = {
        "path": str(Path(__file__).resolve().relative_to(ROOT)),
        "sha256": sha256(Path(__file__).resolve()),
    }
    artifacts["public_sentinel_binding_regression_audit"] = {
        "path": str(sentinel_validation_path.relative_to(ROOT)),
        "sha256": sha256(sentinel_validation_path),
    }
    closeout = {
        "schema_version": "four-task-predeployment-offline-closeout-v2",
        "status": "rejected_infrastructure_and_behavior_development_evidence_only",
        "device": "cpu", "performance_evidence": False,
        "confirmation_states_opened": 0, "old_invalid_outcomes_used": False,
        "artifacts": artifacts, "counts": {
            **report["counts"],
            "successes": sum(bool(row.get("success")) for row in outcomes),
            "timeouts": sum(row.get("outcome") == "timeout" for row in outcomes),
            "constraint_violations": sum(
                bool(row.get("constraint_violation")) for row in outcomes
            ),
            "infrastructure_errors": len(infrastructure),
            "individually_audited_episode_evidence": len(evidence_rows),
        },
        "whole_run_independent_integrity_passed": False,
        "individually_audited_development_evidence": True,
        "public_geometry_sentinel": {
            "episodes": 8, "successes": 0, "passed": False,
            "failure": "terminal-interface parameter routing before physics",
            "binding_repair_validation": {
                "episodes": 8, "successes": 8, "passed": True,
                "independent_integrity_audit_passed": True,
            },
        },
        "row_statistics": row_statistics,
        "v14_failure_attribution": attribution,
        "interpretation_limits": attribution["causal_limits"],
    }

    output.mkdir(parents=True, exist_ok=False)
    (output / "closeout.json").write_text(
        json.dumps(closeout, indent=2, sort_keys=True) + "\n"
    )
    (output / "diagnosis_plan.json").write_text(
        json.dumps(diagnosis, indent=2, sort_keys=True) + "\n"
    )
    (output / "v9_trigger_screen.json").write_text(
        json.dumps(trigger_screen, indent=2, sort_keys=True) + "\n"
    )
    with (output / "row_statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "row_id", "task", "family", "key", "method_identity",
            "admission_status", "episodes", "successes", "success_rate",
            "constraint_violations", "outcomes",
        ])
        writer.writeheader()
        for row in row_statistics:
            writer.writerow({
                **{key: row[key] for key in writer.fieldnames if key != "outcomes"},
                "outcomes": json.dumps(row["outcomes"], sort_keys=True),
            })
    manifest = {
        "schema_version": "derived-artifact-manifest-v1",
        "files": {
            path.name: sha256(path) for path in sorted(output.iterdir())
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    try:
        output_label = str(output.relative_to(ROOT))
    except ValueError:
        output_label = str(output)
    print(json.dumps({
        "output": output_label,
        "status": closeout["status"],
        "audited_development_episodes": len(evidence_rows),
        "unique_failed_physical_traces": 7,
    }, indent=2, sort_keys=True))


def seal_terminal_candidate_preflights() -> dict[str, dict[str, str]]:
    """Seal the two bounded terminal-selector preflights as rejected candidates."""
    candidates = {
        "terminal_transport_v5": (
            "terminal_transport_exact_replay",
            "public-T2-clearance-terminal-transport-v5",
            0.0021206531768598134,
        ),
        "terminal_strict_progress_v6": (
            "terminal_strict_progress_exact_replay",
            "public-T2-clearance-terminal-strict-progress-v6",
            0.002014132198842158,
        ),
    }
    sealed = {}
    for label, (directory_name, version, expected_error) in candidates.items():
        directory = (
            ROOT / "results/arm/guided_contact_alignment/validation" / directory_name
        )
        files = {
            "plan": directory / "plan.json",
            "progress": directory / "progress.jsonl",
            "report": directory / "report.json",
        }
        for path in files.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        report = json.loads(files["report"].read_text())
        rows = [json.loads(line) for line in files["progress"].read_text().splitlines()]
        if not (
            report.get("diagnostic_only") is True
            and report.get("passed") is False
            and report["counts"]["completed_physical_jobs"] == 2
            and report["counts"]["execution_errors"] == 0
            and len(rows) == 2
            and [row["outcome"] for row in rows] == ["timeout", "success"]
            and all(row["evidence"]["audit_passed"] for row in rows)
            and math.isclose(
                rows[0]["episode_metrics"]["final_position_error_m"],
                expected_error, rel_tol=0., abs_tol=1e-15,
            )
        ):
            raise ValueError(f"bounded preflight changed: {directory_name}")
        marker = directory / "REJECTED_CANDIDATE.json"
        payload = {
            "schema_version": "t2-terminal-interface-rejection-v1",
            "status": "rejected_after_bounded_exact_preflight",
            "candidate": version,
            "device": "cpu",
            "diagnostic_only": True,
            "completed_physical_jobs": 2,
            "planned_physical_jobs": report["counts"]["planned_physical_jobs"],
            "full_plan_cancelled": True,
            "reason": (
                "The frozen regression request remains a timeout, so the candidate cannot "
                "satisfy the preserve-success gate. The successful companion request does not "
                "override that regression."
            ),
            "regression": {
                "record_id": rows[0]["record_id"],
                "outcome": rows[0]["outcome"],
                "final_position_error_m": rows[0]["episode_metrics"][
                    "final_position_error_m"
                ],
                "evidence": rows[0]["evidence"],
            },
            "companion": {
                "record_id": rows[1]["record_id"],
                "outcome": rows[1]["outcome"],
                "evidence": rows[1]["evidence"],
            },
            "artifacts": {
                name: {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                }
                for name, path in files.items()
            },
            "confirmation_states_opened": 0,
            "performance_claim": False,
        }
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if marker.exists() and marker.read_text() != serialized:
            raise ValueError(f"existing rejection marker differs: {marker}")
        marker.write_text(serialized)
        sealed[label] = {
            "path": str(marker.relative_to(ROOT)),
            "sha256": sha256(marker),
        }
    return sealed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument(
        "--paired-diagnostic", type=Path,
        help="exact recorded candidate directory for a CPU-only paired closeout",
    )
    parser.add_argument(
        "--terminal-admission-diagnostic", type=Path,
        help="early-stopped v4 directory for a CPU-only terminal admission closeout",
    )
    parser.add_argument(
        "--seal-terminal-preflights", action="store_true",
        help="seal the bounded v5/v6 terminal-selector preflights as rejected",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.terminal_admission_diagnostic is not None:
        if args.paired_diagnostic is None:
            parser.error("--terminal-admission-diagnostic requires --paired-diagnostic")
        terminal_set_admission_closeout(
            args.qualification,
            args.paired_diagnostic,
            args.terminal_admission_diagnostic,
            args.output,
        )
        return
    if args.paired_diagnostic is not None:
        paired_terminal_authority_closeout(
            args.qualification, args.paired_diagnostic, args.output,
        )
        return

    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "-1"}:
        raise RuntimeError("offline closeout requires CUDA_VISIBLE_DEVICES='' or -1")
    sealed_preflights = (
        seal_terminal_candidate_preflights()
        if args.seal_terminal_preflights else {}
    )
    qualification = args.qualification.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"derived output already exists: {output}")

    files = {
        name: qualification / filename
        for name, filename in {
            "plan": "plan.json",
            "progress": "progress.jsonl",
            "report": "report.json",
            "independent_audit": "independent_audit.json",
        }.items()
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    plan = json.loads(files["plan"].read_text())
    report = json.loads(files["report"].read_text())
    audit = json.loads(files["independent_audit"].read_text())
    outcomes = [json.loads(line) for line in files["progress"].read_text().splitlines()]

    if (
        report.get("schema_version") == "four-task-behavioral-predeployment-v14"
        and report.get("passed") is False
    ):
        v14_rejected_predeployment_closeout(
            qualification=qualification, output=output, plan=plan,
            report=report, audit=audit, outcomes=outcomes, files=files,
        )
        return

    if report.get("passed") is not True or report.get("formal_evaluation_allowed") is not True:
        raise ValueError("qualification report is not sealed as passed/admitted")
    if audit.get("passed") is not True:
        raise ValueError("independent qualification audit did not pass")
    if len(outcomes) != int(report["counts"]["completed_physical_jobs"]):
        raise ValueError("journal length differs from sealed report")
    if len(outcomes) != int(plan["counts"]["physical_jobs"]):
        raise ValueError("journal length differs from frozen plan")
    if any(not row.get("evidence", {}).get("audit_passed") for row in outcomes):
        raise ValueError("journal contains unaudited evidence")
    if any(row.get("request", {}).get("performance_evidence") for row in outcomes):
        raise ValueError("predeployment journal contains performance evidence")

    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        by_row[row["row_id"]].append(row)

    row_statistics = []
    for sealed in report["rows"]:
        rows = by_row.get(sealed["row_id"], [])
        successes = sum(bool(row["success"]) for row in rows)
        modes: dict[str, Any] = {}
        for mode in (0, 1):
            selected = [row for row in rows if int(row["request"]["requested_mode"]) == mode]
            count = len(selected)
            success_count = sum(bool(row["success"]) for row in selected)
            modes[str(mode)] = {
                "episodes": count,
                "successes": success_count,
                "success_rate": success_count / count if count else None,
                "outcomes": dict(sorted(Counter(row["outcome"] for row in selected).items())),
                "wilson95_descriptive": wilson(success_count, count),
            }
        row_statistics.append({
            "row_id": sealed["row_id"],
            "task": sealed["task"],
            "family": sealed["family"],
            "key": sealed["key"],
            "method_identity": sealed["method_identity"],
            "admission_status": sealed["admission_status"],
            "episodes": len(rows),
            "successes": successes,
            "success_rate": successes / len(rows) if rows else None,
            "constraint_violations": sum(bool(row["constraint_violation"]) for row in rows),
            "outcomes": dict(sorted(Counter(row["outcome"] for row in rows).items())),
            "requested_mode": modes,
            "wilson95_descriptive": wilson(successes, len(rows)),
        })

    failure_cells: dict[tuple[str, str, str, str, int], int] = Counter()
    for row in outcomes:
        if row["outcome"] == "success":
            continue
        failure_cells[
            row["task"], row["family"], row["key"], row["outcome"],
            int(row["request"]["requested_mode"]),
        ] += 1

    timeouts = [row for row in outcomes if row["outcome"] == "timeout"]
    overforces = [row for row in outcomes if row["outcome"] == "overforce"]
    contact_contract = [
        row for row in outcomes if row["outcome"] in {
            "guide_contact_not_established_before_outlet", "guide_contact_loss",
        }
    ]
    violations = [row for row in outcomes if bool(row["constraint_violation"])]
    failures = [row for row in outcomes if row["outcome"] != "success"]
    classified = timeouts + overforces + contact_contract
    if (len(classified) != len(failures)
            or {row["record_id"] for row in classified}
            != {row["record_id"] for row in failures}):
        raise ValueError("fresh failures are not exhausted by the frozen three classes")

    def terminal_handoff_signature(row: dict[str, Any]) -> dict[str, Any] | None:
        pointer = row.get("evidence", {})
        path = ROOT / pointer.get("path", "")
        if not path.is_file() or sha256(path) != pointer.get("sha256"):
            raise ValueError(f"missing or changed episode evidence: {row['record_id']}")
        evidence = json.loads(gzip.decompress(path.read_bytes()))
        records = evidence.get("records", [])
        if len(records) != 1:
            raise ValueError(f"episode evidence cardinality differs: {row['record_id']}")
        diagnostics = records[0].get("control_diagnostics", [])
        if not diagnostics:
            return None
        base = _wrapper_common(diagnostics[-1])
        completion = base.get("completion", {})
        route = base.get("route", {})
        if not (
            base.get("terminal_pose_active") is True
            and int(base.get("stage", -1)) == 3
            and int(completion.get("stage", -1)) == 2
        ):
            return None
        return {
            "record_id": row["record_id"],
            "row_id": row["row_id"],
            "outcome": row["outcome"],
            "instance": row["instance"],
            "request_id": row["request"]["request_id"],
            "requested_mode": int(row["request"]["requested_mode"]),
            "force_peak_N": float(row["force_peak_N"]),
            "lateral_force_peak_N": float(row["lateral_force_peak_N"]),
            "route_stage": route.get("stage"),
            "completion_stage": completion.get("stage"),
            "terminal_distance_m": base.get("terminal_distance_m"),
        }

    terminal_handoff_failures = [
        signature for row in failures
        if (signature := terminal_handoff_signature(row)) is not None
    ]

    def failure_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "episodes": len(rows),
            "unique_public_requests": len({
                row["request"]["request_id"] for row in rows
            }),
            "rows": dict(sorted(Counter(row["row_id"] for row in rows).items())),
            "tasks": dict(sorted(Counter(row["task"] for row in rows).items())),
            "methods": dict(sorted(Counter(row["key"] for row in rows).items())),
            "requested_modes": dict(sorted(Counter(
                str(row["request"]["requested_mode"]) for row in rows
            ).items())),
        }
    failure_classes = [
        {
            "class": "finite_time_progress_failure",
            **failure_summary(timeouts),
            "selection": "outcome == timeout",
            "diagnostic_observables": [
                "phase/stage occupancy", "issued and applied task-space action",
                "accepted public-reference progress", "terminal position/orientation error",
                "mode match and hold count",
            ],
        },
        {
            "class": "force_safety_failure",
            **failure_summary(overforces),
            "selection": "outcome == overforce",
            "diagnostic_observables": [
                "last safe and first violating physics substep", "normal/lateral force margin",
                "issued/applied request delta", "accepted reference and governor stage",
                "candidate/reference delta and optimizer or selector decision",
            ],
        },
        {
            "class": "guide_contact_continuity_failure",
            **failure_summary(contact_contract),
            "selection": (
                "outcome in {guide_contact_not_established_before_outlet, "
                "guide_contact_loss}"
            ),
            "diagnostic_observables": [
                "first contact and outlet substeps", "pre-outlet axial-support coverage",
                "longest continuous contact loss", "mode-specific crossing coordinate",
                "task-owned request equality",
            ],
        },
    ]

    sentinel = [row for row in outcomes if row["key"] == "public_geometry"]
    source_checks = source_contract_checks()
    row_lookup = {row["row_id"]: row for row in row_statistics}
    t2_ours = row_lookup["T2/main/cross_ebdm"]
    t2_gated = row_lookup["T2/ablation/gated_full"]
    final_actor_v8 = (
        report.get("schema_version") == "four-task-behavioral-predeployment-v8"
    )
    outlet_latched_v12 = (
        report.get("schema_version") == "four-task-behavioral-predeployment-v12"
    )
    task_contract_v5_v13 = (
        report.get("schema_version") == "four-task-behavioral-predeployment-v13"
    )
    v13_attribution = None
    v13_profiles: list[dict[str, Any]] = []
    if task_contract_v5_v13:
        v13_attribution, v13_profiles = v13_shared_failure_attribution(outcomes)

    supplemental_artifacts: dict[str, dict[str, str]] = {}
    joint_report_path = ROOT / (
        "results/arm/guided_contact_alignment/training/"
        "joint_admission_models_v2/report.json"
    )
    guard_replay_path = ROOT / (
        "results/arm/guided_contact_alignment/validation/contact_guard_cross_ebdm/"
        "exact_v12_failure_reproduction_v2/episodes/"
        "a60c4fad2f49c3fe064be7eae220e884204891a2a84aff9de70a38a062a32ce9/"
        "result.json"
    )
    joint_summary: dict[str, Any] | None = None
    if joint_report_path.is_file():
        joint_report = json.loads(joint_report_path.read_text())
        failed_gates = Counter(
            gate
            for instance in joint_report.get("instances", {}).values()
            for gate, passed in instance.get("gates", {}).items()
            if not passed
        )
        joint_summary = {
            "status": joint_report.get("status"),
            "passed_instances": sum(
                bool(instance.get("passed"))
                for instance in joint_report.get("instances", {}).values()
            ),
            "instances": len(joint_report.get("instances", {})),
            "failed_gate_counts": dict(sorted(failed_gates.items())),
            "eligible_for_binding": False,
        }
        supplemental_artifacts["failed_joint_admission_report"] = {
            "path": str(joint_report_path.relative_to(ROOT)),
            "sha256": sha256(joint_report_path),
        }
    if guard_replay_path.is_file():
        supplemental_artifacts["contact_guard_exact_replay"] = {
            "path": str(guard_replay_path.relative_to(ROOT)),
            "sha256": sha256(guard_replay_path),
        }
    supplemental_artifacts.update(sealed_preflights)

    failing_contact_rows = sorted({row["row_id"] for row in contact_contract})
    if task_contract_v5_v13:
        bounded_cpu_stages = [
            {
                "order": 1,
                "name": "public augmented-state handoff debt",
                "work": (
                    "Replace the clearance-distance-only authority schedule with a causal, "
                    "monotone mode-1 handoff margin over remaining deadline, public reference "
                    "lag, TCP twist, measured wrench, terminal dwell and the delayed executor "
                    "state. Candidate residual authority may only decrease after outlet."
                ),
                "gate": (
                    "The offline screen must cover all five unique failed mode-1 traces while "
                    "retaining nonzero candidate authority on successful development traces. "
                    "This is trigger coverage only and cannot claim repaired outcomes."
                ),
            },
            {
                "order": 2,
                "name": "bounded exact-request counterfactual",
                "work": (
                    "Run the six unique failure/comparator requests: one Cross/gated convergence "
                    "trace, two response-MPC late handoffs, two delayed impacts, their task-only "
                    "comparators, and fixed public sentinels. Preserve every success and require "
                    "zero force violations before expanding."
                ),
                "gate": (
                    "Reject on the first success regression, timeout, overforce, incomplete trace "
                    "or evidence-audit failure. Do not substitute a different reset."
                ),
            },
            {
                "order": 3,
                "name": "complete exact v13 replay",
                "work": (
                    "Only after the bounded gate, replay all 576 comparable learned/model episodes "
                    "and the eight public sentinels with the frozen v13 resets."
                ),
                "gate": (
                    "Require all previously successful episodes to remain successful and all shared "
                    "mode-1 failures to be removed, with zero new constraint violations."
                ),
            },
            {
                "order": 4,
                "name": "fresh seed-disjoint release and predeployment",
                "work": (
                    "Freeze a new task contract only after exact replay, then run a seed-disjoint "
                    "public/Cross gate followed by a fresh full behavioral predeployment."
                ),
                "gate": (
                    "P6 stays blocked until Cross-EBDM is 64/64, the public sentinel is 8/8, all "
                    "evidence audits pass and package/device audits bind the exact manifest."
                ),
            },
        ]
        next_revision = {
            "candidate_interface": "public-T2-handoff-viability-v7",
            "candidate_contract": "cross-embodiment-t2-shared-execution-v6",
            "authority_domain": (
                "Mode 1 only. Start viability accounting at outlet; keep task-owned coordinates "
                "exactly common, admit only transport-coordinate residuals, make authority "
                "monotone nonincreasing, and issue exact common actions after the viability margin "
                "is exhausted or completion stage 2 begins."
            ),
            "state": [
                "remaining controls to the frozen 20 s deadline",
                "15-control terminal dwell reserve",
                "public reference minus TCP pose",
                "TCP twist and measured wrench",
                "common completion stage and terminal-set distance",
                "issued/applied reference debt represented through the allowed public reference",
            ],
            "decision_rule": (
                "Use a deterministic augmented-state handoff-debt bound. It must reserve enough "
                "controls for common completion and dwell and must brake transport residuals before "
                "the reference/TCP lag can produce delayed recontact. Do not choose terminal "
                "residuals with another one-step pose proxy."
            ),
            "unchanged": [
                "1 ms physics, 20 ms control and 40 ms contact dynamics",
                "20 s deadline, 25 N total and 10 N lateral limits",
                "modes, reset cells, evaluator and common route/controller",
                "mode-0 Cross-EBDM contact guard and the ungated ablation definition",
            ],
            "explicit_nonchanges": (
                "Do not repair ungated-full: its mode-0 pre-outlet collision is an ablation-specific "
                "failure and the matched gated actor succeeds. Do not modify FADA, DIAL or source "
                "prior mechanisms in this shared-handoff revision."
            ),
            "threshold_policy": (
                "Treat the rejected v13 data as development evidence. Derive an initial reserve from "
                "the public/task-only successful handoff envelope, record it in a task-local config, "
                "and require a fresh seed-disjoint gate; never tune on the fresh gate."
            ),
        }
        theory_revision = [
            "Replace terminal-action dominance with augmented-state handoff viability: exact common actions do not erase delayed reference, velocity or contact state.",
            "Add a finite-time completion debt containing approach, reference settling and terminal dwell; eventual common authority is insufficient under a deadline.",
            "State the handoff condition over allowed public observations and a monotone authority schedule, without conditioning on delay, payload, servo gain or method identity.",
            "Separate shared mode-1 compatibility failures from the mode-0 ungated ablation failure; the latter supports the gating mechanism and should remain in the denominator.",
            "Keep transport benefit unclaimed until seed-disjoint formal performance beats task-only and strong baselines under the same released interface.",
        ]
        formal_evaluation_decision = (
            "Reject task contract v5 for P6: Cross-EBDM is 63/64 and the same trace fails gated-full. "
            "The runner's sentinel-only admission flag is insufficient for the paper gate. Build and "
            "qualify the bounded v7 handoff-viability interface before any formal performance roster."
        )
        completed_cpu_stages = [
            "The fresh 720-episode v13 run and all 720 evidence records pass independent integrity audit.",
            "The 576 comparable method/ablation traces are paired by instance and public request.",
            "Three shared mode-1 failure mechanisms contain six logical failures and five unique physical traces.",
            "The sole mode-0 ungated collision is isolated as an ablation-specific failure.",
            "The bounded v5 and v6 terminal selector candidates are sealed as rejected." if sealed_preflights else "The bounded v5/v6 preflight evidence is available but rejection markers were not requested.",
        ]
    elif outlet_latched_v12:
        terminal_rows = dict(sorted(Counter(
            row["row_id"] for row in terminal_handoff_failures
        ).items()))
        bounded_cpu_stages = [
            {
                "order": 1,
                "name": "terminal handoff ownership attribution",
                "work": (
                    "Replay every v12 learned-policy timeout or overforce whose final audited "
                    "state is route stage 3, completion stage 2 and terminal_pose_active. Compare "
                    "the unchanged candidate with a diagnostic profile that returns the exact "
                    "public common action only after terminal capture starts."
                ),
                "gate": (
                    "Admit a common task-interface revision only if it preserves every paired v12 "
                    "success, removes every terminal-handoff failure on the exact requests, and "
                    "then passes a fresh-seed public and learned-actor gate. Do not retrain first."
                ),
            },
            {
                "order": 2,
                "name": "FADA source-to-closed-loop attribution",
                "work": (
                    "For each frozen FADA instance, verify source fixture reconstruction, feature "
                    "normalization, decoder and Box semantics, then align the first target "
                    "closed-loop divergence with the first unsafe physics substep."
                ),
                "gate": (
                    "Repair only a reproducible fixture, normalization, decoder or adapter defect. "
                    "A correctly decoded early collision remains a FADA limitation on T2."
                ),
            },
            {
                "order": 3,
                "name": "DIAL and direct-prior finite-progress attribution",
                "work": (
                    "Align phase occupancy, candidate-to-accepted-reference displacement, optimizer "
                    "objective reduction and remaining distance per remaining second against the "
                    "successful public geometry request with the same reset."
                ),
                "gate": (
                    "Repair only objective, unit or projection plumbing. Internally consistent "
                    "stagnation under the frozen deadline remains a baseline limitation."
                ),
            },
        ]
        next_revision = {
            "shared_protocol": (
                "Keep controller-v50, route-v4, force limits, deadlines, modes, resets and the "
                "evaluator fixed while running the terminal-handoff counterfactual."
            ),
            "task_completion_ownership": (
                "Test one common lifecycle rule: after public terminal capture starts, the task "
                "completion controller owns the full request and every method returns the exact "
                "public common action. If the paired gate passes, version this once in the T2 task "
                "controller and apply it uniformly to ours, ablations and baselines."
            ),
            "cross_ebdm_training": (
                "Do not retrain yet. The v12 failures shared with task-only occur after transport's "
                "intended contact-transfer interval; first remove post-handoff method authority. "
                "Retrain only if fresh failures remain while method authority is legitimately active."
            ),
            "external_baselines": (
                "Keep FADA and DIAL mechanisms unchanged until their source/optimizer audits finish. "
                "Preserve a clean collision or stagnation as measured baseline behavior."
            ),
            "formal_evaluation": (
                "Keep P6 formal performance execution blocked until the main Cross-EBDM row passes "
                "a fresh full behavior qualification and the package/device audits are regenerated."
            ),
        }
        theory_revision = [
            "Define a finite authority interval for transported actions; terminal completion is a task-owned invariant regime.",
            "Separate transport quality, finite-time completion, contact continuity and force safety in the claims and metrics.",
            "Treat behavior qualification as compatibility evidence only; v12 still provides no task-performance advantage for transport.",
            "Attribute a baseline failure only after source semantics and task projection are audited; do not improve a baseline by attaching an ours-only recovery rule.",
            "Require seed-disjoint Cross-EBDM versus task-only comparisons after the common lifecycle contract is frozen.",
        ]
        formal_evaluation_decision = (
            "The public infrastructure gate passes if its sentinel remains 8/8, but the main "
            f"Cross-EBDM actor is {t2_ours['successes']}/{t2_ours['episodes']} and therefore is not "
            "ready for P6 performance sampling. Complete the bounded terminal-handoff diagnosis, "
            "freeze the resulting common contract, and rerun behavior qualification first."
        )
        completed_cpu_stages = [
            "All four static T2 adapter/interface checks pass.",
            (
                f"The fresh {len(outcomes)}-episode T2 qualification is complete and independently "
                "audited with no confirmation-state access."
            ),
            (
                f"The public geometry sentinel passes {sum(r['success'] for r in sentinel)}/"
                f"{len(sentinel)} under controller-v50 and route-v4."
            ),
            (
                f"{len(terminal_handoff_failures)} non-success episodes share the audited terminal "
                f"handoff signature across {terminal_rows}."
            ),
            "The outcome-level timeout, force-safety and guide-contact classes are mutually exclusive and exhaustive.",
        ]
    elif final_actor_v8:
        bounded_cpu_stages = [
            {
                "order": 1,
                "name": "FADA source-to-closed-loop attribution",
                "work": (
                    "For each frozen FADA instance, verify teacher-forced source error, "
                    "normalization, decoder and Box semantics, then locate the first target "
                    "closed-loop divergence and first unsafe substep when present."
                ),
                "gate": (
                    "Repair only a reproducible fixture, normalization, decoder, objective or "
                    "adapter defect. Otherwise retain the timeout or overforce as a method limitation."
                ),
            },
            {
                "order": 2,
                "name": "DIAL finite-progress attribution",
                "work": (
                    "For the frozen T2 failures, align optimizer objective reduction, phase "
                    "occupancy, candidate-to-accepted-reference displacement and remaining "
                    "distance per remaining second with the paired public executor."
                ),
                "gate": (
                    "Fix only objective or projection plumbing. Internally consistent stagnation "
                    "under the frozen public executor remains a DIAL limitation."
                ),
            },
            {
                "order": 3,
                "name": "response-MPC force attribution",
                "work": (
                    "For every T2 response-MPC overforce, align predicted force margins, optimizer "
                    "acceptance, fallback state and candidate/reference deltas with the last safe "
                    "and first violating 1 ms physics substeps."
                ),
                "gate": (
                    "Repair indexing, units or objective plumbing only. Out-of-support prediction "
                    "after a clean audit remains a baseline limitation."
                ),
            },
            {
                "order": 4,
                "name": "direct source-prior semantics",
                "work": (
                    "Verify unit, frame, mode and Box semantics for T1 and T2 and compare issued, "
                    "accepted and applied references against public geometry on the exact failing cells."
                ),
                "gate": (
                    "No target fitting is allowed. Failure after a clean schema audit is the intended "
                    "no-target-learning result."
                ),
            },
        ]
        next_revision = {
            "shared_protocol": (
                "unchanged: the 16/16 public sentinel and zero contact-contract failures provide "
                "no evidence for changing deadlines, limits, modes, routes, resets or evaluators"
            ),
            "cross_ebdm": (
                "freeze the final actor and completion projection while confirmation_v8 is active; "
                "make no behavior change until the complete seed-disjoint paired results are audited"
            ),
            "external_baselines": (
                "keep each native mechanism and the common executor unchanged; apply only a proven "
                "implementation repair and otherwise retain its qualified failure"
            ),
            "ablation_interpretation": (
                "matching_no_band has one isolated T2 overforce; retain it in that ablation's "
                "denominator and do not widen the common force contract"
            ),
            "formal_runner": (
                "treat post-reset runner interruptions as infrastructure failures, never algorithm "
                "failures; retain them without replay in v8 and improve receipt-to-outcome durability "
                "only in a later protocol version"
            ),
        }
        theory_revision = [
            "Keep shared executor feasibility, candidate transport and action admission as separate layers.",
            "Treat force safety, guide-contact continuity and finite-time completion as independent properties; the zero contact-failure count is a regression result, not a safety proof.",
            "A behavior-qualification pass establishes deployment compatibility only and is not paper performance evidence.",
            "Attribute transport only from complete seed-disjoint paired formal comparisons against task-only and frozen ablations under the same executor and admission rules.",
            "Exclude retained runner interruptions from mechanism claims and report them as infrastructure outcomes in their original denominators.",
        ]
        formal_evaluation_decision = (
            "The exact final-actor qualification and independent audit admit the already sealed "
            "confirmation_v8 package. While that launch is active, do not regenerate its roster, "
            "change actor bindings or use partial outcomes for a next-version method decision."
        )
        completed_cpu_stages = [
            "All four static T2 adapter/interface checks pass.",
            "The final-actor 1,440-episode qualification is complete and independently audited.",
            (
                f"T2 Cross-EBDM and gated-full pass {t2_ours['successes']}/"
                f"{t2_ours['episodes']} and {t2_gated['successes']}/"
                f"{t2_gated['episodes']} under the frozen public executor."
            ),
            "The three failure classes are mutually exclusive and exhaustive; guide-contact continuity has zero events and remains a hard regression gate.",
        ]
    else:
        bounded_cpu_stages = [
            {
                "order": 1,
                "name": "FADA source-to-closed-loop attribution",
                "work": (
                    "For each frozen FADA instance, persist teacher-forced source error, Box and "
                    "normalization checks, first target closed-loop divergence tick, and the first "
                    "unsafe substep when present."
                ),
                "gate": (
                    "Repair only a reproducible source fixture, normalization, decoder or adapter "
                    "defect; otherwise retain the timeout/overforce outcomes."
                ),
            },
            {
                "order": 2,
                "name": "DIAL finite-progress attribution",
                "work": (
                    "On four fresh paired T2 development requests, record objective reduction, phase "
                    "occupancy, accepted-reference displacement and remaining distance per remaining "
                    "second against the public geometry actor."
                ),
                "gate": (
                    "Fix only objective or projection plumbing. If the optimizer is internally "
                    "consistent but stagnates under contact, retain timeout as a mechanism limitation."
                ),
            },
            {
                "order": 3,
                "name": "response MPC force-model attribution",
                "work": (
                    "For every T2 response-MPC failure, align predicted force margins with the last "
                    "safe and first violating 1 ms substep, optimizer acceptance and fallback state."
                ),
                "gate": (
                    "Repair indexing, units or objective plumbing only. Out-of-support force prediction "
                    "after a clean audit remains a baseline limitation."
                ),
            },
            {
                "order": 4,
                "name": "contact-guard scope attribution",
                "work": (
                    "Compare guard activation windows, contact-loss intervals and candidate/base request "
                    "deltas for Cross-EBDM, gated-full, task-only and band-only on the exact failing cells."
                ),
                "gate": (
                    "The guard remains frozen if Cross-EBDM/gated-full stay qualified. Residual ablation "
                    "failures are not repaired by widening the guard or weakening the contract."
                ),
            },
        ]
        next_revision = {
            "shared_protocol": "unchanged",
            "ours_T2": (
                "Freeze public_measured_contact_guard_v1 for Cross-EBDM and the matched ablation "
                "interface; do not add another behavior change before independent evaluation."
            ),
            "joint_admission_v2": joint_summary or {
                "status": "artifact_missing",
                "eligible_for_binding": False,
            },
            "external_baselines": (
                "Keep native mechanisms and the common executor unchanged. Apply only a proven "
                "implementation repair; otherwise carry the qualified failure into the comparison."
            ),
            "ablation_interpretation": (
                "Treat guard, transport, matching and band as separate mechanisms. A guarded ablation "
                "can still fail force or contact, so no single component may be claimed as a safety "
                "certificate."
            ),
        }
        theory_revision = [
            "Factor the deployed policy into shared executor, candidate generator, public contact-continuity guard and optional statistical risk admission.",
            "State the contact guard as a reactive invariance heuristic driven by public wrench history; do not call it a calibrated safety certificate.",
            "State substep force safety and guide-contact continuity as distinct constraints because v13 observes each without the other.",
            "The failed joint model shows that marginal block coverage plus high-risk recall can coexist with unusable safe rejection; theory must include utility or nonempty-progress conditions.",
            "A main-method qualification pass establishes deployment compatibility only. Transport benefit still requires a seed-disjoint matched comparison against task-only and frozen ablations.",
        ]
        formal_evaluation_decision = (
            "The fresh qualification and independent audit admit formal evaluation under the frozen "
            "protocol. Before opening a new roster, rerun package/device gates against the exact guarded "
            "manifest. Do not bind the failed joint-admission checkpoints or reuse any invalid outcome."
        )
        completed_cpu_stages = [
            "All four static T2 adapter/interface checks pass.",
            "The fresh 1,440-episode qualification is complete and independently audited.",
            (
                f"T2 Cross-EBDM and gated-full both pass {t2_ours['successes']}/"
                f"{t2_ours['episodes']} and {t2_gated['successes']}/{t2_gated['episodes']} "
                "under the public-measured contact guard."
            ),
        ]

    diagnosis_plan = {
        "schema_version": "four-task-method-failure-diagnosis-plan-v2",
        "scope": "CPU development diagnosis only; no confirmation state and no performance claim",
        "shared_contract_decision": {
            "change_required": task_contract_v5_v13,
            "reason": (
                "The v5 terminal actor interface is rejected because the same mode-1 handoff "
                "boundary produces finite-time and delayed-impact failures across Cross-EBDM, "
                "response MPC and two ablations. The public sentinel still passes, so the frozen "
                "plant, evaluator, force limits, deadline and common route remain unchanged."
                if task_contract_v5_v13 else
                f"The common public-geometry sentinel passed {sum(r['success'] for r in sentinel)}/"
                f"{len(sentinel)} episodes, infrastructure errors are zero, and every fresh failure "
                "is confined to a method or ablation row. The contact failures occur in "
                f"{failing_contact_rows}; none occurs in the public sentinel."
            ),
            "frozen": [
                "T2 1 ms physics / 20 ms control / 40 ms contact unit and task-owned coordinates",
                "T3 15 mm insertion-only lead and safe-settled release",
                "T4 folded reset, joint-margin executor and public planar body intent",
                "all force limits, deadlines, mode definitions and evaluators",
            ],
        },
        "common_mechanism_diagnosis": {
            "terminal_handoff_signature": {
                "episodes": len(terminal_handoff_failures),
                "rows": dict(sorted(Counter(
                    row["row_id"] for row in terminal_handoff_failures
                ).items())),
                "outcomes": dict(sorted(Counter(
                    row["outcome"] for row in terminal_handoff_failures
                ).items())),
                "requests": len({
                    row["request_id"] for row in terminal_handoff_failures
                }),
                "requested_modes": dict(sorted(Counter(
                    str(row["requested_mode"]) for row in terminal_handoff_failures
                ).items())),
                "request_ids": dict(sorted(Counter(
                    row["request_id"] for row in terminal_handoff_failures
                ).items())),
                "selection": (
                    "non-success with final audited route stage 3, completion stage 2 "
                    "and terminal_pose_active == true"
                ),
                "episodes_detail": terminal_handoff_failures,
            },
            "interpretation": (
                "This signature localizes a shared lifecycle boundary across learned or optimized "
                "actors. It does not by itself prove that the common terminal-handoff revision will "
                "remove the failure; the exact paired counterfactual is the decision experiment."
            ),
            "v13_shared_failure_attribution": v13_attribution,
        },
        "three_class_decision": {
            "classes_are_mutually_exclusive_and_exhaustive": True,
            "failed_logical_episodes": len(failures),
            "classified_logical_episodes": len(classified),
            "finite_time_progress": {
                **failure_summary(timeouts),
                "diagnosis": (
                    "The unchanged public sentinel completes, while DIAL, FADA and direct source "
                    "prior time out in method-specific rows. Attribute by phase occupancy and accepted "
                    "progress; do not extend the common deadline."
                ),
                "revision_scope": (
                    "Repair only a demonstrated fixture, normalization, decoder, objective or adapter "
                    "defect. Retain clean closed-loop stagnation as a baseline limitation."
                ),
            },
            "force_safety": {
                **failure_summary(overforces),
                "diagnosis": (
                    "FADA, response MPC, direct source prior and any learned ablation failures must "
                    "be separated by their first unsafe substep and candidate/reference history; "
                    "a shared outcome label alone does not establish a shared mechanism."
                ),
                "revision_scope": (
                    "Keep frozen limits and the shared executor. Do not attach an ours-only safety "
                    "head to an external baseline under its original name. Preserve failures after "
                    "source and optimizer plumbing pass."
                ),
            },
            "guide_contact_continuity": {
                **failure_summary(contact_contract),
                "diagnosis": (
                    "A zero count is a negative regression result. Preserve exact substep evidence "
                    "and the frozen threshold; do not describe this class as a repaired failure."
                    if not contact_contract else
                    "The public-measured recontact guard removes the prior Cross-EBDM/gated-full "
                    "regression, but it is not a general contact certificate: residual losses remain."
                ),
                "revision_scope": (
                    "Keep this as a hard zero-event regression gate and do not change the 150 ms "
                    "limit or shared route."
                    if not contact_contract else
                    "Keep the guard as a method-side public-feedback component and retain residual "
                    "row failures without changing the 150 ms limit."
                ),
            },
        },
        "confirmed_adapter_defects": {
            "T2_dial_task_owned_projection_missing": not source_checks["checks"][
                "dial_projects_T2_task_owned_coordinates"
            ],
            "T2_public_source_prior_task_owned_projection_missing": not source_checks["checks"][
                "public_source_prior_projects_T2_task_owned_coordinates"
            ],
            "required_change": (
                "None. DIAL, public source prior, learned dense and ordinary response MPC preserve "
                "the frozen T2 task-owned coordinates."
                if source_checks["passed"] else
                "Repair the failing static interface check before any further physics."
            ),
        },
        "completed_cpu_stages": completed_cpu_stages,
        "bounded_cpu_stages": bounded_cpu_stages,
        "next_revision": next_revision,
        "theory_revision": theory_revision,
        "formal_evaluation_decision": formal_evaluation_decision,
        "supplemental_artifacts": supplemental_artifacts,
    }

    input_artifacts = {
        name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
        for name, path in files.items()
    }
    input_artifacts["analysis_script"] = {
        "path": str(Path(__file__).resolve().relative_to(ROOT)),
        "sha256": sha256(Path(__file__).resolve()),
    }
    closeout = {
        "schema_version": "four-task-predeployment-offline-closeout-v1",
        "status": (
            "complete_with_shared_handoff_revision_required"
            if task_contract_v5_v13 and source_checks["passed"] else
            "complete_with_method_diagnosis_frozen" if source_checks["passed"]
            else "complete_with_method_adapter_repair_required"
        ),
        "scope": "descriptive qualification statistics; not target performance evidence",
        "device": "cpu",
        "old_invalid_outcomes_used": False,
        "confirmation_states_opened": 0,
        "artifacts": input_artifacts,
        "counts": {
            **report["counts"],
            "successes": sum(bool(row["success"]) for row in outcomes),
            "timeouts": len(timeouts),
            "constraint_violations": len(violations),
            "contact_contract_failures": len(contact_contract),
            "infrastructure_errors": report["infrastructure_error_count"],
        },
        "public_geometry_sentinel": {
            "episodes": len(sentinel),
            "successes": sum(bool(row["success"]) for row in sentinel),
            "passed": bool(sentinel) and all(row["success"] for row in sentinel),
        },
        "source_contract_checks": source_checks,
        "terminal_handoff_failures": {
            "episodes": len(terminal_handoff_failures),
            "rows": dict(sorted(Counter(
                row["row_id"] for row in terminal_handoff_failures
            ).items())),
            "outcomes": dict(sorted(Counter(
                row["outcome"] for row in terminal_handoff_failures
            ).items())),
        },
        "failure_classes": failure_classes,
        "v13_shared_failure_attribution": v13_attribution,
        "row_statistics": row_statistics,
        "interpretation_limits": [
            "Qualification episodes decide admission and are not formal performance samples.",
            "Wilson intervals are descriptive only; rows share task cells and are not an IID paper comparison.",
            "A passing public sentinel localizes failures away from the shared task contract but does not by itself distinguish a baseline adapter defect from a method limitation.",
        ],
    }

    output.mkdir(parents=True, exist_ok=False)
    (output / "closeout.json").write_text(json.dumps(closeout, indent=2, sort_keys=True) + "\n")
    (output / "diagnosis_plan.json").write_text(
        json.dumps(diagnosis_plan, indent=2, sort_keys=True) + "\n"
    )
    with (output / "row_statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "row_id", "task", "family", "key", "method_identity", "admission_status",
            "episodes", "successes", "success_rate", "constraint_violations", "outcomes",
        ])
        writer.writeheader()
        for row in row_statistics:
            writer.writerow({**{key: row[key] for key in writer.fieldnames if key != "outcomes"},
                             "outcomes": json.dumps(row["outcomes"], sort_keys=True)})
    with (output / "failure_cells.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "family", "key", "outcome", "requested_mode", "episodes"])
        for cell, count in sorted(failure_cells.items()):
            writer.writerow([*cell, count])
    if task_contract_v5_v13:
        paired_rows = []
        for pair in v13_attribution["paired_failure_profiles"]:
            for role in ("failure", "task_only_comparator"):
                profile = pair[role]
                entry = profile.get("exact_common_entry") or {}
                violation = profile.get("first_force_violation") or {}
                paired_rows.append({
                    "pair_failure_record_id": pair["failure"]["record_id"],
                    "role": role,
                    "row_id": profile["row_id"],
                    "instance": profile["instance"],
                    "request_id": profile["request_id"],
                    "requested_mode": profile["requested_mode"],
                    "outcome": profile["outcome"],
                    "evidence_sha256": profile["evidence_sha256"],
                    "first_exact_common_control": profile["first_exact_common_control"],
                    "deadline_reserve_controls": entry.get("deadline_reserve_controls"),
                    "entry_tracking_lag_m": entry.get("tracking_lag_m"),
                    "entry_public_reference_lead_m": entry.get(
                        "public_reference_lead_m"
                    ),
                    "first_violation_control": violation.get("control_tick"),
                    "first_violation_substep": violation.get("physics_substep"),
                    "violation_selector_code": violation.get("selector_code"),
                    "final_position_error_m": profile["final_position_error_m"],
                    "force_peak_N": profile["force_peak_N"],
                    "lateral_force_peak_N": profile["lateral_force_peak_N"],
                })
        with (output / "paired_failure_profiles.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
            writer.writeheader()
            writer.writerows(paired_rows)

    manifest = {
        "schema_version": "derived-artifact-manifest-v1",
        "files": {
            path.name: sha256(path)
            for path in sorted(output.iterdir()) if path.name != "manifest.json"
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        output_label = str(output.relative_to(ROOT))
    except ValueError:
        output_label = str(output)
    print(json.dumps({
        "output": output_label,
        "status": closeout["status"],
        "counts": closeout["counts"],
        "adapter_checks_passed": source_checks["passed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
