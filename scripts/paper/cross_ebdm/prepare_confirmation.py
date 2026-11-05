#!/usr/bin/env python3
"""Seal the fresh v14 GPU1 four-task request roster and evaluation schedule."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess

from genedynamics.tasks.transfer.controllers.bindings import (
    resolve_transfer_binding,
    validate_transfer_actor_manifest,
)
from genedynamics.experiments.framework import ExperimentConfig


ROOT = Path(__file__).resolve().parents[3]
TASK_ROOTS = {
    "T1": Path("configs/arm/obstacle_transfer"),
    "T2": Path("configs/arm/guided_contact_alignment"),
    "T3": Path("configs/arm/keyed_insertion_recovery"),
    "T4": Path("configs/quadruped/mobile_keyed_insertion_recovery"),
}
EXPECTED_REQUESTS = {"T1": 40, "T2": 40, "T3": 40, "T4": 80}
CONFIRMATION_REVISION = 14
CONFIRMATION_LABEL = f"CONFIRMATION-v{CONFIRMATION_REVISION}"
PACKAGE_VERSION = (
    f"four-task-transfer-confirmation-package-v{CONFIRMATION_REVISION}"
)
SEED_ANCHORS = {"T1": 9_681_000, "T2": 9_682_000, "T3": 9_683_000, "T4": 9_684_000}
PHYSICAL_GPU_INDEX = 1
PHYSICAL_GPU_UUID = "GPU-58b54684-3577-11d3-f039-0e8469190739"
MECHANISMS = (
    ("latency_gain_shift", {"delay_steps": 3, "payload_mass": .5, "servo_gain": .7}),
    ("payload_shift", {"delay_steps": 1, "payload_mass": .8, "servo_gain": .85}),
)
MAIN_KEYS = (
    "cross_ebdm", "task_only", "fada", "acp", "human2sim2robot",
    "dial_transfer", "response_mpc", "public_geometry", "public_source_prior",
)
TRANSPORT_CLAIM_CONTRACT = {
    "version": "four-task-stable-transport-and-strong-baseline-v1",
    "primary_metrics": [
        "constrained_success_rate",
        "constraint_violation_rate",
        "capped_normalized_completion_time",
        "worst_group_constrained_success",
    ],
    "task_weighting": "equal_macro_average_over_T1_T2_T3_T4",
    "transport_control": "task_only",
    "stable_transport_gate": {
        "minimum_nonnegative_task_success_deltas": 4,
        "minimum_strictly_positive_task_success_deltas": 1,
        "minimum_macro_success_delta": 0.0,
        "maximum_macro_constraint_violation_delta": 0.0,
        "maximum_macro_capped_completion_time_delta": 0.0,
        "require_strict_macro_success_gain": True,
        "require_transport_to_preserve_every_task_only_success": True,
    },
    "strong_executable_baselines": [
        "fada",
        "dial_transfer",
        "response_mpc",
    ],
    "strong_baseline_gate": {
        "selection_rule": (
            "highest_macro_constrained_success_then_lowest_macro_constraint_"
            "violation_then_lowest_macro_capped_completion_time_then_"
            "lexicographically_ascending_method_id"
        ),
        "require_positive_macro_success_delta_against_every_baseline": True,
        "require_nonpositive_macro_constraint_delta_against_every_baseline": True,
        "require_positive_paired_bootstrap_95pct_lower_success_bound_against_selected_strongest": True,
    },
    "coverage_only_rows": ["acp", "human2sim2robot"],
    "coverage_only_rule": (
        "Keep predeployment rejections in the main-table denominator, but do not "
        "interpret their zero score as empirical algorithmic superiority."
    ),
    "public_references": ["public_geometry", "public_source_prior"],
    "failure_replacement": False,
}
ABLATION_KEYS = (
    "frozen_warm", "task_only", "matching_no_band", "band_only",
    "ungated_full", "gated_full",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ))


def prior_seed_conflicts(candidates: set[int]) -> set[int]:
    """Find exact candidate reuse without reading the entire result tree in Python."""
    if not candidates:
        return set()
    result_root = ROOT / "results"
    conflicts = {
        int(path.name.removeprefix("seed_"))
        for path in result_root.rglob("seed_*")
        if (
            path.is_dir()
            and path.name.removeprefix("seed_").isdigit()
            and int(path.name.removeprefix("seed_")) in candidates
        )
    }
    alternatives = "|".join(str(value) for value in sorted(candidates))
    search = subprocess.run(
        [
            os.environ.get("RG_EXECUTABLE", "rg"),
            "--files-with-matches", "--no-messages",
            "--glob", "*.json", "--glob", "*.jsonl",
            rf'"(?:reset_seed|confirmation_seed)"\s*:\s*(?:{alternatives})(?:[^0-9]|$)',
            str(result_root),
        ],
        capture_output=True, text=True, check=False,
    )
    if search.returncode not in {0, 1}:
        raise RuntimeError(f"historical seed search failed: {search.stderr.strip()}")
    field = re.compile(r'"(?:reset_seed|confirmation_seed)"\s*:\s*([0-9]+)')
    for name in search.stdout.splitlines():
        path = Path(name)
        try:
            conflicts.update(
                value for value in map(int, field.findall(path.read_text()))
                if value in candidates
            )
        except (OSError, UnicodeDecodeError):
            continue
    return conflicts


def request_roster() -> list[dict]:
    rows = []
    for task in TASK_ROOTS:
        supports = (
            (("xarm7_fixed_base", {
                "robot": "xarm7", "support_controller": "fixed_base",
            }),)
            if task != "T4" else (
                ("support_nominal", {
                    "robot": "go2_z1", "body_servo_gain": 1.,
                    "support_controller": "unchanged host OSQP stance WBC",
                }),
                ("support_derated_0p90", {
                    "robot": "go2_z1", "body_servo_gain": .9,
                    "support_controller": "unchanged host OSQP stance WBC",
                }),
            )
        )
        sequence = 0
        for mechanism_group, mechanism_parameters in MECHANISMS:
            for body_support_cell, body_support_parameters in supports:
                for slot in range(10):
                    for mode in (0, 1):
                        request_id = (
                            f"{CONFIRMATION_LABEL}/{task}/{mechanism_group}/"
                            f"{body_support_cell}/i{slot:02d}/m{mode}"
                        )
                        rows.append({
                            "request_id": request_id, "task": task,
                            "mechanism_group": mechanism_group,
                            "mechanism_parameters": mechanism_parameters,
                            "body_support_cell": body_support_cell,
                            "body_support_parameters": body_support_parameters,
                            "physical_initial_slot": slot,
                            "requested_mode": mode,
                            "reset_seed": SEED_ANCHORS[task] + sequence,
                            "confirmation_accessed": False,
                            "status": "sealed_unaccessed_request_specification",
                            "expected_runtime_receipts": [
                                "post_reset_physical_state_sha256",
                                "post_reset_controller_state_sha256",
                                "post_reset_episode_record_sha256",
                            ],
                        })
                        sequence += 1
    if Counter(row["task"] for row in rows) != Counter(EXPECTED_REQUESTS):
        raise AssertionError("fresh request roster cardinality differs")
    if len({row["reset_seed"] for row in rows}) != len(rows):
        raise AssertionError("fresh reset seeds are not globally unique")
    reused = sorted(prior_seed_conflicts({row["reset_seed"] for row in rows}))
    if reused:
        raise ValueError(f"fresh roster reuses existing seed directories: {reused}")
    return rows


def binding_hash(binding: dict) -> str | None:
    for label in ("checkpoint", "policy_checkpoint", "response_checkpoint"):
        value = binding.get(label)
        if isinstance(value, dict) and isinstance(value.get("sha256"), str):
            return value["sha256"]
    return None


def evaluation_schedule(roster: list[dict], qualification: dict) -> list[dict]:
    by_task = {
        task: [row for row in roster if row["task"] == task]
        for task in TASK_ROOTS
    }
    rows = []
    admissions = {
        (row["task"], row["family"], row["key"]): row
        for row in qualification["rows"]
    }
    if len(admissions) != 60:
        raise ValueError("behavioral qualification must contain exactly 60 rows")
    for task, relative in TASK_ROOTS.items():
        base = ExperimentConfig.from_yaml(ROOT / relative / "_base.yaml")
        manifest_pointer = base.method_params["actor_binding"]["manifest"]
        manifest_path = ROOT / manifest_pointer["path"]
        if sha(manifest_path) != manifest_pointer["sha256"]:
            raise ValueError(f"{task} base config actor manifest hash changed")
        document = validate_transfer_actor_manifest(manifest_pointer)
        for family, keys in (("main", MAIN_KEYS), ("ablation", ABLATION_KEYS)):
            for key in keys:
                manifest_row = document[family][key]
                admitted = admissions[(task, family, key)]
                if admitted.get("final_for_current_protocol") is not True:
                    raise ValueError(f"qualification is not final for {task}/{family}/{key}")
                if manifest_row["status"] == "deterministic_single_identity":
                    instances = ["fixed"]
                else:
                    instances = list(manifest_row.get("logical_instance_slots") or (
                        f"instance_{index:02d}" for index in range(1, 9)
                    ))
                for instance in instances:
                    selection = {
                        "version": document["binding_version"],
                        "manifest": {
                            "path": str(manifest_path.relative_to(ROOT)),
                            "sha256": sha(manifest_path),
                        },
                        "family": family, "key": key,
                    }
                    if instance != "fixed":
                        selection["instance"] = instance
                    qualification_status = admitted["admission_status"]
                    dispatch = (
                        "execute_once_after_admission"
                        if qualification_status == "behavior_qualified"
                        else "score_predeployment_rejection_without_physical_execution"
                    )
                    for request in by_task[task]:
                        binding = resolve_transfer_binding(
                            {
                                "actor_binding": selection,
                                "mode": int(request["requested_mode"]),
                            },
                            task_alias=task,
                        )
                        if (
                            dispatch == "execute_once_after_admission"
                            and binding["status"] == "predeployment_rejection"
                        ):
                            raise ValueError(
                                f"qualified row has no actor: {task}/{family}/{key}"
                            )
                        checkpoint = binding_hash(binding)
                        schedule_id = (
                            f"{family}/{key}/{instance}/{request['request_id']}"
                        )
                        row = {
                            "schedule_id": schedule_id,
                            "request_id": request["request_id"],
                            "task": task, "family": family, "method": key,
                            "instance": instance, "dispatch": dispatch,
                            "qualification_status": qualification_status,
                            "qualification_row_id": admitted["row_id"],
                            "qualification_reason": (
                                admitted.get("manifest_reason")
                                or ";".join(sorted((admitted.get("outcomes") or {}).keys()))
                                or qualification_status
                            ),
                            "rejection_scoring": {
                                "constrained_success": 0.0,
                                "constraint_violation": 1.0,
                                "capped_normalized_completion_time": 1.0,
                                "predeployment_rejection": 1.0,
                            },
                            "failure_replacement": False,
                        }
                        if checkpoint is not None:
                            row["checkpoint_sha256"] = checkpoint
                        rows.append(row)
    random.Random(2_027_104_0).shuffle(rows)
    if len(rows) != 21_200 or len({row["schedule_id"] for row in rows}) != len(rows):
        raise AssertionError(f"fresh evaluation schedule has {len(rows)} rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/_protocol/four_task_transfer/confirmation_v14/package",
    )
    parser.add_argument("--qualification-report", type=Path, required=True)
    parser.add_argument("--qualification-audit", type=Path, required=True)
    parser.add_argument("--architecture-report", type=Path, required=True)
    parser.add_argument("--cpu-package-report", type=Path, required=True)
    parser.add_argument("--gpu-calibration-report", type=Path, required=True)
    parser.add_argument("--invalid-predecessor", type=Path, required=True)
    parser.add_argument("--abandoned-predecessor", type=Path, required=True)
    parser.add_argument("--accessed-predecessor", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    evidence_paths = {
        name: (path if path.is_absolute() else ROOT / path)
        for name, path in {
            "behavioral_predeployment": args.qualification_report,
            "behavioral_predeployment_audit": args.qualification_audit,
            "zero_physics_architecture": args.architecture_report,
            "cpu_package_qualification": args.cpu_package_report,
            "gpu_package_calibration": args.gpu_calibration_report,
        }.items()
    }
    evidence = {name: json.loads(path.read_text()) for name, path in evidence_paths.items()}
    invalid_path = (
        args.invalid_predecessor
        if args.invalid_predecessor.is_absolute()
        else ROOT / args.invalid_predecessor
    )
    invalid = json.loads(invalid_path.read_text())
    if (
        invalid.get("status") not in {
            "invalid_incomplete_run", "invalid_incomplete_qualification",
        }
        or invalid.get("performance_evidence_admitted") is not False
        or invalid.get("reusable_records") != 0
        or invalid.get("failure_replacement") is not False
    ):
        raise ValueError("invalid predecessor is not a zero-reuse invalid artifact")
    abandoned_path = (
        args.abandoned_predecessor
        if args.abandoned_predecessor.is_absolute()
        else ROOT / args.abandoned_predecessor
    )
    abandoned = json.loads(abandoned_path.read_text())
    if not (
        abandoned.get("status") == "abandoned_before_confirmation_access"
        and abandoned.get("performance_evidence_admitted") is False
        and abandoned.get("reusable_records") == 0
        and abandoned.get("confirmation_states_opened") == 0
        and abandoned.get("run_directory_created") is False
        and abandoned.get("physical_steps") == 0
    ):
        raise ValueError("abandoned predecessor is not a zero-access package")
    accessed_path = (
        args.accessed_predecessor
        if args.accessed_predecessor.is_absolute()
        else ROOT / args.accessed_predecessor
    )
    accessed = json.loads(accessed_path.read_text())
    if (
        accessed.get("passed") is not True
        or accessed.get("status") != "passed"
        or accessed.get("counts", {}).get("outcomes") != 21_200
    ):
        raise ValueError("accessed predecessor is not a complete audited confirmation")
    if evidence["behavioral_predeployment"].get("formal_evaluation_allowed") is not True:
        raise ValueError("behavioral qualification did not admit formal evaluation")
    if any(
        document.get("passed", document.get("status") == "passed") is not True
        for document in evidence.values()
    ):
        raise ValueError("every qualification, audit, architecture, and package report must pass")
    if evidence["gpu_package_calibration"].get("gpu_binding", {}).get("cuda_visible_devices") != "1":
        raise ValueError("GPU package calibration was not bound to physical GPU1")
    if evidence["gpu_package_calibration"].get("gpu_binding", {}).get("physical_index") != PHYSICAL_GPU_INDEX:
        raise ValueError("GPU package calibration physical index differs from GPU1")
    if evidence["gpu_package_calibration"].get("gpu_binding", {}).get("uuid") != PHYSICAL_GPU_UUID:
        raise ValueError("GPU package calibration UUID differs from physical GPU1")
    roster = request_roster()
    schedule = evaluation_schedule(roster, evidence["behavioral_predeployment"])
    roster_path = output / "request_roster.jsonl"
    schedule_path = output / "evaluation_schedule.jsonl"
    write_jsonl(roster_path, roster)
    write_jsonl(schedule_path, schedule)
    audit_inputs = {name: pointer(path) for name, path in evidence_paths.items()}
    audit_inputs["invalid_predecessor"] = pointer(invalid_path)
    audit_inputs["accessed_predecessor"] = pointer(accessed_path)
    package = {
        "version": PACKAGE_VERSION,
        "status": "sealed_unaccessed", "passed": True,
        "confirmation_accessed": False, "confirmation_states_opened": 0,
        "old_invalid_run_reused": False,
        "physical_gpu_index": PHYSICAL_GPU_INDEX, "logical_cuda_device": "cuda:0",
        "physical_gpu_uuid": PHYSICAL_GPU_UUID,
        "failure_replacement": False, "launches_allowed": 1,
        "old_confirmation_reused": False,
        "counts": {
            "requests": len(roster), "scheduled_method_requests": len(schedule),
            "main": sum(row["family"] == "main" for row in schedule),
            "ablation": sum(row["family"] == "ablation" for row in schedule),
            "dispatch_physics": sum(row["dispatch"] == "execute_once_after_admission" for row in schedule),
            "predeployment_rejections": sum(row["dispatch"] != "execute_once_after_admission" for row in schedule),
        },
        "seed_policy": {
            "anchors": SEED_ANCHORS, "all_unique": True,
            "existing_seed_directory_overlap": 0,
            "old_confirmation_seed_range_reused": False,
        },
        "transport_claim_contract": TRANSPORT_CLAIM_CONTRACT,
        "prior_invalid_run": pointer(invalid_path),
        "prior_abandoned_package": pointer(abandoned_path),
        "prior_accessed_confirmation": {
            "audit": pointer(accessed_path),
            "status": "historical_accessed_confirmation_excluded",
            "performance_evidence_reused": False,
            "records_reused": 0,
        },
        "audit_inputs": audit_inputs,
        "artifacts": {
            "request_roster": pointer(roster_path),
            "evaluation_schedule": pointer(schedule_path),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    unique_bindings = {
        (row["task"], row["family"], row["method"], row["instance"])
        for row in schedule
    }
    task_counts = Counter(row["task"] for row in roster)
    audit = {
        "schema_version": "four-task-transfer-confirmation-preaccess-audit-v6",
        "status": "passed", "passed": True, "physics_steps": 0,
        "confirmation_states_opened": 0,
        "package_report": pointer(report_path),
        "checks": {
            "package_sealed_unaccessed": package["status"] == "sealed_unaccessed",
            "old_invalid_run_unused": package["old_invalid_run_reused"] is False,
            "old_accessed_confirmation_unused": (
                package["old_confirmation_reused"] is False
                and package["prior_accessed_confirmation"]["records_reused"] == 0
                and package["prior_accessed_confirmation"]
                ["performance_evidence_reused"] is False
            ),
            "invalid_predecessor_bound_and_zero_reuse": (
                package["prior_invalid_run"] == pointer(invalid_path)
                and invalid["reusable_records"] == 0
            ),
            "abandoned_predecessor_bound_and_zero_access": (
                package["prior_abandoned_package"] == pointer(abandoned_path)
                and abandoned["reusable_records"] == 0
                and abandoned["confirmation_states_opened"] == 0
                and abandoned["physical_steps"] == 0
            ),
            "all_request_seeds_unique": len({row["reset_seed"] for row in roster}) == len(roster),
            "historical_seed_overlap_zero": package["seed_policy"]["existing_seed_directory_overlap"] == 0,
            "request_count_and_identity": len(roster) == 200 and len({row["request_id"] for row in roster}) == 200,
            "task_request_counts": task_counts == Counter(EXPECTED_REQUESTS),
            "schedule_coverage": len(schedule) == 21_200 and len({row["schedule_id"] for row in schedule}) == 21_200,
            "all_unique_bindings_resolve": len(unique_bindings) == 424,
            "dispatch_counts_exact": len(schedule) == package["counts"]["dispatch_physics"] + package["counts"]["predeployment_rejections"],
            "gpu1_binding_exact": package["physical_gpu_uuid"] == PHYSICAL_GPU_UUID and package["physical_gpu_index"] == PHYSICAL_GPU_INDEX,
            "run_directory_absent": not (output.parent / "run").exists(),
            "physics_steps_zero": True,
        },
        "counts": {
            "requests": len(roster), "schedule": len(schedule),
            "unique_bindings": len(unique_bindings),
            "dispatch_physics": package["counts"]["dispatch_physics"],
            "predeployment_rejections": package["counts"]["predeployment_rejections"],
            "historical_seed_overlap": 0,
        },
    }
    if not all(audit["checks"].values()):
        raise ValueError("confirmation preaccess audit failed")
    audit_path = output / "preaccess_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": package["status"], "report": str(report_path),
        "sha256": sha(report_path), "preaccess_audit_sha256": sha(audit_path),
        "counts": package["counts"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
