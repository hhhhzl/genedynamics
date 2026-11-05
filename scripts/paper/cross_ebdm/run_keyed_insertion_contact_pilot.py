#!/usr/bin/env python3
"""Run the fresh paired keyed-insertion contact and clean-ablation pilot."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.tasks.transfer.controllers.bindings import resolve_transfer_binding
from scripts.paper.cross_ebdm.qualify_method_behavior import (
    _execute_job,
    _runtime_device_proof,
)


ROOT = REPOSITORY_ROOT
VERSION = "keyed-insertion-contact-pilot-v1"
CLEAN_TREATMENTS = (
    "no_calibration",
    "no_response_anchoring",
    "unconditional_response_anchoring",
)
TRAINED_TREATMENTS = ("full", *CLEAN_TREATMENTS)
METHODS = (
    "full",
    "adaptsim",
    "asid",
    "response_mpc",
    "task_only",
    *CLEAN_TREATMENTS,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "sha256": sha256(path),
    }


def checked(value: dict[str, str], label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact path/SHA pointer")
    path = Path(value["path"])
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file() or sha256(path) != value["sha256"]:
        raise ValueError(f"{label} changed")
    return path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def resolve_manifest_binding(
    task_config: Path, *, key: str, mode: int,
) -> dict[str, Any]:
    config = ExperimentConfig.from_yaml(task_config)
    selection = deepcopy(config.method_params["actor_binding"])
    selection.update(family="main", key=key, instance="instance_01")
    return resolve_transfer_binding(
        {"actor_binding": selection, "mode": mode},
        task_alias="T3", instance_id="instance_01",
    )


def baseline_binding(plan_path: Path, *, method: str) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text())
    expected_instance = {
        "adaptsim": "independent_8813001",
        "asid": "independent_8803001",
    }[method]
    if (
        plan.get("version") != "transfer-independent-main-table-evaluation-v2"
        or plan.get("task") != "T3"
        or plan.get("method") != method
        or plan.get("instance_id") != expected_instance
        or plan.get("actor_binding", {}).get("status") != "instances_ready"
        or plan.get("replacement_requests_allowed") is not False
        or plan.get("all_failures_retained") is not True
    ):
        raise ValueError(f"{method} formal actor plan changed")
    return deepcopy(plan["actor_binding"])


def treatment_binding(
    full: dict[str, Any], report_path: Path, *, treatment: str, mode: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(report_path.read_text())
    row = report.get("instances", {}).get("instance_01")
    metadata = report.get("clean_ablation")
    if (
        report.get("schema_version")
        != "t3-task-first-mode-specific-recorded-completion-training-v9"
        or report.get("status") not in {"complete", "selection_failed"}
        or report.get("pilot_instance_count") != 1
        or report.get("required_validation_modes") != [mode]
        or report.get("new_physics_calls") != 0
        or report.get("confirmation_states_opened") != 0
        or report.get("target_reward_or_truth_opened") is not False
        or not isinstance(row, dict)
        or metadata != row.get("clean_ablation")
        or metadata.get("treatment") != treatment
        or report.get("control_checkpoint_policy") != "frozen_selector"
    ):
        raise ValueError(f"{treatment} mode {mode} training report changed")
    checkpoint = checked(row["control_checkpoint"], f"{treatment} policy")
    task_only = checked(
        row["inputs"]["task_only_checkpoint"], f"{treatment} task-only start",
    )
    response = checked(
        row["inputs"]["response_checkpoint"], f"{treatment} response model",
    )
    if pointer(task_only) != full["admission_task_only_checkpoint"]:
        raise ValueError(f"{treatment} changed the paired task-only admission policy")
    calibrated = metadata["calibration"] == "target_calibrated"
    if calibrated and pointer(response) != full["response_checkpoint"]:
        raise ValueError(f"{treatment} changed the calibrated response checkpoint")
    if not calibrated and pointer(response) == full["response_checkpoint"]:
        raise ValueError("no-calibration treatment retained the calibrated response")
    binding = deepcopy(full)
    binding.update({
        "policy_checkpoint": pointer(checkpoint),
        "response_checkpoint": pointer(response),
        "method_identity": f"clean_{treatment}_instance_01",
        "clean_ablation": deepcopy(metadata),
    })
    lineage = {
        "training_report": pointer(report_path),
        "policy_checkpoint": pointer(checkpoint),
        "response_checkpoint": pointer(response),
        "selector_accepted": bool(row["selection"]["accepted"]),
        "selected_update": row.get("selected_update"),
        "sample_stream_sha256": row["sample_stream_sha256"],
        "selected_is_task_only_fallback": (
            row["control_checkpoint"] == row["inputs"]["task_only_checkpoint"]
        ),
        "treatment": deepcopy(metadata),
    }
    return binding, lineage


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty pilot cell")
    latencies = [
        float(row["actor_action_latency_seconds"]["p95"])
        for row in rows
        if row.get("actor_action_latency_seconds", {}).get("p95") is not None
    ]
    return {
        "episodes": len(rows),
        "constrained_success_rate": sum(bool(row["passed"]) for row in rows) / len(rows),
        "constraint_violation_rate": sum(
            bool(row["constraint_violation"]) for row in rows
        ) / len(rows),
        "failure_penalized_normalized_time": sum(
            float(row["capped_normalized_completion_time"]) for row in rows
        ) / len(rows),
        "median_episode_actor_action_p95_latency_seconds": (
            statistics.median(latencies) if latencies else None
        ),
    }


def run(config_path: Path, *, plan_only: bool, resume: bool) -> None:
    config = json.loads(config_path.read_text())
    expected_keys = {
        "version", "status", "task", "legacy_task_alias", "task_configs",
        "actor_manifest", "baseline_actor_plans", "treatment_training_reports",
        "response_calibration_seconds",
        "scene_cells", "requested_modes", "reset_slots_per_cell", "seed_anchor",
        "execution", "limits", "output", "policy",
    }
    if set(config) != expected_keys or config["version"] != VERSION:
        raise ValueError("contact-pilot config schema changed")
    if (
        config["status"] != "frozen_unexecuted"
        or config["task"] != "keyed_insertion_recovery"
        or config["legacy_task_alias"] != "T3"
        or config["requested_modes"] != [0, 1]
        or config["reset_slots_per_cell"] != 4
        or set(config["treatment_training_reports"]) != set(TRAINED_TREATMENTS)
        or config["response_calibration_seconds"] != {
            "full": 60,
            "no_calibration": 0,
            "no_response_anchoring": 60,
            "unconditional_response_anchoring": 60,
        }
        or set(config["baseline_actor_plans"]) != {"adaptsim", "asid"}
        or config["policy"] != {
            "diagnostic_only": True,
            "performance_evidence": False,
            "confirmation_states_opened": 0,
            "failure_replacement": False,
            "all_failures_retained": True,
        }
    ):
        raise ValueError("contact-pilot protocol changed")
    expected_cells = {
        "latency_gain_shift": {
            "contact_time_constant": 0.02,
            "delay_steps": 3,
            "payload_mass": 0.5,
            "servo_gain": 0.7,
        },
        "payload_shift": {
            "contact_time_constant": 0.02,
            "delay_steps": 0,
            "payload_mass": 0.8,
            "servo_gain": 1.0,
        },
    }
    if {
        row["id"]: row["mechanism_parameters"] for row in config["scene_cells"]
    } != expected_cells or any(row.get("robot") != "xarm7" for row in config["scene_cells"]):
        raise ValueError("contact-pilot scene cells changed")
    task_configs = {
        name: checked(value, f"{name} task config")
        for name, value in config["task_configs"].items()
    }
    if set(task_configs) != {
        "full", "task_only", "response_mpc", "adaptsim", "asid",
    }:
        raise ValueError("contact-pilot task config roster changed")
    manifest_path = checked(config["actor_manifest"], "actor manifest")
    full_config = ExperimentConfig.from_yaml(task_configs["full"])
    if full_config.method_params["actor_binding"]["manifest"] != pointer(manifest_path):
        raise ValueError("full task config uses a different actor manifest")
    baseline_bindings = {
        method: baseline_binding(
            checked(value, f"{method} actor plan"), method=method,
        )
        for method, value in config["baseline_actor_plans"].items()
    }

    requests = []
    seed = int(config["seed_anchor"])
    for cell in config["scene_cells"]:
        for slot in range(config["reset_slots_per_cell"]):
            for mode in config["requested_modes"]:
                requests.append({
                    "request_id": (
                        f"DEVELOPMENT/keyed_insertion_contact/{cell['id']}/"
                        f"i{slot:02d}/m{mode}"
                    ),
                    "task": "T3",
                    "robot": "xarm7",
                    "requested_mode": mode,
                    "reset_seed": seed + len(requests),
                    "physical_initial_slot": slot,
                    "mechanism_parameters": deepcopy(cell["mechanism_parameters"]),
                    "body_support_parameters": {
                        "robot": "xarm7", "support_controller": "fixed_base",
                    },
                    "mechanism_group": cell["id"],
                    "body_support_group": "xarm7_fixed_base",
                    "confirmation_state": False,
                    "performance_evidence": False,
                })
    if len(requests) != 16 or len({row["reset_seed"] for row in requests}) != 16:
        raise ValueError("contact-pilot request denominator changed")

    bindings: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    lineage: dict[str, Any] = {}
    for mode in config["requested_modes"]:
        full = resolve_manifest_binding(
            task_configs["full"], key="cross_ebdm", mode=mode,
        )
        bindings["task_only"][mode] = resolve_manifest_binding(
            task_configs["task_only"], key="task_only", mode=mode,
        )
        bindings["response_mpc"][mode] = resolve_manifest_binding(
            task_configs["response_mpc"], key="response_mpc", mode=mode,
        )
        mode_streams = set()
        for treatment in TRAINED_TREATMENTS:
            report_path = checked(
                config["treatment_training_reports"][treatment][f"mode_{mode}"],
                f"{treatment} mode {mode} training report",
            )
            binding, row = treatment_binding(
                full, report_path, treatment=treatment, mode=mode,
            )
            bindings[treatment][mode] = binding
            lineage[f"{treatment}/mode_{mode}"] = row
            mode_streams.add(row["sample_stream_sha256"])
        if len(mode_streams) != 1:
            raise ValueError(f"mode {mode} treatments changed the batch-index stream")
        for method in ("adaptsim", "asid"):
            bindings[method][mode] = deepcopy(baseline_bindings[method])

    jobs = []
    config_for_method = {
        "full": task_configs["full"],
        "task_only": task_configs["task_only"],
        "response_mpc": task_configs["response_mpc"],
        "adaptsim": task_configs["adaptsim"],
        "asid": task_configs["asid"],
        **{name: task_configs["full"] for name in CLEAN_TREATMENTS},
    }
    for method in METHODS:
        for request in requests:
            mode = int(request["requested_mode"])
            jobs.append({
                "record_id": f"keyed_insertion_contact/{method}/{request['request_id']}",
                "row_id": f"keyed_insertion_contact/{method}",
                "task": "T3", "family": "main", "key": method,
                "method": method, "instance": "instance_01",
                "config": str(config_for_method[method].relative_to(ROOT)),
                "actor_binding": bindings[method][mode],
                "device": "cuda",
                "cuda_visible_devices": config["execution"]["cuda_visible_devices"],
                "physical_uuid": config["execution"]["physical_uuid"],
                "request": deepcopy(request),
            })
    plan = {
        "schema_version": VERSION,
        "status": "planned_unexecuted",
        "task": config["task"],
        "methods": list(METHODS),
        "requests": requests,
        "jobs": jobs,
        "treatment_lineage": lineage,
        "response_calibration_seconds": config["response_calibration_seconds"],
        "policy": config["policy"],
        "execution": config["execution"],
        "artifacts": {
            "config": pointer(config_path),
            "runner": pointer(Path(__file__).resolve()),
            "actor_manifest": pointer(manifest_path),
        },
    }
    _runtime_device_proof(plan)
    output = ROOT / config["output"]
    plan_path = output / "plan.json"
    if plan_only:
        print(json.dumps({
            "status": plan["status"],
            "methods": len(METHODS),
            "requests_per_method": len(requests),
            "planned_episodes": len(jobs),
            "plan_sha256": hashlib.sha256(
                json.dumps(plan, sort_keys=True).encode()
            ).hexdigest(),
        }, indent=2, sort_keys=True))
        return
    if plan_path.is_file():
        if not resume or json.loads(plan_path.read_text()) != plan:
            raise ValueError("existing contact-pilot plan differs; use exact --resume")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing nonempty output: {output}")
        output.mkdir(parents=True, exist_ok=True)
        write_json(plan_path, plan)

    begun = time.monotonic()
    completed = {}
    for result_path in output.glob("episodes/*/result.json"):
        row = json.loads(result_path.read_text())
        completed[row["record_id"]] = row
    pending = [job for job in jobs if job["record_id"] not in completed]
    workers = int(config["limits"]["workers"])
    if workers < 1 or workers > 4:
        raise ValueError("contact pilot workers must be in 1..4")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_execute_job, (job, str(output))): job["record_id"]
            for job in pending
        }
        for future in as_completed(futures):
            result = future.result()
            completed[result["record_id"]] = result
            write_json(output / "progress.json", [
                completed[job["record_id"]]
                for job in jobs if job["record_id"] in completed
            ])
            if time.monotonic() - begun > float(config["limits"]["maximum_wall_seconds"]):
                raise TimeoutError("contact pilot exceeded its wall ceiling")
    results = [completed[job["record_id"]] for job in jobs]
    integrity = len(results) == len(jobs) and all(
        row.get("outcome") != "reset_or_execution_error"
        and row.get("evidence", {}).get("audit_passed", False)
        for row in results
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cells: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for job, result in zip(jobs, results):
        method = job["key"]
        grouped[method].append(result)
        request = job["request"]
        cells[(method, request["mechanism_group"], request["requested_mode"])].append(result)
    method_summary = {method: summarize(grouped[method]) for method in METHODS}
    budget_metrics = (
        "constrained_success_rate",
        "constraint_violation_rate",
        "failure_penalized_normalized_time",
    )
    preliminary_budget_contrast = {
        "effective_response_calibration_seconds": {
            "full": config["response_calibration_seconds"]["full"],
            "no_calibration": config["response_calibration_seconds"][
                "no_calibration"
            ],
        },
        "full_minus_no_calibration": {
            metric: method_summary["full"][metric]
            - method_summary["no_calibration"][metric]
            for metric in budget_metrics
        },
        "scope": (
            "Single-instance two-point physical development contrast; it is not a "
            "matched multi-budget sample-efficiency curve."
        ),
    }
    cell_summary = [
        {
            "method": method,
            "mechanism_group": mechanism,
            "requested_mode": mode,
            **summarize(rows),
        }
        for (method, mechanism, mode), rows in sorted(cells.items())
    ]
    report = {
        "schema_version": VERSION,
        "status": "complete_development_pilot" if integrity else "integrity_failed",
        "passed": integrity,
        "formal_performance_evidence": False,
        "task": config["task"],
        "methods": list(METHODS),
        "metrics": [
            "constrained_success_rate",
            "constraint_violation_rate",
            "failure_penalized_normalized_time",
            "median_episode_actor_action_p95_latency_seconds",
        ],
        "method_summary": method_summary,
        "preliminary_response_calibration_contrast": preliminary_budget_contrast,
        "cell_summary": cell_summary,
        "treatment_lineage": lineage,
        "response_calibration_seconds": config["response_calibration_seconds"],
        "counts": {
            "planned_episodes": len(jobs),
            "completed_episodes": len(results),
            "execution_errors": sum(
                row.get("outcome") == "reset_or_execution_error" for row in results
            ),
        },
        "claim_scope": (
            "Fresh single-policy-instance development pilot. It can reject a treatment "
            "or motivate the multi-instance experiment; it cannot establish the final "
            "paper ranking by itself."
        ),
        "policy": config["policy"],
        "runtime_device": _runtime_device_proof(plan),
        "artifacts": {"plan": pointer(plan_path), **plan["artifacts"]},
        "wall_seconds": time.monotonic() - begun,
    }
    write_json(output / "report.json", report)
    print(json.dumps({
        "status": report["status"],
        "output": str(output / "report.json"),
        "sha256": sha256(output / "report.json"),
        "method_summary": method_summary,
    }, indent=2, sort_keys=True))
    if not integrity:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    run(config_path, plan_only=args.plan_only, resume=args.resume)


if __name__ == "__main__":
    main()
