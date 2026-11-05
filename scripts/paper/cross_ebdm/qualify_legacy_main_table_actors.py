#!/usr/bin/env python3
"""Audit eight legacy learned actors on the same source/nominal requests."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from genedynamics.core.artifacts import atomic_json
from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.tasks.transfer.controllers.bindings import (
    resolve_transfer_binding, validate_transfer_actor_manifest,
)
from genedynamics.tasks.transfer.main_table_evaluation import (
    ADMISSION_VERSION, ROOT, TASK_ROOTS, VERSION as FORMAL_VERSION,
    _admission_path, _binding, _roster, _task_config, formal_plan,
)
from genedynamics.tasks.transfer.method_evaluation import (
    GPU1_UUID, _source_nominal_method_config, audit_evaluation,
    evaluation_plan, run_evaluation,
)


VERSION = "transfer-legacy-independent-source-nominal-cohort-v1"
METHODS = ("cross_ebdm", "task_only", "fada", "dial_transfer", "response_mpc")
ANCHORS = {"T1": 3301000000, "T2": 3301100000,
           "T3": 3301300000, "T4": 3301400000}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pointer(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": _sha(path)}


def _check(pointer: dict, label: str) -> Path:
    if not isinstance(pointer, dict) or set(pointer) != {"path", "sha256"}:
        raise ValueError(f"{label} must bind a file and SHA256")
    path = ROOT / pointer["path"]
    if not path.is_file() or _sha(path) != pointer["sha256"]:
        raise ValueError(f"{label} missing or changed")
    return path


def _paths(task: str, method: str) -> tuple[Path, Path, Path]:
    if task not in TASK_ROOTS or method not in METHODS:
        raise ValueError("legacy cohort task or method differs")
    experiment = ExperimentConfig.from_yaml(_task_config(task, method))
    config_path = ROOT / TASK_ROOTS[task] / "validation" / (
        f"{method}_independent_source_nominal_cohort.json"
    )
    output = experiment.output_dir / "validation/independent_source_nominal_cohort_v1"
    return config_path, output / "launch_receipt.json", output


def _source(task: str, method: str) -> tuple[dict, Path, dict, Path, dict]:
    experiment = ExperimentConfig.from_yaml(_task_config(task, method))
    original = experiment.output_dir / "validation" / (
        f"source_nominal_actor_screen_{ANCHORS[task]}"
    )
    report_path = original / "report.json"
    audit_path = original / "independent_audit.json"
    report = json.loads(report_path.read_text())
    audit = json.loads(audit_path.read_text())
    if (report.get("status") != "complete"
            or report.get("task") != task or report.get("method") != method
            or report.get("counts", {}).get("requested") != 8
            or report.get("counts", {}).get("execution_errors") != 0
            or audit.get("status") != "passed"
            or audit.get("report_sha256") != _sha(report_path)
            or audit.get("physical_evidence_verified") != 8
            or audit.get("runner_failures_retained") != 0
            or audit.get("formal_performance_evidence") is not False
            or report.get("instance_id") != "instance_01"):
        raise ValueError("first legacy actor instance has no complete shared screen")
    roster_path = _check(report["request_roster"], "first-instance roster")
    roster = json.loads(roster_path.read_text())
    if (roster.get("task") != task or method not in roster.get("methods", ())
            or len(roster.get("requests", ())) != 8):
        raise ValueError("legacy source/nominal request roster differs")
    base_config, _ = _source_nominal_method_config(
        roster_path, method, roster["version"],
    )
    if (evaluation_plan(base_config)["output"] != str(original.relative_to(ROOT))
            or base_config["method_identity"] != report["method_identity"]):
        raise ValueError("first-instance actor config or identity differs")
    return base_config, roster_path, report, audit_path, audit


def _instance_config(base: dict, output: Path, instance: int) -> dict:
    config = dict(base)
    config["instance_id"] = f"instance_{instance:02d}"
    location = output / config["instance_id"]
    config["output"] = str(location.relative_to(ROOT))
    config["plan_output"] = str(location.relative_to(ROOT)) + "_plan.json"
    plan = evaluation_plan(config)
    if (plan.get("instance_id") != config["instance_id"]
            or len(plan["requests"]) != 8):
        raise ValueError("legacy actor instance does not resolve eight requests")
    return config


def freeze(task: str, method: str) -> dict:
    config_path, receipt_path, output = _paths(task, method)
    if config_path.exists() or output.exists():
        raise FileExistsError("legacy actor cohort already frozen or started")
    base, roster_path, report, audit_path, _ = _source(task, method)
    for instance in range(2, 9):
        _instance_config(base, output, instance)
    config = {
        "version": VERSION, "task": task, "method": method,
        "method_identity": report["method_identity"],
        "task_config": _pointer(_task_config(task, method)),
        "source_roster": _pointer(roster_path),
        "first_instance_report": _pointer(ROOT / report["output"] / "report.json"),
        "first_instance_audit": _pointer(audit_path),
        "instance_ids": [f"instance_{instance:02d}" for instance in range(1, 9)],
        "output": str(output.relative_to(ROOT)),
        "maximum_new_physical_episodes": 56,
        "maximum_cohort_physical_episodes": 64,
        "formal_request_seed_access": 0,
        "failure_replacement": False,
        "execution": {"device": "cuda", "cuda_visible_devices": "1",
                      "physical_uuid": GPU1_UUID},
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "version": VERSION, "status": "frozen_unexecuted",
        "task": task, "method": method,
        "config": _pointer(config_path),
        "maximum_new_physical_episodes": 56,
        "formal_request_seed_access": 0,
        "failure_replacement": False,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def _frozen(task: str, method: str) -> tuple[dict, dict, Path, Path]:
    config_path, receipt_path, output = _paths(task, method)
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("version") != VERSION
            or receipt.get("task") != task or receipt.get("method") != method
            or receipt.get("config") != _pointer(config_path)
            or receipt.get("maximum_new_physical_episodes") != 56
            or receipt.get("formal_request_seed_access") != 0
            or receipt.get("failure_replacement") is not False):
        raise ValueError("legacy cohort launch receipt differs")
    config = json.loads(config_path.read_text())
    base, roster_path, report, audit_path, _ = _source(task, method)
    if (config.get("version") != VERSION
            or config.get("task") != task or config.get("method") != method
            or config.get("method_identity") != base["method_identity"]
            or config.get("task_config") != _pointer(_task_config(task, method))
            or config.get("source_roster") != _pointer(roster_path)
            or config.get("first_instance_report") != _pointer(ROOT / report["output"] / "report.json")
            or config.get("first_instance_audit") != _pointer(audit_path)
            or config.get("instance_ids") != [f"instance_{i:02d}" for i in range(1, 9)]
            or config.get("output") != str(output.relative_to(ROOT))
            or config.get("maximum_new_physical_episodes") != 56
            or config.get("maximum_cohort_physical_episodes") != 64
            or config.get("formal_request_seed_access") != 0
            or config.get("failure_replacement") is not False):
        raise ValueError("legacy source/nominal cohort config differs")
    return config, base, receipt_path, output


def run(task: str, method: str) -> dict:
    config, base, receipt_path, output = _frozen(task, method)
    for instance in range(2, 9):
        child = _instance_config(base, output, instance)
        if (ROOT / child["output"]).exists():
            raise FileExistsError("legacy instance output already exists; no replacement")
        report = run_evaluation(child)
        audit = audit_evaluation(child)
        if audit["status"] != "passed":
            failure = {
                "version": VERSION, "status": "failed_retained",
                "task": task, "method": method,
                "instance_id": child["instance_id"],
                "report": _pointer(ROOT / child["output"] / "report.json"),
                "audit": _pointer(ROOT / child["output"] / "independent_audit.json"),
                "formal_request_seed_access": 0,
            }
            atomic_json(output / "failure_retained.json", failure)
            return failure
    return audit_cohort(task, method)


def audit_cohort(task: str, method: str) -> dict:
    config, base, receipt_path, output = _frozen(task, method)
    first_audit_path = _check(config["first_instance_audit"], "first instance audit")
    first = json.loads(first_audit_path.read_text())
    reference = {row["request_id"]: row["reset_physical_hash"]
                 for row in first["episodes"]}
    if len(reference) != 8 or any(not value for value in reference.values()):
        raise ValueError("first instance has incomplete physical reset evidence")
    instances = [{"instance_id": "instance_01",
                  "report": config["first_instance_report"],
                  "independent_audit": config["first_instance_audit"]}]
    outcomes = Counter()
    old_report = json.loads(_check(config["first_instance_report"], "first report").read_text())
    outcomes.update(row["outcome"] for row in old_report["results"])
    for instance in range(2, 9):
        child = _instance_config(base, output, instance)
        report_path = ROOT / child["output"] / "report.json"
        audit_path = ROOT / child["output"] / "independent_audit.json"
        checked = audit_evaluation(child, write=False)
        saved = json.loads(audit_path.read_text())
        report = json.loads(report_path.read_text())
        if (checked != saved or saved.get("status") != "passed"
                or saved.get("physical_evidence_verified") != 8
                or saved.get("runner_failures_retained") != 0
                or report.get("counts", {}).get("requested") != 8
                or report.get("counts", {}).get("execution_errors") != 0):
            raise ValueError(f"{task}/{method}/{instance} evidence audit differs")
        for row in saved["episodes"]:
            if row["reset_physical_hash"] != reference[row["request_id"]]:
                raise ValueError("legacy cohort physical resets differ")
        outcomes.update(row["outcome"] for row in report["results"])
        instances.append({"instance_id": child["instance_id"],
                          "report": _pointer(report_path),
                          "independent_audit": _pointer(audit_path)})
    result = {
        "version": VERSION, "status": "passed_development_only",
        "task": task, "method": method,
        "method_identity": config["method_identity"],
        "launch": _pointer(receipt_path),
        "source_roster": config["source_roster"],
        "instances": instances,
        "physical_evidence_verified": 64,
        "reset_hashes_matched_to_first_instance": 64,
        "outcomes": dict(sorted(outcomes.items())),
        "success_is_not_an_admission_gate": True,
        "physical_calls": 0, "gpu_calls": 0,
        "formal_request_seed_access": 0,
        "formal_performance_evidence": False,
    }
    destination = output / "independent_audit.json"
    if destination.exists():
        if json.loads(destination.read_text()) != result:
            raise ValueError("legacy cohort audit changed")
    else:
        atomic_json(destination, result)
    return result


def audit_fada_source_lineage(task: str) -> dict:
    """Prove all target instances descend from one qualified Panda source pair."""
    method = "fada"
    experiment = ExperimentConfig.from_yaml(_task_config(task, method))
    binding = experiment.method_params["actor_binding"]
    source_identity = None
    source_seeds, target_seeds = set(), set()
    instances = []
    for index in range(1, 9):
        instance_id = f"instance_{index:02d}"
        selected = dict(binding, instance=instance_id)
        resolved = resolve_transfer_binding(
            {"actor_binding": selected}, task_alias=task,
        )
        target = _check(resolved["checkpoint"], f"FADA {task}/{instance_id} target")
        recipe_path = target.parent.parent / "recipe.json"
        target_report_path = target.parent / "report.json"
        recipe = json.loads(recipe_path.read_text())
        target_report = json.loads(target_report_path.read_text())
        source = recipe["source_checkpoint"]
        qualification = recipe["source_qualification_report"]
        source_audit = recipe["source_audit"]
        source_path = _check(source, "FADA qualified Panda source checkpoint")
        qualification_report = json.loads(_check(
            qualification, "FADA Panda source qualification",
        ).read_text())
        audit = json.loads(_check(source_audit, "FADA Panda source audit").read_text())
        identity = (source["sha256"], qualification["sha256"], source_audit["sha256"])
        if source_identity is None:
            source_identity = identity
        if (identity != source_identity
                or qualification_report.get("status") != "complete"
                or qualification_report.get("task") != ("T3" if task == "T4" else task)
                or qualification_report.get("source_only") is not True
                or qualification_report.get("source_final_qualified") is not True
                or qualification_report.get("source_final_successes") != 4
                or qualification_report.get("source_final_requested") != 4
                or qualification_report.get("checkpoint_sha256") != source["sha256"]
                or Path(qualification_report["checkpoint"]).resolve() != source_path.resolve()
                or audit.get("passed") is not True
                or target_report.get("checkpoint_sha256") != resolved["checkpoint"]["sha256"]
                or recipe.get("task") not in {task, None}
                or (index > 1 and (recipe.get("instance_id") != instance_id
                                   or recipe.get("p6_confirmation_accessed") is not False))):
            raise ValueError(f"FADA {task}/{instance_id} source lineage differs")
        source_seed, target_seed = recipe.get("source_seed"), recipe.get("positive_budget_seed")
        if (not isinstance(source_seed, int) or not isinstance(target_seed, int)
                or source_seed in source_seeds or target_seed in target_seeds):
            raise ValueError("FADA source or target adaptation seed is reused")
        source_seeds.add(source_seed)
        target_seeds.add(target_seed)
        instances.append({
            "instance_id": instance_id,
            "target_checkpoint": resolved["checkpoint"],
            "recipe": _pointer(recipe_path),
            "target_report": _pointer(target_report_path),
            "source_seed": source_seed,
            "positive_budget_seed": target_seed,
        })
    result = {
        "version": "transfer-fada-source-lineage-v1",
        "status": "qualified_source_lineage_only",
        "task": task, "method": method,
        "method_identity": resolved["method_identity"],
        "source_checkpoint": source,
        "source_qualification_report": qualification,
        "source_audit": source_audit,
        "source_successes": 4, "source_requests": 4,
        "instances": instances,
        "distinct_target_adaptation_seeds": 8,
        "target_method_fidelity_qualified": False,
        "formal_request_seed_access": 0,
        "physical_calls": 0, "gpu_calls": 0,
    }
    output = experiment.output_dir / "validation/independent_source_lineage_v1/report.json"
    if output.exists():
        if json.loads(output.read_text()) != result:
            raise ValueError("FADA source-lineage audit changed")
    else:
        atomic_json(output, result)
    return result


def _dial_refresh_paths(task: str) -> tuple[Path, Path, Path]:
    if task not in TASK_ROOTS:
        raise ValueError("unknown DIAL refresh task")
    task_root = ROOT / TASK_ROOTS[task]
    config = task_root / "validation/dial_transfer_independent_source_nominal_refresh.json"
    output = (ROOT / "results" / TASK_ROOTS[task].relative_to("configs") /
              "baseline/dial/validation/independent_source_nominal_cohort_v2")
    return config, output / "launch_receipt.json", output


def freeze_dial_refresh(task: str) -> dict:
    """Pin current DIAL bytes before rerunning all eight task development actors."""
    config_path, receipt_path, output = _dial_refresh_paths(task)
    if config_path.exists() or output.exists():
        raise FileExistsError("DIAL current-source refresh already frozen or started")
    base, roster_path, old_report, _, _ = _source(task, "dial_transfer")
    controller = ROOT / "genedynamics/tasks/transfer/controllers/dial.py"
    source_manifest = ROOT / "genedynamics/solvers/single/dial/source_manifest.json"
    old_cohort = (ROOT / "results" / TASK_ROOTS[task].relative_to("configs") /
                  "baseline/dial/validation/independent_source_nominal_cohort_v1/independent_audit.json")
    old = json.loads(old_cohort.read_text())
    if (old.get("status") != "passed_development_only"
            or old.get("physical_evidence_verified") != 64
            or old.get("method_identity") != base["method_identity"]):
        raise ValueError("old DIAL development cohort is not an intact reference")
    for instance in range(1, 9):
        _instance_config(base, output, instance)
    config = {
        "version": "transfer-dial-current-source-nominal-cohort-v2",
        "task": task, "method": "dial_transfer",
        "method_identity": old_report["method_identity"],
        "task_config": _pointer(_task_config(task, "dial_transfer")),
        "source_roster": _pointer(roster_path),
        "previous_cohort": _pointer(old_cohort),
        "controller_source": _pointer(controller),
        "source_manifest": _pointer(source_manifest),
        "instance_ids": [f"instance_{index:02d}" for index in range(1, 9)],
        "output": str(output.relative_to(ROOT)),
        "maximum_physical_episodes": 64,
        "formal_request_seed_access": 0,
        "failure_replacement": False,
        "execution": {"device": "cuda", "cuda_visible_devices": "1",
                      "physical_uuid": GPU1_UUID},
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "version": config["version"], "status": "frozen_unexecuted",
        "config": _pointer(config_path),
        "controller_source": config["controller_source"],
        "source_manifest": config["source_manifest"],
        "maximum_physical_episodes": 64,
        "formal_request_seed_access": 0,
        "failure_replacement": False,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def _frozen_dial_refresh(task: str) -> tuple[dict, dict, Path]:
    config_path, receipt_path, output = _dial_refresh_paths(task)
    config = json.loads(config_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    base, roster_path, old_report, _, _ = _source(task, "dial_transfer")
    if (config.get("version") != "transfer-dial-current-source-nominal-cohort-v2"
            or config.get("task") != task or config.get("method") != "dial_transfer"
            or config.get("method_identity") != old_report["method_identity"]
            or config.get("task_config") != _pointer(_task_config(task, "dial_transfer"))
            or config.get("source_roster") != _pointer(roster_path)
            or config.get("instance_ids") != [f"instance_{i:02d}" for i in range(1, 9)]
            or config.get("output") != str(output.relative_to(ROOT))
            or config.get("maximum_physical_episodes") != 64
            or config.get("formal_request_seed_access") != 0
            or config.get("failure_replacement") is not False
            or config.get("execution") != {"device": "cuda", "cuda_visible_devices": "1",
                                          "physical_uuid": GPU1_UUID}
            or receipt.get("config") != _pointer(config_path)
            or receipt.get("controller_source") != config.get("controller_source")
            or receipt.get("source_manifest") != config.get("source_manifest")
            or receipt.get("maximum_physical_episodes") != 64
            or receipt.get("formal_request_seed_access") != 0):
        raise ValueError("DIAL current-source refresh freeze differs")
    _check(config["previous_cohort"], "old DIAL cohort")
    _check(config["controller_source"], "current DIAL controller")
    _check(config["source_manifest"], "pinned DIAL author source manifest")
    return config, base, output


def run_dial_refresh(task: str) -> dict:
    config, base, output = _frozen_dial_refresh(task)
    for instance in range(1, 9):
        _frozen_dial_refresh(task)
        child = _instance_config(base, output, instance)
        if (ROOT / child["output"]).exists():
            raise FileExistsError("DIAL refreshed instance already exists; no replacement")
        report = run_evaluation(child)
        audit = audit_evaluation(child)
        if report.get("status") != "complete" or audit.get("status") != "passed":
            failure = {
                "version": config["version"], "status": "failed_retained",
                "task": task, "method": "dial_transfer",
                "instance_id": child["instance_id"],
                "formal_request_seed_access": 0,
            }
            atomic_json(output / "failure_retained.json", failure)
            return failure
    return audit_dial_refresh(task)


def audit_dial_refresh(task: str) -> dict:
    config, base, output = _frozen_dial_refresh(task)
    instances, reference, outcomes = [], None, Counter()
    for instance in range(1, 9):
        child = _instance_config(base, output, instance)
        report_path = ROOT / child["output"] / "report.json"
        audit_path = ROOT / child["output"] / "independent_audit.json"
        report = json.loads(report_path.read_text())
        saved = json.loads(audit_path.read_text())
        checked = audit_evaluation(child, write=False)
        if (saved != checked or saved.get("status") != "passed"
                or saved.get("physical_evidence_verified") != 8
                or saved.get("runner_failures_retained") != 0
                or report.get("counts", {}).get("requested") != 8
                or report.get("counts", {}).get("execution_errors") != 0):
            raise ValueError("DIAL refreshed physical evidence differs")
        hashes = {row["request_id"]: row["reset_physical_hash"]
                  for row in saved["episodes"]}
        if len(hashes) != 8 or not all(hashes.values()):
            raise ValueError("DIAL refreshed reset evidence incomplete")
        if reference is None:
            reference = hashes
        elif hashes != reference:
            raise ValueError("DIAL refreshed instance resets differ")
        outcomes.update(row["outcome"] for row in report["results"])
        instances.append({"instance_id": child["instance_id"],
                          "report": _pointer(report_path),
                          "independent_audit": _pointer(audit_path)})
    result = {
        "version": config["version"], "status": "passed_development_only",
        "task": task, "method": "dial_transfer",
        "method_identity": config["method_identity"],
        "controller_source": config["controller_source"],
        "source_manifest": config["source_manifest"],
        "source_roster": config["source_roster"],
        "previous_cohort": config["previous_cohort"],
        "instances": instances,
        "physical_evidence_verified": 64,
        "reset_hashes_matched_to_first_instance": 64,
        "outcomes": dict(sorted(outcomes.items())),
        "success_is_not_an_admission_gate": True,
        "physical_calls": 0, "gpu_calls": 0,
        "formal_request_seed_access": 0,
        "formal_performance_evidence": False,
    }
    destination = output / "independent_audit.json"
    if destination.exists():
        if json.loads(destination.read_text()) != result:
            raise ValueError("DIAL current-source cohort audit changed")
    else:
        atomic_json(destination, result)
    return result


def audit_t1_shared_training_lineage() -> dict:
    """Check the eight paired T1 policy streams and common response models."""
    task = "T1"
    task_root = ROOT / TASK_ROOTS[task]
    experiment = ExperimentConfig.from_yaml(task_root / "_base.yaml")
    manifest_pointer = experiment.method_params["actor_binding"]["manifest"]
    manifest = validate_transfer_actor_manifest(manifest_pointer)
    artifacts = manifest["artifacts"]
    if any(manifest["main"][method].get("policy_set") != policy_set
           or manifest["main"][method].get("response_set") != "response_models"
           for method, policy_set in {
               "cross_ebdm": "cross_ebdm", "task_only": "task_only",
               "dial_transfer": "task_only", "response_mpc": "task_only",
           }.items()):
        raise ValueError("T1 current main-row actor bindings differ")
    policy = artifacts["policy_sets"]
    response = artifacts["response_models"]
    streams = artifacts["sample_streams"]["task_only"]
    if (manifest["task"] != task
            or len(streams) != 8
            or set(streams) != {f"instance_{i:02d}" for i in range(1, 9)}
            or len(set(streams.values())) != 8):
        raise ValueError("T1 paired training stream roster differs")
    rows = []
    hashes = {"ours": set(), "task_only": set(), "response": set()}
    for index in range(1, 9):
        instance = f"instance_{index:02d}"
        ours = policy["cross_ebdm"][instance]
        task_only = policy["task_only"][instance]
        model = response[instance]
        ours_path = _check(ours, f"T1/{instance} ours policy")
        task_path = _check(task_only, f"T1/{instance} task-only policy")
        model_path = _check(model, f"T1/{instance} response model")
        ours_report_path = ours_path.parent / "report.json"
        task_report_path = task_path.parent / "report.json"
        model_report_path = model_path.parent / "report.json"
        ours_report = json.loads(ours_report_path.read_text())
        task_report = json.loads(task_report_path.read_text())
        model_report = json.loads(model_report_path.read_text())
        if (ours_report.get("status") != "complete"
                or ours_report.get("passed") is not True
                or ours_report.get("task") != task
                or ours_report.get("main_updates") != 2000
                or ours_report.get("resume_exact") is not True
                or ours_report.get("matching_task_only_sample_stream") is not True
                or ours_report.get("sample_stream_sha256") != streams[instance]
                or ours_report.get("candidate_checkpoint") != ours
                or ours_report.get("task_only_checkpoint") != task_only
                or ours_report.get("selection", {}).get("accepted") is not True
                or ours_report.get("p6_accessed") is not False
                or ours_report.get("new_unknown_target_calls") != 0
                or task_report.get("main_updates") != 2000
                or task_report.get("resume_exact") is not True
                or task_report.get("sample_stream_sha256") != streams[instance]
                or task_report.get("checkpoint_sha256") != task_only["sha256"]
                or model_report.get("updates") != 2000
                or model_report.get("resume_exact") is not True
                or model_report.get("checkpoint_sha256") != model["sha256"]
                or model_report.get("held_out_source_nominal_validation", {}).get("windows", 0) <= 0):
            raise ValueError(f"T1/{instance} paired training lineage differs")
        hashes["ours"].add(ours["sha256"])
        hashes["task_only"].add(task_only["sha256"])
        hashes["response"].add(model["sha256"])
        rows.append({
            "instance_id": instance, "sample_stream_sha256": streams[instance],
            "ours_policy": ours, "ours_training_report": _pointer(ours_report_path),
            "task_only_policy": task_only,
            "task_only_training_report": _pointer(task_report_path),
            "response_model": model,
            "response_training_report": _pointer(model_report_path),
        })
    if any(len(values) != 8 for values in hashes.values()):
        raise ValueError("T1 learned actor artifacts reuse an instance identity")
    result = {
        "version": "transfer-t1-shared-training-lineage-v1",
        "status": "passed_training_lineage_only",
        "task": task,
        "methods": ["cross_ebdm", "task_only", "dial_transfer", "response_mpc"],
        "manifest": manifest_pointer,
        "instances": rows,
        "distinct_sample_streams": 8,
        "distinct_policy_and_response_artifacts": {key: len(value) for key, value in hashes.items()},
        "pairing": "same instance task-only policy and response model are used by Ours, DIAL, and ordinary response MPC",
        "physical_calls": 0, "gpu_calls": 0,
        "formal_request_seed_access": 0,
        "formal_performance_evidence": False,
    }
    output = (ROOT / "results" / TASK_ROOTS[task].relative_to("configs") /
              "validation/independent_shared_training_lineage_v1/report.json")
    if output.exists():
        if json.loads(output.read_text()) != result:
            raise ValueError("T1 shared training lineage audit changed")
    else:
        atomic_json(output, result)
    return result


def audit_t2_shared_training_lineage() -> dict:
    """Check the audited T2 eight-instance policy and shared-response cohort."""
    task = "T2"
    task_root = ROOT / TASK_ROOTS[task]
    experiment = ExperimentConfig.from_yaml(task_root / "_base.yaml")
    manifest_pointer = experiment.method_params["actor_binding"]["manifest"]
    manifest = validate_transfer_actor_manifest(manifest_pointer)
    artifacts = manifest["artifacts"]
    if any(manifest["main"][method].get("policy_set") != policy_set
           or manifest["main"][method].get("response_set") != "response_models"
           for method, policy_set in {
               "cross_ebdm": "cross_ebdm", "task_only": "task_only",
               "dial_transfer": "task_only",
               "response_mpc": "fixed_response_companion_policy",
           }.items()):
        raise ValueError("T2 current main-row actor bindings differ")
    companion = artifacts["fixed_response_companion_policy"]
    companion_path = _check(companion, "T2 response MPC fixed companion policy")
    companion_report_path = companion_path.parent / "report.json"
    companion_report = json.loads(companion_report_path.read_text())
    if companion_report.get("checkpoint_sha256") != companion["sha256"]:
        raise ValueError("T2 response MPC fixed companion training report differs")
    cohort = (ROOT / "results" / TASK_ROOTS[task].relative_to("configs") /
              "training/policy_instances_v3")
    report_path = cohort / "report.json"
    audit_path = cohort / "artifact_audit.json"
    report = json.loads(report_path.read_text())
    audit = json.loads(audit_path.read_text())
    if (manifest.get("task") != task
            or report.get("status") != "complete"
            or report.get("passed") is not True
            or report.get("new_physics_calls") != 0
            or report.get("confirmation_states_opened") != 0
            or report.get("old_invalid_outcomes_used") is not False
            or report.get("training_updates") != 96000
            or audit.get("status") != "passed"
            or audit.get("passed") is not True
            or audit.get("issues") != []
            or audit.get("training_report") != _pointer(report_path)
            or audit.get("counts") != {
                "instances": 8, "selected_checkpoints": 48,
                "training_updates": 96000, "unique_sample_streams": 8,
            }
            or audit.get("data_policy") != {
                "confirmation_states_opened": 0,
                "new_physics_calls": 0,
                "old_invalid_outcomes_used": False,
            }):
        raise ValueError("T2 independent policy cohort or artifact audit differs")
    for item in audit["inventory"]:
        _check({"path": item["path"], "sha256": item["sha256"]},
               "T2 audited selected checkpoint")
    streams = artifacts["sample_streams"]["task_only"]
    instances = report.get("instances", {})
    expected = {f"instance_{index:02d}" for index in range(1, 9)}
    if (set(instances) != expected or set(streams) != expected
            or len({row["seed"] for row in instances.values()}) != 8
            or len({row["sample_stream_sha256"] for row in instances.values()}) != 8):
        raise ValueError("T2 independent training seed or sample stream reused")
    rows = []
    for instance in sorted(expected):
        row = instances[instance]
        ours = artifacts["policy_sets"]["cross_ebdm"][instance]
        task_only = artifacts["policy_sets"]["task_only"][instance]
        response = artifacts["response_models"][instance]
        if (row.get("status") != "complete"
                or row.get("instance") != instance
                or row.get("sample_stream_sha256") != streams[instance]
                or row.get("response_checkpoint") != response
                or row["branches"]["gated_full"]["selected_checkpoint"] != ours
                or row["branches"]["gated_full"]["selection"]["accepted"] is not True
                or row["branches"]["task_only"]["selected_checkpoint"] != task_only):
            raise ValueError(f"T2/{instance} paired checkpoint lineage differs")
        for name, pointer in (("ours", ours), ("task_only", task_only),
                              ("response", response)):
            _check(pointer, f"T2/{instance} {name} checkpoint")
        rows.append({
            "instance_id": instance, "training_seed": row["seed"],
            "sample_stream_sha256": streams[instance],
            "ours_policy": ours, "task_only_policy": task_only,
            "response_model": response,
        })
    result = {
        "version": "transfer-t2-shared-training-lineage-v1",
        "status": "passed_training_lineage_only",
        "task": task,
        "methods": ["cross_ebdm", "task_only", "dial_transfer", "response_mpc"],
        "manifest": manifest_pointer,
        "training_report": _pointer(report_path),
        "artifact_audit": _pointer(audit_path),
        "fixed_response_companion_policy": companion,
        "fixed_response_companion_training_report": _pointer(companion_report_path),
        "instances": rows,
        "distinct_sample_streams": 8,
        "distinct_training_seeds": 8,
        "pairing": "Ours and DIAL use the per-instance task-only policy and response model; ordinary response MPC uses one fixed companion policy plus the per-instance response model",
        "physical_calls": 0, "gpu_calls": 0,
        "formal_request_seed_access": 0,
        "formal_performance_evidence": False,
    }
    output = (ROOT / "results" / TASK_ROOTS[task].relative_to("configs") /
              "validation/independent_shared_training_lineage_v1/report.json")
    if output.exists():
        if json.loads(output.read_text()) != result:
            raise ValueError("T2 shared training lineage audit changed")
    else:
        atomic_json(output, result)
    return result


def qualify_shared_model_formal_cell(task: str, method: str) -> dict:
    """Admit a source-audited T1/T2 actor without opening formal requests."""
    if task not in {"T1", "T2"} or method not in {
        "cross_ebdm", "task_only", "dial_transfer", "response_mpc",
    }:
        raise ValueError("shared-model formal admission covers T1/T2 learned actors")
    experiment = ExperimentConfig.from_yaml(_task_config(task, method))
    manifest_pointer = experiment.method_params["actor_binding"]["manifest"]
    manifest = validate_transfer_actor_manifest(manifest_pointer)
    row = manifest["main"][method]
    identity = row["method_identity"]
    if (experiment.method != method or row["status"] != "instances_ready"
            or row["actor"] != method):
        raise ValueError("current learned main-row binding differs")
    result_root = ROOT / "results" / TASK_ROOTS[task].relative_to("configs")
    lineage_path = (result_root / "validation" /
                    "independent_shared_training_lineage_v1/report.json")
    lineage = json.loads(lineage_path.read_text())
    if (lineage.get("status") != "passed_training_lineage_only"
            or lineage.get("task") != task or method not in lineage.get("methods", ())
            or lineage.get("manifest") != manifest_pointer
            or len(lineage.get("instances", ())) != 8
            or lineage.get("formal_request_seed_access") != 0):
        raise ValueError("shared-model training lineage is not qualified")
    base, output_folder = (
        ("main/cross_ebdm", "main/cross_ebdm") if method == "cross_ebdm" else
        ("baseline/dial", "baseline/dial") if method == "dial_transfer" else
        (f"baseline/{method}", f"baseline/{method}")
    )
    cohort = "v2" if method == "dial_transfer" else "v1"
    source_path = (result_root / base / "validation" /
                   f"independent_source_nominal_cohort_{cohort}" /
                   "independent_audit.json")
    source = json.loads(source_path.read_text())
    if (source.get("status") != "passed_development_only"
            or source.get("task") != task or source.get("method") != method
            or source.get("method_identity") != identity
            or source.get("physical_evidence_verified") != 64
            or source.get("formal_performance_evidence") is not False
            or source.get("formal_request_seed_access") != 0
            or len(source.get("instances", ())) != 8):
        raise ValueError("learned source/nominal qualification differs")
    implementation = [ROOT / "genedynamics/tasks/transfer/controllers/bindings.py"]
    if method in {"cross_ebdm", "task_only"}:
        implementation.append(ROOT / "genedynamics/tasks/transfer/controllers/cross_ebdm/arm_dense.py")
    elif method == "response_mpc":
        implementation.append(ROOT / "genedynamics/solvers/single/response_mpc/arm.py")
    else:
        implementation.extend([
            ROOT / "genedynamics/tasks/transfer/controllers/dial.py",
            ROOT / "genedynamics/solvers/single/dial/source_manifest.json",
            ROOT / "genedynamics/solvers/single/dial/backends/dial_jax.py",
            ROOT / "test/unit/test_dial_author_source_parity.py",
        ])
        if (source.get("controller_source") != _pointer(implementation[1])
                or source.get("source_manifest") != _pointer(implementation[2])):
            raise ValueError("DIAL source/nominal audit is stale against deployed source")
    roster_pointer, _ = _roster(task)
    binding_hashes = {}
    for index in range(1, 9):
        instance = f"instance_{index:02d}"
        binding = _binding(task, method, instance,
                           {"method_identity": identity}, experiment)
        binding_hashes[instance] = hashlib.sha256(
            json.dumps(binding, sort_keys=True).encode()
        ).hexdigest()
    fidelity_path = (result_root / base / "validation" /
                     "independent_method_fidelity_v1/report.json")
    admission_path = _admission_path(task, method)
    formal_dir = ROOT / TASK_ROOTS[task] / "confirmation/independent_main_v1"
    formal_paths = [formal_dir / f"{method}_instance_{index:02d}.json"
                    for index in range(1, 9)]
    if any(path.exists() for path in (fidelity_path, admission_path, *formal_paths)):
        raise FileExistsError("learned actor formal admission already exists")
    fidelity = {
        "version": "transfer-shared-model-learned-fidelity-v1",
        "status": "passed", "task": task, "method": method,
        "method_identity": identity,
        "scope": ("task-adapted DIAL reverse update with public response rollout"
                  if method == "dial_transfer" else
                  "internal method or ablation identity on the shared task interface"),
        "task_config": _pointer(_task_config(task, method)),
        "manifest": manifest_pointer,
        "training_lineage": _pointer(lineage_path),
        "source_nominal_qualification": _pointer(source_path),
        "source_files": [_pointer(path) for path in implementation],
        "physical_calls": 0, "gpu_calls": 0,
        "formal_request_seed_access": 0,
    }
    atomic_json(fidelity_path, fidelity)
    admission = {
        "version": ADMISSION_VERSION, "status": "passed",
        "independent_table_eligible": True,
        "task": task, "method": method, "method_identity": identity,
        "instance_ids": list(binding_hashes),
        "request_roster": roster_pointer,
        "source_nominal_qualification": _pointer(source_path),
        "method_fidelity_review": _pointer(fidelity_path),
        "actor_binding_sha256": binding_hashes,
        "formal_request_seed_access": 0,
        "physical_calls": 0, "gpu_calls": 0,
    }
    atomic_json(admission_path, admission)
    for instance, formal_path in zip(binding_hashes, formal_paths):
        output = (Path("results") / TASK_ROOTS[task].relative_to("configs") /
                  output_folder / "confirmation/independent_main_v1" / instance)
        config = {
            "version": FORMAL_VERSION, "task": task, "method": method,
            "method_identity": identity, "instance_id": instance,
            "task_config": _pointer(_task_config(task, method)),
            "request_roster": roster_pointer,
            "cell_admission": _pointer(admission_path),
            "output": str(output),
            "plan_output": str(output.parent / f"{instance}_plan.json"),
            "execution": {"device": "cuda", "cuda_visible_devices": "1",
                          "physical_uuid": GPU1_UUID},
        }
        formal_plan(config)
        atomic_json(formal_path, config)
    return {
        "status": "passed", "task": task, "method": method,
        "fidelity": _pointer(fidelity_path),
        "admission": _pointer(admission_path),
        "formal_configs": [_pointer(path) for path in formal_paths],
        "formal_policy_episodes_accessed": 0,
    }


def qualify_fada_task_adapted_formal_cell(task: str) -> dict:
    """Admit the source-derived Planner-IDM port with its adaptation scope explicit."""
    method = "fada"
    experiment = ExperimentConfig.from_yaml(_task_config(task, method))
    manifest_pointer = experiment.method_params["actor_binding"]["manifest"]
    manifest = validate_transfer_actor_manifest(manifest_pointer)
    binding_row = manifest["main"][method]
    identity = binding_row["method_identity"]
    if (experiment.method != method or binding_row["status"] != "instances_ready"
            or binding_row["actor"] != method
            or binding_row["checkpoint_set"] != "fada"
            or identity != "FADA_task_adapted"):
        raise ValueError("FADA current main-row identity differs")
    parity_path = (ROOT / "results/_analysis/four_task_transfer" /
                   "fada_task_adapted_parity_20261031/report.json")
    parity = json.loads(parity_path.read_text())
    if (parity.get("status") != "passed"
            or parity.get("test_result", {}).get("passed") != 34
            or parity.get("test_result", {}).get("skipped") != 0
            or parity.get("physical_calls") != 0
            or parity.get("formal_request_seed_access") != 0):
        raise ValueError("FADA author-source and interface parity receipt differs")
    _check(parity["source_manifest"], "FADA source manifest")
    for test in parity["test_files"]:
        _check(test, "FADA source/interface parity test")
    site = parity["environment"]["isolated_mujoco_pytest_site"]
    tests = [row["path"] for row in parity["test_files"]]
    code = ("import numpy, scipy, sys; sys.path.insert(0, " + repr(site) +
            "); import pytest; raise SystemExit(pytest.main(" +
            repr(["-q", *tests]) + "))")
    environment = os.environ.copy()
    environment.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", PYTHONPATH=".")
    verification = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=environment,
        capture_output=True, text=True, timeout=120,
    )
    if (verification.returncode != 0 or "34 passed" not in verification.stdout
            or "skipped" in verification.stdout):
        raise ValueError("live FADA source/interface parity no longer passes")
    result_root = ROOT / "results" / TASK_ROOTS[task].relative_to("configs")
    source_lineage_path = (result_root / "baseline/fada/validation" /
                           "independent_source_lineage_v1/report.json")
    lineage = json.loads(source_lineage_path.read_text())
    if (lineage.get("status") != "qualified_source_lineage_only"
            or lineage.get("task") != task or lineage.get("method") != method
            or lineage.get("method_identity") != identity
            or lineage.get("source_successes") != 4
            or lineage.get("source_requests") != 4
            or lineage.get("distinct_target_adaptation_seeds") != 8
            or lineage.get("formal_request_seed_access") != 0
            or len(lineage.get("instances", ())) != 8):
        raise ValueError("FADA source and target lineage differs")
    for label in ("source_checkpoint", "source_qualification_report", "source_audit"):
        _check(lineage[label], f"FADA {label}")
    expected = {f"instance_{index:02d}" for index in range(1, 9)}
    rows = lineage["instances"]
    if ({row["instance_id"] for row in rows} != expected
            or len({row["positive_budget_seed"] for row in rows}) != 8):
        raise ValueError("FADA target adaptation instances or seeds reused")
    artifacts = manifest["artifacts"]["checkpoint_sets"]["fada"]
    target_reports = []
    for row in rows:
        instance = row["instance_id"]
        if artifacts[instance] != row["target_checkpoint"]:
            raise ValueError(f"FADA/{task}/{instance} main checkpoint differs")
        _check(row["target_checkpoint"], f"FADA/{task}/{instance} target checkpoint")
        recipe = json.loads(_check(row["recipe"], "FADA target recipe").read_text())
        report = json.loads(_check(row["target_report"], "FADA target training report").read_text())
        if (recipe.get("source_checkpoint") != lineage["source_checkpoint"]
                or recipe.get("source_qualification_report") != lineage["source_qualification_report"]
                or recipe.get("source_audit") != lineage["source_audit"]
                or recipe.get("positive_budget_seed") != row["positive_budget_seed"]
                or recipe.get("updates_per_budget") != 2000
                or 60 not in recipe.get("budgets_seconds", ())
                or recipe.get("native_recipe", {}).get("finetune_method") != "full"
                or recipe.get("new_target_calls") != 0
                or recipe.get("new_physics_calls") != 0
                or report.get("checkpoint_sha256") != row["target_checkpoint"]["sha256"]
                or report.get("main_updates") != 2000
                or report.get("resume_exact") is not True
                or report.get("planner_frozen") is not True
                or report.get("normalization_frozen") is not True
                or report.get("all_target_controls") != 3000
                or report.get("knownnominal_validation", {}).get("windows", 0) <= 0):
            raise ValueError(f"FADA/{task}/{instance} native adaptation differs")
        target_reports.append(row["target_report"])
    source_path = (result_root / "baseline/fada/validation" /
                   "independent_source_nominal_cohort_v1/independent_audit.json")
    source = json.loads(source_path.read_text())
    if (source.get("status") != "passed_development_only"
            or source.get("task") != task or source.get("method") != method
            or source.get("method_identity") != identity
            or source.get("physical_evidence_verified") != 64
            or source.get("formal_performance_evidence") is not False
            or source.get("formal_request_seed_access") != 0
            or len(source.get("instances", ())) != 8):
        raise ValueError("FADA eight-instance source/nominal qualification differs")
    roster_pointer, _ = _roster(task)
    binding_hashes = {}
    for index in range(1, 9):
        instance = f"instance_{index:02d}"
        binding = _binding(task, method, instance,
                           {"method_identity": identity}, experiment)
        binding_hashes[instance] = hashlib.sha256(
            json.dumps(binding, sort_keys=True).encode()
        ).hexdigest()
    fidelity_path = (result_root / "baseline/fada/validation" /
                     "independent_method_fidelity_v1/report.json")
    admission_path = _admission_path(task, method)
    formal_dir = ROOT / TASK_ROOTS[task] / "confirmation/independent_main_v1"
    formal_paths = [formal_dir / f"fada_instance_{index:02d}.json"
                    for index in range(1, 9)]
    if any(path.exists() for path in (fidelity_path, admission_path, *formal_paths)):
        raise FileExistsError("FADA formal admission already exists")
    source_files = [
        ROOT / "genedynamics/solvers/single/fada/source_manifest.json",
        ROOT / "genedynamics/solvers/single/fada/backends/model.py",
        ROOT / "genedynamics/solvers/single/fada/learning/adaptation.py",
        ROOT / "genedynamics/solvers/single/fada/learning/trainer.py",
        ROOT / "genedynamics/tasks/transfer/controllers/bindings.py",
    ]
    fidelity = {
        "version": "transfer-fada-task-adapted-fidelity-v1",
        "status": "passed", "task": task, "method": method,
        "method_identity": identity,
        "scope": "author-derived Planner-IDM and native full-IDM target adaptation on public task records; original oracle PPO and DAgger collection excluded",
        "full_original_fada_equivalence": False,
        "task_config": _pointer(_task_config(task, method)),
        "manifest": manifest_pointer,
        "source_lineage": _pointer(source_lineage_path),
        "target_training_reports": target_reports,
        "source_parity": _pointer(parity_path),
        "source_nominal_qualification": _pointer(source_path),
        "source_files": [_pointer(path) for path in source_files],
        "physical_calls": 0, "gpu_calls": 0,
        "formal_request_seed_access": 0,
    }
    atomic_json(fidelity_path, fidelity)
    admission = {
        "version": ADMISSION_VERSION, "status": "passed",
        "independent_table_eligible": True,
        "task": task, "method": method, "method_identity": identity,
        "instance_ids": list(binding_hashes), "request_roster": roster_pointer,
        "source_nominal_qualification": _pointer(source_path),
        "method_fidelity_review": _pointer(fidelity_path),
        "actor_binding_sha256": binding_hashes,
        "formal_request_seed_access": 0, "physical_calls": 0, "gpu_calls": 0,
    }
    atomic_json(admission_path, admission)
    for instance, formal_path in zip(binding_hashes, formal_paths):
        output = (Path("results") / TASK_ROOTS[task].relative_to("configs") /
                  "baseline/fada/confirmation/independent_main_v1" / instance)
        config = {
            "version": FORMAL_VERSION, "task": task, "method": method,
            "method_identity": identity, "instance_id": instance,
            "task_config": _pointer(_task_config(task, method)),
            "request_roster": roster_pointer,
            "cell_admission": _pointer(admission_path),
            "output": str(output),
            "plan_output": str(output.parent / f"{instance}_plan.json"),
            "execution": {"device": "cuda", "cuda_visible_devices": "1",
                          "physical_uuid": GPU1_UUID},
        }
        formal_plan(config)
        atomic_json(formal_path, config)
    return {
        "status": "passed", "task": task, "method": method,
        "fidelity": _pointer(fidelity_path),
        "admission": _pointer(admission_path),
        "formal_configs": [_pointer(path) for path in formal_paths],
        "formal_policy_episodes_accessed": 0,
    }


def audit_current_t3_t4_actor_lineage(task: str) -> dict:
    """Check the deployed T3/T4 actor sets, including their later selectors."""
    if task not in {"T3", "T4"}:
        raise ValueError("current actor lineage covers T3/T4 only")
    experiment = ExperimentConfig.from_yaml(ROOT / TASK_ROOTS[task] / "_base.yaml")
    manifest_pointer = experiment.method_params["actor_binding"]["manifest"]
    manifest = validate_transfer_actor_manifest(manifest_pointer)
    artifacts = manifest["artifacts"]
    main = manifest["main"]
    sets = artifacts["policy_sets"]
    expected = {f"instance_{index:02d}" for index in range(1, 9)}
    checks = {}
    for name, values in (
        ("task_only", sets[main["task_only"]["policy_set"]]),
        ("dial", sets[main["dial_transfer"]["policy_set"]]),
        ("response_mpc", sets[main["response_mpc"]["policy_set"]]),
        ("response_models", artifacts["response_models"]),
    ):
        if set(values) != expected or len({p["sha256"] for p in values.values()}) != 8:
            raise ValueError(f"{task}/{name} current actor identity reused")
        for instance, pointer in values.items():
            _check(pointer, f"{task}/{name}/{instance}")
        checks[name] = 8
    if task == "T3":
        ours = main["cross_ebdm"]
        if (ours.get("method_identity") != "ours_task_first_mode_specific_dual_physical_selected_v21"
                or ours.get("admission_policy_set") != "task_only_shared_start_v9"
                or ours.get("policy_sets_by_mode") != {
                    "0": "cross_ebdm_mode_0_v21", "1": "cross_ebdm_mode_1_v21",
                }):
            raise ValueError("T3 current mode-specific main actor differs")
        reports = {}
        for name, pointer in ours["training_reports"].items():
            report = json.loads(_check(pointer, f"T3 {name} training report").read_text())
            if (report.get("status") != "complete" or report.get("passed") is not True
                    or report.get("new_physics_calls") != 0
                    or set(report.get("instances", {})) != expected):
                raise ValueError(f"T3 {name} training report differs")
            reports[name] = report
        start = sets["task_only_shared_start_v9"]
        for instance in expected:
            if reports["shared_start"]["instances"][instance]["selected_checkpoint"] != start[instance]:
                raise ValueError(f"T3/{instance} shared-start checkpoint differs")
        final = json.loads(_check(ours["qualification_report"],
                                  "T3 current qualification report").read_text())
        if (final.get("status") != "complete_recorded_transport_confirmation"
                or final.get("passed") is not True
                or final.get("contribution_contrast", {}).get(
                    "transport_claim_supported_confirmation") is not True):
            raise ValueError("T3 final mode-specific actor qualification differs")
        selector_matches = {}
        fallback = {}
        for mode in (0, 1):
            key = f"mode_{mode}"
            selected = sets[ours["policy_sets_by_mode"][str(mode)]]
            selector = json.loads(_check(ours["selection_reports"][key],
                                         f"T3 {key} physical selector").read_text())
            if (selector.get("passed") is not True
                    or selector.get("status") != "complete_recorded_physical_checkpoint_selection"):
                raise ValueError(f"T3 {key} physical selector differs")
            chosen = selector["physical_checkpoint_selection"]["instances"]
            if set(chosen) != expected or set(selected) != expected:
                raise ValueError(f"T3 {key} selected instance roster differs")
            matched = []
            backed_off = []
            for instance in sorted(expected):
                _check(selected[instance], f"T3/{key}/{instance} selected policy")
                if selected[instance] == chosen[instance]["selected_checkpoint"]:
                    matched.append(instance)
                elif selected[instance] == start[instance]:
                    backed_off.append(instance)
                else:
                    raise ValueError(f"T3/{key}/{instance} not selected or fallback")
            selector_matches[key] = matched
            fallback[key] = backed_off
        safety = json.loads(_check(ours["selection_reports"]["mode_1_safety"],
                                   "T3 mode-1 safety contrast").read_text())
        if safety.get("passed") is not True:
            raise ValueError("T3 mode-1 safety contrast differs")
        if (len(selector_matches["mode_0"]) != 8
                or len(selector_matches["mode_1"]) != 7
                or fallback != {"mode_0": [], "mode_1": ["instance_05"]}):
            raise ValueError("T3 final mode-specific fallback roster differs")
        detail = {
            "training_reports": ours["training_reports"],
            "selection_reports": ours["selection_reports"],
            "qualification_report": ours["qualification_report"],
            "selector_matches": selector_matches,
            "task_only_fallback": fallback,
            "mode_1_same_checkpoint_as_task_only": sum(
                sets["cross_ebdm_mode_1_v21"][instance] == start[instance]
                for instance in expected
            ),
            "raw_unprojected_selection": None,
        }
    else:
        ours = main["cross_ebdm"]
        if (ours.get("method_identity") != "ours_gated_v2_completion_projected"
                or ours.get("policy_set") != "cross_ebdm_completion_projected"
                or ours.get("actor_parameters", {}).get(
                    "completion_admission_version") != "public_completion_dominance_v1"):
            raise ValueError("T4 current completion-projected actor differs")
        result_root = ROOT / "results" / TASK_ROOTS[task].relative_to("configs")
        base = result_root / "training/completion_tail_aligned_transport/checkpoints"
        raw_path = base / "report.json"
        raw = json.loads(raw_path.read_text())
        if raw.get("status") != "selection_failed" or raw.get("passed") is not False:
            raise ValueError("T4 raw unprojected training rejection changed")
        preflight_path = (result_root / "validation/completion_projected_transport" /
                          "preflight/report.json")
        cohort_path = (result_root / "validation/completion_projected_transport" /
                       "all_instances/report.json")
        preflight = json.loads(preflight_path.read_text())
        cohort = json.loads(cohort_path.read_text())
        selected = sets[ours["policy_set"]]
        if (preflight.get("status") != "complete" or preflight.get("passed") is not True
                or set(preflight.get("instances", {})) != expected
                or cohort.get("status") != "complete" or cohort.get("passed") is not True
                or cohort.get("counts", {}).get("completed_episodes") != 64
                or cohort.get("counts", {}).get("task_successes") != 64
                or cohort.get("counts", {}).get("constraint_violations") != 0
                or set(cohort.get("instances", {})) != expected):
            raise ValueError("T4 completion-projected actor qualification differs")
        for instance in sorted(expected):
            _check(selected[instance], f"T4/{instance} completion-projected policy")
            if (preflight["instances"][instance].get("accepted") is not True
                    or preflight["instances"][instance]["candidate_checkpoint"] != selected[instance]
                    or cohort["instances"][instance]["episodes"] != 8
                    or cohort["instances"][instance]["successes"] != 8
                    or cohort["instances"][instance]["candidate_admission_fraction"] < .25):
                raise ValueError(f"T4/{instance} composed actor failed qualification")
        detail = {
            "raw_training_report": _pointer(raw_path),
            "projection_preflight": _pointer(preflight_path),
            "composed_actor_cohort": _pointer(cohort_path),
            "raw_unprojected_selection": "selection_failed",
            "composed_projection_qualified": True,
            "minimum_instance_candidate_admission_fraction": min(
                row["candidate_admission_fraction"] for row in cohort["instances"].values()
            ),
        }
    result = {
        "version": f"transfer-{task.lower()}-current-actor-lineage-v1",
        "status": "passed_current_actor_lineage_only",
        "task": task, "manifest": manifest_pointer,
        "methods": ["cross_ebdm", "task_only", "dial_transfer", "response_mpc"],
        "current_actor_artifact_counts": checks,
        "single_factor_transport_attribution": False,
        **detail,
        "physical_calls": 0, "gpu_calls": 0,
        "formal_request_seed_access": 0,
        "formal_performance_evidence": False,
    }
    output = (ROOT / "results" / TASK_ROOTS[task].relative_to("configs") /
              "validation/independent_current_actor_lineage_v1/report.json")
    if output.exists():
        if json.loads(output.read_text()) != result:
            raise ValueError(f"{task} current actor lineage changed")
    else:
        atomic_json(output, result)
    return result


def qualify_current_t3_t4_formal_cell(task: str, method: str) -> dict:
    """Admit a current T3/T4 actor after its independent physical cohort."""
    if task not in {"T3", "T4"} or method not in {
        "cross_ebdm", "task_only", "dial_transfer", "response_mpc",
    }:
        raise ValueError("current actor formal admission covers T3/T4 legacy methods")
    experiment = ExperimentConfig.from_yaml(_task_config(task, method))
    manifest_pointer = experiment.method_params["actor_binding"]["manifest"]
    manifest = validate_transfer_actor_manifest(manifest_pointer)
    row = manifest["main"][method]
    identity = row["method_identity"]
    if (experiment.method != method or row.get("status") != "instances_ready"
            or row.get("actor") != method):
        raise ValueError("current main actor binding differs")
    result_root = ROOT / "results" / TASK_ROOTS[task].relative_to("configs")
    lineage_path = (result_root / "validation" /
                    "independent_current_actor_lineage_v1/report.json")
    lineage = json.loads(lineage_path.read_text())
    if (lineage.get("status") != "passed_current_actor_lineage_only"
            or lineage.get("task") != task or lineage.get("manifest") != manifest_pointer
            or method not in lineage.get("methods", ())
            or lineage.get("single_factor_transport_attribution") is not False
            or lineage.get("formal_request_seed_access") != 0):
        raise ValueError("current T3/T4 actor lineage differs")
    folder = ("main/cross_ebdm" if method == "cross_ebdm" else
              "baseline/dial" if method == "dial_transfer" else
              f"baseline/{method}")
    cohort = "v2" if method == "dial_transfer" else "v1"
    source_path = (result_root / folder / "validation" /
                   f"independent_source_nominal_cohort_{cohort}" /
                   "independent_audit.json")
    source = json.loads(source_path.read_text())
    if (source.get("status") != "passed_development_only"
            or source.get("task") != task or source.get("method") != method
            or source.get("method_identity") != identity
            or source.get("physical_evidence_verified") != 64
            or source.get("formal_performance_evidence") is not False
            or source.get("formal_request_seed_access") != 0
            or len(source.get("instances", ())) != 8):
        raise ValueError("current T3/T4 source/nominal cohort differs")
    sources = [ROOT / "genedynamics/tasks/transfer/controllers/bindings.py"]
    if method in {"cross_ebdm", "task_only"}:
        sources.append(ROOT / "genedynamics/tasks/transfer/controllers/cross_ebdm" /
                       ("dense.py" if task == "T3" else "mobile_dense.py"))
        if method == "cross_ebdm" and task == "T4":
            sources.append(ROOT / "genedynamics/tasks/transfer/controllers/recovery_execution.py")
    elif method == "response_mpc":
        sources.append(ROOT / ("genedynamics/solvers/single/response_mpc/contact.py"
                               if task == "T3" else
                               "genedynamics/tasks/transfer/controllers/cross_ebdm/mobile_dense.py"))
    else:
        sources.extend([
            ROOT / "genedynamics/tasks/transfer/controllers/dial.py",
            ROOT / "genedynamics/solvers/single/dial/source_manifest.json",
            ROOT / "genedynamics/solvers/single/dial/backends/dial_jax.py",
            ROOT / "test/unit/test_dial_author_source_parity.py",
        ])
        if (source.get("controller_source") != _pointer(sources[1])
                or source.get("source_manifest") != _pointer(sources[2])):
            raise ValueError("T3/T4 DIAL source audit is stale")
    roster_pointer, _ = _roster(task)
    binding_hashes = {}
    for index in range(1, 9):
        instance = f"instance_{index:02d}"
        selected = _binding(task, method, instance,
                            {"method_identity": identity}, experiment)
        binding_hashes[instance] = hashlib.sha256(
            json.dumps(selected, sort_keys=True).encode()
        ).hexdigest()
    fidelity_path = (result_root / folder / "validation" /
                     "independent_method_fidelity_v1/report.json")
    admission_path = _admission_path(task, method)
    formal_dir = ROOT / TASK_ROOTS[task] / "confirmation/independent_main_v1"
    formal_paths = [formal_dir / f"{method}_instance_{index:02d}.json"
                    for index in range(1, 9)]
    if any(path.exists() for path in (fidelity_path, admission_path, *formal_paths)):
        raise FileExistsError("current legacy formal admission already exists")
    fidelity = {
        "version": "transfer-current-t3-t4-learned-fidelity-v1",
        "status": "passed", "task": task, "method": method,
        "method_identity": identity,
        "scope": ("task-adapted DIAL with public response rollout" if method == "dial_transfer"
                  else "current mode-specific or projected composite actor" if method == "cross_ebdm"
                  else "current internal task-only or ordinary response MPC baseline"),
        "single_factor_transport_attribution": False,
        "task_config": _pointer(_task_config(task, method)),
        "manifest": manifest_pointer,
        "current_actor_lineage": _pointer(lineage_path),
        "source_nominal_qualification": _pointer(source_path),
        "source_files": [_pointer(path) for path in sources],
        "physical_calls": 0, "gpu_calls": 0,
        "formal_request_seed_access": 0,
    }
    atomic_json(fidelity_path, fidelity)
    admission = {
        "version": ADMISSION_VERSION, "status": "passed",
        "independent_table_eligible": True,
        "task": task, "method": method, "method_identity": identity,
        "instance_ids": list(binding_hashes), "request_roster": roster_pointer,
        "source_nominal_qualification": _pointer(source_path),
        "method_fidelity_review": _pointer(fidelity_path),
        "actor_binding_sha256": binding_hashes,
        "formal_request_seed_access": 0,
        "physical_calls": 0, "gpu_calls": 0,
    }
    atomic_json(admission_path, admission)
    for instance, formal_path in zip(binding_hashes, formal_paths):
        output = (Path("results") / TASK_ROOTS[task].relative_to("configs") /
                  folder / "confirmation/independent_main_v1" / instance)
        config = {
            "version": FORMAL_VERSION, "task": task, "method": method,
            "method_identity": identity, "instance_id": instance,
            "task_config": _pointer(_task_config(task, method)),
            "request_roster": roster_pointer,
            "cell_admission": _pointer(admission_path),
            "output": str(output),
            "plan_output": str(output.parent / f"{instance}_plan.json"),
            "execution": {"device": "cuda", "cuda_visible_devices": "1",
                          "physical_uuid": GPU1_UUID},
        }
        formal_plan(config)
        atomic_json(formal_path, config)
    return {
        "status": "passed", "task": task, "method": method,
        "fidelity": _pointer(fidelity_path),
        "admission": _pointer(admission_path),
        "formal_configs": [_pointer(path) for path in formal_paths],
        "formal_policy_episodes_accessed": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASK_ROOTS)
    parser.add_argument("--method", required=True, choices=METHODS)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--audit", action="store_true")
    action.add_argument("--source-lineage", action="store_true")
    action.add_argument("--refresh-dial-freeze", action="store_true")
    action.add_argument("--refresh-dial-run", action="store_true")
    action.add_argument("--refresh-dial-audit", action="store_true")
    action.add_argument("--audit-shared-training-lineage", action="store_true")
    action.add_argument("--qualify-formal", action="store_true")
    action.add_argument("--audit-current-actor-lineage", action="store_true")
    args = parser.parse_args()
    if args.source_lineage and args.method != "fada":
        raise ValueError("source-lineage audit applies only to FADA")
    if (args.refresh_dial_freeze or args.refresh_dial_run or args.refresh_dial_audit):
        if args.method != "dial_transfer":
            raise ValueError("current-source refresh applies only to DIAL")
    if args.audit_shared_training_lineage and args.task not in {"T1", "T2"}:
        raise ValueError("shared-training-lineage audit covers T1 and T2 only")
    result = (audit_current_t3_t4_actor_lineage(args.task)
              if args.audit_current_actor_lineage else
              (qualify_fada_task_adapted_formal_cell(args.task)
               if args.method == "fada" else
               qualify_shared_model_formal_cell(args.task, args.method)
               if args.task in {"T1", "T2"} else
               qualify_current_t3_t4_formal_cell(args.task, args.method))
              if args.qualify_formal else
              freeze_dial_refresh(args.task) if args.refresh_dial_freeze else
              run_dial_refresh(args.task) if args.refresh_dial_run else
              audit_dial_refresh(args.task) if args.refresh_dial_audit else
              (audit_t1_shared_training_lineage() if args.task == "T1" else
               audit_t2_shared_training_lineage())
              if args.audit_shared_training_lineage else
              freeze(args.task, args.method) if args.freeze else
              run(args.task, args.method) if args.run else
              audit_fada_source_lineage(args.task) if args.source_lineage else
              audit_cohort(args.task, args.method))
    print(json.dumps({"task": args.task, "method": args.method,
                      "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
