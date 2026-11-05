#!/usr/bin/env python3
"""Inventory independent-table method cells without scoring missing executions."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import yaml

from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.tasks.transfer.main_table_evaluation import formal_plan
from genedynamics.tasks.transfer.main_table_statistics import METHODS, VERSION as STATS_VERSION


ROOT = Path(__file__).resolve().parents[3]
TASK_ROOTS = {
    "T1": Path("configs/arm/obstacle_transfer"),
    "T2": Path("configs/arm/guided_contact_alignment"),
    "T3": Path("configs/arm/keyed_insertion_recovery"),
    "T4": Path("configs/quadruped/mobile_keyed_insertion_recovery"),
}
LEGACY_ROWS = (
    "cross_ebdm", "task_only", "fada",
    "dial_transfer", "response_mpc", "public_geometry", "public_source_prior",
)
ADDED_ROWS = ("adaptsim", "asid")
ARCHIVED_ROWS = ("acp", "human2sim2robot")
VERSION = "four-task-independent-main-table-admission-v9"
REQUEST_FREEZE = Path(
    "results/_analysis/four_task_transfer/independent_main_table_request_freeze_v1/receipt.json"
)
TRAINING_LAUNCHES = {
    "asid": Path("results/_analysis/four_task_transfer/asid_independent_training_cohort_20261031/launch_receipt.json"),
    "adaptsim": Path("results/_analysis/four_task_transfer/adaptsim_independent_training_cohort_20261031/launch_receipt.json"),
}
ASID_T4_CONTINUATION = Path(
    "results/_analysis/four_task_transfer/"
    "asid_t4_independent_training_continuation_v2/launch_receipt.json"
)
ADAPTSIM_T4_CONTINUATION = Path(
    "results/_analysis/four_task_transfer/"
    "adaptsim_t4_independent_training_continuation_v2/launch_receipt.json"
)
SCREEN_ANCHORS = {
    "T1": "3301000000", "T2": "3301100000",
    "T3": "3301300000", "T4": "3301400000",
}
SCREEN_METHODS = set(LEGACY_ROWS)


def _method_config(root: Path, method: str) -> Path:
    if method == "cross_ebdm":
        return ROOT / root / "main/cross_ebdm.yaml"
    if method == "dial_transfer":
        return ROOT / root / "baseline/dial.yaml"
    return ROOT / root / "baseline" / f"{method}.yaml"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pointer(pointer: dict, label: str) -> dict:
    if not isinstance(pointer, dict) or set(pointer) != {"path", "sha256"}:
        raise ValueError(f"{label} requires path and SHA256")
    path = ROOT / pointer["path"]
    if not path.is_file() or _sha(path) != pointer["sha256"]:
        raise ValueError(f"{label} missing or changed: {path}")
    return json.loads(path.read_text()) if path.suffix == ".json" else {}


def _behavioral_screen(task: str, root: Path) -> tuple[dict, dict]:
    """Bind old development behavior to its exact current actor/config bytes."""
    result_root = Path("results") / root.relative_to("configs")
    anchor = SCREEN_ANCHORS[task]
    screen_path = ROOT / result_root / "validation" / (
        f"method_source_nominal_actor_screen_{anchor}/independent_audit.json"
    )
    screen = json.loads(screen_path.read_text())
    if (screen.get("status") != "complete_development_evidence"
            or screen.get("task") != task
            or set(screen.get("method_order", [])) != SCREEN_METHODS
            or set(screen.get("methods", {})) != SCREEN_METHODS
            or screen.get("formal_performance_evidence") is not False):
        raise ValueError(f"{task} archived actor screen contract differs")
    evidence = {}
    for method in SCREEN_METHODS:
        row = screen["methods"][method]
        location = (
            Path("main/cross_ebdm") if method == "cross_ebdm" else
            Path("baseline/dial") if method == "dial_transfer" else
            Path("baseline") / method
        )
        report_path = ROOT / result_root / location / "validation" / (
            f"source_nominal_actor_screen_{anchor}/report.json"
        )
        report = json.loads(report_path.read_text())
        task_config = ROOT / report["task_config"]
        if (row.get("audit_status") != "passed"
                or row.get("physical_evidence_verified") != 8
                or _sha(report_path) != row.get("report_sha256")
                or report.get("status") != "complete"
                or report.get("task") != task or report.get("method") != method
                or report.get("method_identity") != row.get("method_identity")
                or report.get("evidence_audits_passed") is not True
                or _sha(task_config) != report.get("task_config_sha256")
                or report.get("counts", {}).get("requested") != 8
                or report.get("counts", {}).get("execution_errors") != 0):
            raise ValueError(f"{task}/{method} source/nominal evidence differs")
        manifest = report.get("actor_binding", {}).get("manifest")
        if manifest is not None:
            _pointer(manifest, f"{task}/{method} behavioral manifest")
        evidence[method] = {
            "screen_audit": {
                "path": str(screen_path.relative_to(ROOT)),
                "sha256": _sha(screen_path),
            },
            "report": {
                "path": str(report_path.relative_to(ROOT)),
                "sha256": _sha(report_path),
            },
            "task_config_sha256": report["task_config_sha256"],
            "instance_id": row["instance_id"],
            "source": row["scene_cells"]["source"],
            "nominal": row["scene_cells"]["nominal"],
            "runner_failures_retained": row["runner_failures_retained"],
            "scope": "audited eight-request development screen; not formal performance",
        }
    return screen, evidence


def _legacy(task: str, root: Path) -> tuple[list[dict], list[dict]]:
    config = ExperimentConfig.from_yaml(ROOT / root / "_base.yaml")
    manifest_pointer = config.method_params["actor_binding"]["manifest"]
    manifest = _pointer(manifest_pointer, f"{task} actor manifest")
    if set(manifest.get("main", {})) != set(LEGACY_ROWS + ARCHIVED_ROWS):
        raise ValueError(f"{task} legacy manifest main-row roster differs")
    screen, behavior = _behavioral_screen(task, root)
    rows = []
    excluded = []
    for method in ARCHIVED_ROWS:
        binding = manifest["main"][method]
        if binding["status"] != "predeployment_rejection":
            raise ValueError(f"{task}/{method} archived source status differs")
        excluded.append({
            "task": task, "method": method,
            "method_identity": binding["method_identity"],
            "source_status": binding["status"],
            "table_status": "excluded_predeployment_rejection",
            "reason": "no qualified executable four-task actor; historical rejection is not a measured algorithm score",
            "manifest": manifest_pointer,
            "performance": None,
        })
    for method in LEGACY_ROWS:
        binding = manifest["main"][method]
        source_status = binding["status"]
        if source_status in {"instances_ready", "deterministic_single_identity"}:
            status = "artifact_resolvable_qualification_pending"
            reason = "one audited source/nominal development screen does not establish method fidelity, independent training repeats, or formal performance"
            if binding["method_identity"] != screen["methods"][method]["method_identity"]:
                raise ValueError(f"{task}/{method} behavioral method identity changed")
        else:
            raise ValueError(f"{task}/{method} unknown source status {source_status}")
        eligible = False
        formal_config = None
        formal_configs = None
        independent_source_nominal = None
        stale_source_nominal = None
        source_lineage = None
        shared_training_lineage = None
        current_actor_lineage = None
        if source_status == "instances_ready":
            location = (
                Path("main/cross_ebdm") if method == "cross_ebdm" else
                Path("baseline/dial") if method == "dial_transfer" else
                Path("baseline") / method
            )
            cohort_path = (
                ROOT / "results" / root.relative_to("configs") / location /
                "validation/independent_source_nominal_cohort_v1/independent_audit.json"
            )
            if cohort_path.is_file():
                cohort = json.loads(cohort_path.read_text())
                instances = cohort.get("instances", ())
                if (cohort.get("status") != "passed_development_only"
                        or cohort.get("task") != task
                        or cohort.get("method") != method
                        or cohort.get("method_identity") != binding["method_identity"]
                        or cohort.get("physical_evidence_verified") != 64
                        or cohort.get("reset_hashes_matched_to_first_instance") != 64
                        or cohort.get("formal_request_seed_access") != 0
                        or cohort.get("success_is_not_an_admission_gate") is not True
                        or len(instances) != 8
                        or {row.get("instance_id") for row in instances} != {
                            f"instance_{index:02d}" for index in range(1, 9)
                        }
                        or instances[0].get("report") != behavior[method]["report"]):
                    raise ValueError(f"{task}/{method} legacy cohort differs")
                for instance in instances:
                    _pointer(instance["report"], f"{task}/{method} cohort report")
                    _pointer(instance["independent_audit"], f"{task}/{method} cohort evidence")
                status = "eight_instance_source_nominal_development_qualified"
                reason = "eight distinct actor artifacts and 64 physically audited source/nominal development episodes; training provenance, fidelity review, and formal execution remain"
                independent_source_nominal = {
                    "path": str(cohort_path.relative_to(ROOT)),
                    "sha256": _sha(cohort_path),
                }
            if method == "dial_transfer":
                # The author-source clipping correction changed deployed DIAL
                # bytes after the v1 screens. Retain their evidence as history,
                # but only the source-pinned v2 rerun can qualify current code.
                stale_source_nominal = independent_source_nominal
                independent_source_nominal = None
                status = "controller_source_refresh_pending"
                reason = "DIAL author-source clipping correction requires a new eight-instance source/nominal screen under current controller bytes"
                refresh_path = (
                    ROOT / "results" / root.relative_to("configs") /
                    "baseline/dial/validation/independent_source_nominal_cohort_v2/independent_audit.json"
                )
                if refresh_path.is_file():
                    refreshed = json.loads(refresh_path.read_text())
                    instances = refreshed.get("instances", ())
                    controller = (ROOT / "genedynamics/tasks/transfer/controllers/dial.py")
                    manifest_path = (ROOT / "genedynamics/solvers/single/dial/source_manifest.json")
                    if (refreshed.get("status") != "passed_development_only"
                            or refreshed.get("task") != task
                            or refreshed.get("method") != method
                            or refreshed.get("method_identity") != binding["method_identity"]
                            or refreshed.get("physical_evidence_verified") != 64
                            or refreshed.get("reset_hashes_matched_to_first_instance") != 64
                            or refreshed.get("formal_request_seed_access") != 0
                            or refreshed.get("formal_performance_evidence") is not False
                            or refreshed.get("controller_source") != {
                                "path": str(controller.relative_to(ROOT)),
                                "sha256": _sha(controller),
                            }
                            or refreshed.get("source_manifest") != {
                                "path": str(manifest_path.relative_to(ROOT)),
                                "sha256": _sha(manifest_path),
                            }
                            or len(instances) != 8
                            or {row.get("instance_id") for row in instances} != {
                                f"instance_{index:02d}" for index in range(1, 9)
                            }):
                        raise ValueError(f"{task}/DIAL current-source cohort differs")
                    for instance in instances:
                        _pointer(instance["report"], f"{task}/DIAL refreshed report")
                        _pointer(instance["independent_audit"], f"{task}/DIAL refreshed evidence")
                    independent_source_nominal = {
                        "path": str(refresh_path.relative_to(ROOT)),
                        "sha256": _sha(refresh_path),
                    }
                    status = "eight_instance_source_nominal_development_qualified"
                    reason = "64 source/nominal development episodes audited under source-pinned DIAL controller; training provenance, fidelity review, and formal execution remain"
            if method == "fada":
                lineage_path = (
                    ROOT / "results" / root.relative_to("configs") /
                    "baseline/fada/validation/independent_source_lineage_v1/report.json"
                )
                if lineage_path.is_file():
                    lineage = json.loads(lineage_path.read_text())
                    if (lineage.get("status") != "qualified_source_lineage_only"
                            or lineage.get("task") != task
                            or lineage.get("method") != method
                            or lineage.get("method_identity") != binding["method_identity"]
                            or lineage.get("source_successes") != 4
                            or lineage.get("source_requests") != 4
                            or lineage.get("distinct_target_adaptation_seeds") != 8
                            or lineage.get("target_method_fidelity_qualified") is not False
                            or lineage.get("formal_request_seed_access") != 0):
                        raise ValueError(f"{task}/FADA source lineage differs")
                    source_lineage = {
                        "path": str(lineage_path.relative_to(ROOT)),
                        "sha256": _sha(lineage_path),
                    }
            if task in {"T1", "T2"} and method in {
                "cross_ebdm", "task_only", "dial_transfer", "response_mpc",
            }:
                lineage_path = (
                    ROOT / "results" / root.relative_to("configs") /
                    "validation/independent_shared_training_lineage_v1/report.json"
                )
                if lineage_path.is_file():
                    lineage = json.loads(lineage_path.read_text())
                    instances = lineage.get("instances", ())
                    if (lineage.get("status") != "passed_training_lineage_only"
                            or lineage.get("task") != task
                            or method not in lineage.get("methods", ())
                            or lineage.get("manifest") != manifest_pointer
                            or lineage.get("distinct_sample_streams") != 8
                            or lineage.get("formal_request_seed_access") != 0
                            or len(instances) != 8):
                        raise ValueError(f"{task} shared training lineage differs")
                    if task == "T1" and lineage.get(
                        "distinct_policy_and_response_artifacts"
                    ) != {"ours": 8, "task_only": 8, "response": 8}:
                        raise ValueError(f"{task} shared training artifact counts differ")
                    if task == "T2":
                        if lineage.get("distinct_training_seeds") != 8:
                            raise ValueError("T2 independent training seed count differs")
                        _pointer(lineage["training_report"], "T2 shared training report")
                        _pointer(lineage["artifact_audit"], "T2 training artifact audit")
                        _pointer(lineage["fixed_response_companion_policy"],
                                 "T2 response MPC companion policy")
                        _pointer(lineage["fixed_response_companion_training_report"],
                                 "T2 response MPC companion training report")
                    for instance in instances:
                        keys = ["ours_policy", "task_only_policy", "response_model"]
                        if task == "T1":
                            keys += ["ours_training_report", "task_only_training_report",
                                     "response_training_report"]
                        for key in keys:
                            _pointer(instance[key], f"{task}/{method} training {key}")
                    shared_training_lineage = {
                        "path": str(lineage_path.relative_to(ROOT)),
                        "sha256": _sha(lineage_path),
                    }
            if task in {"T3", "T4"} and method in {
                "cross_ebdm", "task_only", "dial_transfer", "response_mpc",
            }:
                lineage_path = (
                    ROOT / "results" / root.relative_to("configs") /
                    "validation/independent_current_actor_lineage_v1/report.json"
                )
                if lineage_path.is_file():
                    lineage = json.loads(lineage_path.read_text())
                    if (lineage.get("status") != "passed_current_actor_lineage_only"
                            or lineage.get("task") != task
                            or lineage.get("manifest") != manifest_pointer
                            or method not in lineage.get("methods", ())
                            or lineage.get("current_actor_artifact_counts") != {
                                "task_only": 8, "dial": 8,
                                "response_mpc": 8, "response_models": 8,
                            }
                            or lineage.get("single_factor_transport_attribution") is not False
                            or lineage.get("formal_request_seed_access") != 0):
                        raise ValueError(f"{task} current actor lineage differs")
                    current_actor_lineage = {
                        "path": str(lineage_path.relative_to(ROOT)),
                        "sha256": _sha(lineage_path),
                    }
        if source_status == "deterministic_single_identity":
            formal_path = ROOT / root / "confirmation/independent_main_v1" / f"{method}_fixed.json"
            if formal_path.is_file():
                candidate = json.loads(formal_path.read_text())
                plan = formal_plan(candidate)
                if (plan["task"] != task or plan["method"] != method
                        or plan["instance_id"] != "fixed"
                        or plan["method_identity"] != binding["method_identity"]):
                    raise ValueError(f"{task}/{method} fixed formal plan differs")
                status = "independent_formal_cell_qualified"
                reason = "deterministic reference has audited source/nominal behavior and task-local method fidelity review; formal requests remain unexecuted"
                eligible = True
                formal_config = {"path": str(formal_path.relative_to(ROOT)),
                                 "sha256": _sha(formal_path)}
        if source_status == "instances_ready":
            formal_dir = ROOT / root / "confirmation/independent_main_v1"
            paths = [formal_dir / f"{method}_instance_{index:02d}.json"
                     for index in range(1, 9)]
            present = [path.is_file() for path in paths]
            if any(present):
                if not all(present):
                    raise ValueError(f"{task}/{method} learned formal config roster incomplete")
                for index, path in enumerate(paths, start=1):
                    instance = f"instance_{index:02d}"
                    plan = formal_plan(json.loads(path.read_text()))
                    if (plan["task"] != task or plan["method"] != method
                            or plan["instance_id"] != instance
                            or plan["method_identity"] != binding["method_identity"]):
                        raise ValueError(f"{task}/{method}/{instance} formal plan differs")
                status = "independent_formal_cell_qualified"
                reason = "eight learned actor bindings, source/nominal physical audit, training lineage, and task-local fidelity review passed; formal requests remain unexecuted"
                eligible = True
                formal_configs = [
                    {"path": str(path.relative_to(ROOT)), "sha256": _sha(path)}
                    for path in paths
                ]
        rows.append({
            "task": task, "method": method,
            "method_identity": binding["method_identity"],
            "source_status": source_status, "admission_status": status,
            "reason": reason, "performance": None,
            "independent_table_eligible": eligible,
            "manifest": manifest_pointer,
            "behavioral_development_evidence": behavior.get(method),
            "independent_source_nominal_qualification": independent_source_nominal,
            "stale_source_nominal_qualification": stale_source_nominal,
            "source_lineage_qualification": source_lineage,
            "shared_training_lineage_qualification": shared_training_lineage,
            "current_actor_lineage_qualification": current_actor_lineage,
            "formal_config": formal_config,
            "formal_configs": formal_configs,
        })
    return rows, excluded


def _added(task: str, root: Path, method: str, cohort: dict) -> dict:
    path = _method_config(root, method)
    base = {
        "task": task, "method": method,
        "config": str(path.relative_to(ROOT)), "performance": None,
        "independent_table_eligible": False,
    }
    if not path.is_file():
        return {**base, "admission_status": "no_task_actor",
                "reason": "no trained task-owned direct actor binding"}
    config = ExperimentConfig.from_yaml(path)
    binding = config.method_params.get("actor_binding")
    if (config.method != method or not isinstance(binding, dict)
            or binding.get("manifest") is not None
            or binding.get("actor") != method
            or binding.get("task") != task
            or binding.get("status") != "instances_ready"):
        return {**base, "admission_status": "invalid_direct_binding",
                "reason": "task-local direct actor contract differs"}
    try:
        _pointer(binding["checkpoint"], f"{task}/{method} checkpoint")
        recipe = _pointer(binding["training_recipe"], f"{task}/{method} recipe")
        report = _pointer(binding["training_report"], f"{task}/{method} report")
        if (report.get("formal_performance_evidence") is not False
                or recipe.get("task") != task or report.get("status") not in {
                    "complete_development", "complete_meta_development",
                    "complete_development_chain",
                }):
            raise ValueError("training identity or development scope differs")
        audit_pointer = binding.get("training_audit")
        if audit_pointer is not None:
            audit = _pointer(audit_pointer, f"{task}/{method} audit")
            if (audit.get("status") not in {"passed", "passed_development_only"}
                    or audit.get("formal_performance_evidence") is True):
                raise ValueError("training audit has not passed")
    except (KeyError, ValueError) as exc:
        return {**base, "admission_status": "training_artifact_unverified",
                "reason": str(exc)}
    if cohort["source_nominal_qualification"] is not None:
        status = "eight_instance_source_nominal_development_qualified"
        reason = "eight true training instances and 64 paired source/nominal development episodes are audited; method fidelity review and independent four-mechanism formal execution remain"
    elif cohort["actor_package_qualification"] is not None:
        status = "eight_instance_package_qualified"
        reason = "eight true training instances construct valid actors; source/nominal physical behavior, method fidelity review, and formal execution remain"
    else:
        status = "trained_actor_development_only"
        reason = "single development actor; eight independent training instances, source/nominal behavior, method fidelity review, and formal execution remain"
    formal_cohort_config = None
    formal_path = ROOT / root / "confirmation/independent_main_v1" / f"{method}_cohort.json"
    if formal_path.is_file():
        if cohort["source_nominal_qualification"] is None:
            raise ValueError(f"{task}/{method} formal cohort lacks source/nominal audit")
        template = json.loads(formal_path.read_text())
        instances = template.get("instance_ids", ())
        if len(instances) != 8 or len(set(instances)) != 8:
            raise ValueError(f"{task}/{method} formal actor roster differs")
        for instance in instances:
            candidate = dict(
                template, instance_id=instance,
                output=template["output"].replace("{instance_id}", instance),
                plan_output=template["plan_output"].replace("{instance_id}", instance),
            )
            plan = formal_plan(candidate)
            if (plan["task"] != task or plan["method"] != method
                    or plan["instance_id"] != instance
                    or plan["method_identity"] != binding["method_identity"]):
                raise ValueError(f"{task}/{method}/{instance} formal plan differs")
        status = "independent_formal_cell_qualified"
        reason = "eight independent training instances, source/nominal physical audit, task-adapted source parity, and task-local fidelity review passed; formal requests remain unexecuted"
        formal_cohort_config = {"path": str(formal_path.relative_to(ROOT)),
                                "sha256": _sha(formal_path)}
    return {
        **base, "method_identity": binding["method_identity"],
        "development_instance_id": binding["instance_id"],
        "admission_status": status, "reason": reason,
        "independent_table_eligible": formal_cohort_config is not None,
        "development_checkpoint": binding["checkpoint"],
        "independent_cohort": cohort,
        "formal_cohort_config": formal_cohort_config,
    }


def _shared_contract(task: str, root: Path) -> dict:
    """Check every present active leaf against its task-owned common contract."""
    base = ExperimentConfig.from_yaml(ROOT / root / "_base.yaml")
    common_execution = base.env_params.get("shared_execution")
    common_contract = base.metadata.get("shared_task_contract")
    if task != "T1":
        if not isinstance(common_execution, dict) or common_contract is None:
            raise ValueError(f"{task} common execution or task contract missing")
        _pointer(common_contract, f"{task} common task contract")
    elif common_execution is not None or common_contract is not None:
        raise ValueError("T1 unexpectedly acquired a contact execution contract")
    verified = []
    missing = []
    for method in METHODS:
        path = _method_config(root, method)
        if not path.is_file():
            missing.append(method)
            continue
        raw = yaml.safe_load(path.read_text()) or {}
        leaf = ExperimentConfig.from_yaml(path)
        if ("shared_execution" in (raw.get("env_params") or {})
                or "shared_task_contract" in (raw.get("metadata") or {})
                or leaf.env_params.get("shared_execution") != common_execution
                or leaf.metadata.get("shared_task_contract") != common_contract):
            raise ValueError(f"{task}/{method} overrides or changes common task contract")
        verified.append(method)
    return {
        "task": task,
        "base_config_sha256": _sha(ROOT / root / "_base.yaml"),
        "shared_task_contract": common_contract,
        "verified_present_methods": verified,
        "missing_actor_methods": missing,
        "physics_calls": 0,
    }


def _frozen_requests() -> dict:
    receipt_path = ROOT / REQUEST_FREEZE
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("status") != "sealed_unaccessed"
            or receipt.get("method_order") != list(METHODS)
            or receipt.get("task_order") != list(TASK_ROOTS)
            or receipt.get("formal_policy_episodes_accessed") != 0
            or receipt.get("request_count") != 200):
        raise ValueError("independent main-table request freeze differs")
    for task, root in TASK_ROOTS.items():
        roster = _pointer(receipt["request_files"][task], f"{task} independent requests")
        expected = 80 if task == "T4" else 40
        rows = roster.get("requests", [])
        if (roster.get("task") != task or len(rows) != expected
                or roster.get("request_count") != expected
                or roster.get("formal_performance_evidence") is not False
                or roster.get("failure_replacement") is not False
                or roster.get("task_config", {}).get("sha256") != _sha(ROOT / root / "_base.yaml")
                or len({row["reset_seed"] for row in rows}) != expected):
            raise ValueError(f"{task} independent request roster differs")
    return {"path": str(REQUEST_FREEZE), "sha256": _sha(receipt_path),
            "status": "sealed_unaccessed", "request_count": 200}


def _training_cohorts() -> dict:
    cohorts = {}
    for method, relative in TRAINING_LAUNCHES.items():
        path = ROOT / relative
        launch = json.loads(path.read_text())
        if (launch.get("task_order") != list(TASK_ROOTS)
                or launch.get("total_training_instances") != 32
                or launch.get("formal_request_seed_access") != 0
                or launch.get("failure_replacement") is not False):
            raise ValueError(f"{method} independent training launch contract differs")
        task_rows = {}
        continuations = {}
        for task in TASK_ROOTS:
            recipes = launch["tasks"][task]["recipes"]
            cohort_version = "independent_main_table_training_v1"
            if method == "asid" and task == "T4":
                for row in recipes:
                    if _sha(ROOT / row["path"]) != row["sha256"]:
                        raise ValueError("ASID/T4 retained v1 recipe changed")
                continuation_path = ROOT / ASID_T4_CONTINUATION
                continuation = json.loads(continuation_path.read_text())
                failure_pointer = continuation.get("previous_failure", {})
                failed = _pointer(failure_pointer, "ASID/T4 retained v1 failure")
                if (continuation.get("version") != "asid-t4-independent-training-continuation-v2"
                        or continuation.get("task") != "T4"
                        or continuation.get("training_cohort") != "independent_main_table_training_v2"
                        or continuation.get("gpu_physical_index") != 1
                        or continuation.get("total_training_instances") != 8
                        or continuation.get("maximum_simulator_episodes") != 2464
                        or continuation.get("maximum_target_episodes") != 24
                        or continuation.get("formal_request_seed_access") != 0
                        or continuation.get("failure_replacement") is not False
                        or continuation.get("previous_launch") != {
                            "path": str(relative), "sha256": _sha(path),
                        }
                        or failed.get("status") != "failed_retained"
                        or failed.get("training_cohort") != "independent_main_table_training_v1"
                        or failed.get("output", "").endswith("/independent_training_8804003") is False
                        or failed.get("recipe_sha256") != next(
                            row["sha256"] for row in launch["tasks"]["T4"]["recipes"]
                            if row["seed"] == 8804003
                        )):
                    raise ValueError("ASID/T4 v2 continuation contract differs")
                continuations[task] = {
                    "launch": {"path": str(ASID_T4_CONTINUATION),
                               "sha256": _sha(continuation_path)},
                    "retained_v1_failure": failure_pointer,
                }
                recipes = continuation["recipes"]
                cohort_version = "independent_main_table_training_v2"
            if method == "adaptsim" and task == "T4":
                for row in recipes:
                    if _sha(ROOT / row["path"]) != row["sha256"]:
                        raise ValueError("AdaptSim/T4 retained v1 recipe changed")
                continuation_path = ROOT / ADAPTSIM_T4_CONTINUATION
                continuation = json.loads(continuation_path.read_text())
                failure_pointer = continuation.get("previous_failure", {})
                failed = _pointer(failure_pointer, "AdaptSim/T4 retained v1 failure")
                if (continuation.get("version") != "adaptsim-t4-independent-training-continuation-v2"
                        or continuation.get("task") != "T4"
                        or continuation.get("training_cohort") != "independent_main_table_training_v2"
                        or continuation.get("gpu_physical_index") != 1
                        or continuation.get("total_training_instances") != 8
                        or continuation.get("maximum_physics_episodes") != 7056
                        or continuation.get("maximum_deployment_target_episodes") != 40
                        or continuation.get("formal_request_seed_access") != 0
                        or continuation.get("failure_replacement") is not False
                        or continuation.get("previous_launch") != {
                            "path": str(relative), "sha256": _sha(path),
                        }
                        or failed.get("status") != "failed_retained"
                        or not failure_pointer["path"].endswith(
                            "/independent_training_8814002/report.json"
                        )
                        or _sha((ROOT / failure_pointer["path"]).parent / "recipe.json") != next(
                            row["sha256"] for row in launch["tasks"]["T4"]["recipes"]
                            if row["seed"] == 8814002
                        )):
                    raise ValueError("AdaptSim/T4 v2 continuation contract differs")
                continuations[task] = {
                    "launch": {"path": str(ADAPTSIM_T4_CONTINUATION),
                               "sha256": _sha(continuation_path)},
                    "retained_v1_failure": failure_pointer,
                }
                recipes = continuation["recipes"]
                cohort_version = "independent_main_table_training_v2"
            if len(recipes) != 8 or len({row["seed"] for row in recipes}) != 8:
                raise ValueError(f"{method}/{task} true training seed roster differs")
            states = Counter()
            for row in recipes:
                recipe_path = ROOT / row["path"]
                if _sha(recipe_path) != row["sha256"]:
                    raise ValueError(f"{method}/{task} frozen training recipe changed")
                recipe = json.loads(recipe_path.read_text())
                if (recipe.get("task") != task
                        or recipe.get("training_cohort") != cohort_version):
                    raise ValueError(f"{method}/{task} training cohort identity differs")
                output = ROOT / recipe["output"]
                report_path = output / "report.json"
                audit_path = output / "independent_audit.json"
                if audit_path.exists():
                    audit = json.loads(audit_path.read_text())
                    expected_status = "passed_development_only" if method == "asid" or task == "T1" else "passed"
                    if (audit.get("status") != expected_status
                            or json.loads((output / "recipe.json").read_text()) != recipe
                            or (audit.get("recipe_sha256") is not None
                                and audit["recipe_sha256"] != _sha(output / "recipe.json"))
                            or audit.get("report_sha256") != _sha(report_path)):
                        raise ValueError(f"{method}/{task} independent training audit differs")
                    states["audited_development_instance"] += 1
                elif report_path.exists():
                    status = json.loads(report_path.read_text()).get("status")
                    states["failed_retained" if status == "failed_retained" else
                           "training_complete_audit_pending"] += 1
                elif output.exists():
                    states["training_in_progress"] += 1
                else:
                    states["queued"] += 1
            package_path = (ROOT / "results" / TASK_ROOTS[task].relative_to("configs") /
                            "baseline" / method / "validation" /
                            "independent_actor_package_v1/report.json")
            package = None
            if package_path.exists():
                checked = json.loads(package_path.read_text())
                by_seed = {row["seed"]: row for row in recipes}
                instances = checked.get("instances", [])
                if (checked.get("status") != "passed_development_only"
                        or checked.get("task") != task or checked.get("method") != method
                        or checked.get("training_cohort") != cohort_version
                        or checked.get("physical_calls") != 0
                        or checked.get("formal_request_seed_access") != 0
                        or len(instances) != 8
                        or {row["training_seed"] for row in instances} != set(by_seed)
                        or states["audited_development_instance"] != 8):
                    raise ValueError(f"{method}/{task} independent actor package differs")
                for instance in instances:
                    seed = instance["training_seed"]
                    if (instance["instance_id"] != f"independent_{seed}"
                            or instance["recipe_config"] != {
                                "path": by_seed[seed]["path"],
                                "sha256": by_seed[seed]["sha256"],
                            }):
                        raise ValueError(f"{method}/{task}/{seed} package identity differs")
                    for name, pointer in instance["artifacts"].items():
                        _pointer(pointer, f"{method}/{task}/{seed}/{name}")
                package = {"path": str(package_path.relative_to(ROOT)),
                           "sha256": _sha(package_path),
                           "status": "passed_development_only"}
            source_nominal_path = (package_path.parent.parent /
                                   "independent_source_nominal_cohort_v1/independent_audit.json")
            source_nominal = None
            if source_nominal_path.exists():
                checked = json.loads(source_nominal_path.read_text())
                if (package is None or checked.get("status") != "passed_development_only"
                        or checked.get("task") != task or checked.get("method") != method
                        or checked.get("physical_evidence_verified") != 64
                        or checked.get("reset_hashes_matched_to_public_reference") != 64
                        or checked.get("formal_request_seed_access") != 0
                        or checked.get("success_is_not_an_admission_gate") is not True
                        or {row["instance_id"] for row in checked.get("instances", ())}
                        != {f"independent_{row['seed']}" for row in recipes}):
                    raise ValueError(f"{method}/{task} source/nominal cohort differs")
                source_nominal = {"path": str(source_nominal_path.relative_to(ROOT)),
                                  "sha256": _sha(source_nominal_path),
                                  "status": "passed_development_only"}
            task_rows[task] = {"instances": 8, "training_cohort": cohort_version,
                               "states": dict(sorted(states.items())),
                               "actor_package_qualification": package,
                               "source_nominal_qualification": source_nominal}
        cohorts[method] = {"launch": {"path": str(relative), "sha256": _sha(path)},
                           "active_continuations": continuations,
                           "tasks": task_rows}
    return cohorts


def audit() -> dict:
    if LEGACY_ROWS + ADDED_ROWS != METHODS:
        raise RuntimeError("admission and statistics main-table method rosters differ")
    frozen_requests = _frozen_requests()
    training_cohorts = _training_cohorts()
    rows = []
    excluded = []
    shared_checks = []
    for task, root in TASK_ROOTS.items():
        shared_checks.append(_shared_contract(task, root))
        legacy, archived = _legacy(task, root)
        rows.extend(legacy)
        excluded.extend(archived)
        rows.extend(
            _added(task, root, method, training_cohorts[method]["tasks"][task])
            for method in ADDED_ROWS
        )
    required_cells = len(METHODS) * len(TASK_ROOTS)
    if len(rows) != required_cells or len({(row["task"], row["method"]) for row in rows}) != required_cells:
        raise RuntimeError("independent main-table inventory must cover every active method and task")
    counts = Counter(row["admission_status"] for row in rows)
    unexecuted = [
        {"task": row["task"], "method": row["method"],
         "admission_status": row["admission_status"]}
        for row in rows if row["admission_status"] == "no_task_actor"
    ]
    return {
        "version": VERSION, "status": "admission_pending",
        "task_order": list(TASK_ROOTS),
        "main_rows": list(LEGACY_ROWS + ADDED_ROWS),
        "primary_tasks": ["T1", "T2", "T3"],
        "shape_extension_task": "T4",
        "primary_metrics": [
            "constrained_success_rate", "constraint_violation_rate",
            "capped_normalized_completion_time", "worst_group_constrained_success",
        ],
        "statistics_contract": STATS_VERSION,
        "frozen_independent_requests": frozen_requests,
        "independent_training_cohorts": training_cohorts,
        "missing_execution_score": None,
        "missing_execution_in_macro_denominator": False,
        "required_execution_cells": required_cells,
        "independent_table_eligible_cells": sum(
            row["independent_table_eligible"] for row in rows
        ),
        "behaviorally_screened_development_cells": sum(
            row.get("behavioral_development_evidence") is not None for row in rows
        ),
        "unexecuted_cells": unexecuted,
        "excluded_methods": excluded,
        "shared_contract_checks": shared_checks,
        "shared_contract_verified_present_cells": sum(
            len(check["verified_present_methods"]) for check in shared_checks
        ),
        "admission_counts": dict(sorted(counts.items())),
        "rows": rows,
        "physics_calls": 0, "gpu_calls": 0,
        "formal_performance_evidence": False,
        "interpretation": "availability and artifact inventory only; archived ACP/Human2Sim2Robot rejections remain outside the active table and are not measured task scores",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() or not output.is_relative_to(ROOT / "results/_analysis/four_task_transfer"):
        raise ValueError("main-table admission output must be fresh and analysis-local")
    report = audit()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"], "output": str(output.relative_to(ROOT)),
        "sha256": _sha(output), "counts": report["admission_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
