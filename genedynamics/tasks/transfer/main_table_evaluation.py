"""Task-owned execution and physical audit for the sealed independent main table.

Formal access requires a task-local cell admission.  A runner error is retained
in the report, but cannot become a scored physical failure.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from genedynamics.core.artifacts import atomic_json
from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.tasks.transfer.controllers.bindings import resolve_transfer_binding
from genedynamics.tasks.transfer.episode import (
    audit_episode_evidence, load_episode_evidence,
)
from genedynamics.tasks.transfer.main_table_statistics import (
    FIXED_METHODS, METHODS, TASKS, TRUE_TRAINING_INSTANCES,
)
from genedynamics.tasks.transfer.method_evaluation import (
    GPU1_UUID, _gpu1, _run_plan,
)


VERSION = "transfer-independent-main-table-evaluation-v1"
ADMISSION_VERSION = "transfer-independent-main-table-cell-admission-v1"
VERSION_V2 = "transfer-independent-main-table-evaluation-v2"
ADMISSION_VERSION_V2 = "transfer-independent-main-table-cell-admission-v2"
GPU0_UUID = "GPU-253f0813-4432-2d51-a149-3e73b38bf77a"
ROOT = Path(__file__).resolve().parents[3]
TASK_ROOTS = {
    "T1": Path("configs/arm/obstacle_transfer"),
    "T2": Path("configs/arm/guided_contact_alignment"),
    "T3": Path("configs/arm/keyed_insertion_recovery"),
    "T4": Path("configs/quadruped/mobile_keyed_insertion_recovery"),
}
FREEZE_RECEIPT = Path(
    "results/_analysis/four_task_transfer/independent_main_table_request_freeze_v1/receipt.json"
)
FREEZE_RECEIPT_V2 = Path(
    "results/_analysis/four_task_transfer/independent_main_table_request_freeze_v2/receipt.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pointer(pointer: dict[str, str], label: str) -> Path:
    if not isinstance(pointer, dict) or set(pointer) != {"path", "sha256"}:
        raise ValueError(f"{label} requires path and SHA256")
    path = ROOT / pointer["path"]
    if not path.is_file() or _sha(path) != pointer["sha256"]:
        raise ValueError(f"{label} missing or changed: {path}")
    return path


def _task_config(task: str, method: str) -> Path:
    root = TASK_ROOTS[task]
    if method == "cross_ebdm":
        suffix = "main/cross_ebdm.yaml"
    elif method == "dial_transfer":
        suffix = "baseline/dial.yaml"
    else:
        suffix = f"baseline/{method}.yaml"
    return ROOT / root / suffix


def _admission_path(task: str, method: str, revision: str = "v1") -> Path:
    if revision not in {"v1", "v2"}:
        raise ValueError("unknown formal cohort revision")
    return (ROOT / "results" / TASK_ROOTS[task].relative_to("configs") /
            f"validation/independent_main_table_admission_{revision}" /
            f"{method}.json")


def _roster(task: str, revision: str = "v1") -> tuple[dict, dict]:
    if revision not in {"v1", "v2"}:
        raise ValueError("unknown formal cohort revision")
    receipt_path = FREEZE_RECEIPT if revision == "v1" else FREEZE_RECEIPT_V2
    receipt = json.loads((ROOT / receipt_path).read_text())
    if (receipt.get("status") != "sealed_unaccessed"
            or receipt.get("version") != f"four-task-independent-main-table-requests-{revision}"
            or receipt.get("formal_policy_episodes_accessed") != 0
            or receipt.get("method_order") != list(METHODS)
            or receipt.get("task_order") != list(TASKS)
            or receipt.get("request_count") != 200):
        raise ValueError("independent main-table request freeze differs")
    if revision == "v2" and receipt.get("execution") != {
        "device": "cuda", "cuda_visible_devices": "0", "physical_uuid": GPU0_UUID,
    }:
        raise ValueError("GPU0 independent request freeze device differs")
    pointer = receipt["request_files"][task]
    name = ("independent_main_table_requests.json" if revision == "v1"
            else "independent_main_table_requests_v2.json")
    expected = TASK_ROOTS[task] / "protocol" / name
    if pointer["path"] != str(expected):
        raise ValueError("task-local formal request path differs")
    roster = json.loads(_pointer(pointer, "frozen formal request roster").read_text())
    expected_count = 80 if task == "T4" else 40
    if (roster.get("task") != task
            or roster.get("version") != receipt["version"]
            or roster.get("request_count") != expected_count
            or len(roster.get("requests", ())) != expected_count
            or any(not row.get("request_id", "").startswith(
                       f"INDEPENDENT-MAIN-{revision}/{task}/")
                   or row.get("formal_policy_episodes_accessed") != 0
                   for row in roster["requests"])):
        raise ValueError("task-local formal requests differ")
    return pointer, roster


def _binding(task: str, method: str, instance_id: str, config: dict, experiment: ExperimentConfig) -> dict:
    binding = deepcopy(experiment.method_params.get("actor_binding"))
    if not isinstance(binding, dict):
        raise ValueError("task config has no actor binding")
    if method in {"asid", "adaptsim"}:
        package_pointer = config.get("actor_package")
        package = json.loads(_pointer(package_pointer, "independent actor package").read_text())
        instances = package.get("instances", ())
        selected = [row for row in instances if row.get("instance_id") == instance_id]
        if (package.get("status") != "passed_development_only"
                or package.get("task") != task or package.get("method") != method
                or package.get("method_identity") != config.get("method_identity")
                or len(instances) != TRUE_TRAINING_INSTANCES
                or len(selected) != 1
                or binding.get("actor") != method
                or binding.get("manifest") is not None):
            raise ValueError("independent baseline package or direct binding differs")
        for label, pointer in selected[0]["artifacts"].items():
            _pointer(pointer, f"independent actor {label}")
            binding[label] = deepcopy(pointer)
        binding["instance_id"] = instance_id
    else:
        if binding.get("manifest") is None:
            raise ValueError("legacy main-table actor requires the task manifest")
        binding.update(family="main", key=method)
        if method not in FIXED_METHODS:
            binding["instance"] = instance_id
        for mode in (0, 1):
            resolved = resolve_transfer_binding(
                {"actor_binding": binding, "mode": mode}, task_alias=task,
            )
            if (resolved.get("actor") != method
                    or resolved.get("instance_id") != instance_id
                    or resolved.get("method_identity") != config.get("method_identity")):
                raise ValueError("manifest actor or true instance differs")
    return binding


def formal_plan(config: dict[str, Any]) -> dict[str, Any]:
    """Validate a signed cell and construct a zero-physics formal plan."""
    version = config.get("version")
    if version not in {VERSION, VERSION_V2}:
        raise ValueError("independent main-table evaluation version differs")
    revision = "v1" if version == VERSION else "v2"
    task, method = config.get("task"), config.get("method")
    if task not in TASKS or method not in METHODS:
        raise ValueError("unknown task or active main-table method")
    instance_id = config.get("instance_id")
    if method in FIXED_METHODS:
        if instance_id != "fixed":
            raise ValueError("deterministic reference requires fixed identity")
    elif (method in {"asid", "adaptsim"} and
          (not isinstance(instance_id, str) or not instance_id)):
        raise ValueError("independent trained actor requires a nonempty instance ID")
    elif method not in {"asid", "adaptsim"} and instance_id not in {
        f"instance_{index:02d}" for index in range(1, 9)
    }:
        raise ValueError("legacy learned actor requires a true training instance")
    task_path = _task_config(task, method)
    if config.get("task_config") != {"path": str(task_path.relative_to(ROOT)),
                                      "sha256": _sha(task_path)}:
        raise ValueError("formal task config is not pinned to current bytes")
    experiment = ExperimentConfig.from_yaml(task_path)
    if experiment.method != method:
        raise ValueError("task config method differs")
    roster_pointer, roster = _roster(task, revision)
    if config.get("request_roster") != roster_pointer:
        raise ValueError("formal config does not bind sealed task requests")
    admission_path = _pointer(config.get("cell_admission"), "cell admission")
    expected_admission = _admission_path(task, method, revision)
    if admission_path != expected_admission:
        raise ValueError("cell admission must be task-local")
    admission = json.loads(admission_path.read_text())
    expected_admission_version = (
        ADMISSION_VERSION if revision == "v1" else ADMISSION_VERSION_V2
    )
    if (admission.get("version") != expected_admission_version
            or admission.get("status") != "passed"
            or admission.get("independent_table_eligible") is not True
            or admission.get("task") != task or admission.get("method") != method
            or admission.get("method_identity") != config.get("method_identity")
            or admission.get("request_roster") != roster_pointer
            or admission.get("formal_request_seed_access") != 0
            or instance_id not in admission.get("instance_ids", ())
            or len(admission.get("instance_ids", ())) != (
                1 if method in FIXED_METHODS else TRUE_TRAINING_INSTANCES
            )):
        raise ValueError("cell has no passed independent formal admission")
    source_nominal = json.loads(_pointer(
        admission.get("source_nominal_qualification"),
        "source/nominal qualification",
    ).read_text())
    fidelity = json.loads(_pointer(
        admission.get("method_fidelity_review"), "method fidelity review",
    ).read_text())
    expected_episodes = 8 if method in FIXED_METHODS else 64
    if (source_nominal.get("task") != task
            or source_nominal.get("method") != method
            or source_nominal.get("status") not in {"passed", "passed_development_only"}
            or source_nominal.get("physical_evidence_verified") != expected_episodes
            or source_nominal.get("formal_performance_evidence") is not False
            or fidelity.get("task") != task or fidelity.get("method") != method
            or fidelity.get("status") != "passed"
            or fidelity.get("method_identity") != config.get("method_identity")
            or fidelity.get("source_nominal_qualification") != admission[
                "source_nominal_qualification"
            ]
            or fidelity.get("formal_request_seed_access") != 0
            or fidelity.get("physical_calls") != 0):
        raise ValueError("source/nominal evidence or method fidelity is not qualified")
    if method not in FIXED_METHODS and (
        source_nominal.get("method_identity") != config.get("method_identity")
        or source_nominal.get("formal_request_seed_access") != 0
        or len(source_nominal.get("instances", ())) != TRUE_TRAINING_INSTANCES
    ):
        raise ValueError("learned actor lacks eight-instance source/nominal qualification")
    sources = fidelity.get("source_files")
    if sources is None:
        sources = [fidelity.get("controller_source")]
    if not isinstance(sources, list) or not sources:
        raise ValueError("method fidelity review lacks pinned implementation sources")
    if (fidelity.get("task_config") != config["task_config"]
            or fidelity.get("manifest") != experiment.method_params["actor_binding"].get("manifest")):
        raise ValueError("method fidelity review differs from the current task binding")
    for key in ("training_lineage", "source_lineage", "source_parity",
                "current_actor_lineage"):
        if key in fidelity:
            _pointer(fidelity[key], f"method fidelity {key}")
    for index, report in enumerate(fidelity.get("target_training_reports", ())):
        _pointer(report, f"method fidelity target training report {index}")
    for index, source in enumerate(sources):
        _pointer(source, f"method fidelity source {index}")
    binding = _binding(task, method, instance_id, config, experiment)
    binding_sha = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    if (admission.get("actor_binding_sha256", {}).get(instance_id) != binding_sha
            or (method in FIXED_METHODS and fidelity.get("actor_binding_sha256") != binding_sha)):
        raise ValueError("admitted actor binding differs")
    output = ROOT / config.get("output", "")
    location = Path("results") / TASK_ROOTS[task].relative_to("configs")
    location /= "main/cross_ebdm" if method == "cross_ebdm" else (
        "baseline/dial" if method == "dial_transfer" else f"baseline/{method}"
    )
    expected_output = ROOT / location / "confirmation" / f"independent_main_{revision}" / instance_id
    if output != expected_output:
        raise ValueError("formal output must be task-local and instance-specific")
    gpu_index, gpu_uuid = (("1", GPU1_UUID) if revision == "v1"
                           else ("0", GPU0_UUID))
    if config.get("execution") != {
        "device": "cuda", "cuda_visible_devices": gpu_index,
        "physical_uuid": gpu_uuid,
    }:
        raise ValueError(f"formal execution requires physical GPU{gpu_index}")
    requests = []
    for row in roster["requests"]:
        if (row.get("task") != task or row.get("requested_mode") not in (0, 1)
                or row.get("request_id", "").split("/")[1:2] != [task]):
            raise ValueError("formal request task or mode differs")
        requests.append({
            "request_id": row["request_id"], "task": task,
            "robot": row["body_support_parameters"]["robot"],
            "requested_mode": row["requested_mode"],
            "reset_seed": row["reset_seed"],
            "physical_initial_slot": row["physical_initial_slot"],
            "mechanism_parameters": deepcopy(row["mechanism_parameters"]),
            "body_support_parameters": deepcopy(row["body_support_parameters"]),
            "mechanism_group": row["mechanism_group"],
            "body_support_group": row["body_support_group"],
            "confirmation_state": True, "performance_evidence": True,
        })
    if len({row["request_id"] for row in requests}) != len(requests):
        raise ValueError("formal request IDs are duplicated")
    return {
        "version": version, "status": "planned_unexecuted",
        "scope": "independent formal main table", "task": task,
        "method": method, "instance_id": instance_id,
        "method_identity": config["method_identity"],
        "task_config": str(task_path.relative_to(ROOT)),
        "task_config_sha256": _sha(task_path),
        "request_roster": deepcopy(roster_pointer),
        "cell_admission": deepcopy(config["cell_admission"]),
        "actor_binding": binding, "actor_binding_sha256": binding_sha,
        "execution": deepcopy(config["execution"]),
        "requests": requests, "output": str(output.relative_to(ROOT)),
        "all_failures_retained": True, "replacement_requests_allowed": False,
    }


def run_formal(config: dict[str, Any], *, continue_claimed: bool = False) -> dict[str, Any]:
    plan = formal_plan(config)
    if plan["task"] in {"T3", "T4"}:
        import ctypes

        native = ROOT / ".runtime/cross_ebdm/native/libcross_ebdm.so"
        if not native.is_file():
            raise FileNotFoundError("T3/T4 native ABI library is missing")
        configured = os.environ.get("CROSS_EBDM_NATIVE_LIBRARY")
        if configured is not None and Path(configured).resolve() != native.resolve():
            raise ValueError("native ABI binding differs")
        library = ctypes.CDLL(str(native))
        library.cross_ebdm_abi_version.restype = ctypes.c_int
        if library.cross_ebdm_abi_version() != 1:
            raise ValueError("T3/T4 native ABI version differs")
        os.environ["CROSS_EBDM_NATIVE_LIBRARY"] = str(native.resolve())
    if plan["version"] == VERSION:
        _gpu1()
    else:
        _gpu0()
    return _run_plan(plan, continue_claimed=continue_claimed)


def _gpu0() -> None:
    """Reject an accidental fallback or an incorrectly exposed physical GPU."""
    import torch

    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("formal evaluation must expose exactly physical GPU0")
    actual = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    if actual.removeprefix("GPU-").lower() != GPU0_UUID.removeprefix("GPU-").lower():
        raise RuntimeError(f"visible GPU UUID differs: {actual} != {GPU0_UUID}")
    torch.set_num_threads(1)


def audit_formal(config: dict[str, Any], *, write: bool = True) -> dict[str, Any]:
    """Reevaluate every saved episode on CPU; keep runner errors unscored."""
    plan = formal_plan(config)
    output = ROOT / plan["output"]
    if json.loads((output / "plan.json").read_text()) != plan:
        raise ValueError("saved formal plan differs")
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    if (report.get("status") != "complete"
            or any(report.get(key) != value for key, value in plan.items() if key != "status")
            or len(report.get("results", ())) != len(plan["requests"])):
        raise ValueError("formal report does not cover its complete plan")
    outcomes, failures = [], []
    for request, row in zip(plan["requests"], report["results"], strict=True):
        request_id = request["request_id"]
        episode_dir = output / "episodes" / hashlib.sha256(request_id.encode()).hexdigest()
        if (row.get("record_id") != request_id or row.get("request") != request
                or json.loads((episode_dir / "result.json").read_text()) != row):
            raise ValueError(f"formal result identity differs: {request_id}")
        receipt_path = episode_dir / "reset_receipt.json"
        if row.get("evidence") is None:
            if row.get("outcome") != "reset_or_execution_error":
                raise ValueError("missing formal evidence without retained runner error")
            if row.get("reset_succeeded") is True and not receipt_path.is_file():
                raise ValueError("post-reset runner failure lacks pre-actuation receipt")
            failures.append({"request_id": request_id, "error_type": row.get("error_type"),
                             "reset_succeeded": row.get("reset_succeeded")})
            continue
        evidence = row["evidence"]
        path = _pointer({"path": evidence["path"], "sha256": evidence["sha256"]},
                        "formal episode evidence")
        if not path.is_relative_to(episode_dir):
            raise ValueError("formal episode evidence left its task-local run")
        checked = audit_episode_evidence(path)
        physical = checked["evaluation"]["episodes"][-1]
        records, _, _ = load_episode_evidence(path)
        support_ok = True
        if plan["task"] == "T4":
            valid = [step for step in records[-1].traces if step.get("valid")]
            support_ok = bool(valid) and min(
                int(step.get("support_count", 0)) for step in valid
            ) >= 2
        expected_passed = bool(
            physical["success"] and not physical["constraint_violation"]
            and physical["mode_match"]
            and physical.get("exact_physics_contact_evidence", True)
            and support_ok and checked["passed"]
        )
        receipt = json.loads(receipt_path.read_text())
        reset_hash = checked["reset_physical_hashes"][-1]
        if (not checked["passed"] or evidence.get("audit_passed") is not True
                or receipt != {
                    "request_id": request_id, "task": plan["task"],
                    "method": plan["method"], "instance_id": plan["instance_id"],
                    "reset_seed": request["reset_seed"],
                    "reset_physical_hash": reset_hash,
                    "written_before_first_actuation": True,
                }
                or row["outcome"] != physical["outcome"]
                or row["passed"] != expected_passed
                or row["success"] != physical["success"]
                or row["constraint_violation"] != physical["constraint_violation"]
                or row["mode_match"] != physical["mode_match"]
                or row["physical_control_steps"] != physical["physical_control_steps"]
                or abs(row["capped_normalized_completion_time"] -
                       physical["capped_normalized_completion_time"]) > 1e-12):
            raise ValueError(f"formal physical evidence differs: {request_id}")
        success = int(bool(row["passed"]))
        outcomes.append({
            "task": plan["task"], "method": plan["method"],
            "instance_id": plan["instance_id"], "request_id": request_id,
            "dispatch_status": "executed", "target_physical_execution": 1,
            "constrained_success": success,
            "constraint_violation": bool(row["constraint_violation"]),
            "capped_normalized_completion_time": (
                row["capped_normalized_completion_time"] if success else 1.
            ),
            "outcome": (row["outcome"] if row["outcome"] != "success" or success
                        else "task_contract_failure"),
            "evidence_sha256": evidence["sha256"],
            "reset_physical_hash": reset_hash,
            "evidence_audit_status": "physical_evidence_verified",
            "mechanism_group": request["mechanism_group"],
            "requested_mode": request["requested_mode"],
            "body_support_group": request["body_support_group"],
        })
    status = "passed" if not failures else "retained_runner_failures_unscorable"
    outcomes_sha = hashlib.sha256(
        (json.dumps(outcomes, indent=2, allow_nan=False) + "\n").encode()
    ).hexdigest()
    audit = {
        "version": plan["version"], "status": status,
        "task": plan["task"], "method": plan["method"],
        "instance_id": plan["instance_id"],
        "report_sha256": _sha(report_path),
        "request_roster": plan["request_roster"],
        "physical_evidence_verified": len(outcomes),
        "verified_outcomes_sha256": outcomes_sha,
        "runner_failures_retained": failures,
        "formal_performance_evidence": not failures,
        "physical_calls": 0, "gpu_calls": 0,
    }
    if write:
        atomic_json(output / "independent_audit.json", audit)
        atomic_json(output / "verified_outcomes.json", outcomes)
    return audit


def prepare_gpu0_revision() -> dict[str, Any]:
    """Rebind qualified actors to fresh GPU0 requests without touching v1 evidence."""
    invalidation_path = (ROOT / "results/_analysis/four_task_transfer/"
                         "independent_main_table_execution_v1/invalidation.json")
    invalidation = json.loads(invalidation_path.read_text())
    if invalidation.get("status") != "invalid_incomplete_ambiguous_physical_attempt":
        raise ValueError("the old GPU1 formal run must remain invalid and unscored")
    for task in TASKS:
        _roster(task, "v2")

    jobs = []
    cells = []
    for task in TASKS:
        base = ROOT / TASK_ROOTS[task] / "confirmation"
        for method in METHODS:
            if method in FIXED_METHODS:
                names = [f"{method}_fixed.json"]
            elif method in {"asid", "adaptsim"}:
                names = [f"{method}_cohort.json"]
            else:
                names = [f"{method}_instance_{index:02d}.json"
                         for index in range(1, TRUE_TRAINING_INSTANCES + 1)]
            old_admission_path = _admission_path(task, method)
            old_admission = json.loads(old_admission_path.read_text())
            if (old_admission.get("status") != "passed"
                    or old_admission.get("independent_table_eligible") is not True):
                raise ValueError(f"{task}/{method} old actor admission differs")
            new_admission_path = _admission_path(task, method, "v2")
            if new_admission_path.exists():
                raise FileExistsError(new_admission_path)
            old_roster, _ = _roster(task)
            new_roster, _ = _roster(task, "v2")
            if old_admission.get("request_roster") != old_roster:
                raise ValueError(f"{task}/{method} old request admission differs")
            entries = []
            for name in names:
                old_path = base / "independent_main_v1" / name
                new_path = base / "independent_main_v2" / name
                if not old_path.is_file() or new_path.exists():
                    raise FileExistsError(f"old formal config absent or GPU0 config exists: {name}")
                old = json.loads(old_path.read_text())
                identities = old.get("instance_ids", [old.get("instance_id")])
                if (len(identities) != (1 if method in FIXED_METHODS else
                                        TRUE_TRAINING_INSTANCES if method in
                                        {"asid", "adaptsim"} else 1)):
                    raise ValueError(f"{task}/{method} actor instance roster differs")
                for identity in identities:
                    candidate = deepcopy(old)
                    if "instance_ids" in candidate:
                        candidate["instance_id"] = identity
                        for key in ("output", "plan_output"):
                            candidate[key] = candidate[key].replace("{instance_id}", identity)
                    old_plan = formal_plan(candidate)
                    if old_plan["version"] != VERSION:
                        raise ValueError("historical formal plan version differs")
                new = deepcopy(old)
                new["version"] = VERSION_V2
                new["request_roster"] = new_roster
                new["execution"] = {
                    "device": "cuda", "cuda_visible_devices": "0",
                    "physical_uuid": GPU0_UUID,
                }
                for key in ("output", "plan_output"):
                    if new[key].count("/independent_main_v1/") != 1:
                        raise ValueError(f"{task}/{method} formal {key} path differs")
                    new[key] = new[key].replace(
                        "/independent_main_v1/", "/independent_main_v2/", 1
                    )
                if any((ROOT / new[key].replace("{instance_id}", identity)).exists()
                       for identity in identities for key in ("output", "plan_output")):
                    raise FileExistsError(f"{task}/{method} GPU0 formal output exists")
                entries.append((new_path, new, identities))
                jobs.extend((task, method, identity) for identity in identities)
            cells.append((task, method, old_admission_path, old_admission,
                          new_admission_path, new_roster, entries))
    if (len(cells) != len(TASKS) * len(METHODS) or len(jobs) != 232):
        raise ValueError("GPU0 formal cell or actor denominator differs")

    for task, method, old_path, old, new_path, roster, _ in cells:
        revised = deepcopy(old)
        revised["version"] = ADMISSION_VERSION_V2
        revised["request_roster"] = roster
        revised["previous_gpu1_admission"] = {
            "path": str(old_path.relative_to(ROOT)), "sha256": _sha(old_path),
        }
        atomic_json(new_path, revised)
    for task, method, _, _, admission_path, _, entries in cells:
        pointer = {"path": str(admission_path.relative_to(ROOT)),
                   "sha256": _sha(admission_path)}
        for path, config, identities in entries:
            config["cell_admission"] = pointer
            for identity in identities:
                candidate = deepcopy(config)
                if "instance_ids" in candidate:
                    candidate["instance_id"] = identity
                    for key in ("output", "plan_output"):
                        candidate[key] = candidate[key].replace("{instance_id}", identity)
                formal_plan(candidate)
            atomic_json(path, config)
    return {
        "version": "transfer-independent-main-table-gpu0-revision-v2",
        "status": "qualified_unexecuted", "cells": len(cells), "actor_jobs": len(jobs),
        "expected_physical_outcomes": 11600, "physical_gpu_index": 0,
        "physical_gpu_uuid": GPU0_UUID, "formal_policy_episodes_accessed": 0,
        "previous_invalid_cohort": {
            "path": str(invalidation_path.relative_to(ROOT)),
            "sha256": _sha(invalidation_path),
        },
    }


__all__ = [
    "VERSION", "VERSION_V2", "ADMISSION_VERSION", "ADMISSION_VERSION_V2",
    "GPU0_UUID", "formal_plan", "run_formal", "audit_formal",
    "prepare_gpu0_revision",
]
