#!/usr/bin/env python3
"""Qualify deterministic public references for the sealed nine-method table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.tasks.transfer.controllers.bindings import (
    validate_transfer_actor_manifest,
)
from genedynamics.tasks.transfer.main_table_evaluation import (
    ADMISSION_VERSION, ROOT, TASK_ROOTS, VERSION, _admission_path,
    _binding, _roster, _task_config, formal_plan,
)
from genedynamics.tasks.transfer.method_evaluation import GPU1_UUID


ANCHORS = {"T1": 3301000000, "T2": 3301100000,
           "T3": 3301300000, "T4": 3301400000}
METHODS = ("public_geometry", "public_source_prior")
FIDELITY_VERSION = "transfer-fixed-reference-fidelity-v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pointer(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": _sha(path)}


def _canonical(value: dict) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def qualify(task: str, method: str) -> dict:
    """Write a zero-physics task-local admission and checked formal config."""
    if task not in TASK_ROOTS or method not in METHODS:
        raise ValueError("fixed reference task or method differs")
    task_root = TASK_ROOTS[task]
    config_path = _task_config(task, method)
    experiment = ExperimentConfig.from_yaml(config_path)
    selection = experiment.method_params["actor_binding"]
    manifest_pointer = selection["manifest"]
    manifest = validate_transfer_actor_manifest(manifest_pointer)
    row = manifest["main"][method]
    if (experiment.method != method or row["status"] != "deterministic_single_identity"
            or row["instance_id"] != "fixed"
            or row["actor"] != method):
        raise ValueError("public reference is not a deterministic main-table actor")
    identity = row["method_identity"]
    binding = _binding(task, method, "fixed", {"method_identity": identity}, experiment)
    binding_sha = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    result_root = ROOT / "results" / task_root.relative_to("configs")
    screen_path = (result_root / "baseline" / method / "validation" /
                   f"source_nominal_actor_screen_{ANCHORS[task]}/independent_audit.json")
    screen = json.loads(screen_path.read_text())
    report_path = screen_path.parent / "report.json"
    report = json.loads(report_path.read_text())
    if (screen.get("status") != "passed"
            or screen.get("task") != task or screen.get("method") != method
            or screen.get("physical_evidence_verified") != 8
            or screen.get("runner_failures_retained") != 0
            or screen.get("formal_performance_evidence") is not False
            or screen.get("report_sha256") != _sha(report_path)
            or report.get("method_identity") != identity
            or report.get("task_config_sha256") != _sha(config_path)
            or report.get("counts", {}).get("requested") != 8
            or report.get("counts", {}).get("execution_errors") != 0):
        raise ValueError("public reference source/nominal screen is not qualified")
    roster_pointer, _ = _roster(task)
    fidelity_path = (result_root / "baseline" / method / "validation" /
                     "independent_method_fidelity_v1/report.json")
    admission_path = _admission_path(task, method)
    formal_config_path = (ROOT / task_root / "confirmation/independent_main_v1" /
                          f"{method}_fixed.json")
    if any(path.exists() for path in (fidelity_path, admission_path, formal_config_path)):
        raise FileExistsError("fixed-reference admission or formal config already exists")
    fidelity = {
        "version": FIDELITY_VERSION, "status": "passed",
        "scope": "internal deterministic public reference; no learned actor",
        "task": task, "method": method, "method_identity": identity,
        "task_config": _pointer(config_path),
        "manifest": manifest_pointer,
        "source_nominal_qualification": _pointer(screen_path),
        "actor_binding_sha256": binding_sha,
        "controller_source": _pointer(
            ROOT / "genedynamics/tasks/transfer/controllers/bindings.py"
        ),
        "physical_calls": 0, "gpu_calls": 0,
        "formal_request_seed_access": 0,
    }
    fidelity_path.parent.mkdir(parents=True, exist_ok=True)
    fidelity_path.write_text(_canonical(fidelity))
    admission = {
        "version": ADMISSION_VERSION, "status": "passed",
        "independent_table_eligible": True,
        "task": task, "method": method, "method_identity": identity,
        "instance_ids": ["fixed"],
        "request_roster": roster_pointer,
        "source_nominal_qualification": _pointer(screen_path),
        "method_fidelity_review": _pointer(fidelity_path),
        "actor_binding_sha256": {"fixed": binding_sha},
        "formal_request_seed_access": 0,
        "physical_calls": 0, "gpu_calls": 0,
    }
    admission_path.parent.mkdir(parents=True, exist_ok=True)
    admission_path.write_text(_canonical(admission))
    output = (Path("results") / task_root.relative_to("configs") /
              "baseline" / method / "confirmation/independent_main_v1/fixed")
    formal_config = {
        "version": VERSION, "task": task, "method": method,
        "method_identity": identity, "instance_id": "fixed",
        "task_config": _pointer(config_path),
        "request_roster": roster_pointer,
        "cell_admission": _pointer(admission_path),
        "output": str(output),
        "plan_output": str(output.parent / "fixed_plan.json"),
        "execution": {"device": "cuda", "cuda_visible_devices": "1",
                      "physical_uuid": GPU1_UUID},
    }
    formal_plan(formal_config)
    formal_config_path.parent.mkdir(parents=True, exist_ok=True)
    formal_config_path.write_text(_canonical(formal_config))
    return {
        "task": task, "method": method,
        "fidelity": _pointer(fidelity_path),
        "admission": _pointer(admission_path),
        "formal_config": _pointer(formal_config_path),
        "formal_policy_episodes_accessed": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASK_ROOTS)
    parser.add_argument("--method", required=True, choices=METHODS)
    args = parser.parse_args()
    print(json.dumps(qualify(args.task, args.method), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
