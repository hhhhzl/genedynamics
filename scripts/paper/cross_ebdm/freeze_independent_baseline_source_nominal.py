#!/usr/bin/env python3
"""Freeze task-local development screens for eight audited baseline actors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from genedynamics.core.artifacts import atomic_json
from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.tasks.transfer.main_table_evaluation import (
    ADMISSION_VERSION, VERSION as FORMAL_VERSION, _admission_path,
    _binding, _roster, _task_config, formal_plan,
)
from genedynamics.tasks.transfer.method_evaluation import (
    GPU1_UUID, INDEPENDENT_ACTOR_VERSION, evaluation_plan,
)


ROOT = Path(__file__).resolve().parents[3]
TASKS = {
    "T1": "arm/obstacle_transfer",
    "T2": "arm/guided_contact_alignment",
    "T3": "arm/keyed_insertion_recovery",
    "T4": "quadruped/mobile_keyed_insertion_recovery",
}


def pointer(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(value: dict, label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} requires path and SHA256")
    path = ROOT / value["path"]
    if not path.is_file() or pointer(path) != value:
        raise ValueError(f"{label} missing or changed")
    return path


def freeze(task: str, method: str) -> list[dict]:
    subdir = TASKS[task]
    package_path = (ROOT / "results" / subdir / "baseline" / method /
                    "validation/independent_actor_package_v1/report.json")
    package = json.loads(package_path.read_text())
    if (package.get("status") != "passed_development_only"
            or package.get("task") != task or package.get("method") != method
            or len(package.get("instances", ())) != 8
            or len({row["training_seed"] for row in package["instances"]}) != 8):
        raise ValueError("eight independently trained actors are not package-qualified")
    roster_path = (ROOT / "configs" / subdir / "validation" /
                   f"{method}_independent_actor_source_nominal_roster.json")
    roster = json.loads(roster_path.read_text())
    formal_path = ROOT / "configs" / subdir / "protocol/independent_main_table_requests.json"
    formal_seeds = {row["reset_seed"] for row in json.loads(formal_path.read_text())["requests"]}
    if (roster.get("task") != task or roster.get("methods") != [method]
            or roster.get("instance_count") != 8
            or roster.get("maximum_total_physics_episodes") != 64
            or len(roster.get("requests", ())) != 8
            or {row["reset_seed"] for row in roster["requests"]} & formal_seeds):
        raise ValueError("source/nominal roster conflicts with independent formal requests")
    frozen = []
    for instance in package["instances"]:
        seed = instance["training_seed"]
        config_path = (ROOT / "configs" / subdir / "validation" /
                       f"{method}_independent_source_nominal_{seed}.json")
        output = f"results/{subdir}/baseline/{method}/validation/independent_source_nominal_{seed}"
        config = {
            "version": INDEPENDENT_ACTOR_VERSION,
            "task": task, "method": method,
            "method_identity": package["method_identity"],
            "actor_binding_source": "independent_actor_package",
            "actor_package": pointer(package_path),
            "instance_id": instance["instance_id"],
            "task_config": package["task_config"]["path"],
            "checkpoint": instance["artifacts"]["checkpoint"],
            "request_roster": pointer(roster_path),
            "execution": {"device": "cuda", "cuda_visible_devices": "1",
                          "physical_uuid": GPU1_UUID},
            "output": output,
            "plan_output": (f"results/{subdir}/baseline/{method}/validation/plans/"
                            f"independent_source_nominal_{seed}.json"),
        }
        plan = evaluation_plan(config)
        if len(plan["requests"]) != 8 or plan["instance_id"] != instance["instance_id"]:
            raise ValueError("independent actor source/nominal plan differs")
        if config_path.exists():
            if json.loads(config_path.read_text()) != config:
                raise ValueError(f"existing frozen recipe differs: {config_path}")
        else:
            if (ROOT / output).exists():
                raise ValueError("unfrozen independent actor output already exists")
            atomic_json(config_path, config)
        frozen.append({"seed": seed, **pointer(config_path)})
    return sorted(frozen, key=lambda row: row["seed"])


def admit_formal(task: str, method: str) -> dict:
    """Freeze a direct baseline cohort after eight audited actor identities."""
    subdir = TASKS[task]
    result_root = ROOT / "results" / subdir / "baseline" / method
    package_path = result_root / "validation/independent_actor_package_v1/report.json"
    source_path = (result_root / "validation/independent_source_nominal_cohort_v1" /
                   "independent_audit.json")
    package = json.loads(package_path.read_text())
    source = json.loads(source_path.read_text())
    instances = package.get("instances", ())
    identity = package.get("method_identity")
    expected = {row.get("instance_id") for row in instances}
    if (package.get("status") != "passed_development_only"
            or package.get("task") != task or package.get("method") != method
            or package.get("physical_calls") != 0
            or package.get("formal_request_seed_access") != 0
            or len(instances) != 8 or len(expected) != 8
            or len({row["training_seed"] for row in instances}) != 8
            or source.get("status") != "passed_development_only"
            or source.get("task") != task or source.get("method") != method
            or source.get("method_identity") != identity
            or source.get("physical_evidence_verified") != 64
            or source.get("reset_hashes_matched_to_public_reference") != 64
            or source.get("success_is_not_an_admission_gate") is not True
            or source.get("formal_performance_evidence") is not False
            or source.get("formal_request_seed_access") != 0
            or {row.get("instance_id") for row in source.get("instances", ())} != expected):
        raise ValueError("eight-instance direct baseline source/nominal qualification differs")
    parity_name = ("asid_adaptsim_task_adapted_parity_20261031_v2"
                   if (task, method) == ("T1", "adaptsim") else
                   "asid_adaptsim_task_adapted_parity_20261031")
    parity_path = (ROOT / "results/_analysis/four_task_transfer" /
                   parity_name / "report.json")
    parity = json.loads(parity_path.read_text())
    if (parity.get("status") != "passed"
            or parity.get("test_result", {}).get("passed") != 38
            or parity.get("test_result", {}).get("skipped") != 0
            or parity.get("formal_request_seed_access") != 0):
        raise ValueError("ASID/AdaptSim source and public actor parity differs")
    checked(parity["test_file"], "ASID/AdaptSim parity test")
    for entry in parity["source_manifests"]:
        checked(entry, "ASID/AdaptSim source manifest")
    site = parity["environment"]["isolated_mujoco_pytest_site"]
    code = ("import numpy, scipy, sys; sys.path.insert(0, " + repr(site) +
            "); import pytest; raise SystemExit(pytest.main(" +
            repr(["-q", parity["test_file"]["path"]]) + "))")
    environment = os.environ.copy()
    environment.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", PYTHONPATH=".")
    verification = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=environment,
        capture_output=True, text=True, timeout=120,
    )
    if (verification.returncode != 0 or "38 passed" not in verification.stdout
            or "skipped" in verification.stdout):
        raise ValueError("live ASID/AdaptSim source and actor parity no longer passes")
    source_manifest = next(entry for entry in parity["source_manifests"]
                           if entry["path"] == f"genedynamics/solvers/single/{method}/SOURCE.json")
    experiment = ExperimentConfig.from_yaml(_task_config(task, method))
    binding = experiment.method_params["actor_binding"]
    if (experiment.method != method or binding.get("manifest") is not None
            or binding.get("actor") != method or binding.get("task") != task
            or binding.get("method_identity") != identity):
        raise ValueError("direct development actor binding differs")
    for row in instances:
        checked(row["recipe_config"], "direct baseline training recipe")
        for label, value in row["artifacts"].items():
            checked(value, f"direct baseline {label}")
        training_audit = json.loads(checked(
            row["artifacts"]["training_audit"], "direct training audit"
        ).read_text())
        if (training_audit.get("status") not in {"passed", "passed_development_only"}
                or training_audit.get("formal_performance_evidence") is True):
            raise ValueError("direct training instance audit is not qualified")
    for row in source["instances"]:
        for label in ("config", "report", "independent_audit"):
            checked(row[label], f"direct source/nominal {label}")
    roster_pointer, _ = _roster(task)
    package_pointer = pointer(package_path)
    binding_hashes = {}
    for instance in sorted(expected):
        selected = _binding(task, method, instance,
                            {"method_identity": identity,
                             "actor_package": package_pointer}, experiment)
        binding_hashes[instance] = hashlib.sha256(
            json.dumps(selected, sort_keys=True).encode()
        ).hexdigest()
    source_files = [
        ROOT / f"genedynamics/solvers/single/{method}/SOURCE.json",
        ROOT / f"genedynamics/solvers/single/{method}/{method}.py",
        ROOT / "genedynamics/tasks/transfer/controllers/bindings.py",
    ]
    if method == "asid":
        source_files.append(ROOT / "genedynamics/solvers/single/asid/learning/search.py")
    else:
        source_files.extend([
            ROOT / "genedynamics/tasks/transfer/controllers/adaptsim.py",
            ROOT / "genedynamics/solvers/single/adaptsim/learning/meta_training.py",
            ROOT / "genedynamics/solvers/single/adaptsim/learning/adaptation.py",
        ])
    fidelity_path = result_root / "validation/independent_method_fidelity_v1/report.json"
    admission_path = _admission_path(task, method)
    config_path = ROOT / "configs" / subdir / "confirmation/independent_main_v1" / f"{method}_cohort.json"
    if any(path.exists() for path in (fidelity_path, admission_path, config_path)):
        raise FileExistsError("direct baseline formal qualification already exists")
    fidelity = {
        "version": "transfer-direct-baseline-task-adapted-fidelity-v1",
        "status": "passed", "task": task, "method": method,
        "method_identity": identity,
        "scope": "task-adapted source-derived parameter search and public actor; official simulator excluded",
        "full_original_system_equivalence": False,
        "task_config": pointer(_task_config(task, method)),
        "manifest": None,
        "actor_package": package_pointer,
        "source_parity": pointer(parity_path),
        "source_manifest": source_manifest,
        "source_nominal_qualification": pointer(source_path),
        "source_files": [pointer(path) for path in source_files],
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
        "source_nominal_qualification": pointer(source_path),
        "method_fidelity_review": pointer(fidelity_path),
        "actor_binding_sha256": binding_hashes,
        "formal_request_seed_access": 0,
        "physical_calls": 0, "gpu_calls": 0,
    }
    atomic_json(admission_path, admission)
    output = (Path("results") / TASKS[task] / "baseline" / method /
              "confirmation/independent_main_v1/{instance_id}")
    config = {
        "version": FORMAL_VERSION, "task": task, "method": method,
        "method_identity": identity,
        "instance_ids": list(binding_hashes),
        "task_config": pointer(_task_config(task, method)),
        "request_roster": roster_pointer,
        "cell_admission": pointer(admission_path),
        "actor_package": package_pointer,
        "output": str(output),
        "plan_output": str(output.parent / "{instance_id}_plan.json"),
        "execution": {"device": "cuda", "cuda_visible_devices": "1",
                      "physical_uuid": GPU1_UUID},
    }
    for instance in binding_hashes:
        selected = dict(config, instance_id=instance,
                        output=config["output"].replace("{instance_id}", instance),
                        plan_output=config["plan_output"].replace("{instance_id}", instance))
        plan = formal_plan(selected)
        if (plan["task"] != task or plan["method"] != method
                or plan["instance_id"] != instance):
            raise ValueError("direct baseline formal instance differs")
    atomic_json(config_path, config)
    return {
        "status": "passed", "task": task, "method": method,
        "fidelity": pointer(fidelity_path),
        "admission": pointer(admission_path),
        "formal_cohort_config": pointer(config_path),
        "formal_policy_episodes_accessed": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--method", required=True, choices=("asid", "adaptsim"))
    parser.add_argument("--admit-formal", action="store_true")
    args = parser.parse_args()
    if args.admit_formal:
        print(json.dumps(admit_formal(args.task, args.method), indent=2))
        return
    rows = freeze(args.task, args.method)
    print(json.dumps({"task": args.task, "method": args.method,
                      "frozen_recipes": rows}, indent=2))


if __name__ == "__main__":
    main()
