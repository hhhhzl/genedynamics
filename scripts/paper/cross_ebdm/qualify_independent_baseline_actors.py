#!/usr/bin/env python3
"""CPU-only package check for eight independent ASID or AdaptSim task actors."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from genedynamics.core.artifacts import atomic_json
from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.tasks.transfer.controllers.bindings import create_transfer_actor
from genedynamics.tasks.transfer.method_evaluation import _task


ROOT = Path(__file__).resolve().parents[3]
TASKS = {
    "T1": ("arm/obstacle_transfer", "method_source_nominal_actor_roster.json"),
    "T2": ("arm/guided_contact_alignment", "method_source_nominal_actor_roster.json"),
    "T3": ("arm/keyed_insertion_recovery", "method_source_nominal_actor_roster_native.json"),
    "T4": ("quadruped/mobile_keyed_insertion_recovery", "method_source_nominal_actor_roster_native.json"),
}
SEEDS = {
    "asid": {"T1": range(8801002, 8801010), "T2": range(8802001, 8802009),
             "T3": range(8803001, 8803009), "T4": range(8804101, 8804109)},
    "adaptsim": {"T1": range(8811001, 8811009), "T2": range(8812001, 8812009),
                 "T3": range(8813001, 8813009), "T4": range(8814101, 8814109)},
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def qualify(task: str, method: str) -> dict:
    subdir, roster_name = TASKS[task]
    config_path = ROOT / "configs" / subdir / "baseline" / f"{method}.yaml"
    config = ExperimentConfig.from_yaml(config_path)
    binding_base = config.method_params["actor_binding"]
    if (config.method != method or binding_base.get("actor") != method
            or binding_base.get("task") != task
            or binding_base.get("status") != "instances_ready"
            or binding_base.get("manifest") is not None):
        raise ValueError("task-local direct actor binding differs")
    roster_path = ROOT / "configs" / subdir / "validation" / roster_name
    roster = json.loads(roster_path.read_text())
    if (roster["task"] != task or len(roster["requests"]) != 8
            or roster["formal_performance_evidence"] is not False):
        raise ValueError("task-local source/nominal development roster differs")
    if task in {"T3", "T4"}:
        native_pointer = roster["execution"]["native_library"]
        native = ROOT / native_pointer["path"]
        if sha(native) != native_pointer["sha256"]:
            raise ValueError("task native ABI differs")
        os.environ["CROSS_EBDM_NATIVE_LIBRARY"] = str(native)
    requests = [row for row in roster["requests"] if row["requested_mode"] == 0]
    requests += [row for row in roster["requests"] if row["requested_mode"] == 1]
    if {row["requested_mode"] for row in requests} != {0, 1}:
        raise ValueError("both public modes must be present")
    selected = {}
    for row in requests:
        key = (row["requested_mode"], json.dumps(row["body_support_parameters"], sort_keys=True))
        selected.setdefault(key, row)
    instances = []
    for seed in SEEDS[method][task]:
        recipe_path = ROOT / "configs" / subdir / "training" / (
            f"{method}_independent_training_{seed}.json" if method == "adaptsim" else
            f"asid_grid_seeded_independent_training_{seed}.json" if task == "T1" else
            f"asid_independent_training_{seed}.json"
        )
        recipe = json.loads(recipe_path.read_text())
        cohort = (
            "independent_main_table_training_v2" if task == "T4" else
            "independent_main_table_training_v1"
        )
        if (recipe.get("task") != task or recipe.get("training_cohort") != cohort
                or recipe.get("formal_performance_evidence") is True):
            raise ValueError(f"{method}/{task}/{seed} independent recipe differs")
        output = ROOT / recipe["output"]
        if json.loads((output / "recipe.json").read_text()) != recipe:
            raise ValueError(f"{method}/{task}/{seed} frozen output recipe differs")
        checkpoint_name = "checkpoint.json" if method == "asid" else "inference_and_policy.pt"
        artifacts = {name: pointer(output / filename) for name, filename in (
            ("checkpoint", checkpoint_name), ("training_recipe", "recipe.json"),
            ("training_report", "report.json"), ("training_audit", "independent_audit.json"),
        )}
        report = json.loads((output / "report.json").read_text())
        audit = json.loads((output / "independent_audit.json").read_text())
        expected_audit = "passed_development_only" if method == "asid" or task == "T1" else "passed"
        if (audit.get("status") != expected_audit
                or audit.get("report_sha256") != artifacts["training_report"]["sha256"]
                or report.get("formal_performance_evidence") is not False):
            raise ValueError(f"{method}/{task}/{seed} training audit differs")
        binding = deepcopy(binding_base)
        binding.update(artifacts)
        binding["instance_id"] = f"independent_{seed}"
        actor_classes = set()
        for request in selected.values():
            mode = request["requested_mode"]
            physical = _task(config, {**request, "task": task})
            actor = create_transfer_actor(
                physical, method, {"mode": mode, "device": "cpu", "actor_binding": binding},
            )
            if (not callable(getattr(actor, "act", None))
                    or any(physical.accounting.values())
                    or any(physical.lifetime_accounting.values())):
                raise RuntimeError(f"{method}/{task}/{seed} package check used physics or has no actor")
            actor_classes.add(f"{type(actor).__module__}.{type(actor).__qualname__}")
        instances.append({
            "instance_id": binding["instance_id"], "training_seed": seed,
            "recipe_config": pointer(recipe_path), "artifacts": artifacts,
            "actor_classes": sorted(actor_classes),
            "public_modes_checked": [0, 1],
            "body_support_groups_checked": len({
                json.dumps(row["body_support_parameters"], sort_keys=True)
                for row in selected.values()
            }),
        })
    return {
        "version": "independent-baseline-actor-package-qualification-v1",
        "status": "passed_development_only", "task": task, "method": method,
        "method_identity": binding_base["method_identity"],
        "task_config": pointer(config_path), "source_nominal_roster": pointer(roster_path),
        "training_cohort": cohort, "instances": instances,
        "physical_calls": 0, "gpu_calls": 0,
        "formal_request_seed_access": 0, "formal_performance_evidence": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--method", required=True, choices=SEEDS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    subdir = TASKS[args.task][0]
    allowed = ROOT / "results" / subdir / "baseline" / args.method / "validation"
    if not output.is_relative_to(allowed) or output.exists():
        raise ValueError("output must be fresh and task-local")
    report = qualify(args.task, args.method)
    atomic_json(output, report)
    print(json.dumps({"status": report["status"], "instances": len(report["instances"]),
                      "output": str(output.relative_to(ROOT)), "sha256": sha(output)}, indent=2))


if __name__ == "__main__":
    main()
