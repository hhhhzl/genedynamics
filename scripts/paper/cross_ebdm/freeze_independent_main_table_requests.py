#!/usr/bin/env python3
"""Seal task-local, seed-disjoint requests for the nine-method main table."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.tasks.transfer.main_table_statistics import METHODS, TASKS
from scripts.paper.cross_ebdm.prepare_confirmation import prior_seed_conflicts


ROOT = Path(__file__).resolve().parents[3]
VERSION = "four-task-independent-main-table-requests-v1"
VERSION_V2 = "four-task-independent-main-table-requests-v2"
TASK_ROOTS = {
    "T1": Path("configs/arm/obstacle_transfer"),
    "T2": Path("configs/arm/guided_contact_alignment"),
    "T3": Path("configs/arm/keyed_insertion_recovery"),
    "T4": Path("configs/quadruped/mobile_keyed_insertion_recovery"),
}
SEED_ANCHORS = {"T1": 9_691_000, "T2": 9_692_000,
                "T3": 9_693_000, "T4": 9_694_000}
SEED_ANCHORS_V2 = {"T1": 9_701_000, "T2": 9_702_000,
                   "T3": 9_703_000, "T4": 9_704_000}
BASE_MECHANISMS = (
    ("source", {"servo_gain": .85, "payload_mass": .65, "delay_steps": 0}),
    ("nominal", {"servo_gain": 1., "payload_mass": .5, "delay_steps": 0}),
    ("latency_gain_shift", {"servo_gain": .7, "payload_mass": .5, "delay_steps": 3}),
    ("payload_shift", {"servo_gain": .85, "payload_mass": .8, "delay_steps": 1}),
)
OUTPUT = ROOT / "results/_analysis/four_task_transfer/independent_main_table_request_freeze_v1"
OUTPUT_V2 = ROOT / "results/_analysis/four_task_transfer/independent_main_table_request_freeze_v2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def task_roster(task: str, revision: str = "v1") -> dict:
    if revision not in {"v1", "v2"}:
        raise ValueError("unknown independent main-table request revision")
    version = VERSION if revision == "v1" else VERSION_V2
    seed_anchors = SEED_ANCHORS if revision == "v1" else SEED_ANCHORS_V2
    root = TASK_ROOTS[task]
    base_path = ROOT / root / "_base.yaml"
    base = ExperimentConfig.from_yaml(base_path)
    supports = (
        (("xarm7_fixed_base", {"robot": "xarm7", "support_controller": "fixed_base"}),)
        if task != "T4" else (
            ("support_nominal", {"robot": "go2_z1", "body_servo_gain": 1.,
                                  "support_controller": "unchanged host OSQP stance WBC"}),
            ("support_derated_0p90", {"robot": "go2_z1", "body_servo_gain": .9,
                                       "support_controller": "unchanged host OSQP stance WBC"}),
        )
    )
    contact_time = {"T2": .04, "T3": .02}.get(task)
    requests = []
    for mechanism_group, parameters in BASE_MECHANISMS:
        mechanism = dict(parameters)
        if contact_time is not None:
            mechanism["contact_time_constant"] = contact_time
        for body_group, support in supports:
            for slot in range(5):
                for mode in (0, 1):
                    request_id = (
                        f"INDEPENDENT-MAIN-{revision}/{task}/{mechanism_group}/"
                        f"{body_group}/i{slot:02d}/m{mode}"
                    )
                    requests.append({
                        "request_id": request_id,
                        "task": task,
                        "mechanism_group": mechanism_group,
                        "mechanism_parameters": mechanism,
                        "body_support_group": body_group,
                        "body_support_parameters": support,
                        "physical_initial_slot": slot,
                        "requested_mode": mode,
                        "reset_seed": seed_anchors[task] + len(requests),
                        "formal_policy_episodes_accessed": 0,
                    })
    expected = 80 if task == "T4" else 40
    groups = Counter((r["mechanism_group"], r["body_support_group"],
                      r["requested_mode"]) for r in requests)
    if (len(requests) != expected or len(set(groups.values())) != 1
            or set(groups.values()) != {5}
            or len({r["reset_seed"] for r in requests}) != expected):
        raise AssertionError(f"{task} balanced independent request roster differs")
    return {
        "version": version, "task": task,
        "task_config": {"path": str((root / "_base.yaml")), "sha256": sha(base_path)},
        "shared_execution": base.env_params.get("shared_execution"),
        "shared_task_contract": base.metadata.get("shared_task_contract"),
        "requests": requests,
        "request_count": expected,
        "request_access": "sealed_unaccessed; no training, tuning, or pilot may use these seeds",
        "failure_replacement": False,
        "formal_performance_evidence": False,
    }


def freeze(revision: str = "v1") -> dict:
    if revision not in {"v1", "v2"}:
        raise ValueError("unknown independent main-table request revision")
    version = VERSION if revision == "v1" else VERSION_V2
    output = OUTPUT if revision == "v1" else OUTPUT_V2
    rosters = {task: task_roster(task, revision) for task in TASKS}
    seeds = {row["reset_seed"] for roster in rosters.values()
             for row in roster["requests"]}
    if len(seeds) != 200 or prior_seed_conflicts(seeds):
        raise ValueError("independent main-table reset seed reused in earlier results")
    previous_invalidation = None
    if revision == "v2":
        old = json.loads((OUTPUT / "receipt.json").read_text())
        old_seeds = {row["reset_seed"] for task in TASKS for row in
                     json.loads((ROOT / old["request_files"][task]["path"]).read_text())["requests"]}
        if seeds & old_seeds:
            raise ValueError("GPU0 v2 request seeds overlap invalid GPU1 v1")
        invalidation = (ROOT / "results/_analysis/four_task_transfer/"
                        "independent_main_table_execution_v1/invalidation.json")
        prior = json.loads(invalidation.read_text())
        if prior.get("status") != "invalid_incomplete_ambiguous_physical_attempt":
            raise ValueError("invalid GPU1 cohort has not been retained")
        previous_invalidation = {"path": str(invalidation.relative_to(ROOT)),
                                 "sha256": sha(invalidation)}
    name = ("independent_main_table_requests.json" if revision == "v1"
            else "independent_main_table_requests_v2.json")
    paths = {task: ROOT / TASK_ROOTS[task] / "protocol" / name
             for task in TASKS}
    receipt_path = output / "receipt.json"
    if any(path.exists() for path in paths.values()) or output.exists():
        raise FileExistsError("independent request freeze already exists")
    for task, path in paths.items():
        path.write_text(canonical(rosters[task]))
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "version": version, "status": "sealed_unaccessed",
        "task_order": list(TASKS), "method_order": list(METHODS),
        "primary_tasks": ["T1", "T2", "T3"], "shape_extension_task": "T4",
        "request_count": 200,
        "per_task_request_counts": {task: rosters[task]["request_count"] for task in TASKS},
        "eight_true_training_instances_required_for_learned_methods": True,
        "fixed_methods": ["public_geometry", "public_source_prior"],
        "maximum_formal_episodes_if_all_methods_qualify": 11600,
        "request_files": {task: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
                          for task, path in paths.items()},
        "historical_reset_seed_overlap": 0,
        "formal_policy_episodes_accessed": 0,
        "performance_metrics": None,
        "failure_replacement": False,
    }
    if revision == "v2":
        receipt["execution"] = {
            "device": "cuda", "cuda_visible_devices": "0",
            "physical_uuid": "GPU-253f0813-4432-2d51-a149-3e73b38bf77a",
        }
        receipt["previous_invalid_cohort"] = previous_invalidation
    receipt_path.write_text(canonical(receipt))
    return receipt


def verify(revision: str = "v1") -> dict:
    if revision not in {"v1", "v2"}:
        raise ValueError("unknown independent main-table request revision")
    version = VERSION if revision == "v1" else VERSION_V2
    output = OUTPUT if revision == "v1" else OUTPUT_V2
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("version") != version or receipt.get("status") != "sealed_unaccessed"
            or receipt.get("method_order") != list(METHODS)
            or receipt.get("formal_policy_episodes_accessed") != 0):
        raise ValueError("independent main-table freeze receipt differs")
    if revision == "v2" and receipt.get("execution") != {
        "device": "cuda", "cuda_visible_devices": "0",
        "physical_uuid": "GPU-253f0813-4432-2d51-a149-3e73b38bf77a",
    }:
        raise ValueError("GPU0 v2 device identity differs")
    for task in TASKS:
        pointer = receipt["request_files"][task]
        path = ROOT / pointer["path"]
        if (sha(path) != pointer["sha256"]
                or json.loads(path.read_text()) != task_roster(task, revision)):
            raise ValueError(f"{task} task-local independent request freeze changed")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--freeze-v2", action="store_true")
    mode.add_argument("--verify-v2", action="store_true")
    mode.add_argument("--prepare-v2", action="store_true")
    args = parser.parse_args()
    if args.prepare_v2:
        from genedynamics.tasks.transfer.main_table_evaluation import prepare_gpu0_revision

        result = prepare_gpu0_revision()
    else:
        result = (freeze("v2") if args.freeze_v2 else verify("v2") if args.verify_v2
                  else freeze() if args.freeze else verify())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
