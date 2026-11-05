#!/usr/bin/env python3
"""Audit eight independently trained baseline actors on shared development resets."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from genedynamics.core.artifacts import atomic_json


ROOT = Path(__file__).resolve().parents[3]
TASKS = {
    "T1": ("arm/obstacle_transfer", 3301000000),
    "T2": ("arm/guided_contact_alignment", 3301100000),
    "T3": ("arm/keyed_insertion_recovery", 3301300000),
    "T4": ("quadruped/mobile_keyed_insertion_recovery", 3301400000),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def audit(task: str, method: str, launch_path: Path) -> dict:
    subdir, reference_seed = TASKS[task]
    launch = json.loads(launch_path.read_text())
    task_launch = launch["tasks"][task]
    if (launch.get("formal_request_seed_access") != 0
            or launch.get("failure_replacement") is not False
            or task_launch.get("maximum_physical_episodes") != 64
            or len(task_launch.get("recipes", ())) != 8):
        raise ValueError("independent actor development launch contract differs")
    roster_path = ROOT / task_launch["request_roster"]["path"]
    package_path = ROOT / task_launch["actor_package"]["path"]
    if (sha(roster_path) != task_launch["request_roster"]["sha256"]
            or sha(package_path) != task_launch["actor_package"]["sha256"]):
        raise ValueError("independent actor launch roster or package changed")
    roster = json.loads(roster_path.read_text())
    package = json.loads(package_path.read_text())
    if (roster.get("task") != task or roster.get("methods") != [method]
            or roster.get("instance_count") != 8
            or roster.get("maximum_total_physics_episodes") != 64
            or package.get("task") != task or package.get("method") != method
            or package.get("status") != "passed_development_only"):
        raise ValueError("independent actor development method identity differs")
    requests = roster["requests"]
    reference_path = (ROOT / "results" / subdir / "baseline/public_geometry/validation" /
                      f"source_nominal_actor_screen_{reference_seed}/independent_audit.json")
    reference = json.loads(reference_path.read_text())
    reference_hashes = {row["request_id"]: row["reset_physical_hash"]
                        for row in reference["episodes"]}
    if (reference.get("status") != "passed"
            or len(reference_hashes) != len(requests) == 8):
        raise ValueError("shared source/nominal physical reference differs")
    checked = []
    counts = Counter()
    cells = Counter()
    for declared in task_launch["recipes"]:
        config_path = ROOT / declared["path"]
        if sha(config_path) != declared["sha256"]:
            raise ValueError("independent actor source/nominal recipe changed")
        config = json.loads(config_path.read_text())
        if (config.get("task") != task or config.get("method") != method
                or config.get("request_roster") != task_launch["request_roster"]
                or config.get("actor_package") != task_launch["actor_package"]
                or config.get("instance_id") != f"independent_{declared['seed']}"):
            raise ValueError("independent actor source/nominal instance identity differs")
        output = ROOT / config["output"]
        report_path = output / "report.json"
        audit_path = output / "independent_audit.json"
        report = json.loads(report_path.read_text())
        evidence = json.loads(audit_path.read_text())
        if (report.get("status") != "complete"
                or report.get("method_identity") != package["method_identity"]
                or report.get("instance_id") != config["instance_id"]
                or evidence.get("status") != "passed"
                or evidence.get("report_sha256") != sha(report_path)
                or evidence.get("physical_evidence_verified") != 8
                or evidence.get("runner_failures_retained") != 0
                or evidence.get("formal_performance_evidence") is not False
                or report.get("counts", {}).get("requested") != 8
                or report.get("counts", {}).get("execution_errors") != 0
                or len(evidence.get("episodes", ())) != 8):
            raise ValueError("independent actor source/nominal evidence incomplete")
        for request, row, physical in zip(
            requests, report["results"], evidence["episodes"], strict=True,
        ):
            request_id = request["request_id"]
            if (row["record_id"] != request_id
                    or physical["request_id"] != request_id
                    or physical["status"] != "physical_evidence_verified"
                    or physical["reset_physical_hash"] != reference_hashes[request_id]
                    or row["outcome"] != physical["outcome"]
                    or row["passed"] != physical["passed"]):
                raise ValueError("independent actor physical reset or outcome differs")
            counts[row["outcome"]] += 1
            cells[(request["scene_cell"], request["requested_mode"], row["outcome"])] += 1
        checked.append({"instance_id": config["instance_id"],
                        "training_seed": declared["seed"],
                        "config": pointer(config_path),
                        "report": pointer(report_path),
                        "independent_audit": pointer(audit_path)})
    if len({row["instance_id"] for row in checked}) != 8:
        raise ValueError("independent actor instances are duplicated")
    return {
        "version": "independent-actor-source-nominal-cohort-audit-v1",
        "status": "passed_development_only", "task": task, "method": method,
        "method_identity": package["method_identity"],
        "launch": pointer(launch_path), "reference": pointer(reference_path),
        "instances": checked, "physical_evidence_verified": 64,
        "reset_hashes_matched_to_public_reference": 64,
        "outcomes": dict(sorted(counts.items())),
        "scene_mode_outcomes": [
            {"scene_cell": scene, "requested_mode": mode,
             "outcome": outcome, "count": count}
            for (scene, mode, outcome), count in sorted(cells.items())
        ],
        "success_is_not_an_admission_gate": True,
        "physical_calls": 0, "gpu_calls": 0,
        "formal_request_seed_access": 0,
        "formal_performance_evidence": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--method", required=True, choices=("asid", "adaptsim"))
    parser.add_argument("--launch-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    launch = args.launch_receipt if args.launch_receipt.is_absolute() else ROOT / args.launch_receipt
    output = args.output if args.output.is_absolute() else ROOT / args.output
    subdir = TASKS[args.task][0]
    allowed = ROOT / "results" / subdir / "baseline" / args.method / "validation"
    if not output.is_relative_to(allowed) or output.exists():
        raise ValueError("cohort audit output must be fresh and task-local")
    result = audit(args.task, args.method, launch)
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "task": args.task,
                      "method": args.method, "sha256": sha(output)}, indent=2))


if __name__ == "__main__":
    main()
