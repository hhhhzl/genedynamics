#!/usr/bin/env python3
"""Independently reconcile frozen AdaptSim population-development ledgers."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
VERSIONS = {
    "adaptsim-mating-feedback-population-development-v2": {
        "task": "T3", "maximum_wall_seconds": 5400,
        "source_snapshots": 16,
        "schema_version": "adaptsim-t3-population-independent-ledger-audit-v1",
    },
    "adaptsim-guide-policy-population-development-v1": {
        "task": "T2", "maximum_wall_seconds": 16200,
        "source_snapshots": 24,
        "schema_version": "adaptsim-t2-population-independent-ledger-audit-v1",
    },
    "adaptsim-mobile-mating-population-development-v2": {
        "task": "T4", "maximum_wall_seconds": 7200,
        "source_snapshots": 16,
        "schema_version": "adaptsim-t4-population-independent-ledger-audit-v1",
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_episode(row: dict, task: str, *, simulator: bool = True) -> None:
    required = {"mode", "reset_seed", "success", "constraint_violation",
                "outcome", "control_steps", "reward"}
    if not required <= set(row):
        raise ValueError("candidate or target episode lacks an outcome field")
    reset_failure = (simulator and task == "T4" and row["outcome"] == "reset_unreachable"
                     and row["control_steps"] == 0 and row["reward"] == -2.
                     and row["success"] is False
                     and row["constraint_violation"] is False
                     and str(row.get("reset_error", "")).startswith((
                         "Go2+Z1 source pose unreachable:",
                         "floating reset stance preparation failed:",
                     )))
    if (row["mode"] not in (0, 1) or not isinstance(row["success"], bool)
            or not isinstance(row["constraint_violation"], bool)
            or (row["control_steps"] < 1 and not reset_failure)
            or not np.isfinite(row["reward"])):
        raise ValueError("invalid episode outcome or physical count")
    if row["success"] != (row["outcome"] == "success"):
        raise ValueError("episode success/outcome disagree")


def _check_target_rollout(row: dict, task: str, *, simulator: bool) -> None:
    episodes = row["episodes"]
    if len(episodes) != 2 or [episode["mode"] for episode in episodes] != [0, 1]:
        raise ValueError("target rollout must retain both modes")
    for episode in episodes:
        _check_episode(episode, task, simulator=simulator)
    if row["control_steps"] != sum(episode["control_steps"] for episode in episodes):
        raise ValueError("target rollout physical controls differ")
    if not np.isclose(row["reward"], np.mean([episode["reward"] for episode in episodes]),
                      rtol=0, atol=1e-12):
        raise ValueError("target rollout reward differs")


def audit(config_path: Path, *, write: bool = True) -> dict:
    config = json.loads(config_path.read_text())
    version = config["version"]
    settings = VERSIONS.get(version)
    if settings is None:
        raise ValueError("unknown AdaptSim population recipe version")
    task = settings["task"]
    output = ROOT / config["output"]
    recipe_path = output / "recipe.json"
    plan_path = output / "plan.json"
    report_path = output / "report.json"
    recipe = json.loads(recipe_path.read_text())
    plan = json.loads(plan_path.read_text())
    report = json.loads(report_path.read_text())
    if recipe != config or report["version"] != version or report["status"] != "complete_development":
        raise ValueError("frozen recipe or completed report differs")
    if task in {"T2", "T4"} and (report.get("task") != task or plan.get("task") != task):
        raise ValueError(f"{task} task identity differs")
    if {key: plan[key] for key in (
        "policy_training_calls", "simulated_target_rollouts",
        "deployment_target_rollouts", "candidate_physics_episodes",
        "maximum_physics_episodes", "maximum_wall_seconds",
    )} != {
        "policy_training_calls": 21, "simulated_target_rollouts": 16,
        "deployment_target_rollouts": 5, "candidate_physics_episodes": 840,
        "maximum_physics_episodes": 882,
        "maximum_wall_seconds": settings["maximum_wall_seconds"],
    } or report["wall_seconds"] > settings["maximum_wall_seconds"]:
        raise ValueError("AdaptSim population budget exceeded or altered")
    failures_retained = (
        plan.get("all_failures_retained") is True if task in {"T3", "T4"}
        else plan.get("failure_replacement") is False
    )
    if report.get("formal_performance_evidence") is not False or not failures_retained:
        raise ValueError("AdaptSim development evidence scope differs")
    bindings = config["source_bindings"]
    if (len(bindings) != settings["source_snapshots"]
            or len({row["path"] for row in bindings}) != len(bindings)):
        raise ValueError("AdaptSim source snapshot roster differs")
    for row in bindings:
        path = output / "source_snapshot" / row["path"]
        if _sha(path) != row["sha256"]:
            raise ValueError(f"AdaptSim source snapshot differs: {row['path']}")

    trainings = report["policy_training_ledger"]
    simulated = report["simulated_target_ledger"]
    deployment = report["deployment_target_ledger"]
    if (len(trainings), len(simulated), len(deployment)) != (21, 16, 5):
        raise ValueError("AdaptSim call roster incomplete")
    outcomes = Counter()
    candidate_count = 0
    physical_count = 0
    for training in trainings:
        candidates = training["candidates"]
        if len(candidates) != 10:
            raise ValueError("AdaptSim CEM candidate roster incomplete")
        if training["simulation_steps"] != sum(row["simulation_steps"] for row in candidates):
            raise ValueError("AdaptSim CEM simulation accounting differs")
        for row in candidates:
            episodes = row["episodes"]
            if len(episodes) != 4 or [episode["mode"] for episode in episodes] != [0, 1, 0, 1]:
                raise ValueError("AdaptSim candidate mode/task roster differs")
            if row["simulation_steps"] != sum(episode["control_steps"] for episode in episodes):
                raise ValueError("AdaptSim candidate physical controls differ")
            if not np.isclose(row["reward"], np.mean([episode["reward"] for episode in episodes]),
                              rtol=0, atol=1e-12):
                raise ValueError("AdaptSim candidate mean reward differs")
            for episode in episodes:
                _check_episode(episode, task)
                outcomes[episode["outcome"]] += 1
            candidate_count += 1
            physical_count += 4
    for row in simulated:
        _check_target_rollout(row, task, simulator=True)
        for episode in row["episodes"]:
            outcomes[episode["outcome"]] += 1
        physical_count += 2
    for row in deployment:
        _check_target_rollout(row, task, simulator=False)
        for episode in row["episodes"]:
            outcomes[episode["outcome"]] += 1
        physical_count += 2
    if candidate_count != 210 or physical_count != 882:
        raise ValueError("AdaptSim candidate or physical episode count differs")
    meta = report["meta_training"]
    selection = report["deployment"]["selection"]
    final = report["deployment"]
    if (meta["iterations"] != 2 or meta["inner_episodes"] != 4
            or len(meta["records"]) != 2 or meta["simulated_target_rollouts"] != 16
            or len(meta["final_particles"]) != 2
            or len(selection["records"]) != 2 or len(final["records"]) != 3):
        raise ValueError("AdaptSim population/update structure differs")
    if (meta["simulation_steps"] != sum(row["simulation_steps"] for row in trainings[:16])
            or selection["simulation_steps"] != sum(row["simulation_steps"] for row in trainings[16:18])
            or final["simulation_steps"] != sum(row["simulation_steps"] for row in trainings[18:])
            or meta["simulated_target_control_steps"] != sum(row["control_steps"] for row in simulated)
            or selection["target_control_steps"] != sum(row["control_steps"] for row in deployment[:2])
            or final["target_control_steps"] != sum(row["control_steps"] for row in deployment[2:])):
        raise ValueError("AdaptSim stage resource accounting differs")
    checkpoint = ROOT / report["checkpoint"]["path"]
    if _sha(checkpoint) != report["checkpoint"]["sha256"]:
        raise ValueError("AdaptSim checkpoint digest differs")
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if weights["version"] != version or weights["final_policy"] != final["final_policy"]:
        raise ValueError("AdaptSim checkpoint identity differs")
    if not all(torch.isfinite(value).all() for value in weights["q_network"].values()):
        raise ValueError("AdaptSim Q network has nonfinite weights")
    progress = json.loads((output / "progress.json").read_text())
    if (progress["status"] != "complete_development"
            or progress["candidate_physics_episodes"] != 840):
        raise ValueError("AdaptSim terminal progress differs")
    result = {
        "schema_version": settings["schema_version"],
        "status": "passed", "report_sha256": _sha(report_path),
        "recipe_sha256": _sha(recipe_path), "checkpoint_sha256": _sha(checkpoint),
        "source_snapshots_verified": len(bindings),
        "policy_training_calls": len(trainings), "candidates": candidate_count,
        "simulated_target_rollouts": len(simulated),
        "deployment_target_rollouts": len(deployment),
        "physical_episodes_accounted": physical_count,
        "reset_unreachable_attempts": outcomes.get("reset_unreachable", 0),
        "physical_episodes_with_actions": physical_count - outcomes.get("reset_unreachable", 0),
        "outcomes": dict(sorted(outcomes.items())),
        "physical_trace_replay_audit": False,
        "scope": "independent ledger and snapshot reconciliation; development, not performance evidence",
    }
    if write:
        (output / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = audit(args.config, write=not args.check_only)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
