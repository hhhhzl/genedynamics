#!/usr/bin/env python3
"""Audit a source-snapshotted, independent T1 AdaptSim meta-training run."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
VERSION = "adaptsim-t1-independent-training-ledger-audit-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target(row: dict) -> Counter:
    episodes = row["episodes"]
    if (len(episodes) != 2 or [episode["mode"] for episode in episodes] != [0, 1]
            or row["control_steps"] != sum(episode["control_steps"] for episode in episodes)
            or not np.isclose(row["reward"], np.mean([episode["reward"] for episode in episodes]),
                              atol=1e-12, rtol=0)):
        raise ValueError("T1 AdaptSim target rollout accounting differs")
    counts = Counter()
    for episode in episodes:
        if (not 1 <= episode["control_steps"] <= 500
                or episode["reset_seed"] != row["seed"] + episode["mode"]
                or episode["success"] != (episode["outcome"] == "success")
                or episode["success"] and episode["constraint_violation"]
                or not np.isfinite(episode["reward"])):
            raise ValueError("T1 AdaptSim target episode ledger differs")
        counts[episode["outcome"]] += 1
    return counts


def audit(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    output = ROOT / config["output"]
    destination = output / "independent_audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    if (config.get("task") != "T1"
            or config.get("version") != "adaptsim-obstacle-meta-development-v2"
            or config.get("training_cohort") != "independent_main_table_training_v1"
            or not 8811001 <= config.get("seed", -1) <= 8811008
            or not config["output"].endswith(f"/independent_training_{config['seed']}")
            or json.loads((output / "recipe.json").read_text()) != config):
        raise ValueError("T1 AdaptSim independent training recipe differs")
    plan = json.loads((output / "plan.json").read_text())
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    if (plan.get("status") != "planned_unexecuted"
            or plan.get("maximum_physics_episodes") != 672
            or plan.get("maximum_wall_seconds") != 1500
            or report.get("version") != config["version"]
            or report.get("plan") != plan
            or report.get("status") != "complete_meta_development"
            or report.get("formal_performance_evidence") is not False
            or report.get("wall_seconds", 1501) > 1500):
        raise ValueError("T1 AdaptSim meta-training budget or completed status differs")
    bindings = config["source_bindings"]
    if len(bindings) != 16 or len({row["path"] for row in bindings}) != 16:
        raise ValueError("T1 AdaptSim source manifest differs")
    for row in bindings:
        relative = Path(row["path"])
        if (relative.is_absolute() or ".." in relative.parts
                or sha(output / "source_snapshot" / relative) != row["sha256"]
                or sha(ROOT / relative) != row["sha256"]):
            raise ValueError(f"T1 AdaptSim source snapshot differs: {relative}")
    trainings = report["policy_training_ledger"]
    simulated = report["simulated_target_ledger"]
    deployment = report["deployment_target_ledger"]
    if (len(trainings), len(simulated), len(deployment)) != (42, 36, 6):
        raise ValueError("T1 AdaptSim training or target rollout count differs")
    if (trainings[0]["seed"] != config["seed"] + 50_000_000
            or simulated[0]["seed"] != config["seed"] + 60_000_000
            or deployment[0]["seed"] != config["seed"] + 1_500_000):
        raise ValueError("T1 AdaptSim ledger does not belong to this training seed")
    candidate_count = 0
    candidate_steps = 0
    for training in trainings:
        candidates = training["candidates"]
        if (len(candidates) != 7
                or training["simulation_steps"] != sum(row["simulation_steps"] for row in candidates)
                or not np.all(np.isfinite(training["policy"]))):
            raise ValueError("T1 AdaptSim CEM training ledger differs")
        for row in candidates:
            if (len(row["candidate"]) != 6 or row["simulation_steps"] < 2
                    or row["simulation_steps"] > 1000 or not np.isfinite(row["reward"])):
                raise ValueError("T1 AdaptSim candidate ledger differs")
            candidate_steps += row["simulation_steps"]
        candidate_count += len(candidates)
    outcomes = Counter()
    for row in (*simulated, *deployment):
        outcomes.update(_target(row))
    meta = report["meta_training"]
    target = report["deployment"]
    if (meta["outer_iterations"] != 4 or meta["episodes"] != 8
            or meta["simulated_target_rollouts"] != 36
            or meta["simulation_steps"] != sum(row["simulation_steps"] for row in trainings[:36])
            or meta["simulated_target_control_steps"] != sum(row["control_steps"] for row in simulated)
            or target["selection"]["simulation_steps"] != sum(
                row["simulation_steps"] for row in trainings[36:39])
            or target["simulation_steps"] != sum(
                row["simulation_steps"] for row in trainings[36:])
            or target["target_rollouts"] != 6
            or target["target_control_steps"] != sum(row["control_steps"] for row in deployment)
            or candidate_count != 294 or candidate_count * 2 + 2 * (36 + 6) != 672):
        raise ValueError("T1 AdaptSim meta/deployment resource reconciliation differs")
    checkpoint_pointer = report["checkpoint"]
    checkpoint_path = ROOT / checkpoint_pointer["path"]
    if sha(checkpoint_path) != checkpoint_pointer["sha256"]:
        raise ValueError("T1 AdaptSim checkpoint digest differs")
    weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (weights["version"] != config["version"]
            or weights["final_policy"] != trainings[-1]["policy"]
            or not all(torch.isfinite(value).all() for value in weights["q_network"].values())):
        raise ValueError("T1 AdaptSim final policy or Q weights differ")
    result = {
        "schema_version": VERSION, "status": "passed_development_only",
        "task": "T1", "training_seed": config["seed"],
        "recipe_sha256": sha(output / "recipe.json"),
        "report_sha256": sha(report_path),
        "checkpoint_sha256": sha(checkpoint_path),
        "source_snapshots_verified": len(bindings),
        "policy_training_calls": len(trainings),
        "candidate_physics_episodes_accounted": candidate_count * 2,
        "simulated_target_episodes_accounted": len(simulated) * 2,
        "deployment_target_episodes_accounted": len(deployment) * 2,
        "simulator_candidate_control_steps_accounted": candidate_steps,
        "outcomes": dict(sorted(outcomes.items())),
        "physical_trace_replay_audit": False,
        "formal_performance_evidence": False,
        "scope": "source snapshot, CEM/Q ledger, target summaries and checkpoint; no raw candidate trace replay",
    }
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else ROOT / args.config
    print(json.dumps(audit(path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
