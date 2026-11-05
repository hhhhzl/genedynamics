#!/usr/bin/env python3
"""Independently check a source-snapshotted T1 ASID training instance."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genedynamics.core.artifacts import atomic_json
from genedynamics.tasks.transfer.obstacle_asid import (
    INDEPENDENT_COHORT, _trajectory,
)
from genedynamics.tasks.transfer.obstacle_policy_search import ObstaclePolicySearchTask
from genedynamics.tasks.transfer.runtime import MechanismSpec


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    output = ROOT / config["output"]
    report_path = output / "report.json"
    recipe_path = output / "recipe.json"
    checkpoint_path = output / "checkpoint.json"
    destination = output / "independent_audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    if (config.get("training_cohort") != INDEPENDENT_COHORT
            or json.loads(recipe_path.read_text()) != config):
        raise ValueError("T1 ASID independent recipe differs")
    report = json.loads(report_path.read_text())
    checkpoint = json.loads(checkpoint_path.read_text())
    if (report.get("status") != "complete_development_chain"
            or report.get("formal_performance_evidence") is not False
            or report.get("plan", {}).get("training_cohort") != INDEPENDENT_COHORT
            or checkpoint.get("schema_version") != "asid-obstacle-public-policy-checkpoint-v1"
            or checkpoint.get("source_report") != {
                "path": str(report_path.relative_to(ROOT)), "sha256": sha(report_path),
            }
            or checkpoint.get("policy") != report.get("task_policy", {}).get("best_policy")
            or checkpoint.get("development_only") is not True):
        raise ValueError("T1 ASID report or checkpoint identity differs")
    bindings = config["source_bindings"]
    snapshots = report.get("source_snapshot_manifest")
    if snapshots != bindings or len(bindings) != 16:
        raise ValueError("T1 ASID source snapshot roster differs")
    for binding in bindings:
        path = binding["path"]
        if (sha(output / "source_snapshot" / path) != binding["sha256"]
                or sha(ROOT / path) != binding["sha256"]):
            raise ValueError(f"T1 ASID source snapshot or current replay code differs: {path}")
    ledger = report["interaction_ledger"]
    stages = Counter(row["stage"] for row in ledger)
    if stages != {
            "exploration": 140, "identification": 151,
            "task_policy_training": 56, "target_exploration": 1,
            "deployment": 2,
    }:
        raise ValueError("T1 ASID training stage ledger differs")
    if any(row["target_interaction"] != (row["stage"] in {
            "target_exploration", "deployment",
    }) or row.get("control_steps", 0) <= 0 for row in ledger):
        raise ValueError("T1 ASID interaction kind or control count differs")
    accounting = report["accounting"]
    if (accounting["simulator_episodes"] != 347
            or accounting["target_episodes"] != 3
            or accounting["simulator_control_steps"] != sum(
                row["control_steps"] for row in ledger if not row["target_interaction"]
            )
            or accounting["target_control_steps"] != sum(
                row["control_steps"] for row in ledger if row["target_interaction"]
            )
            or report["plan"]["maximum_wall_seconds"] < report["wall_seconds"]):
        raise ValueError("T1 ASID physical budget or accounting differs")
    target = MechanismSpec(**config["target_mechanism"])
    exploration = report["target_exploration"]
    episode = exploration["episode"]
    actions = np.asarray(exploration["issued_actions"], dtype=np.float32)
    trace, replay_actions, replay = _trajectory(
        None, target, episode["mode"], episode["seed"],
        len(actions), robot=config["robot"], fixed_actions=actions,
    )
    np.testing.assert_array_equal(replay_actions, actions)
    np.testing.assert_allclose(
        trace, np.asarray(exploration["public_trace"], dtype=np.float64),
        rtol=0, atol=1e-6,
    )
    if replay != episode:
        raise ValueError("T1 ASID target exploration physical replay differs")
    task = ObstaclePolicySearchTask(robot=config["robot"], delay_steps=0)
    for mode, original in enumerate(report["deployment"]):
        reward, _, steps, row = task._episode(
            np.asarray(checkpoint["policy"], dtype=np.float64),
            target, mode, original["reset_seed"],
        )
        expected = {"reward": reward, **row}
        if (steps != original["control_steps"]
                or set(expected) != set(original)
                or any(not np.isclose(expected[key], original[key], rtol=0, atol=1e-7)
                       if isinstance(expected[key], float) else expected[key] != original[key]
                       for key in expected)):
            raise ValueError(f"T1 ASID target deployment mode {mode} replay differs")
    result = {
        "schema_version": "asid-t1-independent-training-audit-v1",
        "status": "passed_development_only", "task": "T1",
        "training_cohort": INDEPENDENT_COHORT,
        "report_sha256": sha(report_path),
        "recipe_sha256": sha(recipe_path),
        "checkpoint_sha256": sha(checkpoint_path),
        "source_snapshots_verified": len(bindings),
        "simulator_episodes_accounted": accounting["simulator_episodes"],
        "target_episodes_accounted": accounting["target_episodes"],
        "target_episodes_independently_replayed": 3,
        "formal_performance_evidence": False,
        "scope": "training source, budget, and three target replays; simulator candidate traces were not retained",
    }
    atomic_json(destination, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    print(json.dumps(audit(config_path), indent=2))


if __name__ == "__main__":
    main()
