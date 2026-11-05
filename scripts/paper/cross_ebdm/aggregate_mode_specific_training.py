#!/usr/bin/env python3
"""Aggregate SHA-bound instance shards of one mode-specific T3 training plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_INSTANCES = {f"instance_{index:02d}" for index in range(1, 9)}
TRAINING_SCHEMA = "t3-task-first-mode-specific-recorded-completion-training-v9"
AGGREGATION_SCHEMA = "t3-mode-specific-recorded-completion-aggregation-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "sha256": sha(path),
    }


def checked(value: dict[str, str], label: str) -> Path:
    if set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be one SHA pointer")
    path = Path(value["path"])
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file() or sha(path) != value["sha256"]:
        raise ValueError(f"{label} changed")
    return path


def plan_identity(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": plan.get("version"),
        "task": plan.get("task"),
        "finite_time_progress": plan.get("finite_time_progress"),
        "feedback": plan.get("feedback"),
        "actor_manifest": plan.get("actor_manifest"),
        "data": plan.get("data"),
        "selection": plan.get("selection"),
        "training": {
            key: plan["training"].get(key)
            for key in (
                "variant",
                "initialization",
                "control_checkpoint_policy",
                "response_normalization",
                "shared_start_report",
                "paired_progress",
                "ensemble_task",
                "checkpoint_selection",
                "retain_validation_checkpoints",
                "target_sampling",
                "updates",
                "batch_size",
                "target_state_fraction",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text())
    if set(config) != {"version", "mode", "sources", "output"}:
        raise ValueError("aggregation config has unknown fields")
    if config["version"] != AGGREGATION_SCHEMA:
        raise ValueError("unknown aggregation version")
    mode = config["mode"]
    if isinstance(mode, bool) or mode not in {0, 1}:
        raise ValueError("aggregation mode must be public mode 0 or 1")
    if not isinstance(config["sources"], list) or not config["sources"]:
        raise ValueError("nonempty shard sources required")

    identity = None
    instances: dict[str, Any] = {}
    source_artifacts = []
    for index, source in enumerate(config["sources"], start=1):
        if set(source) != {"plan", "report"}:
            raise ValueError("each source requires one plan and one report")
        plan_path = checked(source["plan"], f"source {index} plan")
        report_path = checked(source["report"], f"source {index} report")
        plan = json.loads(plan_path.read_text())
        report = json.loads(report_path.read_text())
        current_identity = plan_identity(plan)
        if current_identity["version"] != TRAINING_SCHEMA:
            raise ValueError("source plan uses a different training schema")
        paired = current_identity["training"]["paired_progress"]
        sampling = current_identity["training"]["target_sampling"]
        required = current_identity["selection"].get("required_modes")
        if (
            paired.get("requested_mode") != mode
            or sampling != {"kind": "mode_specific", "modes": [mode]}
            or required != [mode]
        ):
            raise ValueError("source plan differs from the requested public mode")
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ValueError("training shard identities differ")
        if report.get("schema_version") != TRAINING_SCHEMA:
            raise ValueError("source report uses a different training schema")
        if report.get("status") == "failed":
            if report.get("error") != (
                "TimeoutError('terminal-progress training wall ceiling')"
            ):
                raise ValueError("failed shard is not the admitted wall-time split")
        elif report.get("status") not in {"complete", "selection_failed"}:
            raise ValueError("unknown shard report status")
        shard_instances = report.get("instances", {})
        if not shard_instances or not set(shard_instances) <= set(plan["instances"]):
            raise ValueError("shard report instances differ from its plan")
        overlap = set(instances) & set(shard_instances)
        if overlap:
            raise ValueError(f"duplicate instances across shards: {sorted(overlap)}")
        instances.update(shard_instances)
        source_artifacts.append({
            "plan": pointer(plan_path),
            "report": pointer(report_path),
            "reported_status": report["status"],
        })

    if set(instances) != EXPECTED_INSTANCES:
        raise ValueError("aggregation does not cover exactly eight instances")
    selected = {}
    for name, row in sorted(instances.items()):
        decision = row.get("selection", {})
        checks = decision.get("checks", {})
        per_mode = decision.get("per_mode", {}).get(str(mode), {})
        if (
            row.get("status") != "complete"
            or decision.get("accepted") is not True
            or not checks
            or not all(value is True for value in checks.values())
            or per_mode.get("accepted") is not True
        ):
            raise ValueError(f"{name} did not pass every frozen selection check")
        update = row.get("selected_update")
        if (
            isinstance(update, bool)
            or not isinstance(update, int)
            or update < 100
            or update > int(identity["training"]["updates"])
            or update % 100
        ):
            raise ValueError(f"{name} selected update differs from the frozen grid")
        checkpoint_path = checked(
            row["selected_checkpoint"], f"{name} selected checkpoint",
        )
        selected[name] = {
            "selected_update": update,
            "selected_checkpoint": pointer(checkpoint_path),
            "task_only_relative_gap": decision["task_only_relative_gap"],
            "linear_deficit_p95": decision["linear_deficit_p95"],
            "angular_deficit_p95": decision["angular_deficit_p95"],
            "useful_fraction": decision["useful_fraction"],
            "checks": checks,
        }

    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": AGGREGATION_SCHEMA,
        "status": "complete",
        "passed": True,
        "mode": mode,
        "instances": selected,
        "training_identity": identity,
        "artifacts": {
            "config": pointer(config_path),
            "runner": pointer(Path(__file__).resolve()),
            "sources": source_artifacts,
        },
        "training_updates": len(selected) * int(identity["training"]["updates"]),
        "selection_gate_passed": True,
        "new_physics_calls": 0,
        "confirmation_states_opened": 0,
        "target_reward_or_truth_opened": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "passed": result["passed"],
        "output": str(report_path),
        "sha256": sha(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
