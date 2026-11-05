#!/usr/bin/env python3
"""Verify that each declared matrix cell has complete per-seed evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TASK_ROOTS = (
    Path("configs/arm/obstacle_transfer"),
    Path("configs/arm/guided_contact_alignment"),
    Path("configs/arm/keyed_insertion_recovery"),
    Path("configs/quadruped/mobile_keyed_insertion_recovery"),
)
REQUIRED_METRICS = {
    "evaluation_version",
    "constrained_success_rate",
    "constraint_violation_rate",
    "capped_normalized_completion_time",
    "worst_group_constrained_success",
    "predeployment_rejection_rate",
    "planning_time_seconds",
    "physical_control_steps",
    "preparation_control_steps",
    "model_calls",
    "policy_network_calls",
    "candidate_rollouts",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", type=Path)
    parser.add_argument("--evidence", nargs="+", type=Path,
                        help="Reevaluate compressed transfer episode evidence on CPU; no physics")
    args = parser.parse_args()
    if args.evidence:
        if args.configs:
            parser.error("choose config verification or episode evidence verification")
        from genedynamics.tasks.transfer.episode import audit_episode_evidence
        audits = [audit_episode_evidence(path) for path in args.evidence]
        print(json.dumps({"passed": all(row["passed"] for row in audits), "audits": audits},
                         indent=2, sort_keys=True, allow_nan=False))
        raise SystemExit(0 if all(row["passed"] for row in audits) else 1)
    from genedynamics.experiments.framework import ExperimentConfig
    configs = list(args.configs)
    if not configs:
        configs = [
            path
            for root in TASK_ROOTS
            for category in ("main", "baseline", "ablation")
            for path in sorted((root / category).glob("*.yaml"))
        ]
    missing = []
    for path in configs:
        config = ExperimentConfig.from_yaml(path)
        levels = [suite["name"] for suite in config.suites] or config.obstacle_levels
        for level in levels:
            for seed in config.seeds:
                result = config.output_dir / f"level_{level}" / f"seed_{seed}" / "results.json"
                if not result.is_file():
                    missing.append(str(result))
                    continue
                payload = json.loads(result.read_text())
                state = (payload.get("execution_status") or {}).get("state", "completed")
                if state != "completed":
                    missing.append(f"{result} [{state}]")
                    continue
                snapshot = payload.get("config_snapshot") or {}
                if any(
                    snapshot.get(key) != expected
                    for key, expected in (
                        ("name", config.name),
                        ("env_name", config.env_name),
                        ("method", config.method),
                        ("output_dir", str(config.output_dir)),
                    )
                ):
                    missing.append(f"{result} [config identity mismatch]")
                    continue
                metrics = (payload.get("metrics") or {}).get("transfer_task")
                absent = REQUIRED_METRICS - set(metrics or {})
                if absent:
                    missing.append(
                        f"{result} [missing transfer metrics: {sorted(absent)}]"
                    )
                    continue
                contract = payload.get("component_contract") or {}
                if contract.get("plugin") != config.method:
                    missing.append(f"{result} [method component mismatch]")
                    continue
                expected_binding = str(
                    (config.method_params.get("actor_binding") or {}).get(
                        "version", "task-reference"
                    )
                )
                if contract.get("binding_version") != expected_binding:
                    missing.append(f"{result} [actor binding mismatch]")
    print(json.dumps({"complete": not missing, "missing": missing}, indent=2))
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
