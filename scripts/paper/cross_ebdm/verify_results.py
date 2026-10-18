#!/usr/bin/env python3
"""Verify that each declared matrix cell has complete per-seed evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from genedynamics.experiments.framework import ExperimentConfig


TASK_ROOTS = (
    Path("configs/arm/obstacle_transfer"),
    Path("configs/arm/guided_contact_alignment"),
    Path("configs/arm/keyed_insertion_recovery"),
    Path("configs/quadruped/mobile_keyed_insertion_recovery"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", type=Path)
    args = parser.parse_args()
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
    print(json.dumps({"complete": not missing, "missing": missing}, indent=2))
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
