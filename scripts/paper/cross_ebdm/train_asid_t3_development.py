#!/usr/bin/env python3
"""Plan or run task-owned T3 ASID exploration, identification, and policy search."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genedynamics.tasks.transfer.mating_asid import run_training, training_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(path.read_text())
    result = training_plan(config, ROOT) if args.plan_only else run_training(config, ROOT)
    print(json.dumps({key: result.get(key) for key in (
        "status", "output", "accounting", "deployment_outcomes", "error",
    )}, indent=2))
    if result["status"] == "failed_retained":
        raise RuntimeError("ASID T3 full chain retained an execution failure")


if __name__ == "__main__":
    main()
