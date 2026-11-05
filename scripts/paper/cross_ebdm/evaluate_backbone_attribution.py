#!/usr/bin/env python3
"""Plan or run the paired T2 backbone-attribution development cohort."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genedynamics.tasks.transfer.attribution_evaluation import (
    PREFLIGHT_VERSION, evaluation_plan, run_evaluation, run_numerical_preflight,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(path.read_text())
    if config.get("version") == PREFLIGHT_VERSION:
        if args.plan_only:
            raise ValueError("numerical preflight requires real checkpoints and GPU1 execution")
        result = run_numerical_preflight(config)
        size = len(result["rows"])
    else:
        result = evaluation_plan(config) if args.plan_only else run_evaluation(config)
        size = len(result["jobs"]) if args.plan_only else len(result["cells"])
    print(json.dumps({"status": result["status"], "output": config["output"],
                      "cells_or_checkpoints": size}, indent=2))


if __name__ == "__main__":
    main()
