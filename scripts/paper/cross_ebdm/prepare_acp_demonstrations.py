#!/usr/bin/env python3
"""Thin paper entry point for ACP's qualified demonstration-data builder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from genedynamics.solvers.single.acp.learning.demonstration_data import (
    prepare_demonstration_data,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-sufficient", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output or Path(config["output"])
    report = prepare_demonstration_data(config, output)
    print(json.dumps({
        "task": report["task"],
        "output": str(output),
        "accepted_episodes": report["accepted_episodes"],
        "coverage_by_mode": report["coverage_by_mode"],
        "training_data_sufficient": report["training_data_sufficient"],
    }, indent=2))
    return int(args.require_sufficient and not report["training_data_sufficient"])


if __name__ == "__main__":
    raise SystemExit(main())
