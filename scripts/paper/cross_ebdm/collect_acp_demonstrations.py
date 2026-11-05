#!/usr/bin/env python3
"""Thin paper entry point for supervised ACP public-teacher collection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from genedynamics.solvers.single.acp.learning.demonstration_collection import (
    collect_demonstrations,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--require-sufficient", action="store_true")
    args = parser.parse_args()
    report = collect_demonstrations(json.loads(args.config.read_text()))
    print(json.dumps({
        "task": report["task"],
        "coverage_by_mode": report["coverage_by_mode"],
        "training_data_sufficient": report["training_data_sufficient"],
        "canonical_duplicates": len(report["canonical_duplicates"]),
    }, indent=2))
    return int(args.require_sufficient and not report["training_data_sufficient"])


if __name__ == "__main__":
    raise SystemExit(main())
