#!/usr/bin/env python3
"""Plan or run the CPU-only T1 public identifiability diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from genedynamics.tasks.transfer.obstacle_identifiability import plan, run


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(path.read_text())
    result = plan(config) if args.plan_only else run(config)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"fisher_rows", "grid_fit"}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
