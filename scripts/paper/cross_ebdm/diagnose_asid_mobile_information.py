#!/usr/bin/env python3
"""Plan, run, or audit T4 ASID early-contact public information."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genedynamics.tasks.transfer.mobile_mating_asid import audit, plan, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(path.read_text())
    report = plan(config, ROOT) if args.plan_only else audit(config, ROOT) if args.audit else run(config, ROOT)
    print(json.dumps({key: report.get(key) for key in (
        "status", "output", "physical_episodes_charged",
        "physical_control_steps_charged", "error",
    )}, indent=2))
    if report["status"] == "failed_retained":
        raise RuntimeError("ASID T4 information probe retained an execution failure")


if __name__ == "__main__":
    main()
