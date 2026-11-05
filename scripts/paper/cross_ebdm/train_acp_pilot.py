#!/usr/bin/env python3
"""Thin paper entry point for a qualified ACP target-task GPU pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from genedynamics.solvers.single.acp.learning.target_pilot import (
    heldout_reconstruction_plan,
    profile_heldout_reconstruction,
    train_target_pilot,
    training_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--heldout-profile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    report_path = Path(config["output"]) / "report.json"
    training_report_path = Path(config["gpu_training"]["output"]) / "report.json"
    if args.heldout_profile:
        demonstrations = json.loads(report_path.read_text())
        training = json.loads(training_report_path.read_text())
        if args.plan_only:
            print(json.dumps(heldout_reconstruction_plan(
                config, demonstrations, training, output=args.output,
            ), indent=2))
            return 0
        report = profile_heldout_reconstruction(
            config, report_path, training_report_path, output=args.output,
        )
        print(json.dumps({
            "task": config["task"], "status": report["status"],
            "checkpoint": report["checkpoint"],
            "profiled_windows": report["profiled_windows"],
            "output": report["output"],
        }, indent=2))
        return 0
    if args.plan_only:
        print(json.dumps(training_plan(config, json.loads(report_path.read_text())), indent=2))
        return 0
    report = train_target_pilot(config, report_path, output=args.output)
    print(json.dumps({
        "task": config["task"], "status": report["status"],
        "updates": report["completed_updates"], "wall_seconds": report["wall_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
