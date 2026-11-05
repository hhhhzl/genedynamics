#!/usr/bin/env python3
"""Plan or run one task-owned learned-baseline development evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genedynamics.core.artifacts import atomic_json
from genedynamics.tasks.transfer.method_evaluation import (
    SOURCE_POSITIVE_CONTROL_VERSION,
    audit_evaluation,
    audit_paired_evaluations,
    audit_source_nominal_roster,
    audit_source_nominal_resources,
    manifest_roster_config,
    evaluation_plan,
    run_evaluation,
    run_source_positive_control,
    source_positive_control_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", type=str)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--audit", action="store_true")
    action.add_argument("--audit-pair", type=Path, metavar="OTHER_CONFIG")
    action.add_argument("--audit-roster", action="store_true")
    action.add_argument("--audit-roster-resources", action="store_true")
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(path.read_text())
    if args.method is not None:
        config = manifest_roster_config(path, args.method)
    source_positive_control = config.get("version") == SOURCE_POSITIVE_CONTROL_VERSION
    if args.audit_roster_resources:
        report = audit_source_nominal_resources(path)
        destination = ROOT / report["output"] / "resource_reconciliation.json"
    elif args.audit_roster:
        report = audit_source_nominal_roster(path)
        destination = ROOT / report["output"] / "independent_audit.json"
    elif args.audit_pair is not None:
        if source_positive_control:
            raise ValueError("source positive controls cannot enter a target paired contrast")
        other_path = args.audit_pair if args.audit_pair.is_absolute() else ROOT / args.audit_pair
        report = audit_paired_evaluations(config, json.loads(other_path.read_text()))
        destination = ROOT / config["output"] / "paired_development_audit.json"
    elif args.audit:
        if source_positive_control:
            raise ValueError("source positive controls use their own audit")
        report = audit_evaluation(config)
        destination = ROOT / config["output"] / "independent_audit.json"
    elif args.plan_only:
        report = (
            source_positive_control_plan(config)
            if source_positive_control else evaluation_plan(config)
        )
        destination = ROOT / config["plan_output"]
        atomic_json(destination, report)
    else:
        report = (
            run_source_positive_control(config)
            if source_positive_control else run_evaluation(config)
        )
        destination = ROOT / report["output"] / "report.json"
    print(json.dumps({
        "status": report["status"], "task": report["task"],
        "method": report["method"], "output": str(destination),
    }, indent=2))
    if args.audit and report["status"] != "passed":
        raise RuntimeError("development evaluation retained runner failures; see saved independent audit")


if __name__ == "__main__":
    main()
