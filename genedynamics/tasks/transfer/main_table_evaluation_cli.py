"""CLI for one admitted independent-table actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from genedynamics.core.artifacts import atomic_json
from .main_table_evaluation import ROOT, audit_formal, formal_plan, run_formal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--instance", type=str)
    parser.add_argument("--continue-claimed", action="store_true")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.continue_claimed and not args.run:
        parser.error("--continue-claimed requires --run")
    path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(path.read_text())
    if args.instance is not None:
        if args.instance not in config.get("instance_ids", ()):
            raise ValueError("selected instance is not in the frozen cell cohort")
        config["instance_id"] = args.instance
        for key in ("output", "plan_output"):
            config[key] = config[key].replace("{instance_id}", args.instance)
    elif "instance_ids" in config:
        raise ValueError("learned-method cohort requires --instance")
    if args.plan_only:
        report = formal_plan(config)
        destination = ROOT / config["plan_output"]
        if not destination.is_relative_to(ROOT / "results"):
            raise ValueError("formal plan output must be results-local")
        atomic_json(destination, report)
    elif args.run:
        report = run_formal(config, continue_claimed=args.continue_claimed)
        destination = ROOT / report["output"] / "report.json"
    else:
        report = audit_formal(config)
        destination = ROOT / config["output"] / "independent_audit.json"
    print(json.dumps({"status": report["status"],
                      "output": str(destination.relative_to(ROOT))}, indent=2))
    if args.audit and report["status"] != "passed":
        raise RuntimeError("formal runner failures retained; cell remains unscorable")


if __name__ == "__main__":
    main()
