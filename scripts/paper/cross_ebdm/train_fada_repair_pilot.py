#!/usr/bin/env python3
"""Thin paper entry for the solver-owned FADA repair pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genedynamics.core.artifacts import atomic_json
from genedynamics.solvers.single.fada.learning.repair_pilot import admit_repair_pilot, run_repair_pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.output is not None:
        config["output"] = str(args.output)
    if args.preflight:
        if args.output is None or args.plan_only:
            raise ValueError("--preflight requires an explicit --output and executes training")
        if config["source_training"].get("strategy") != "preserve_qualified_checkpoint":
            config["source_training"]["updates"] = 1
        config["source_training"]["batch_size"] = 16
        config["source_training"]["log_interval"] = 1
        config["target_adaptation"]["updates"] = 1
        config["target_adaptation"]["batch_size"] = 16
        config["target_adaptation"]["log_interval"] = 1
        config["evaluation_windows"] = 2
        config["wall_seconds"] = min(float(config["wall_seconds"]), 120.0)
    if args.plan_only:
        report = admit_repair_pilot(config)
        destination = Path(config["plan_output"])
    else:
        report = run_repair_pilot(config, device=args.device)
        destination = Path(config["output"]) / "report.json"
    atomic_json(destination, report)
    print(json.dumps({"status": report["status"], "task": report["task"], "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
