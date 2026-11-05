#!/usr/bin/env python3
"""Thin paper entry point for solver-owned FADA closed-loop attribution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genedynamics.core.artifacts import atomic_json
from genedynamics.solvers.single.fada.learning.closed_loop_diagnostics import diagnose_closed_loop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("version") != "fada-closed-loop-attribution-config-v1":
        raise ValueError("unknown FADA closed-loop diagnostic config version")
    if args.plan_only:
        print(json.dumps({
            "config": str(args.config),
            "task": config["task"],
            "device": args.device,
            "checkpoint": config["checkpoint"],
            "episode": config["episode"],
            "output": config["output"],
            "new_physics_calls": 0,
        }, indent=2))
        return
    if args.preflight:
        if args.output is None:
            raise ValueError("--preflight requires an explicit --output so it cannot occupy the GPU evidence path")
        config["sample_limit"] = min(int(config.get("sample_limit", 64)), 2)
        config["batch_size"] = 2
        config["replay_control_limit"] = 2
    report = diagnose_closed_loop(config, device=args.device)
    output = args.output if args.output is not None else ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "report.json", report)
    print(json.dumps({"output": str(output / "report.json"), "task": report["task"]}, indent=2))


if __name__ == "__main__":
    main()
