"""Run a task-local CPU H2S2R nominal-learning diagnostic."""
import argparse
import json
from pathlib import Path

from genedynamics.solvers.single.human2sim2robot.learning.nominal_diagnostics import (
    TERMINATION_VERSION, run_nominal_diagnostic, run_reference_termination_diagnostic,
    LEARNING_VERSION, paired_learning_plan, run_paired_learning,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / args.config).read_text())
    if config.get("version") == LEARNING_VERSION:
        result = paired_learning_plan(config, root) if args.plan_only else run_paired_learning(config, root)
        print(json.dumps({"task": result["task"], "status": result["status"], "output": config["output"]}))
        return
    if args.plan_only:
        raise ValueError("--plan-only is supported for the bounded paired-learning recipe")
    if config.get("version") == TERMINATION_VERSION:
        result = run_reference_termination_diagnostic(config, root)
        print(json.dumps({"task": result["task"], "status": result["status"],
                          "results": result["results"]}))
        return
    result = run_nominal_diagnostic(config, root)
    print(json.dumps({"task": result["task"], "status": result["status"], "results": [
        {"role": r["role"], "outcomes": r["outcomes"]} for r in result["results"]
    ]}))


if __name__ == "__main__":
    main()
