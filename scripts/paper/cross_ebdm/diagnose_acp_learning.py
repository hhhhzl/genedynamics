"""Run a task-local CPU ACP nominal-learning diagnostic."""
import argparse
import json
from pathlib import Path

from genedynamics.solvers.single.acp.learning.nominal_diagnostics import (
    OVERFIT_VERSION, overfit_plan, run_nominal_diagnostic, run_overfit_diagnostic,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / args.config).read_text())
    if config.get("version") == OVERFIT_VERSION:
        result = overfit_plan(config, root) if args.plan_only else run_overfit_diagnostic(config, root)
        print(json.dumps({"task": result["task"], "status": result["status"],
                          "output": config["output"]}))
        return
    if args.plan_only:
        raise ValueError("--plan-only is supported for the explicitly bounded training recipe")
    result = run_nominal_diagnostic(config, root)
    print(json.dumps({"task": result["task"], "status": result["status"], "results": [
        {"purpose": r["purpose"], "outcome": r["evaluation"]["outcome"]} for r in result["results"]
    ]}))


if __name__ == "__main__":
    main()
