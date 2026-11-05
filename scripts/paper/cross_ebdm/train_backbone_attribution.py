"""Run the frozen T2 four-cell backbone comparison through the method package."""
import argparse
import json
from pathlib import Path

from genedynamics.solvers.single.cross_ebdm.learning.backbone_attribution import (
    run_training, training_plan,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    contract = json.loads((root / args.config).read_text())
    result = (training_plan(contract, root) if args.plan_only
              else run_training(contract, root, progress=lambda row: print(json.dumps(row), flush=True)))
    print(json.dumps({"status": result["status"], "output": contract["output"]}), flush=True)


if __name__ == "__main__":
    main()
