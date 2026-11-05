"""Train a versioned public fixture response ensemble from a task-local recipe."""
import argparse
import json
from pathlib import Path

from genedynamics.solvers.single.cross_ebdm.learning.fixture_response import (
    training_plan, train_fixture_response,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / args.config).read_text())
    report = training_plan(config, root) if args.plan_only else train_fixture_response(config, root)
    print(json.dumps({key: report[key] for key in ("status", "updates")}, indent=2))


if __name__ == "__main__":
    main()
