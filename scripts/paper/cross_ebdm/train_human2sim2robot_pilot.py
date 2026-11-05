"""Plan or run one task-local H2S2R robot-source GPU pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from genedynamics.solvers.single.human2sim2robot.learning.pilot import (
    pilot_plan,
    run_training_pilot,
)


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(path.read_text())
    result = pilot_plan(config, ROOT) if args.plan_only else run_training_pilot(config, ROOT)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
