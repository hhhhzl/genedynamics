"""Run one task-local CPU audit before an H2S2R GPU pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from genedynamics.core.artifacts import atomic_json
from genedynamics.solvers.single.human2sim2robot.learning.diagnostics import audit_learning_chain


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text())
    report = audit_learning_chain(config, ROOT)
    output = ROOT / config["output"] / "report.json"
    atomic_json(output, report)
    print(json.dumps({
        "status": report["status"],
        "task": report["task"],
        "selected_epoch": report["historical_monitor"]["selected_epoch"],
        "historical_success": report["failure_diagnosis"]["historical_monitor_has_any_success"],
        "output": str(output.relative_to(ROOT)),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
