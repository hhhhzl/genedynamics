#!/usr/bin/env python3
"""Audit the completed frozen four-task transfer confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from genedynamics.tasks.transfer.confirmation import audit_confirmation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/_protocol/four_task_transfer/confirmation_v14/run"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    report = audit_confirmation(root, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
