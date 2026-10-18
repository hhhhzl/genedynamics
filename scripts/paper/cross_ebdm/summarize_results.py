#!/usr/bin/env python3
"""Summarize completed main/baseline/ablation results without running physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASK_ROOTS = (
    Path("results/arm/obstacle_transfer"),
    Path("results/arm/guided_contact_alignment"),
    Path("results/arm/keyed_insertion_recovery"),
    Path("results/quadruped/mobile_keyed_insertion_recovery"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, dest="roots")
    args = parser.parse_args()
    rows = []
    roots = args.roots or TASK_ROOTS
    for path in sorted(path for root in roots for path in root.glob("**/results.json")):
        parts = path.parts
        if not any(section in parts for section in ("main", "baseline", "ablation")):
            continue
        payload = json.loads(path.read_text())
        rows.append({
            "path": str(path),
            "level": payload.get("level"),
            "seed": payload.get("seed"),
            "metrics": payload.get("metrics", {}),
            "execution_status": payload.get("execution_status", {"state": "completed"}),
        })
    print(json.dumps({"runs": rows, "count": len(rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
