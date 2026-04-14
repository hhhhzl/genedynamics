#!/usr/bin/env python3
"""
Walk a `results/3dgs/` tree and emit a consolidated CSV + JSON summary.

Collects per-run test_psnr / test_lpips / nll / ece (if present) from:
  - metrics.json               (run_full_experiment.py output)
  - <run>/metrics.json          (train_gsplat.py output)
  - <run>/active_log.json       (run_active_selection.py output — best round)
  - <run>/calibration_curve.png side-car ignored; metric must be in JSON.

Usage:
  python scripts/tasks/3dgs/aggregate_results.py results/3dgs \
      --csv results/3dgs/summary.csv --json results/3dgs/summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _gather_run(run_dir: Path) -> Optional[Dict]:
    metrics_json = run_dir / "metrics.json"
    active_log = run_dir / "active_log.json"
    if not metrics_json.exists() and not active_log.exists():
        return None

    row: Dict = {
        "run": str(run_dir),
        "name": run_dir.name,
        "category": run_dir.parent.name,
        "kind": "mbd" if "mbd" in run_dir.name.lower()
                else ("gsplat" if "gsplat" in run_dir.name.lower() else "other"),
    }

    if metrics_json.exists():
        with open(metrics_json) as f:
            m = json.load(f)
        for k in ("test_psnr", "train_psnr", "test_lpips", "nll",
                   "ece", "test_mse", "psnr", "lpips"):
            if k in m:
                row[k] = m[k]

    if active_log.exists():
        with open(active_log) as f:
            log = json.load(f)
        records = log.get("records", [])
        if records:
            last = records[-1]
            row.setdefault("test_psnr", last["metrics"].get("psnr"))
            row["n_views_final"] = len(last["active_indices"])
            row["scorer"] = log.get("scorer")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, help="e.g. results/3dgs")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[error] {root} not found", file=sys.stderr)
        return 1

    rows: List[Dict] = []
    for metrics in sorted(root.rglob("metrics.json")):
        row = _gather_run(metrics.parent)
        if row:
            rows.append(row)
    for log in sorted(root.rglob("active_log.json")):
        row = _gather_run(log.parent)
        if row and row not in rows:
            rows.append(row)

    if args.csv:
        if not rows:
            print("No metrics found.")
            return 1
        keys = sorted({k for r in rows for k in r.keys()})
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})
        print(f"Wrote {args.csv}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Wrote {args.json}")

    # Terminal summary
    if rows:
        print(f"\nFound {len(rows)} runs.")
        print(f"{'run':60s}  {'test_psnr':>9}  {'test_lpips':>10}")
        for r in rows:
            tp = r.get("test_psnr")
            tl = r.get("test_lpips")
            print(
                f"{r['run'][-60:]:60s}  "
                f"{(f'{tp:.3f}' if isinstance(tp, (int, float)) else '-'):>9}  "
                f"{(f'{tl:.3f}' if isinstance(tl, (int, float)) else '-'):>10}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
