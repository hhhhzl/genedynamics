#!/usr/bin/env python3
"""
Aggregate stepping-stones experiment outputs into compact CSV summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def _iter_results(root: Path):
    for p in root.glob("level_*/seed_*/results.json"):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        yield p, data


def _extract_metric(metrics: Dict, key: str, default=0.0):
    if key not in metrics or metrics[key] is None:
        return default
    return metrics[key]


def summarize(root: Path) -> Dict[str, Dict[str, float]]:
    by_level: Dict[int, List[Dict]] = {}
    for _, res in _iter_results(root):
        level = int(res.get("level", -1))
        if level < 0:
            continue
        by_level.setdefault(level, []).append(res)

    out: Dict[str, Dict[str, float]] = {}
    for level, rows in sorted(by_level.items()):
        succ, cvar_foot, cvar_step, ptime = [], [], [], []
        for r in rows:
            m = r.get("metrics", {})
            sm = _extract_metric(m, "stepping_metrics", {})
            succ.append(float(sm.get("success", False)))
            cvar_foot.append(float(sm.get("foothold_violation_cvar95", 0.0)))
            cvar_step.append(float(sm.get("step_violation_cvar95", 0.0)))
            ptime.append(float(sm.get("planning_time", r.get("planning_time", 0.0))))
        arr_pt = np.asarray(ptime, dtype=np.float32)
        out[f"level_{level}"] = {
            "n": int(len(rows)),
            "success_rate": float(np.mean(succ)) if succ else 0.0,
            "foothold_cvar95": float(np.mean(cvar_foot)) if cvar_foot else 0.0,
            "step_cvar95": float(np.mean(cvar_step)) if cvar_step else 0.0,
            "planning_time_mean": float(np.mean(arr_pt)) if arr_pt.size else 0.0,
            "planning_time_p95": float(np.percentile(arr_pt, 95)) if arr_pt.size else 0.0,
        }
    return out


def write_csv(path: Path, summary: Dict[str, Dict[str, float]]) -> None:
    fields = [
        "level",
        "n",
        "success_rate",
        "foothold_cvar95",
        "step_cvar95",
        "planning_time_mean",
        "planning_time_p95",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for lk, row in summary.items():
            rec = {"level": lk}
            rec.update(row)
            w.writerow(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", type=str, help="results/quadruped/stepping_stones_2d/... directory")
    args = ap.parse_args()
    root = Path(args.result_dir).resolve()
    summary = summarize(root)
    out_json = root / "stepping_summary.json"
    out_csv = root / "stepping_summary.csv"
    out_json.write_text(json.dumps(summary, indent=2))
    write_csv(out_csv, summary)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()

