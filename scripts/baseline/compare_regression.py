#!/usr/bin/env python3
"""
Phase 0: Compare baseline vs new results for regression.

Usage:
  python scripts/baseline/compare_regression.py BASELINE_DIR NEW_DIR
  python scripts/baseline/compare_regression.py results_baseline_20250101_1200 results

Expects both dirs to have same structure, e.g.:
  BASELINE_DIR/single2d/mdcoas-f/level_0/seed_0/results.json
  NEW_DIR/single2d/mdcoas-f/level_0/seed_0/results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

THRESHOLDS = {
    "metric_tol": 1e-6,
    "cost_tol": 1e-6,
    "traj_tol": 1e-6,
}


def load_json(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def close(a, b, tol: float) -> bool:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    if aa.shape != bb.shape:
        return False
    return float(np.max(np.abs(aa - bb))) <= tol


def compare_one(seed_dir_a: Path, seed_dir_b: Path) -> tuple[bool, str]:
    ra_path = seed_dir_a / "results.json"
    rb_path = seed_dir_b / "results.json"
    if not ra_path.exists():
        return True, "skip (no baseline results.json)"
    if not rb_path.exists():
        return False, f"missing {rb_path.relative_to(seed_dir_b.parent.parent.parent)}"
    ra = load_json(ra_path)
    rb = load_json(rb_path)

    # 1) Discrete flags must match
    for k in ("success", "collision"):
        va, vb = ra.get(k), rb.get(k)
        if va is not None or vb is not None:
            if va != vb:
                return False, f"{k} mismatch: {va} vs {vb}"

    # 2) SSR / execution_ssr within tolerance
    ma, mb = ra.get("metrics", {}), rb.get("metrics", {})
    for mk in ("ssr", "execution_ssr"):
        if mk not in ma or mk not in mb:
            continue
        mav, mbv = ma[mk], mb[mk]
        if not isinstance(mav, dict) or not isinstance(mbv, dict):
            continue
        va = mav.get("ssr", mav.get("execution_ssr"))
        vb = mbv.get("ssr", mbv.get("execution_ssr"))
        if va is not None and vb is not None:
            if abs(float(va) - float(vb)) > THRESHOLDS["metric_tol"]:
                return False, f"{mk} mismatch: {va} vs {vb}"

    # 3) best_idx and candidate_costs from trajectory.json
    ta = seed_dir_a / "trajectory" / "trajectory.json"
    tb = seed_dir_b / "trajectory" / "trajectory.json"
    if ta.exists() and tb.exists():
        ja, jb = load_json(ta), load_json(tb)
        if int(ja.get("best_idx", -1)) != int(jb.get("best_idx", -1)):
            return False, f"best_idx mismatch: {ja.get('best_idx')} vs {jb.get('best_idx')}"
        ca, cb = ja.get("candidate_costs", []), jb.get("candidate_costs", [])
        if ca and cb and len(ca) == len(cb):
            if not close(ca, cb, THRESHOLDS["cost_tol"]):
                return False, "candidate_costs mismatch"

    return True, "ok"


def find_seed_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("seed_*"):
        if p.is_dir() and (p / "results.json").exists():
            out.append(p)
    return sorted(out)


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python compare_regression.py BASELINE_DIR NEW_DIR")
        sys.exit(2)
    base_root = Path(sys.argv[1]).resolve()
    new_root = Path(sys.argv[2]).resolve()
    if not base_root.exists():
        print(f"Baseline dir not found: {base_root}")
        sys.exit(1)
    if not new_root.exists():
        print(f"New dir not found: {new_root}")
        sys.exit(1)

    base_seeds = {p.relative_to(base_root): p for p in find_seed_dirs(base_root)}
    failed: list[tuple[str, str]] = []
    compared = 0
    for rel, sa in base_seeds.items():
        sb = new_root / rel
        if not sb.exists():
            failed.append((str(rel), "missing in new"))
            continue
        ok, msg = compare_one(sa, sb)
        compared += 1
        if not ok:
            failed.append((str(rel), msg))

    if failed:
        print("REGRESSION FAILED:")
        for rel, msg in failed[:20]:
            print(f"  {rel}: {msg}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
        sys.exit(1)
    print(f"PASS: {compared} seed dirs within threshold.")


if __name__ == "__main__":
    main()
