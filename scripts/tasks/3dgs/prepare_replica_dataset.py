#!/usr/bin/env python3
"""
Verify a downloaded Replica snapshot and optionally write an index file.

This is mostly a sanity check: it walks `data/replica/<sequence>/` and
confirms that:
  - `results/frame000000.jpg` exists
  - `traj.txt` exists and has N lines
  - `cam_params.json` exists near the sequence

Usage:
  python scripts/tasks/3dgs/prepare_replica_dataset.py data/replica
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from genedynamics.data import REPLICA_SEQUENCES


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a Replica snapshot.")
    parser.add_argument("dataset_root", type=str, help="e.g. data/replica")
    parser.add_argument("--index", type=str, default=None, help="Optional JSON index output")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    if not root.exists():
        print(f"[error] not found: {root}", file=sys.stderr)
        return 1

    summary: list[dict] = []
    for seq in REPLICA_SEQUENCES:
        seq_dir = root / seq
        if not seq_dir.exists():
            continue
        traj = seq_dir / "traj.txt"
        results_dir = seq_dir / "results"
        cam_json = seq_dir / "cam_params.json"
        if not cam_json.exists():
            cam_json = root / "cam_params.json"

        status = {"sequence": seq, "root": str(seq_dir), "ok": True, "issues": []}
        if not traj.exists():
            status["ok"] = False
            status["issues"].append("missing traj.txt")
        else:
            with open(traj) as f:
                status["n_frames"] = sum(1 for _ in f)
        if not results_dir.exists():
            status["ok"] = False
            status["issues"].append("missing results/ (rgb frames)")
        if not cam_json.exists():
            status["ok"] = False
            status["issues"].append("missing cam_params.json")
        summary.append(status)

    total = len(summary)
    ok_count = sum(1 for s in summary if s["ok"])
    print(f"Replica snapshot at {root}: {ok_count}/{total} sequences OK")
    for s in summary:
        mark = "✓" if s["ok"] else "✗"
        extra = f" ({', '.join(s['issues'])})" if s["issues"] else ""
        n = s.get("n_frames", "?")
        print(f"  {mark} {s['sequence']:10s}  n_frames={n}{extra}")

    if args.index:
        Path(args.index).parent.mkdir(parents=True, exist_ok=True)
        with open(args.index, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Index written to {args.index}")
    return 0 if ok_count == total and total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
