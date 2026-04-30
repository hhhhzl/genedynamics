#!/usr/bin/env python3
"""
Verify a downloaded TUM RGB-D snapshot.

Checks each sequence directory for rgb.txt / depth.txt / groundtruth.txt.

Usage:
  python scripts/tasks/3dgs/prepare_tum_dataset.py data/tum
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a TUM RGB-D snapshot.")
    parser.add_argument("dataset_root", type=str, help="e.g. data/tum")
    parser.add_argument("--index", type=str, default=None, help="Optional JSON index output")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    if not root.exists():
        print(f"[error] not found: {root}", file=sys.stderr)
        return 1

    summary: list[dict] = []
    for seq_dir in sorted(root.iterdir()):
        if not seq_dir.is_dir() or not seq_dir.name.startswith("rgbd_dataset"):
            continue
        status = {"sequence": seq_dir.name, "ok": True, "issues": []}
        for required in ("rgb.txt", "depth.txt", "groundtruth.txt"):
            if not (seq_dir / required).exists():
                status["ok"] = False
                status["issues"].append(f"missing {required}")
        for d in ("rgb", "depth"):
            if not (seq_dir / d).is_dir():
                status["ok"] = False
                status["issues"].append(f"missing {d}/")
        if status["ok"]:
            with open(seq_dir / "rgb.txt") as f:
                status["n_rgb"] = sum(
                    1 for line in f if line.strip() and not line.startswith("#")
                )
        summary.append(status)

    total = len(summary)
    ok_count = sum(1 for s in summary if s["ok"])
    print(f"TUM snapshot at {root}: {ok_count}/{total} sequences OK")
    for s in summary:
        mark = "✓" if s["ok"] else "✗"
        extra = f" ({', '.join(s['issues'])})" if s["issues"] else ""
        n = s.get("n_rgb", "?")
        print(f"  {mark} {s['sequence']:40s}  n_rgb={n}{extra}")

    if args.index:
        Path(args.index).parent.mkdir(parents=True, exist_ok=True)
        with open(args.index, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Index written to {args.index}")
    return 0 if ok_count == total and total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
