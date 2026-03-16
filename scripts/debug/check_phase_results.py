#!/usr/bin/env python3
"""
Check Phase A and Phase B results.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    print("=== Phase A & B Results Check ===\n")

    # Phase A
    pa = ROOT / "results/phase_a/smoke/results.json"
    if pa.exists():
        with open(pa) as f:
            r = json.load(f)
        r0 = r[0]["result"]
        print("Phase A (crawling_ground, 1 mode, 1 fidelity):")
        print(f"  seeds: {len(r)}")
        print(f"  return_: {[x['result']['return_'] for x in r]}")
        print(f"  num_evaluations: {r0['num_evaluations']}")
        print(f"  wall_time (s): {[round(x['wall_time'], 2) for x in r]}")
        print(f"  bridge_history len: {len(r0['bridge_history'])}")
        print(f"  mode_responsibilities: {len(r0['mode_responsibilities'])} steps, dim={len(r0['mode_responsibilities'][0]) if r0['mode_responsibilities'] else 0}")
        print()
    else:
        print("Phase A: results not found\n")

    # Phase B
    configs = [
        ("s1_id", "full S1, ID (crawling_ground)"),
        ("s1_ood", "full S1, OOD (crawling_desert)"),
        ("nomode_id", "no-mode, ID"),
        ("nomode_ood", "no-mode, OOD"),
    ]
    for key, desc in configs:
        pb = ROOT / f"results/phase_b/{key}/results.json"
        if pb.exists():
            with open(pb) as f:
                r = json.load(f)
            returns = [x["result"]["return_"] for x in r]
            evals = r[0]["result"]["num_evaluations"]
            modes = r[0]["result"].get("mode_responsibilities", [])
            mode_dim = len(modes[0]) if modes else 0
            print(f"Phase B {key} ({desc}):")
            print(f"  seeds: {len(r)}, return_: {returns}")
            print(f"  num_evaluations: {evals}, mode_dim: {mode_dim}")
            print()
        else:
            print(f"Phase B {key}: results not found\n")

    print("Done.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
