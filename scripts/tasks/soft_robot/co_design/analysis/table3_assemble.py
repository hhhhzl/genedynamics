#!/usr/bin/env python3
"""Assemble the full Table 3: one row per prior, all shape/connectivity/diversity
/prior-NLL columns + (optionally) the locomotion-reward column.

Scans each `<bank-root>/<prior>/robotized/` bank, aggregates per-prior shape
metrics (via shape_metrics), and prints a markdown comparison table + JSON. The
locomotion-reward column is read from a `<prior>_loco_reward.json` (written by
the P4 reward sweep) if present.

Usage:
  python scripts/tasks/soft_robot/co_design/analysis/table3_assemble.py \
      --priors pointe:data/asset_banks/pointe shape:data/asset_banks/shape \
               random:data/asset_banks/loco_cpu triposg:data/asset_banks/triposg \
      [--reward-dir results/soft_robot/co_design/table3] \
      [--out results/soft_robot/co_design/table3/table3.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

from genedynamics.morphology.shape_metrics import (
    compute_shape_metrics, diversity_metrics, occupancy_grid_from_voxel_id,
)

# representation tag per prior (for the paper's "across representations" framing)
REPR = {"pointe": "point-cloud", "shape": "implicit-SDF", "trellis": "structured-voxel",
        "splatflow": "3D-Gaussian", "random": "procedural", "triposg": "image/mesh"}


def _bank_metrics(bank_dir):
    files = sorted(glob.glob(os.path.join(bank_dir, "robotized", "*.npz")))
    per, occ = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "voxel_id" not in d.files:
            continue
        g = occupancy_grid_from_voxel_id(d["voxel_id"], d["voxel_dims"])
        per.append(compute_shape_metrics(g))
        occ.append(g.astype(np.float32).ravel())
    if not per:
        return None
    keys = per[0].keys()
    agg = {k: float(np.mean([p[k] for p in per])) for k in keys}
    agg["single_cc_rate"] = float(np.mean([p["single_cc"] for p in per]))
    agg["n_robotized"] = len(per)
    if len({v.shape[0] for v in occ}) == 1:
        agg["diversity"] = diversity_metrics(np.stack(occ))["mean_pairwise_l2"]
    return agg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--priors", nargs="+", required=True,
                    help="name:bank_dir entries")
    ap.add_argument("--reward-dir", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    rows = {}
    for entry in args.priors:
        name, bank = entry.split(":", 1)
        m = _bank_metrics(bank)
        if m is None:
            print(f"[table3] {name}: no robotized assets in {bank}", file=sys.stderr)
            continue
        # raw count for RobotizationSuccess if a manifest exists
        man = os.path.join(bank, "manifest.json")
        if os.path.exists(man):
            mm = json.load(open(man))
            m["robotization_success_rate"] = mm.get("robotization_success_rate",
                                                     mm.get("n_robotized", 0) / max(mm.get("n_raw", 1), 1))
        # per-asset generation time (written by each prior's batch_infer)
        gt = os.path.join(bank, "gen_time.json")
        if os.path.exists(gt):
            m["gen_time_s"] = json.load(open(gt)).get("mean_s_per_asset")
        if args.reward_dir:
            rj = os.path.join(args.reward_dir, f"{name}_loco_reward.json")
            if os.path.exists(rj):
                m["loco_reward"] = json.load(open(rj)).get("mean_reward")
        rows[name] = m

    cols = [("repr", "representation"), ("gen_time_s", "gen-s/asset↓"),
            ("robotization_success_rate", "Robot%"),
            ("single_cc_rate", "1-CC"), ("solidity", "solid"),
            ("bilateral_symmetry", "sym"), ("sa_to_volume", "SA/V↓"),
            ("cavity_count", "cav↓"), ("diversity", "div"),
            ("loco_reward", "loco-R")]
    hdr = "| prior | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    print("\n## Table 3 — prior cross-comparison (shape quality + connectivity)\n")
    print(hdr); print(sep)
    for name, m in rows.items():
        cells = []
        for key, _ in cols:
            if key == "repr":
                cells.append(REPR.get(name, "?"))
            elif key in m and m[key] is not None:
                cells.append(f"{m[key]:.3f}" if isinstance(m[key], float) else str(m[key]))
            else:
                cells.append("—")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print(f"\n(n_robotized: " +
          ", ".join(f"{k}={v.get('n_robotized')}" for k, v in rows.items()) + ")")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=2)
        print(f"\n[table3] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
