#!/usr/bin/env python3
"""Table-3 metrics: per-prior shape / connectivity / diversity aggregation.

Loads a robotized asset bank (one prior) and emits the co-design-relevant quality
columns the prior cross-comparison table reports — beyond raw RobotizationSuccess:

  connectivity : single_cc_rate (DiffuseBot geometry_is_cc), mean n_components,
                 mean largest_cc_fraction
  shape quality: solidity, bilateral_symmetry, sa_to_volume (jaggedness),
                 cavity_count  (does it look like a plausible locomoting body?)
  diversity    : mean pairwise occupancy L2 (mode-collapse detector)
  prior-NLL    : (optional) encode→decode MSE under an A2 decoder = how
                 in-distribution the bodies are (the principled "looks real").

Usage:
  python scripts/tasks/soft_robot/co_design/analysis/table3_metrics.py \
      --bank-root data/asset_banks/loco_cpu [--decoder data/morph_decoders/loco_cpu] \
      [--out results/soft_robot/co_design/analysis/table3_loco_cpu.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

from genedynamics.learning.priors.morphology.shape_metrics import (
    compute_shape_metrics, diversity_metrics, prior_reconstruction_error,
    occupancy_grid_from_voxel_id,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank-root", required=True)
    ap.add_argument("--decoder", default="", help="optional A2 decoder dir for prior-NLL")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.bank_root, "robotized", "*.npz")))
    if not files:
        print(f"[table3] no robotized npz under {args.bank_root}/robotized", file=sys.stderr)
        return 1

    per_asset = []
    occ_vectors = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "voxel_id" not in d.files or "voxel_dims" not in d.files:
            continue
        vd = d["voxel_dims"]
        grid = occupancy_grid_from_voxel_id(d["voxel_id"], vd)
        per_asset.append(compute_shape_metrics(grid))
        occ_vectors.append(grid.astype(np.float32).ravel())

    if not per_asset:
        print("[table3] no usable assets", file=sys.stderr)
        return 1

    keys = list(per_asset[0].keys())
    agg = {k: float(np.mean([a[k] for a in per_asset])) for k in keys}
    agg["single_cc_rate"] = float(np.mean([a["single_cc"] for a in per_asset]))
    agg["n_assets"] = len(per_asset)

    # diversity needs a common occupancy length (same grid across the bank)
    lens = {v.shape[0] for v in occ_vectors}
    if len(lens) == 1:
        agg.update({f"diversity_{k}": v for k, v in
                    diversity_metrics(np.stack(occ_vectors)).items()})

    # optional prior-NLL (in-distribution proxy) under a trained A2 decoder
    if args.decoder and len(lens) == 1:
        try:
            from genedynamics.solvers.single.mrmfmbd.morph_system.decoder import MorphDecoder
            from genedynamics.solvers.single.mrmfmbd.morph_system.specs import MorphDecoderConfig
            with open(os.path.join(args.decoder, "decoder_config.json")) as fp:
                dc = json.load(fp)
            cfg = MorphDecoderConfig(latent_dim=int(dc["latent_dim"]),
                                     hidden_dim=int(dc["hidden_dim"]),
                                     n_voxels=int(dc["n_voxels"]),
                                     x_lo=float(dc["x_lo"]), x_hi=float(dc["x_hi"]))
            dec = MorphDecoder.load(os.path.join(args.decoder, "decoder_params.npz"), cfg)
            occ01 = np.stack(occ_vectors)  # already in {0,1}
            if occ01.shape[1] == cfg.n_voxels:
                nll = prior_reconstruction_error(occ01, dec)
                agg["prior_nll_mean"] = float(nll.mean())
                agg["prior_nll_std"] = float(nll.std())
        except Exception as e:
            print(f"[table3] prior-NLL skipped: {e!r}", file=sys.stderr)

    prior = os.path.basename(os.path.normpath(args.bank_root))
    print(f"=== Table-3 metrics: {prior} ({agg['n_assets']} assets) ===")
    order = ["single_cc_rate", "n_components", "largest_cc_fraction", "solidity",
             "bilateral_symmetry", "sa_to_volume", "cavity_count",
             "diversity_mean_pairwise_l2", "prior_nll_mean"]
    for k in order:
        if k in agg:
            arrow = "↑" if k in ("single_cc_rate", "largest_cc_fraction", "solidity",
                                 "bilateral_symmetry", "diversity_mean_pairwise_l2") else "↓"
            print(f"  {k:30s} {agg[k]:.4f}  ({arrow} better)")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fp:
            json.dump({"prior": prior, "metrics": agg}, fp, indent=2)
        print(f"[table3] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
