#!/usr/bin/env python
"""Train the A2 morphology VAE on a robotized asset bank; save decoder params.

This is the offline, simulator-free side of A2 (joint model-free / model-based).
It learns a low-dim shape latent w whose decoder g(w) → per-voxel occupancy on
the fixed MBD grid; the VAE's N(0,I) latent prior becomes the model-free
morphology prior the joint MBD sampler diffuses over.

Usage:
    python scripts/tasks/soft_robot/morphology/train_morph_ae.py \
        --bank-root data/asset_banks/loco_cpu \
        --out data/morph_decoders/loco_cpu \
        --latent-dim 8 --hidden-dim 32 --n-voxels 27 --steps 2000

Outputs under <out>/:
    decoder_params.npz    VAE params (encoder + decoder)
    decoder_config.json   MorphDecoderConfig fields + dataset size
    train_history.json    (step, loss, recon, kl) samples
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import numpy as np

from genedynamics.solvers.single.mrmfmbd.morph_system import (
    MorphDecoderConfig,
    MorphDecoder,
    build_occupancy_dataset,
    train_vae,
    save_params,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank-root", required=True, help="robotized asset bank dir")
    ap.add_argument("--out", required=True, help="output dir for decoder params")
    ap.add_argument("--latent-dim", type=int, default=8)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--n-voxels", type=int, default=27)
    ap.add_argument("--x-lo", type=float, default=0.2)
    ap.add_argument("--x-hi", type=float, default=1.0)
    ap.add_argument("--beta-kl", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = MorphDecoderConfig(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_voxels=args.n_voxels,
        x_lo=args.x_lo,
        x_hi=args.x_hi,
        beta_kl=args.beta_kl,
    )

    X = build_occupancy_dataset(args.bank_root, n_voxels=cfg.n_voxels)
    print(f"[train_morph_ae] dataset: {X.shape} occupancy vectors "
          f"(mean fill {float(X.mean()):.3f})")

    params, history = train_vae(X, cfg, steps=args.steps, lr=args.lr, seed=args.seed)

    # Reconstruction diagnostic on the (deterministic, μ-based) round trip.
    dec = MorphDecoder(params, cfg)
    recon = np.asarray(dec.reconstruct(X))
    recon_mse = float(np.mean((recon - X) ** 2))
    print(f"[train_morph_ae] final train loss {history[-1][1]:.4f} "
          f"(recon {history[-1][2]:.4f}, kl {history[-1][3]:.4f}); "
          f"mu-recon MSE {recon_mse:.4f}")

    os.makedirs(args.out, exist_ok=True)
    save_params(os.path.join(args.out, "decoder_params.npz"), params)
    with open(os.path.join(args.out, "decoder_config.json"), "w") as f:
        d = dataclasses.asdict(cfg)
        d["dataset_size"] = int(X.shape[0])
        d["recon_mse"] = recon_mse
        json.dump(d, f, indent=2)
    with open(os.path.join(args.out, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train_morph_ae] saved decoder to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
