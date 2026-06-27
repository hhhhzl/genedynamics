#!/usr/bin/env python
"""Robotize a directory of pre-generated 3D assets into an asset bank.

The heavy text-to-3D priors (Point-E / Shap-E / TRELLIS-text / SplatFlow) run in
their own isolated venvs and dump raw assets to `<bank>/raw/`; this MAIN-env
script converts each to a robotized SoftBodySpec on the fixed MBD voxel grid via
the modality-appropriate tail and writes `<bank>/robotized/<id>.npz` + a manifest.
Decouples the prior env from the JAX/MPM env (prior never shares a GPU process).

Raw formats (auto-detected):
  *.npz  with `points` (N,3)                  -> robotize_point_cloud   (Point-E)
  *.npz  with `centers`(+`scales`,`opacities`)-> robotize_gaussians     (SplatFlow)
  *.obj/*.ply/*.glb/*.stl mesh                -> robotize_mesh          (Shap-E/TRELLIS)

Usage:
  python scripts/tasks/soft_robot/morphology/build_bank_from_raw.py \
      --raw-dir data/asset_banks/pointe/raw --bank-root data/asset_banks/pointe \
      --voxel-dims 3,3,3 --n-actuators 10
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

from genedynamics.morphology import (
    MeshRobotizeConfig, robotize_mesh, robotize_point_cloud, robotize_gaussians,
    save_spec_npz,
)


def _triple(s, cast):
    return tuple(cast(x) for x in s.split(","))


def _robotize_one(path, cfg):
    """Dispatch a raw asset file to the right robotize tail → (spec, report)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        d = np.load(path, allow_pickle=True)
        if "points" in d.files:
            return robotize_point_cloud(np.asarray(d["points"], np.float32), cfg)
        if "centers" in d.files:
            centers = np.asarray(d["centers"], np.float32)
            opac = np.asarray(d["opacities"], np.float32).ravel() if "opacities" in d.files else None
            # Multi-view 3DGS dumps ~500k gaussians, ~94% near-transparent
            # background + far floaters. Keep the opaque object gaussians, then
            # trim spatial outliers (floaters) before voxelizing, else the bbox
            # is dominated by junk and the body collapses to nothing.
            if opac is not None:
                keep = opac > 0.5
                if keep.sum() < 32:                       # fall back to a softer cut
                    keep = opac > np.quantile(opac, 0.97)
                centers = centers[keep]
            if len(centers) > 8:
                med = np.median(centers, axis=0)
                dist = np.linalg.norm(centers - med, axis=1)
                centers = centers[dist < np.quantile(dist, 0.92)]   # drop 8% floaters
            # filtered object surface points -> voxelize + fill (cfg.fill_interior)
            return robotize_point_cloud(centers, cfg)
        raise ValueError(f"{path}: .npz has neither 'points' nor 'centers'")
    # mesh
    import trimesh
    mesh = trimesh.load(path, force="mesh")
    return robotize_mesh(mesh, cfg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--bank-root", required=True)
    ap.add_argument("--voxel-dims", default="3,3,3")
    ap.add_argument("--n-actuators", type=int, default=10)
    ap.add_argument("--box-origin", default="0.30,0.05,0.40")
    ap.add_argument("--box-size", default="0.10,0.06,0.10")
    ap.add_argument("--particle-spacing", type=float, default=1.0 / 128.0)
    ap.add_argument("--min-filled-cells", type=int, default=24)
    ap.add_argument("--passive-top-quantile", type=float, default=0.80)
    ap.add_argument("--require-single-component", action="store_true",
                    help="DiffuseBot-strict: reject non-single-CC bodies")
    args = ap.parse_args(argv)

    cfg = MeshRobotizeConfig(
        voxel_dims=_triple(args.voxel_dims, int),
        box_origin=_triple(args.box_origin, float),
        box_size=_triple(args.box_size, float),
        particle_spacing=args.particle_spacing,
        n_actuators=args.n_actuators,
        passive_top_quantile=args.passive_top_quantile,
        min_filled_cells=args.min_filled_cells,
        require_ground_support=False,   # prior bodies are not floor-aligned a priori
        require_single_component=bool(args.require_single_component),
    )

    raw = sorted(
        f for ext in ("*.npz", "*.obj", "*.ply", "*.glb", "*.stl")
        for f in glob.glob(os.path.join(args.raw_dir, ext))
    )
    if not raw:
        print(f"[build_bank] no raw assets under {args.raw_dir}", file=sys.stderr)
        return 1

    out_dir = os.path.join(args.bank_root, "robotized")
    os.makedirs(out_dir, exist_ok=True)
    n_ok, entries = 0, []
    for i, path in enumerate(raw):
        asset_id = f"{os.path.splitext(os.path.basename(path))[0]}"
        try:
            spec, rep = _robotize_one(path, cfg)
        except Exception as e:
            print(f"[build_bank] {asset_id}: ERROR {e!r}", flush=True)
            entries.append({"id": asset_id, "success": False, "error": repr(e)})
            continue
        rec = {"id": asset_id, "success": bool(rep.success),
               "n_filled": int(rep.n_filled_cells), "n_components": int(rep.n_components),
               "reasons": list(rep.failure_reasons)}
        if rep.success and spec is not None:
            save_spec_npz(spec, os.path.join(out_dir, f"{asset_id}.npz"))
            n_ok += 1
        entries.append(rec)
        print(f"[build_bank] {i+1}/{len(raw)} {asset_id}: "
              f"{'OK' if rep.success else 'FAIL'} "
              f"(filled={rep.n_filled_cells}, n_comp={rep.n_components})", flush=True)

    rate = n_ok / max(len(raw), 1)
    manifest = {"raw_dir": args.raw_dir, "n_raw": len(raw), "n_robotized": n_ok,
                "robotization_success_rate": rate,
                "voxel_dims": list(_triple(args.voxel_dims, int)),
                "n_actuators": args.n_actuators, "entries": entries}
    with open(os.path.join(args.bank_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[build_bank] RobotizationSuccess {n_ok}/{len(raw)} = {rate:.1%} "
          f"-> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
