#!/usr/bin/env python
"""Robotize every mesh in an asset bank → SoftBodySpec cache + RobotizationSuccess.

Usage:
    python scripts/morphology/robotize_bank.py \
        --bank-root data/asset_banks/loco_v1 \
        [--n-actuators 10] [--voxel-dims 3,3,3] \
        [--box-origin 0.30,0.05,0.40] [--box-size 0.10,0.06,0.10] \
        [--particle-spacing 0.0078125] \
        [--force]

For each AssetEntry, runs `robotize_mesh(...)` and writes:
    <bank_root>/robotized/<asset_id>.npz   (the SoftBodySpec arrays)
    <bank_root>/robotized/<asset_id>.json  (the RobotizeReport)

Skips assets that are already robotized unless --force is given. Updates the
manifest in place. Final stdout line is the bank's RobotizationSuccess.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from genedynamics.morphology import (
    AssetBank,
    MeshRobotizeConfig,
    robotize_mesh,
)


log = logging.getLogger("robotize_bank")


def _parse_triple(s: str, dtype) -> Tuple:
    parts = s.split(",")
    if len(parts) != 3:
        raise ValueError(f"expected 3 comma-separated values, got: {s!r}")
    return tuple(dtype(p.strip()) for p in parts)


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bank-root", required=True,
                    help="path to a bank directory containing manifest.json")
    ap.add_argument("--n-actuators", type=int, default=10)
    ap.add_argument("--voxel-dims", default="3,3,3", help="vx,vy,vz")
    ap.add_argument("--box-origin", default="0.30,0.05,0.40")
    ap.add_argument("--box-size", default="0.10,0.06,0.10")
    ap.add_argument("--box-margin", type=float, default=0.005)
    ap.add_argument("--particle-spacing", type=float, default=1.0 / 128.0)
    ap.add_argument("--passive-top-quantile", type=float, default=0.80)
    ap.add_argument("--min-filled-cells", type=int, default=64)
    ap.add_argument("--force", action="store_true",
                    help="re-robotize assets that already have a cached spec")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    try:
        import trimesh  # noqa: F401
    except ImportError:
        log.error("trimesh is required (pip install trimesh)")
        return 2

    cfg = MeshRobotizeConfig(
        box_origin=_parse_triple(args.box_origin, float),
        box_size=_parse_triple(args.box_size, float),
        box_margin=float(args.box_margin),
        particle_spacing=float(args.particle_spacing),
        voxel_dims=_parse_triple(args.voxel_dims, int),
        n_actuators=int(args.n_actuators),
        passive_top_quantile=float(args.passive_top_quantile),
        min_filled_cells=int(args.min_filled_cells),
    )

    bank = AssetBank.open(args.bank_root)
    log.info("Bank: %s  (%d assets)", args.bank_root, len(bank.manifest.assets))

    import trimesh
    n_done = 0
    n_skipped = 0
    n_success = 0
    n_failure = 0
    for asset in bank.manifest.assets:
        if asset.robotized and not args.force:
            n_skipped += 1
            continue
        if not asset.mesh_path:
            log.warning("asset %s has no mesh_path; skipping", asset.id)
            continue
        mesh_full = os.path.join(bank.root, asset.mesh_path)
        try:
            loaded = trimesh.load(mesh_full, force="mesh")
        except Exception:
            log.exception("failed to load mesh %s", mesh_full)
            n_failure += 1
            continue
        spec, report = robotize_mesh(loaded, cfg)
        bank.add_robotized(asset, spec, report)
        n_done += 1
        if report.success:
            n_success += 1
            log.debug("OK   %s  particles=%d", asset.id, report.n_filled_cells)
        else:
            n_failure += 1
            log.info("FAIL %s  reasons=%s", asset.id, report.failure_reasons)
    bank.save()

    rate = bank.robotization_success_rate()
    log.info("Robotized %d (%d cached as success, %d failure, %d skipped)",
             n_done, n_success, n_failure, n_skipped)
    log.info("RobotizationSuccess (whole bank): %.1f%%  (writeup target ≥ 30%%)",
             100.0 * rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
