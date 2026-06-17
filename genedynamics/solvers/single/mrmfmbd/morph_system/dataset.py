"""Build the A2 training dataset: per-asset coarse occupancy vectors from a
robotized asset bank (the offline, simulator-free side of A2).

Each robotized `<bank>/robotized/<id>.npz` holds a SoftBodySpec; we bucket its
particles into the fixed coarse voxel grid (via the stored `voxel_id`) and
normalize per-asset so the densest voxel reads 1.0. The result is an
(M, n_voxels) matrix in [0, 1] — the distribution of *plausible* bodies the
VAE learns a latent prior over.
"""

from __future__ import annotations

import glob
import os
from typing import List

import numpy as np


def build_occupancy_dataset(bank_root: str, n_voxels: int = 27) -> np.ndarray:
    """Return (M, n_voxels) float32 occupancy in [0, 1] from robotized npz files.

    occupancy[v] = (#particles bucketed into coarse voxel v) / (densest voxel
    count for that asset). Assets whose stored n_voxels mismatches `n_voxels`
    are skipped with a note (grid must match the MBD body grid).
    """
    files = sorted(glob.glob(os.path.join(bank_root, "robotized", "*.npz")))
    rows: List[np.ndarray] = []
    skipped = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        nv = int(d["n_voxels"]) if "n_voxels" in d.files else n_voxels
        if nv != int(n_voxels):
            skipped += 1
            continue
        vid = np.asarray(d["voxel_id"], dtype=np.int64)
        counts = np.bincount(vid, minlength=int(n_voxels)).astype(np.float32)
        counts = counts[: int(n_voxels)]
        peak = float(counts.max())
        if peak > 0.0:
            counts = counts / peak
        rows.append(counts)
    if not rows:
        raise FileNotFoundError(
            f"no usable robotized npz (n_voxels={n_voxels}) under "
            f"{bank_root}/robotized (skipped {skipped} grid-mismatched)"
        )
    return np.stack(rows, axis=0).astype(np.float32)
