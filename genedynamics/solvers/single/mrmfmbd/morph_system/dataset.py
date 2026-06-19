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


def build_morph_dataset(bank_root: str, n_voxels: int = 27, n_actuators: int = 10,
                        e_lo: float = 0.5, e_hi: float = 3.0):
    """Task-2 multi-head dataset from a robotized bank → dict of per-voxel fields.

    Returns {occ (M,n_voxels)∈[0,1], act (M,n_voxels,K) per-voxel actuator
    simplex, stiff (M,n_voxels)∈[0,1]}. Mirrors `build_occupancy_dataset` for occ;
    additionally aggregates the stored per-particle `actuator_id` into a per-voxel
    actuator distribution (the continuous placement target) and per-particle
    `E_per_particle` (if present) into a normalized per-voxel stiffness target.
    Voxels with no actuated particle get a uniform actuator target; absent E →
    a constant 0.5 stiffness target.
    """
    files = sorted(glob.glob(os.path.join(bank_root, "robotized", "*.npz")))
    occ_rows: List[np.ndarray] = []
    act_rows: List[np.ndarray] = []
    stiff_rows: List[np.ndarray] = []
    skipped = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        nv = int(d["n_voxels"]) if "n_voxels" in d.files else n_voxels
        if nv != int(n_voxels):
            skipped += 1
            continue
        vid = np.asarray(d["voxel_id"], dtype=np.int64)
        aid = np.asarray(d["actuator_id"], dtype=np.int64)

        counts = np.bincount(vid, minlength=int(n_voxels))[: int(n_voxels)].astype(np.float32)
        peak = float(counts.max())
        occ_rows.append(counts / peak if peak > 0.0 else counts)

        act = np.zeros((int(n_voxels), int(n_actuators)), dtype=np.float32)
        m = (aid >= 0) & (aid < int(n_actuators))
        np.add.at(act, (vid[m], aid[m]), 1.0)
        rs = act.sum(axis=1, keepdims=True)
        act = np.where(rs > 0.0, act / np.maximum(rs, 1e-8), 1.0 / float(n_actuators))
        act_rows.append(act.astype(np.float32))

        if "E_per_particle" in d.files:
            E = np.asarray(d["E_per_particle"], dtype=np.float32)
            sE = np.zeros(int(n_voxels), dtype=np.float32)
            cE = np.zeros(int(n_voxels), dtype=np.float32)
            np.add.at(sE, vid, E)
            np.add.at(cE, vid, 1.0)
            stiff = np.where(cE > 0.0, sE / np.maximum(cE, 1e-8), 0.5 * (e_lo + e_hi))
            stiff = np.clip((stiff - e_lo) / max(e_hi - e_lo, 1e-8), 0.0, 1.0)
        else:
            stiff = np.full(int(n_voxels), 0.5, dtype=np.float32)
        stiff_rows.append(stiff.astype(np.float32))

    if not occ_rows:
        raise FileNotFoundError(
            f"no usable robotized npz (n_voxels={n_voxels}) under "
            f"{bank_root}/robotized (skipped {skipped} grid-mismatched)"
        )
    return {
        "occ": np.stack(occ_rows, axis=0).astype(np.float32),
        "act": np.stack(act_rows, axis=0).astype(np.float32),
        "stiff": np.stack(stiff_rows, axis=0).astype(np.float32),
    }
