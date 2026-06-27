"""Shape / topology quality + connectivity metrics for co-design Table 3.

Task-agnostic, numpy-only (skimage for labeling, scipy for the convex hull when
available). Operates on a 3D occupancy GRID (vx, vy, vz) so it works for any
prior's robotized body. Two families:

* connectivity  — absorbs DiffuseBot's single-connected-component criterion
                  (`geometry_is_cc`): n_components, single_cc, largest_cc_fraction.
* shape quality — "does it look like a plausible locomoting soft body" rather
                  than a degenerate sim-exploit: solidity, bilateral symmetry,
                  surface-area-to-volume (jaggedness), enclosed-cavity count.

Plus batch-level diversity (mode-collapse) and a prior-NLL proxy (how in-
distribution a body is under a trained A2 decoder — the principled "looks real"
score, and the same learned prior that regularizes the co-design search).

All metrics are oriented so HIGHER = better EXCEPT where noted (n_components,
sa_to_volume, cavity_count: lower = better).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _require_skimage():
    try:
        from skimage import measure
        return measure
    except Exception as e:  # pragma: no cover
        raise ImportError("shape_metrics needs scikit-image (pip install scikit-image)") from e


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


def occupancy_grid_from_voxel_id(voxel_id: np.ndarray, voxel_dims) -> np.ndarray:
    """(N,) particle→voxel ids + (vx,vy,vz) → bool occupancy grid (vx,vy,vz)."""
    vx, vy, vz = (int(v) for v in voxel_dims)
    counts = np.bincount(np.asarray(voxel_id, dtype=np.int64), minlength=vx * vy * vz)
    return (counts[: vx * vy * vz] > 0).reshape(vx, vy, vz)


def occupancy_grid_from_occ(occ: np.ndarray, voxel_dims, threshold: float = 0.5) -> np.ndarray:
    """(n_voxels,) continuous occupancy + (vx,vy,vz) → bool grid via threshold."""
    vx, vy, vz = (int(v) for v in voxel_dims)
    return (np.asarray(occ, dtype=np.float32).reshape(vx, vy, vz) > float(threshold))


# ---------------------------------------------------------------------------
# Connectivity (DiffuseBot single-CC criterion)
# ---------------------------------------------------------------------------


def connectivity_metrics(grid: np.ndarray) -> Dict[str, float]:
    """6-connected-component stats. single_cc mirrors DiffuseBot's accept gate."""
    measure = _require_skimage()
    g = np.asarray(grid, dtype=bool)
    total = int(g.sum())
    if total == 0:
        return {"n_components": 0.0, "single_cc": 0.0, "largest_cc_fraction": 0.0}
    labels, n = measure.label(g, connectivity=1, return_num=True)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(sizes.max())
    return {
        "n_components": float(n),
        "single_cc": float(n == 1),
        "largest_cc_fraction": float(largest) / float(total),
    }


# ---------------------------------------------------------------------------
# Shape quality
# ---------------------------------------------------------------------------


def solidity(grid: np.ndarray) -> float:
    """filled / convex-hull volume ∈ (0,1]; low = spiky/ugly. Falls back to the
    bounding-box fill ratio if scipy's ConvexHull is unavailable or degenerate."""
    g = np.asarray(grid, dtype=bool)
    filled = int(g.sum())
    if filled == 0:
        return 0.0
    coords = np.argwhere(g).astype(np.float64)
    try:
        from scipy.spatial import ConvexHull
        if coords.shape[0] >= 4 and np.ptp(coords, axis=0).min() > 0:
            # +1 cell padding so a flat slab still has positive hull volume.
            hull = ConvexHull(coords)
            hull_vol = float(hull.volume)
            if hull_vol > 1e-9:
                return float(min(1.0, filled / (hull_vol + filled * 0.0 + 1.0)))
    except Exception:
        pass
    bbox = np.prod(np.ptp(coords, axis=0) + 1.0)
    return float(filled / max(bbox, 1.0))


def bilateral_symmetry(grid: np.ndarray) -> float:
    """Best left-right mirror agreement over the 3 axes ∈ [0,1]; 1 = symmetric.
    Locomoting animals are typically bilaterally symmetric, so this flags the
    'lopsided ugly body' failure mode."""
    g = np.asarray(grid, dtype=np.float32)
    denom = float(g.sum()) * 2.0
    if denom == 0.0:
        return 0.0
    best = 0.0
    for ax in range(3):
        agree = 1.0 - float(np.abs(g - np.flip(g, axis=ax)).sum()) / denom
        best = max(best, agree)
    return best


def surface_area_to_volume(grid: np.ndarray) -> float:
    """Exposed-face count / filled-cell count; high = jagged/spiky (lower better).
    A solid cube → low; a scatter of specks → high."""
    g = np.asarray(grid, dtype=bool)
    filled = int(g.sum())
    if filled == 0:
        return 0.0
    exposed = 0
    for ax in range(3):
        for shift in (1, -1):
            nb = np.roll(g, shift, axis=ax)
            # cells on the rolled-in face count as exposed (outside = empty)
            idx = [slice(None)] * 3
            idx[ax] = (0 if shift == 1 else -1)
            nb[tuple(idx)] = False
            exposed += int((g & ~nb).sum())
    return float(exposed) / float(filled)


def cavity_count(grid: np.ndarray) -> float:
    """Number of fully-enclosed empty pockets (interior holes); lower = better.
    Background reachable from the grid boundary is flood-filled; remaining empty
    connected components are cavities."""
    measure = _require_skimage()
    g = np.asarray(grid, dtype=bool)
    empty = ~g
    labels, n = measure.label(empty, connectivity=1, return_num=True)
    if n == 0:
        return 0.0
    # Labels touching any face are "outside"; the rest are enclosed cavities.
    border = set()
    for ax in range(3):
        for end in (0, -1):
            idx = [slice(None)] * 3
            idx[ax] = end
            border.update(np.unique(labels[tuple(idx)]).tolist())
    border.discard(0)
    interior = set(range(1, n + 1)) - border
    return float(len(interior))


def compute_shape_metrics(grid: np.ndarray) -> Dict[str, float]:
    """All per-body connectivity + shape-quality metrics in one dict."""
    m = connectivity_metrics(grid)
    m.update({
        "solidity": solidity(grid),
        "bilateral_symmetry": bilateral_symmetry(grid),
        "sa_to_volume": surface_area_to_volume(grid),
        "cavity_count": cavity_count(grid),
        "n_filled": float(np.asarray(grid, dtype=bool).sum()),
    })
    return m


# ---------------------------------------------------------------------------
# Batch-level: diversity + prior-NLL proxy
# ---------------------------------------------------------------------------


def diversity_metrics(occ_vectors: np.ndarray) -> Dict[str, float]:
    """Mean pairwise L2 over (M, n_voxels) occupancy → mode-collapse detector."""
    X = np.asarray(occ_vectors, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] < 2:
        return {"mean_pairwise_l2": 0.0, "n_samples": float(X.shape[0] if X.ndim else 0)}
    diffs = X[:, None, :] - X[None, :, :]
    d = np.sqrt((diffs ** 2).sum(-1))
    iu = np.triu_indices(X.shape[0], k=1)
    return {"mean_pairwise_l2": float(d[iu].mean()), "n_samples": float(X.shape[0])}


def prior_reconstruction_error(occ01: np.ndarray, decoder) -> np.ndarray:
    """Per-asset encode→decode MSE under a trained A2 decoder = in-distribution
    proxy (low = the body lies on the learned 'plausible soft-body' manifold).
    occ01: (M, n_voxels) in [0,1]. Returns (M,)."""
    X = np.asarray(occ01, dtype=np.float32)
    recon = np.asarray(decoder.reconstruct(X))
    return ((recon - X) ** 2).mean(axis=-1)
