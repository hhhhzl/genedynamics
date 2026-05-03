"""Mesh → SDF / voxel utilities used by mesh_robotize.

Wraps trimesh and scikit-image so the rest of the morphology layer never has
to touch a 3D file format directly. Lazy imports — these libraries are
optional at the package level (only the mesh-based robotization path needs
them).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .priors.base import MissingDependencyError


def _require_trimesh():
    try:
        import trimesh
        return trimesh
    except ImportError as exc:
        raise MissingDependencyError(
            "trimesh is required for mesh-based robotization",
            install_hint="pip install trimesh",
        ) from exc


def _require_skimage():
    try:
        from skimage import measure
        return measure
    except ImportError as exc:
        raise MissingDependencyError(
            "scikit-image is required for connectivity analysis",
            install_hint="pip install scikit-image",
        ) from exc


# ---------------------------------------------------------------------------
# Mesh repair / normalization
# ---------------------------------------------------------------------------


def repair_mesh(mesh) -> "trimesh.Trimesh":
    """Best-effort cleanup: drop degenerate faces, keep the largest body,
    fix winding, fill simple holes. Returns a NEW mesh.
    """
    trimesh = _require_trimesh()
    m = mesh.copy()
    m.process(validate=True)
    m.remove_unreferenced_vertices()
    # If the mesh contains multiple disconnected bodies, keep only the
    # largest by volume — generated meshes routinely have stray fragments.
    bodies = m.split(only_watertight=False)
    if len(bodies) > 1:
        bodies = sorted(bodies, key=lambda b: float(b.volume), reverse=True)
        m = bodies[0]
    # Try to fix normals so inside/outside is well-defined for SDF queries.
    if not m.is_winding_consistent:
        m.fix_normals()
    # Conservative hole fill — only if cheap.
    if hasattr(m, "fill_holes"):
        try:
            m.fill_holes()
        except Exception:
            pass
    return m


def normalize_mesh_to_box(
    mesh,
    box_origin: Tuple[float, float, float],
    box_size: Tuple[float, float, float],
    *,
    margin: float = 0.0,
):
    """Translate + uniform-scale ``mesh`` to fit inside the AABB defined by
    ``box_origin`` (corner) and ``box_size`` (extents), with optional margin.

    Uses the longest extent for uniform scaling so aspect ratio is preserved
    (the writeup §3 robotization step "normalize scale and orientation" —
    we choose isotropic scale so an elongated worm stays a worm).
    Returns the rescaled mesh.
    """
    trimesh = _require_trimesh()
    m = mesh.copy()
    extents = m.extents.astype(np.float64)
    # Scale so the LONGEST mesh extent matches the LONGEST allowed box extent.
    box_size = np.asarray(box_size, dtype=np.float64)
    if margin > 0:
        box_size = box_size - 2.0 * float(margin)
    if np.any(box_size <= 0):
        raise ValueError(f"box_size after margin is non-positive: {box_size}")
    s = float(np.min(box_size / np.maximum(extents, 1e-9)))
    m.apply_scale(s)
    # Re-center on the box center.
    box_center = np.asarray(box_origin, dtype=np.float64) + 0.5 * np.asarray(box_size, dtype=np.float64)
    if margin > 0:
        box_center = (
            np.asarray(box_origin, dtype=np.float64)
            + np.asarray(margin, dtype=np.float64)
            + 0.5 * box_size
        )
    m.apply_translation(box_center - m.centroid)
    return m


# ---------------------------------------------------------------------------
# Voxelization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VoxelizationResult:
    """Output of `voxelize_mesh`.

    occupancy : (Nx, Ny, Nz) bool — True where the cell center is inside.
    origin : (3,) float64 — world coords of the (0, 0, 0) cell center.
    pitch : float — cell side length.
    """

    occupancy: np.ndarray
    origin: np.ndarray
    pitch: float

    @property
    def n_filled(self) -> int:
        return int(self.occupancy.sum())


def voxelize_mesh(
    mesh,
    pitch: float,
    *,
    fill_interior: bool = True,
) -> VoxelizationResult:
    """Voxelize a watertight-ish mesh on a regular grid of side ``pitch``.

    When ``fill_interior`` is True, runs trimesh's internal hollow→filled
    conversion (uses `voxelized.fill()`) so the body is solid, not a shell.
    """
    trimesh = _require_trimesh()
    if pitch <= 0:
        raise ValueError(f"pitch must be > 0, got {pitch}")
    vox = mesh.voxelized(pitch=float(pitch))
    if fill_interior:
        vox = vox.fill()
    occ = np.asarray(vox.matrix, dtype=bool)
    origin = np.asarray(vox.translation, dtype=np.float64)
    return VoxelizationResult(occupancy=occ, origin=origin, pitch=float(pitch))


def voxel_centers(vox: VoxelizationResult) -> np.ndarray:
    """World-coord centers of all FILLED cells. Shape (N, 3) float32."""
    idx = np.argwhere(vox.occupancy)  # (N, 3) int
    centers = vox.origin[None, :] + (idx.astype(np.float64) + 0.5) * vox.pitch
    return centers.astype(np.float32)


# ---------------------------------------------------------------------------
# Connectivity / support
# ---------------------------------------------------------------------------


def keep_largest_component(occ: np.ndarray) -> Tuple[np.ndarray, int]:
    """Largest 6-connected component of a 3D bool occupancy array.

    Returns ``(occupancy_filtered, n_dropped_cells)``. Used in robotize
    validity to reject designs whose body is fragmented.
    """
    measure = _require_skimage()
    if not occ.any():
        return occ, 0
    labels, n_labels = measure.label(occ, connectivity=1, return_num=True)
    if n_labels <= 1:
        return occ, 0
    # Component sizes.
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # background
    largest = int(np.argmax(sizes))
    keep = labels == largest
    n_dropped = int(occ.sum()) - int(keep.sum())
    return keep, n_dropped


def has_ground_support(centers: np.ndarray, floor_y: float = 0.05,
                        tol: float = 0.02) -> bool:
    """True if at least one particle is within ``tol`` of ``floor_y``.

    Used as a sanity check that the body actually rests on the floor (rather
    than being suspended mid-air).
    """
    if centers.size == 0:
        return False
    return bool(np.any(np.abs(centers[:, 1] - floor_y) <= tol))
