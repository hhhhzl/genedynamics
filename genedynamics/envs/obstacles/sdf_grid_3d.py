"""3D voxel SDF with trilinear interpolation and analytic gradients.

Companion to :mod:`sdf_texture` (2D version). The grid stores signed distances
only — gradients are computed analytically from the trilinear basis at query
time, not finite-differenced from a stored gradient field. This is faster
(grad evaluation costs the same as one extra interp), gives smooth
``sdf_and_grad`` under ``jax.grad``, and avoids the bias of FD near edges.

Three runtime backends — ``numpy`` (default, host), ``jax`` (device,
jit/vmap/grad-safe), ``torch`` (device, autograd) — selected per call. The
voxel grid is built once on host (typically via :meth:`from_mesh`) and is
backend-agnostic.

Conforms to the :class:`Obstacle` protocol (``sdf`` / ``contains`` /
``distance`` / ``gradient`` / ``center`` / ``bounds``) so it plugs straight
into ``ObstacleManager``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jax = None
    jnp = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


Backend = Literal["numpy", "jax", "torch"]


@dataclass
class SDFGrid3D:
    """Trilinear-interpolated 3D voxel SDF.

    Args (and stored fields):
        voxels: ``(Nx, Ny, Nz)`` signed distance field, float32.
        origin: ``(3,)`` world-space coordinate of voxel ``(0, 0, 0)``.
        spacing: ``(3,)`` voxel size along each axis (positive).

    The grid covers ``[origin, origin + (shape-1) * spacing]`` inclusive in
    each axis. Queries outside this AABB are clamped to the boundary face;
    callers wanting a sentinel value at far-field should pad the grid when
    baking.
    """

    voxels: np.ndarray             # (Nx, Ny, Nz) float32
    origin: np.ndarray             # (3,) float32
    spacing: np.ndarray            # (3,) float32

    # Plain instance attributes (filled by __post_init__) — kept as data, not
    # @property, so the Obstacle Protocol's isinstance check sees an ndarray
    # / tuple here rather than a descriptor.
    center: np.ndarray = field(init=False, default=None, repr=False)        # type: ignore[assignment]
    bounds: Tuple[np.ndarray, np.ndarray] = field(init=False, default=None, repr=False)  # type: ignore[assignment]

    # Cached device tensors (lazy).
    _vox_jax: Optional[Any] = field(default=None, repr=False)
    _vox_torch: Optional[Any] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.voxels = np.asarray(self.voxels, dtype=np.float32)
        self.origin = np.asarray(self.origin, dtype=np.float32).reshape(3)
        self.spacing = np.asarray(self.spacing, dtype=np.float32).reshape(3)
        if self.voxels.ndim != 3:
            raise ValueError(f"voxels must be 3D, got shape {self.voxels.shape}")
        if not np.all(self.spacing > 0):
            raise ValueError(f"spacing must be positive, got {self.spacing}")

        Nx, Ny, Nz = self.voxels.shape
        extents = np.array([Nx - 1, Ny - 1, Nz - 1], dtype=np.float32) * self.spacing
        aabb_max = self.origin + extents
        self.center = 0.5 * (self.origin + aabb_max)
        self.bounds = (self.origin.copy(), aabb_max)

    # ----- shape / metadata --------------------------------------------------

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(int(s) for s in self.voxels.shape)  # type: ignore[return-value]

    @property
    def aabb_min(self) -> np.ndarray:
        return self.bounds[0]

    @property
    def aabb_max(self) -> np.ndarray:
        return self.bounds[1]

    # ----- backend caches ----------------------------------------------------

    def _as_jax(self) -> Any:
        if jnp is None:
            raise RuntimeError("JAX is not available; install jax to use the jax backend.")
        if self._vox_jax is None:
            self._vox_jax = jnp.asarray(self.voxels, dtype=jnp.float32)
        return self._vox_jax

    def _as_torch(self, device: Optional[Any] = None) -> Any:
        if torch is None:
            raise RuntimeError("Torch is not available; install torch to use the torch backend.")
        cur = self._vox_torch
        target = torch.device(device) if device is not None else None
        if cur is None or (target is not None and cur.device != target):
            t = torch.as_tensor(self.voxels, dtype=torch.float32)
            if target is not None:
                t = t.to(target)
            self._vox_torch = t.contiguous()
        return self._vox_torch

    # ----- queries -----------------------------------------------------------

    def sdf(self, points: Any, *, backend: Backend = "numpy") -> Any:
        """Trilinear-interpolated SDF, shape ``(...,)`` for input ``(..., 3)``."""
        return self._sample(points, backend=backend, want_grad=False)

    def sdf_and_grad(
        self, points: Any, *, backend: Backend = "numpy"
    ) -> Tuple[Any, Any]:
        """Trilinear SDF and analytic gradient (from trilinear basis).

        Returns ``(phi, grad_phi)`` with ``phi`` shape ``(...,)`` and
        ``grad_phi`` shape ``(..., 3)``. The gradient is exact for the
        piecewise-trilinear field, not a finite-difference of stored grads.
        """
        return self._sample(points, backend=backend, want_grad=True)

    def project(
        self,
        points: Any,
        *,
        n_steps: int = 3,
        backend: Backend = "numpy",
    ) -> Any:
        """Iteratively project points onto the zero level set.

        Each step performs ``p ← p - φ(p) · n(p)`` where
        ``n = ∇φ / ‖∇φ‖``. ``n_steps`` is small in practice (2–4); convergence
        is super-linear because the trilinear field is locally smooth.

        Returns an array of the same shape as ``points``.
        """
        if backend == "jax":
            return self._project_jax(points, n_steps)
        if backend == "torch":
            return self._project_torch(points, n_steps)
        return self._project_numpy(points, n_steps)

    # ----- Obstacle protocol -------------------------------------------------

    def contains(self, point: np.ndarray) -> bool:
        return bool(self.sdf(np.asarray(point), backend="numpy") < 0.0)

    def distance(self, point: np.ndarray) -> float:
        return float(np.abs(self.sdf(np.asarray(point), backend="numpy")))

    def gradient(self, point: np.ndarray) -> np.ndarray:
        _, g = self.sdf_and_grad(np.asarray(point), backend="numpy")
        return np.asarray(g)

    def to_backend(self, backend: str) -> Any:
        """Return the voxel array in the requested runtime backend.

        For ``Obstacle`` protocol compatibility. Concrete callers will usually
        prefer :meth:`sdf` / :meth:`sdf_and_grad` with the ``backend`` keyword
        instead of materialising the raw voxels.
        """
        if backend == "jax":
            return self._as_jax()
        if backend == "torch":
            return self._as_torch()
        if backend == "numpy":
            return self.voxels
        raise ValueError(f"unsupported backend: {backend!r}")

    # ----- backend implementations -------------------------------------------

    def _sample(self, points: Any, *, backend: Backend, want_grad: bool) -> Any:
        if backend == "jax":
            return self._sample_jax(points, want_grad)
        if backend == "torch":
            return self._sample_torch(points, want_grad)
        return self._sample_numpy(points, want_grad)

    # ----- numpy --------------------------------------------------------------

    def _sample_numpy(self, points: np.ndarray, want_grad: bool) -> Any:
        pts = np.asarray(points, dtype=np.float32)
        shp = pts.shape[:-1]
        pts_f = pts.reshape(-1, 3)

        Nx, Ny, Nz = self.shape
        sx, sy, sz = float(self.spacing[0]), float(self.spacing[1]), float(self.spacing[2])
        ox, oy, oz = float(self.origin[0]), float(self.origin[1]), float(self.origin[2])

        ix = np.clip((pts_f[:, 0] - ox) / sx, 0.0, Nx - 1.0)
        iy = np.clip((pts_f[:, 1] - oy) / sy, 0.0, Ny - 1.0)
        iz = np.clip((pts_f[:, 2] - oz) / sz, 0.0, Nz - 1.0)

        x0 = np.floor(ix).astype(np.int32); x1 = np.minimum(x0 + 1, Nx - 1)
        y0 = np.floor(iy).astype(np.int32); y1 = np.minimum(y0 + 1, Ny - 1)
        z0 = np.floor(iz).astype(np.int32); z1 = np.minimum(z0 + 1, Nz - 1)
        wx = (ix - x0).astype(np.float32)
        wy = (iy - y0).astype(np.float32)
        wz = (iz - z0).astype(np.float32)

        v = self.voxels
        c000 = v[x0, y0, z0]; c100 = v[x1, y0, z0]
        c010 = v[x0, y1, z0]; c110 = v[x1, y1, z0]
        c001 = v[x0, y0, z1]; c101 = v[x1, y0, z1]
        c011 = v[x0, y1, z1]; c111 = v[x1, y1, z1]

        # Bilinear in xy at each z slab, then linear in z.
        c00 = c000 * (1 - wx) + c100 * wx
        c10 = c010 * (1 - wx) + c110 * wx
        c01 = c001 * (1 - wx) + c101 * wx
        c11 = c011 * (1 - wx) + c111 * wx
        c0 = c00 * (1 - wy) + c10 * wy
        c1 = c01 * (1 - wy) + c11 * wy
        phi = c0 * (1 - wz) + c1 * wz

        if not want_grad:
            return phi.reshape(shp)

        # Analytic trilinear gradient (closed form on the interior; clamping at
        # the boundary makes ∂φ/∂· silently zero perpendicular to the clamp).
        d_dx_z0 = (c100 - c000) * (1 - wy) + (c110 - c010) * wy
        d_dx_z1 = (c101 - c001) * (1 - wy) + (c111 - c011) * wy
        d_dx = ((d_dx_z0 * (1 - wz) + d_dx_z1 * wz) / sx)

        d_dy_z0 = (c010 - c000) * (1 - wx) + (c110 - c100) * wx
        d_dy_z1 = (c011 - c001) * (1 - wx) + (c111 - c101) * wx
        d_dy = ((d_dy_z0 * (1 - wz) + d_dy_z1 * wz) / sy)

        d_dz = ((c1 - c0) / sz)

        grad = np.stack([d_dx, d_dy, d_dz], axis=-1).astype(np.float32)
        return phi.reshape(shp), grad.reshape(shp + (3,))

    def _project_numpy(self, points: np.ndarray, n_steps: int) -> np.ndarray:
        p = np.asarray(points, dtype=np.float32).copy()
        for _ in range(int(n_steps)):
            phi, g = self._sample_numpy(p, want_grad=True)
            n = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)
            p = p - phi[..., None] * n
        return p

    # ----- jax ---------------------------------------------------------------

    def _sample_jax(self, points: Any, want_grad: bool) -> Any:
        if jnp is None:
            raise RuntimeError("JAX is not available; install jax to use the jax backend.")
        v = self._as_jax()
        pts = jnp.asarray(points, dtype=jnp.float32)
        shp = pts.shape[:-1]
        pts_f = pts.reshape((-1, 3))

        Nx, Ny, Nz = self.shape
        sx, sy, sz = float(self.spacing[0]), float(self.spacing[1]), float(self.spacing[2])
        origin = jnp.asarray(self.origin, dtype=jnp.float32)

        rel = (pts_f - origin) / jnp.asarray([sx, sy, sz], dtype=jnp.float32)
        ix = jnp.clip(rel[:, 0], 0.0, Nx - 1.0)
        iy = jnp.clip(rel[:, 1], 0.0, Ny - 1.0)
        iz = jnp.clip(rel[:, 2], 0.0, Nz - 1.0)

        x0 = jnp.floor(ix).astype(jnp.int32); x1 = jnp.minimum(x0 + 1, Nx - 1)
        y0 = jnp.floor(iy).astype(jnp.int32); y1 = jnp.minimum(y0 + 1, Ny - 1)
        z0 = jnp.floor(iz).astype(jnp.int32); z1 = jnp.minimum(z0 + 1, Nz - 1)
        wx = (ix - x0).astype(jnp.float32)
        wy = (iy - y0).astype(jnp.float32)
        wz = (iz - z0).astype(jnp.float32)

        c000 = v[x0, y0, z0]; c100 = v[x1, y0, z0]
        c010 = v[x0, y1, z0]; c110 = v[x1, y1, z0]
        c001 = v[x0, y0, z1]; c101 = v[x1, y0, z1]
        c011 = v[x0, y1, z1]; c111 = v[x1, y1, z1]

        c00 = c000 * (1 - wx) + c100 * wx
        c10 = c010 * (1 - wx) + c110 * wx
        c01 = c001 * (1 - wx) + c101 * wx
        c11 = c011 * (1 - wx) + c111 * wx
        c0 = c00 * (1 - wy) + c10 * wy
        c1 = c01 * (1 - wy) + c11 * wy
        phi = c0 * (1 - wz) + c1 * wz

        if not want_grad:
            return phi.reshape(shp)

        d_dx_z0 = (c100 - c000) * (1 - wy) + (c110 - c010) * wy
        d_dx_z1 = (c101 - c001) * (1 - wy) + (c111 - c011) * wy
        d_dx = (d_dx_z0 * (1 - wz) + d_dx_z1 * wz) / sx

        d_dy_z0 = (c010 - c000) * (1 - wx) + (c110 - c100) * wx
        d_dy_z1 = (c011 - c001) * (1 - wx) + (c111 - c101) * wx
        d_dy = (d_dy_z0 * (1 - wz) + d_dy_z1 * wz) / sy

        d_dz = (c1 - c0) / sz

        grad = jnp.stack([d_dx, d_dy, d_dz], axis=-1)
        return phi.reshape(shp), grad.reshape(shp + (3,))

    def _project_jax(self, points: Any, n_steps: int) -> Any:
        if jnp is None:
            raise RuntimeError("JAX is not available; install jax to use the jax backend.")
        p = jnp.asarray(points, dtype=jnp.float32)

        def step(carry, _):
            phi, g = self._sample_jax(carry, want_grad=True)
            n = g / jnp.maximum(jnp.linalg.norm(g, axis=-1, keepdims=True), 1e-8)
            return carry - phi[..., None] * n, None

        p, _ = jax.lax.scan(step, p, xs=None, length=int(n_steps))
        return p

    # ----- torch -------------------------------------------------------------

    def _sample_torch(self, points: Any, want_grad: bool) -> Any:
        if torch is None:
            raise RuntimeError("Torch is not available; install torch to use the torch backend.")
        v = self._as_torch(device=getattr(points, "device", None))
        pts = points
        if not isinstance(pts, torch.Tensor):
            pts = torch.as_tensor(pts, dtype=torch.float32, device=v.device)
        else:
            pts = pts.to(device=v.device, dtype=torch.float32)
        shp = pts.shape[:-1]
        pts_f = pts.reshape(-1, 3)

        Nx, Ny, Nz = self.shape
        sx = float(self.spacing[0]); sy = float(self.spacing[1]); sz = float(self.spacing[2])
        origin = torch.as_tensor(self.origin, dtype=torch.float32, device=v.device)

        rel = (pts_f - origin) / torch.tensor([sx, sy, sz], device=v.device)
        ix = rel[:, 0].clamp(0.0, Nx - 1.0)
        iy = rel[:, 1].clamp(0.0, Ny - 1.0)
        iz = rel[:, 2].clamp(0.0, Nz - 1.0)

        x0 = ix.floor().to(torch.int64); x1 = (x0 + 1).clamp(max=Nx - 1)
        y0 = iy.floor().to(torch.int64); y1 = (y0 + 1).clamp(max=Ny - 1)
        z0 = iz.floor().to(torch.int64); z1 = (z0 + 1).clamp(max=Nz - 1)
        wx = (ix - x0.to(torch.float32))
        wy = (iy - y0.to(torch.float32))
        wz = (iz - z0.to(torch.float32))

        c000 = v[x0, y0, z0]; c100 = v[x1, y0, z0]
        c010 = v[x0, y1, z0]; c110 = v[x1, y1, z0]
        c001 = v[x0, y0, z1]; c101 = v[x1, y0, z1]
        c011 = v[x0, y1, z1]; c111 = v[x1, y1, z1]

        c00 = c000 * (1 - wx) + c100 * wx
        c10 = c010 * (1 - wx) + c110 * wx
        c01 = c001 * (1 - wx) + c101 * wx
        c11 = c011 * (1 - wx) + c111 * wx
        c0 = c00 * (1 - wy) + c10 * wy
        c1 = c01 * (1 - wy) + c11 * wy
        phi = c0 * (1 - wz) + c1 * wz

        if not want_grad:
            return phi.reshape(shp)

        d_dx_z0 = (c100 - c000) * (1 - wy) + (c110 - c010) * wy
        d_dx_z1 = (c101 - c001) * (1 - wy) + (c111 - c011) * wy
        d_dx = (d_dx_z0 * (1 - wz) + d_dx_z1 * wz) / sx
        d_dy_z0 = (c010 - c000) * (1 - wx) + (c110 - c100) * wx
        d_dy_z1 = (c011 - c001) * (1 - wx) + (c111 - c101) * wx
        d_dy = (d_dy_z0 * (1 - wz) + d_dy_z1 * wz) / sy
        d_dz = (c1 - c0) / sz

        grad = torch.stack([d_dx, d_dy, d_dz], dim=-1)
        return phi.reshape(shp), grad.reshape(shp + (3,))

    def _project_torch(self, points: Any, n_steps: int) -> Any:
        if torch is None:
            raise RuntimeError("Torch is not available; install torch to use the torch backend.")
        p = points
        if not isinstance(p, torch.Tensor):
            p = torch.as_tensor(p, dtype=torch.float32)
        else:
            p = p.to(torch.float32)
        for _ in range(int(n_steps)):
            phi, g = self._sample_torch(p, want_grad=True)
            n = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            p = p - phi.unsqueeze(-1) * n
        return p

    # ----- baking ------------------------------------------------------------

    @classmethod
    def from_mesh(
        cls,
        mesh: Union[str, Path, Any],
        *,
        spacing: float | Tuple[float, float, float] = 0.01,
        padding: float = 0.05,
    ) -> "SDFGrid3D":
        """Bake a closed mesh into a voxel SDF.

        Args:
            mesh: file path or a ``trimesh.Trimesh`` object.
            spacing: voxel size (uniform if scalar; otherwise per-axis).
            padding: world-space padding around the mesh AABB. The voxel
                grid extends to ``aabb ± padding`` per axis so queries near
                the surface stay inside the bake region.

        Uses ``trimesh.proximity.signed_distance``: exact (slow) for small
        grids, with sign by inside/outside test (winding number under the
        hood). Done once on host; queries are then pure interp.
        """
        try:
            import trimesh
        except ImportError as e:
            raise ImportError(
                "SDFGrid3D.from_mesh requires trimesh. Install with: pip install trimesh"
            ) from e

        if isinstance(mesh, (str, Path)):
            tm = trimesh.load(str(mesh), force="mesh")
        else:
            tm = mesh
        if not isinstance(tm, trimesh.Trimesh):
            raise ValueError(f"loaded object is not a Trimesh: {type(tm)}")

        if isinstance(spacing, (int, float)):
            sxyz = np.array([spacing, spacing, spacing], dtype=np.float32)
        else:
            sxyz = np.asarray(spacing, dtype=np.float32).reshape(3)

        bounds = np.asarray(tm.bounds, dtype=np.float32)         # (2, 3)
        lo = bounds[0] - float(padding)
        hi = bounds[1] + float(padding)
        # Round shape so the grid is aligned to (lo + i*spacing).
        shape = np.maximum(np.ceil((hi - lo) / sxyz).astype(np.int64) + 1, 2)
        Nx, Ny, Nz = (int(s) for s in shape)

        xs = lo[0] + np.arange(Nx, dtype=np.float32) * sxyz[0]
        ys = lo[1] + np.arange(Ny, dtype=np.float32) * sxyz[1]
        zs = lo[2] + np.arange(Nz, dtype=np.float32) * sxyz[2]
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1).astype(np.float32)

        sdf_flat = trimesh.proximity.signed_distance(tm, pts).astype(np.float32)
        # trimesh convention: positive inside, negative outside. We flip so
        # *positive = outside* (standard SDF convention used elsewhere in this
        # package, e.g. SDFTexture2D and Obstacle.sdf).
        sdf_flat = -sdf_flat

        voxels = sdf_flat.reshape(Nx, Ny, Nz).astype(np.float32)
        return cls(voxels=voxels, origin=lo.astype(np.float32), spacing=sxyz)


__all__ = ["SDFGrid3D", "Backend"]
