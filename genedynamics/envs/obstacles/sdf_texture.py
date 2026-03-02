"""
SDF texture (grid) acceleration for obstacles.

This module provides a MDOC-style 2D "texture" representation storing:
  - sdf(x, y)
  - ∂sdf/∂x
  - ∂sdf/∂y

as a dense grid, enabling fast sampling via:
  - Torch: torch.nn.functional.grid_sample (GPU + torch.compile friendly)
  - JAX: pure JAX bilinear sampling (jit/vmap friendly)

The texture is intended as a *fast approximate* SDF backend for planners/filters
that need repeated SDF/gradient queries (e.g., CBF-style safety filters inside
diffusion rollouts). Exact geometry checks can still use ObstacleManager.sdf().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Union

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jax = None
    jnp = None


Backend = Literal["numpy", "torch", "jax"]


@dataclass
class SDFTexture2D:
    """
    2D SDF texture with gradients.

    Coordinates:
      - Grid covers [x_min, x_max] × [y_min, y_max] with resolution `res`.
      - Stored tensor has shape (3, H, W): [sdf, gx, gy].
    """

    tex_np: np.ndarray  # (3, H, W), float32
    x_min: float
    y_min: float
    res: float

    # Cached device tensors
    _tex_torch: Optional["torch.Tensor"] = None  # (1, 3, H, W)
    _tex_jax: Optional["jax.Array"] = None       # (3, H, W)

    @property
    def H(self) -> int:
        return int(self.tex_np.shape[1])

    @property
    def W(self) -> int:
        return int(self.tex_np.shape[2])

    @property
    def x_max(self) -> float:
        return float(self.x_min + (self.W - 1) * self.res)

    @property
    def y_max(self) -> float:
        return float(self.y_min + (self.H - 1) * self.res)

    def to_torch(self, device: Optional[Union[str, "torch.device"]] = None) -> "torch.Tensor":
        if torch is None:
            raise RuntimeError("Torch is not available; cannot convert SDFTexture2D to torch.")
        if self._tex_torch is None or (device is not None and self._tex_torch.device != torch.device(device)):
            t = torch.as_tensor(self.tex_np, dtype=torch.float32)
            t = t.unsqueeze(0)  # (1,3,H,W)
            if device is not None:
                t = t.to(device)
            self._tex_torch = t.contiguous()
        return self._tex_torch

    def to_jax(self) -> "jax.Array":
        if jnp is None:
            raise RuntimeError("JAX is not available; cannot convert SDFTexture2D to JAX.")
        if self._tex_jax is None:
            # Check if we're inside a JAX transformation (would cause tracer leak)
            # This is a best-effort check - if jax is available, try to detect tracers
            try:
                # Convert outside any transformation - this should be called before JIT compilation
                self._tex_jax = jnp.asarray(self.tex_np, dtype=jnp.float32)
            except Exception as e:
                # If conversion fails (e.g., inside a transformation), raise a clearer error
                raise RuntimeError(
                    "Cannot convert SDF texture to JAX inside a JAX transformation. "
                    "Call to_jax() before creating JAX-transformed functions (e.g., before jax.jit)."
                ) from e
        return self._tex_jax

    @staticmethod
    def build_from_sdf_fn(
        sdf_fn,
        *,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        res: float = 0.01,
    ) -> "SDFTexture2D":
        """
        Build texture by evaluating a scalar sdf_fn on a dense grid.
        sdf_fn must accept a point array shape (2,) and return a scalar.
        """
        res = float(res)
        W = max(2, int(round((x_max - x_min) / res)) + 1)
        H = max(2, int(round((y_max - y_min) / res)) + 1)
        xs = np.linspace(x_min, x_max, W, dtype=np.float32)
        ys = np.linspace(y_min, y_max, H, dtype=np.float32)
        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")  # (H,W)
        pts = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)  # (H*W,2)
        sdf = np.asarray([float(sdf_fn(p)) for p in pts], dtype=np.float32).reshape(H, W)

        # Finite difference gradients on the grid
        gx = np.zeros_like(sdf, dtype=np.float32)
        gy = np.zeros_like(sdf, dtype=np.float32)
        # central
        gx[:, 1:-1] = (sdf[:, 2:] - sdf[:, :-2]) / (2 * res)
        gy[1:-1, :] = (sdf[2:, :] - sdf[:-2, :]) / (2 * res)
        # forward/backward edges
        gx[:, 0] = (sdf[:, 1] - sdf[:, 0]) / res
        gx[:, -1] = (sdf[:, -1] - sdf[:, -2]) / res
        gy[0, :] = (sdf[1, :] - sdf[0, :]) / res
        gy[-1, :] = (sdf[-1, :] - sdf[-2, :]) / res

        tex = np.stack([sdf, gx, gy], axis=0).astype(np.float32)  # (3,H,W)
        return SDFTexture2D(tex_np=tex, x_min=float(x_min), y_min=float(y_min), res=res)

    def sample(
        self,
        points: Union[np.ndarray, "torch.Tensor", "jax.Array"],
        *,
        backend: Backend = "numpy",
        device: Optional[Union[str, "torch.device"]] = None,
    ):
        """
        Sample sdf and gradient at 2D points.

        Args:
            points: shape (...,2)
            backend: numpy/torch/jax
        Returns:
            sdf: shape (...)
            grad: shape (...,2)
        """
        if backend == "torch":
            return self._sample_torch(points, device=device)
        if backend == "jax":
            return self._sample_jax(points)
        return self._sample_numpy(points)

    def _sample_numpy(self, points: np.ndarray):
        pts = np.asarray(points, dtype=np.float32)
        shp = pts.shape[:-1]
        pts_f = pts.reshape(-1, 2)

        # grid coords
        ix = (pts_f[:, 0] - self.x_min) / self.res
        iy = (pts_f[:, 1] - self.y_min) / self.res
        ix = np.clip(ix, 0.0, self.W - 1.0)
        iy = np.clip(iy, 0.0, self.H - 1.0)
        x0 = np.floor(ix).astype(np.int32)
        y0 = np.floor(iy).astype(np.int32)
        x1 = np.minimum(x0 + 1, self.W - 1)
        y1 = np.minimum(y0 + 1, self.H - 1)
        wx = (ix - x0).astype(np.float32)
        wy = (iy - y0).astype(np.float32)

        tex = self.tex_np  # (3,H,W)
        v00 = tex[:, y0, x0]
        v10 = tex[:, y0, x1]
        v01 = tex[:, y1, x0]
        v11 = tex[:, y1, x1]
        v0 = v00 * (1.0 - wx) + v10 * wx
        v1 = v01 * (1.0 - wx) + v11 * wx
        v = v0 * (1.0 - wy) + v1 * wy  # (3, N)
        sdf = v[0].reshape(shp)
        grad = np.stack([v[1], v[2]], axis=-1).reshape(shp + (2,))
        return sdf, grad

    def _sample_torch(self, points: "torch.Tensor", device=None):
        if torch is None or F is None:
            raise RuntimeError("Torch is not available; cannot sample SDFTexture2D with torch.")
        tex = self.to_torch(device=device)  # (1,3,H,W)
        pts = points
        if not isinstance(pts, torch.Tensor):
            pts = torch.as_tensor(pts, dtype=torch.float32, device=tex.device)
        else:
            pts = pts.to(device=tex.device, dtype=torch.float32)
        shp = pts.shape[:-1]
        pts_f = pts.reshape(-1, 2)

        # Normalize to [-1, 1] in grid_sample coordinates (align_corners=True)
        x = pts_f[:, 0]
        y = pts_f[:, 1]
        xn = (x - self.x_min) / (self.x_max - self.x_min + 1e-12) * 2.0 - 1.0
        yn = (y - self.y_min) / (self.y_max - self.y_min + 1e-12) * 2.0 - 1.0

        # grid_sample expects (N, H_out, W_out, 2)
        grid = torch.stack([xn, yn], dim=-1).view(1, -1, 1, 2)
        out = F.grid_sample(
            tex, grid, mode="bilinear", padding_mode="border", align_corners=True
        )  # (1,3,N,1)
        out = out.squeeze(0).squeeze(-1).transpose(0, 1)  # (N,3)
        sdf = out[:, 0].view(shp)
        grad = out[:, 1:3].view(shp + (2,))
        return sdf, grad

    def _sample_jax(self, points: "jax.Array"):
        if jnp is None:
            raise RuntimeError("JAX is not available; cannot sample SDFTexture2D with jax.")
        tex = self.to_jax()  # (3,H,W)
        pts = points
        pts = jnp.asarray(pts, dtype=jnp.float32)
        shp = pts.shape[:-1]
        pts_f = pts.reshape((-1, 2))

        ix = (pts_f[:, 0] - self.x_min) / self.res
        iy = (pts_f[:, 1] - self.y_min) / self.res
        ix = jnp.clip(ix, 0.0, self.W - 1.0)
        iy = jnp.clip(iy, 0.0, self.H - 1.0)

        x0 = jnp.floor(ix).astype(jnp.int32)
        y0 = jnp.floor(iy).astype(jnp.int32)
        x1 = jnp.minimum(x0 + 1, self.W - 1)
        y1 = jnp.minimum(y0 + 1, self.H - 1)
        wx = (ix - x0).astype(jnp.float32)
        wy = (iy - y0).astype(jnp.float32)

        v00 = tex[:, y0, x0]
        v10 = tex[:, y0, x1]
        v01 = tex[:, y1, x0]
        v11 = tex[:, y1, x1]
        v0 = v00 * (1.0 - wx) + v10 * wx
        v1 = v01 * (1.0 - wx) + v11 * wx
        v = v0 * (1.0 - wy) + v1 * wy  # (3,N)
        sdf = v[0].reshape(shp)
        grad = jnp.stack([v[1], v[2]], axis=-1).reshape(shp + (2,))
        return sdf, grad

