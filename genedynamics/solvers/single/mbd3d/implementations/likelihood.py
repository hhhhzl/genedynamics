"""
Gaussian observation likelihood for MBD3D.

Supports iid Gaussian noise and nuisance-marginalized correlated noise:
Σ = AA' + σ²I with a low-rank nuisance basis A.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

from genedynamics.core.prob.gaussian_lowrank import LowRankCovariance

from ..protocols import BaseObservationLikelihood
from ..types import ObservationBundle, SceneParams


class GaussianObservationLikelihood(BaseObservationLikelihood):
    """
    log p(y|θ) = -0.5 * r' Σ^{-1} r - 0.5 * log|Σ| + const.

    When use_lowrank=True:
      Σ = AA' + σ²I, where A encodes structured nuisance modes.
    """

    def __init__(
        self,
        renderer: Any,
        sigma2: float = 0.01,
        lowrank_basis: Optional[Any] = None,
        use_lowrank: bool = False,
        backend: str = "jax",
        lowrank_view_rank: int = 4,
        lowrank_spatial_rank: int = 2,
        lowrank_color_rank: int = 3,
        lowrank_max_rank: int = 16,
    ):
        self.renderer = renderer
        self.sigma2 = float(sigma2)
        self.lowrank_basis = lowrank_basis
        self.use_lowrank = bool(use_lowrank)
        self.backend = backend
        self.lowrank_view_rank = int(lowrank_view_rank)
        self.lowrank_spatial_rank = int(lowrank_spatial_rank)
        self.lowrank_color_rank = int(lowrank_color_rank)
        self.lowrank_max_rank = int(lowrank_max_rank)

        self._basis_key: Optional[Tuple[int, ...]] = None
        self._cov: Optional[LowRankCovariance] = None
        self._logdet_cache: Optional[float] = None

    def _is_jax_array(self, x: Any) -> bool:
        return JAX_AVAILABLE and hasattr(x, "block_until_ready")

    def residual(
        self,
        predicted: Any,
        observed: Any,
        **kwargs: Any,
    ) -> Any:
        if self._is_jax_array(predicted) or self._is_jax_array(observed):
            return jnp.ravel(jnp.asarray(observed) - jnp.asarray(predicted))
        return np.ravel(np.asarray(observed) - np.asarray(predicted))

    def _observation_shape_key(self, observations: ObservationBundle) -> Tuple[int, ...]:
        imgs = observations.images
        if hasattr(imgs, "shape"):
            return tuple(int(v) for v in imgs.shape)
        arr = np.asarray(imgs)
        return tuple(int(v) for v in arr.shape)

    def _normalize_mode(self, v: np.ndarray) -> Optional[np.ndarray]:
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-8:
            return None
        return (v / nrm).astype(np.float32)

    def _view_dct_modes(self, n_views: int, rank: int) -> List[np.ndarray]:
        if n_views <= 0 or rank <= 0:
            return []
        t = np.arange(n_views, dtype=np.float32) + 0.5
        modes: List[np.ndarray] = []
        # include low-frequency cosine modes; k=0 is global bias
        for k in range(min(rank, n_views)):
            m = np.cos(np.pi * k * t / float(n_views)).astype(np.float32)
            m = m - np.mean(m)
            nrm = np.linalg.norm(m)
            if nrm > 1e-8:
                modes.append((m / nrm).astype(np.float32))
        if not modes:
            modes.append(np.ones((n_views,), dtype=np.float32) / np.sqrt(float(n_views)))
        return modes

    def _spatial_modes(self, h: int, w: int, rank: int) -> List[np.ndarray]:
        if h <= 0 or w <= 0 or rank <= 0:
            return []
        ys = np.arange(h, dtype=np.float32) + 0.5
        xs = np.arange(w, dtype=np.float32) + 0.5
        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
        modes: List[np.ndarray] = []
        # deterministic low-frequency 2D cosine basis
        candidates: Sequence[Tuple[int, int]] = (
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (2, 0),
            (0, 2),
            (2, 1),
            (1, 2),
        )
        for ky, kx in candidates[: max(rank, 1)]:
            m = np.cos(np.pi * ky * grid_y / float(h)) * np.cos(np.pi * kx * grid_x / float(w))
            m = m.astype(np.float32)
            m = m - np.mean(m)
            nrm = np.linalg.norm(m)
            if nrm > 1e-8:
                modes.append((m / nrm).astype(np.float32))
            if len(modes) >= rank:
                break
        if not modes:
            modes.append(np.ones((h, w), dtype=np.float32) / np.sqrt(float(h * w)))
        return modes

    def _build_nuisance_basis(self, n_views: int, h: int, w: int, c: int) -> np.ndarray:
        view_modes = self._view_dct_modes(n_views, self.lowrank_view_rank)
        spatial_modes = self._spatial_modes(h, w, self.lowrank_spatial_rank)
        color_rank = max(1, min(c, self.lowrank_color_rank))

        ones_view = np.ones((n_views,), dtype=np.float32) / np.sqrt(max(1.0, float(n_views)))
        ones_spatial = np.ones((h, w), dtype=np.float32) / np.sqrt(max(1.0, float(h * w)))
        ones_color = np.ones((c,), dtype=np.float32) / np.sqrt(max(1.0, float(c)))

        cols: List[np.ndarray] = []

        # View-specific drift modes: view x global spatial x global color
        for vm in view_modes:
            tensor = vm[:, None, None, None] * ones_spatial[None, :, :, None] * ones_color[None, None, None, :]
            mode = self._normalize_mode(tensor.reshape(-1))
            if mode is not None:
                cols.append(mode)

        # Color/exposure modes: global view x global spatial x channel basis
        for ch in range(color_rank):
            color = np.zeros((c,), dtype=np.float32)
            color[ch] = 1.0
            color = color / (np.linalg.norm(color) + 1e-8)
            tensor = ones_view[:, None, None, None] * ones_spatial[None, :, :, None] * color[None, None, None, :]
            mode = self._normalize_mode(tensor.reshape(-1))
            if mode is not None:
                cols.append(mode)

        # Spatial structured modes: global view x spatial x global color
        for sm in spatial_modes:
            tensor = ones_view[:, None, None, None] * sm[None, :, :, None] * ones_color[None, None, None, :]
            mode = self._normalize_mode(tensor.reshape(-1))
            if mode is not None:
                cols.append(mode)

        if not cols:
            raise ValueError("Failed to build nuisance low-rank basis.")

        A = np.stack(cols, axis=1).astype(np.float32)
        if A.shape[1] > self.lowrank_max_rank:
            A = A[:, : self.lowrank_max_rank]

        # Re-orthonormalize columns for stable Woodbury computations.
        q, _ = np.linalg.qr(A, mode="reduced")
        return q.astype(np.float32)

    def _set_basis_and_cache(self, basis: Any, p: int, key: Tuple[int, ...]) -> None:
        if self.backend == "jax" and JAX_AVAILABLE:
            basis_arr = jnp.asarray(basis, dtype=jnp.float32)
        else:
            basis_arr = np.asarray(basis, dtype=np.float32)
        self.lowrank_basis = basis_arr
        self._cov = LowRankCovariance(A=basis_arr, sigma2=self.sigma2, p=p)
        self._logdet_cache = float(self._cov.logdet())
        self._basis_key = key

    def ensure_lowrank_basis(self, observations: ObservationBundle) -> None:
        if not self.use_lowrank:
            return
        key = self._observation_shape_key(observations)
        if self._basis_key == key and self._cov is not None:
            return

        obs_arr = np.asarray(observations.images)
        if obs_arr.ndim != 4:
            raise ValueError(f"Expected observations.images to have shape (N,H,W,C), got {obs_arr.shape}")
        n_views, h, w, c = [int(v) for v in obs_arr.shape]
        p = n_views * h * w * c

        if self.lowrank_basis is None:
            basis = self._build_nuisance_basis(n_views=n_views, h=h, w=w, c=c)
        else:
            basis = np.asarray(self.lowrank_basis, dtype=np.float32)
            if basis.shape[0] != p:
                raise ValueError(
                    f"Provided lowrank_basis has incompatible first dimension: {basis.shape[0]} != {p}"
                )
        self._set_basis_and_cache(basis=basis, p=p, key=key)

    def sigma_inv_apply(self, v: Any, observations: ObservationBundle) -> Any:
        if not self.use_lowrank:
            return jnp.asarray(v) / self.sigma2 if self._is_jax_array(v) else np.asarray(v) / self.sigma2
        self.ensure_lowrank_basis(observations)
        if self._cov is None:
            raise RuntimeError("Low-rank covariance cache missing after ensure_lowrank_basis.")
        return self._cov.solve(v)

    def log_likelihood(
        self,
        params: SceneParams,
        observations: ObservationBundle,
        camera_trajectory: Any,
        **kwargs: Any,
    ) -> Any:
        pred = self.renderer.render(params, camera_trajectory, observations.intrinsics, **kwargs)
        pred_is_jax = self._is_jax_array(pred)
        pred_flat = jnp.ravel(jnp.asarray(pred)) if pred_is_jax else np.ravel(np.asarray(pred))
        obs_flat = jnp.ravel(jnp.asarray(observations.images)) if pred_is_jax else np.ravel(np.asarray(observations.images))
        r = self.residual(pred_flat, obs_flat)

        if self.use_lowrank:
            self.ensure_lowrank_basis(observations)
            if self._cov is None:
                raise RuntimeError("Low-rank covariance cache is not initialized.")
            qf = self._cov.quad_form(r)
            ld = float(self._logdet_cache if self._logdet_cache is not None else self._cov.logdet())
            return -0.5 * qf - 0.5 * ld

        if pred_is_jax:
            qf = jnp.sum(jnp.square(r)) / self.sigma2
            ld = r.size * jnp.log(self.sigma2)
            return -0.5 * qf - 0.5 * ld
        qf = float(np.sum(np.square(np.asarray(r))) / self.sigma2)
        ld = float(r.size * np.log(self.sigma2))
        return -0.5 * qf - 0.5 * ld
