"""
Gaussian splatting scene representation for MBD3D.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

from ..types import SceneParams


class GaussianSplatScene:
    """
    Simple 3D Gaussian splatting scene representation.

    Prior: independent Gaussian on each component (means, scales, etc.)
    with configurable scale. When prior_bbox is set and no prior_center,
    means are sampled uniformly in bbox for stronger training-free init.
    """

    def __init__(
        self,
        n_gaussians: int,
        prior_scale_means: float = 1.0,
        prior_scale_scales: float = 0.1,
        prior_scale_quats: float = 1.0,
        prior_scale_opacities: float = 1.0,
        prior_scale_colors: float = 1.0,
        use_spherical_harmonics: bool = False,
        sh_degree: int = 0,
        prior_bbox: Optional[Any] = None,
    ):
        self.n_gaussians = n_gaussians
        self.prior_scale_means = prior_scale_means
        self.prior_scale_scales = prior_scale_scales
        self.prior_scale_quats = prior_scale_quats
        self.prior_scale_opacities = prior_scale_opacities
        self.prior_scale_colors = prior_scale_colors
        self.use_spherical_harmonics = use_spherical_harmonics
        self.sh_degree = sh_degree
        self._sh_dim = (sh_degree + 1) ** 2 if use_spherical_harmonics else 0
        self._dim = (
            n_gaussians * (3 + 3 + 4 + 1 + 3)
            + (n_gaussians * self._sh_dim if self._sh_dim > 0 else 0)
        )
        self._prior_center: Optional[SceneParams] = None
        self._prior_bbox: Optional[np.ndarray] = None
        if prior_bbox is not None:
            b = np.asarray(prior_bbox, dtype=np.float32).ravel()
            if b.size >= 6:
                lo = np.array([b[0], b[2], b[4]], dtype=np.float32)
                hi = np.array([b[1], b[3], b[5]], dtype=np.float32)
                self._prior_bbox = np.stack([lo, hi], axis=0)

    def flatten(self, params: SceneParams) -> Any:
        return params.flatten()

    def unflatten(self, flat: Any) -> SceneParams:
        return SceneParams.from_flattened(
            flat,
            n_gaussians=self.n_gaussians,
            use_sh=self.use_spherical_harmonics,
            sh_dim=self._sh_dim,
        )

    def prior_log_prob(self, params: SceneParams) -> Any:
        """Log p0(θ) = sum of independent Gaussian log-probs."""
        use_jax = JAX_AVAILABLE and (
            hasattr(params.means, "block_until_ready") or getattr(params.means, "aval", None) is not None
        )

        center = self._prior_center

        def _gaussian_log_prob(x: Any, scale: float, c: Optional[Any] = None) -> Any:
            var = scale ** 2
            xc = x if c is None else (x - c)
            if use_jax:
                return -0.5 * jnp.sum(jnp.square(xc)) / var - 0.5 * jnp.size(x) * jnp.log(var)
            return float(-0.5 * np.sum(np.square(np.asarray(xc))) / var - 0.5 * np.size(x) * np.log(var))

        if center is None and self._prior_bbox is not None:
            lo, hi = self._prior_bbox[0], self._prior_bbox[1]
            n_pts = int(params.means.shape[0])
            # Soft box prior: finite everywhere to avoid bridge NaNs.
            if use_jax:
                lo_j = jnp.asarray(lo, dtype=jnp.float32)
                hi_j = jnp.asarray(hi, dtype=jnp.float32)
                box_size = jnp.maximum(hi_j - lo_j, 1e-6)
                box_center = 0.5 * (lo_j + hi_j)
                half = 0.5 * box_size
                vol = jnp.prod(box_size)
                delta = (params.means - box_center) / half
                outside = jnp.maximum(jnp.abs(delta) - 1.0, 0.0)
                soft_penalty = jnp.sum(jnp.square(outside))
                logp = -n_pts * jnp.log(vol) - 0.5 * 10.0 * soft_penalty
            else:
                lo_n = np.asarray(lo, dtype=np.float32)
                hi_n = np.asarray(hi, dtype=np.float32)
                box_size = np.maximum(hi_n - lo_n, 1e-6)
                box_center = 0.5 * (lo_n + hi_n)
                half = 0.5 * box_size
                vol = float(np.prod(box_size))
                delta = (np.asarray(params.means) - box_center) / half
                outside = np.maximum(np.abs(delta) - 1.0, 0.0)
                soft_penalty = float(np.sum(np.square(outside)))
                logp = float(-n_pts * np.log(vol) - 0.5 * 10.0 * soft_penalty)
        else:
            logp = _gaussian_log_prob(params.means, self.prior_scale_means, None if center is None else center.means)
        logp = logp + _gaussian_log_prob(params.scales, self.prior_scale_scales, None if center is None else center.scales)
        logp = logp + _gaussian_log_prob(params.quats, self.prior_scale_quats, None if center is None else center.quats)
        logp = logp + _gaussian_log_prob(
            params.opacities, self.prior_scale_opacities, None if center is None else center.opacities
        )
        logp = logp + _gaussian_log_prob(params.colors, self.prior_scale_colors, None if center is None else center.colors)
        if params.spherical_harmonics is not None:
            logp = logp + _gaussian_log_prob(
                params.spherical_harmonics, 1.0, None if center is None else center.spherical_harmonics
            )
        return logp if use_jax else float(logp)

    def sample_prior(
        self,
        rng: Any,
        n_gaussians: Optional[int] = None,
        center: Optional[Any] = None,
        center_params: Optional[SceneParams] = None,
        jitter_scale: float = 1.0,
        **kwargs: Any,
    ) -> SceneParams:
        n = n_gaussians or self.n_gaussians
        center = center if center is not None else 0.0
        cparams = center_params if center_params is not None else self._prior_center
        jitter = float(max(jitter_scale, 0.0))

        def _sample(shape: tuple, scale: float, key: Any, c: Optional[Any] = None) -> Any:
            c0 = c if c is not None else center
            s = scale * jitter
            if JAX_AVAILABLE and hasattr(key, "block_until_ready"):
                return c0 + s * jax.random.normal(key, shape)
            if hasattr(rng, "normal"):
                return np.asarray(c0) + s * rng.normal(size=shape)
            import numpy as np
            gen = np.random.default_rng(int(rng) if isinstance(rng, (int, np.integer)) else 0)
            return np.asarray(c0) + s * gen.normal(size=shape)

        def _sample_uniform_bbox(shape: tuple, key: Any) -> Any:
            lo, hi = self._prior_bbox[0], self._prior_bbox[1]
            if JAX_AVAILABLE and hasattr(key, "block_until_ready"):
                u = jax.random.uniform(key, shape)
                return lo + u * (hi - lo)
            if hasattr(rng, "random"):
                u = rng.random(size=shape)
            else:
                gen = np.random.default_rng(int(rng) if isinstance(rng, (int, np.integer)) else 0)
                u = gen.random(size=shape)
            return np.asarray(lo) + np.asarray(u) * (np.asarray(hi) - np.asarray(lo))

        use_bbox_means = cparams is None and self._prior_bbox is not None
        if JAX_AVAILABLE:
            keys = jax.random.split(rng, 6)
            if use_bbox_means:
                means = _sample_uniform_bbox((n, 3), keys[0])
            else:
                means = _sample((n, 3), self.prior_scale_means, keys[0], None if cparams is None else cparams.means)
            scales = _sample((n, 3), self.prior_scale_scales, keys[1], None if cparams is None else cparams.scales)
            quats = _sample((n, 4), self.prior_scale_quats, keys[2], None if cparams is None else cparams.quats)
            quats = quats / (jnp.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8)
            opacities = _sample(
                (n, 1), self.prior_scale_opacities, keys[3], None if cparams is None else cparams.opacities
            )
            colors = _sample((n, 3), self.prior_scale_colors, keys[4], None if cparams is None else cparams.colors)
            sh = _sample(
                (n, self._sh_dim), 1.0, keys[5], None if cparams is None else cparams.spherical_harmonics
            ) if self._sh_dim > 0 else None
        else:
            if use_bbox_means:
                means = _sample_uniform_bbox((n, 3), rng)
            else:
                means = _sample((n, 3), self.prior_scale_means, rng, None if cparams is None else cparams.means)
            scales = _sample((n, 3), self.prior_scale_scales, rng, None if cparams is None else cparams.scales)
            quats = _sample((n, 4), self.prior_scale_quats, rng, None if cparams is None else cparams.quats)
            quats = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8)
            opacities = _sample(
                (n, 1), self.prior_scale_opacities, rng, None if cparams is None else cparams.opacities
            )
            colors = _sample((n, 3), self.prior_scale_colors, rng, None if cparams is None else cparams.colors)
            sh = _sample(
                (n, self._sh_dim), 1.0, rng, None if cparams is None else cparams.spherical_harmonics
            ) if self._sh_dim > 0 else None

        return SceneParams(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            colors=colors,
            spherical_harmonics=sh,
        )

    def dim(self) -> int:
        return self._dim

    def set_prior_center(self, center: Optional[SceneParams]) -> None:
        self._prior_center = center

    def get_prior_center(self) -> Optional[SceneParams]:
        return self._prior_center
