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
    with configurable scale.
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
        use_jax = JAX_AVAILABLE and hasattr(params.means, "block_until_ready")

        def _gaussian_log_prob(x: Any, scale: float) -> Any:
            var = scale ** 2
            if use_jax:
                return -0.5 * jnp.sum(jnp.square(x)) / var - 0.5 * jnp.size(x) * jnp.log(var)
            return float(-0.5 * np.sum(np.square(np.asarray(x))) / var - 0.5 * np.size(x) * np.log(var))

        logp = _gaussian_log_prob(params.means, self.prior_scale_means)
        logp = logp + _gaussian_log_prob(params.scales, self.prior_scale_scales)
        logp = logp + _gaussian_log_prob(params.quats, self.prior_scale_quats)
        logp = logp + _gaussian_log_prob(params.opacities, self.prior_scale_opacities)
        logp = logp + _gaussian_log_prob(params.colors, self.prior_scale_colors)
        if params.spherical_harmonics is not None:
            logp = logp + _gaussian_log_prob(params.spherical_harmonics, 1.0)
        return logp if use_jax else float(logp)

    def sample_prior(
        self,
        rng: Any,
        n_gaussians: Optional[int] = None,
        center: Optional[Any] = None,
        **kwargs: Any,
    ) -> SceneParams:
        n = n_gaussians or self.n_gaussians
        center = center if center is not None else 0.0

        def _sample(shape: tuple, scale: float, key: Any) -> Any:
            if JAX_AVAILABLE and hasattr(key, "block_until_ready"):
                return center + scale * jax.random.normal(key, shape)
            if hasattr(rng, "normal"):
                return np.asarray(center) + scale * rng.normal(size=shape)
            import numpy as np
            gen = np.random.default_rng(int(rng) if isinstance(rng, (int, np.integer)) else 0)
            return np.asarray(center) + scale * gen.normal(size=shape)

        if JAX_AVAILABLE:
            keys = jax.random.split(rng, 6)
            means = _sample((n, 3), self.prior_scale_means, keys[0])
            scales = _sample((n, 3), self.prior_scale_scales, keys[1])
            quats = _sample((n, 4), self.prior_scale_quats, keys[2])
            quats = quats / (jnp.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8)
            opacities = _sample((n, 1), self.prior_scale_opacities, keys[3])
            colors = _sample((n, 3), self.prior_scale_colors, keys[4])
            sh = _sample((n, self._sh_dim), 1.0, keys[5]) if self._sh_dim > 0 else None
        else:
            means = _sample((n, 3), self.prior_scale_means, rng)
            scales = _sample((n, 3), self.prior_scale_scales, rng)
            quats = _sample((n, 4), self.prior_scale_quats, rng)
            quats = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8)
            opacities = _sample((n, 1), self.prior_scale_opacities, rng)
            colors = _sample((n, 3), self.prior_scale_colors, rng)
            sh = _sample((n, self._sh_dim), 1.0, rng) if self._sh_dim > 0 else None

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
