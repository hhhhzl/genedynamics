"""
Gaussian observation likelihood for MBD3D.

Supports iid Gaussian and low-rank correlated noise (pose/exposure drift).
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

from genedynamics.core.prob.gaussian_lowrank import quad_form_lowrank, lowrank_logdet, LowRankCovariance

from ..types import ObservationBundle, SceneParams
from ..protocols import BaseObservationLikelihood


class GaussianObservationLikelihood(BaseObservationLikelihood):
    """
    Log p(y|θ) with optional low-rank noise Σ = AA' + σ²I.

    Uses core.prob.gaussian_lowrank for efficient O(pr² + r³) evaluation.
    """

    def __init__(
        self,
        renderer: Any,
        sigma2: float = 0.01,
        lowrank_basis: Optional[Any] = None,
        use_lowrank: bool = False,
        backend: str = "jax",
    ):
        self.renderer = renderer
        self.sigma2 = sigma2
        self.lowrank_basis = lowrank_basis
        self.use_lowrank = use_lowrank
        self.backend = backend

    def residual(
        self,
        predicted: Any,
        observed: Any,
        **kwargs: Any,
    ) -> Any:
        if JAX_AVAILABLE and (hasattr(predicted, "block_until_ready") or hasattr(observed, "block_until_ready")):
            return jnp.ravel(jnp.asarray(observed) - jnp.asarray(predicted))
        return np.ravel(np.asarray(observed) - np.asarray(predicted))

    def log_likelihood(
        self,
        params: SceneParams,
        observations: ObservationBundle,
        camera_trajectory: Any,
        **kwargs: Any,
    ) -> Any:
        """
        log p(y|θ) = -0.5 * r' Σ^{-1} r - 0.5 * log|Σ| + const

        With Σ = AA' + σ²I (low-rank) or σ²I (iid).
        """
        pred = self.renderer.render(params, camera_trajectory, observations.intrinsics, **kwargs)
        pred_flat = jnp.ravel(jnp.asarray(pred)) if (JAX_AVAILABLE and hasattr(pred, "block_until_ready")) else np.ravel(np.asarray(pred))
        obs_flat = jnp.ravel(jnp.asarray(observations.images)) if (JAX_AVAILABLE and hasattr(pred, "block_until_ready")) else np.ravel(np.asarray(observations.images))
        r = self.residual(pred_flat, obs_flat)

        if self.use_lowrank and self.lowrank_basis is not None:
            A = self.lowrank_basis
            qf = quad_form_lowrank(A, self.sigma2, r, backend=self.backend)
            p = r.size
            ld = lowrank_logdet(A, self.sigma2, p, backend=self.backend)
        else:
            if JAX_AVAILABLE and hasattr(r, "block_until_ready"):
                qf = jnp.sum(jnp.square(r)) / self.sigma2
                ld = r.size * jnp.log(self.sigma2)
            else:
                qf = float(np.sum(np.square(np.asarray(r))) / self.sigma2)
                ld = float(r.size * np.log(self.sigma2))

        return -0.5 * qf - 0.5 * ld
