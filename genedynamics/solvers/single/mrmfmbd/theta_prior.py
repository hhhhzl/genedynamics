"""
Prior and parameterization for theta = (x, phi) co-design.

Provides:
- ThetaPrior: log p0(theta) for morphology x + controller phi
- ThetaParametrization: pack/unpack theta <-> (x, phi)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

Array = Any


@dataclass
class ThetaParametrization:
    """
    Pack/unpack theta = (x, phi) for co-design.

    Attributes:
        x_dim: Morphology parameter dimension
        phi_dim: Controller parameter dimension
        x_bounds: Optional (low, high) for clipping x
        phi_bounds: Optional (low, high) for clipping phi
    """

    x_dim: int
    phi_dim: int
    x_bounds: Optional[Tuple[float, float]] = None
    phi_bounds: Optional[Tuple[float, float]] = None
    extra: dict = field(default_factory=dict)

    @property
    def theta_dim(self) -> int:
        return self.x_dim + self.phi_dim

    def pack(self, x: Array, phi: Array) -> Array:
        """Concatenate x and phi into theta."""
        if JAX_AVAILABLE and (hasattr(x, "block_until_ready") or hasattr(phi, "block_until_ready")):
            return jnp.concatenate([jnp.ravel(x), jnp.ravel(phi)])
        return np.concatenate([np.ravel(x), np.ravel(phi)])

    def unpack(self, theta: Array) -> Tuple[Array, Array]:
        """Split theta into (x, phi)."""
        if JAX_AVAILABLE and hasattr(theta, "block_until_ready"):
            x = theta[: self.x_dim]
            phi = theta[self.x_dim :]
            return x, phi
        theta = np.asarray(theta)
        x = theta[: self.x_dim]
        phi = theta[self.x_dim :]
        return x, phi

    def clip(self, theta: Array) -> Array:
        """Clip theta to bounds if specified. Supports (D,) or (M, D)."""
        if self.x_bounds is None and self.phi_bounds is None:
            return theta
        if JAX_AVAILABLE and hasattr(theta, "block_until_ready"):
            out = theta
            if self.x_bounds is not None:
                lo, hi = self.x_bounds
                x_part = theta[..., : self.x_dim]
                out = out.at[..., : self.x_dim].set(jnp.clip(x_part, lo, hi))
            if self.phi_bounds is not None:
                lo, hi = self.phi_bounds
                phi_part = theta[..., self.x_dim :]
                out = out.at[..., self.x_dim :].set(jnp.clip(phi_part, lo, hi))
            return out
        theta = np.asarray(theta, dtype=np.float32)
        if self.x_bounds is not None:
            lo, hi = self.x_bounds
            theta[..., : self.x_dim] = np.clip(theta[..., : self.x_dim], lo, hi)
        if self.phi_bounds is not None:
            lo, hi = self.phi_bounds
            theta[..., self.x_dim :] = np.clip(theta[..., self.x_dim :], lo, hi)
        return theta


@dataclass
class ThetaPriorConfig:
    """
    Configuration for Gaussian prior on theta.

    Attributes:
        x_mean: Prior mean for x (or None for zero)
        phi_mean: Prior mean for phi (or None for zero)
        x_std: Prior std for x
        phi_std: Prior std for phi
    """

    x_mean: Optional[Array] = None
    phi_mean: Optional[Array] = None
    x_std: float = 1.0
    phi_std: float = 1.0


class ThetaPrior:
    """
    Gaussian prior log p0(theta) = log N(theta; mu, diag(sigma^2)).

    Supports both JAX and NumPy backends.
    """

    def __init__(
        self,
        param: ThetaParametrization,
        config: Optional[ThetaPriorConfig] = None,
    ):
        self.param = param
        self.config = config or ThetaPriorConfig()
        self._build()

    def _build(self) -> None:
        x_dim, phi_dim = self.param.x_dim, self.param.phi_dim
        x_mean = self.config.x_mean
        phi_mean = self.config.phi_mean
        x_std = self.config.x_std
        phi_std = self.config.phi_std

        if x_mean is None:
            x_mean = np.zeros(x_dim, dtype=np.float32)
        if phi_mean is None:
            phi_mean = np.zeros(phi_dim, dtype=np.float32)
        self._mean = np.concatenate([np.ravel(x_mean), np.ravel(phi_mean)])
        self._log_std = np.concatenate(
            [np.full(x_dim, np.log(max(x_std, 1e-8))), np.full(phi_dim, np.log(max(phi_std, 1e-8)))]
        )
        self._mean = self._mean.astype(np.float32)
        self._log_std = self._log_std.astype(np.float32)

    def log_prob(self, theta: Array) -> float:
        """Compute log p0(theta)."""
        if JAX_AVAILABLE and hasattr(theta, "block_until_ready"):
            mean = jnp.asarray(self._mean)
            log_std = jnp.asarray(self._log_std)
            diff = theta - mean
            return -0.5 * jnp.sum(jnp.square(diff / jnp.exp(log_std)) + 2 * log_std + np.log(2 * np.pi))
        theta = np.asarray(theta, dtype=np.float32)
        diff = theta - self._mean
        std = np.exp(self._log_std)
        return float(-0.5 * np.sum(np.square(diff / std) + 2 * self._log_std + np.log(2 * np.pi)))

    def log_prob_batch(self, thetas: Array) -> Array:
        """Compute log p0(theta) for batch of thetas (M, D)."""
        if JAX_AVAILABLE and hasattr(thetas, "block_until_ready"):
            mean = jnp.asarray(self._mean)
            log_std = jnp.asarray(self._log_std)
            diff = thetas - mean
            return -0.5 * jnp.sum(
                jnp.square(diff / jnp.exp(log_std)) + 2 * log_std + np.log(2 * np.pi),
                axis=-1,
            )
        return np.array(
            [self.log_prob(thetas[i]) for i in range(thetas.shape[0])],
            dtype=np.float32,
        )

    def sample(self, rng: Any, shape: Tuple[int, ...] = ()) -> Array:
        """Sample from prior."""
        if JAX_AVAILABLE and jax is not None:
            mean = jnp.asarray(self._mean)
            std = jnp.exp(jnp.asarray(self._log_std))
            if shape:
                return mean + std * jax.random.normal(rng, shape + (len(mean),))
            return mean + std * jax.random.normal(rng, (len(mean),))
        rng = np.random.default_rng(rng) if isinstance(rng, (int, type(None))) else rng
        std = np.exp(self._log_std)
        if shape:
            return self._mean + std * rng.standard_normal(shape + (len(self._mean),))
        return self._mean + std * rng.standard_normal(len(self._mean))
