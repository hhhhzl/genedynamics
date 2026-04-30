"""
Mode Marginalizer: theory-correct implementation.

Implements:
  log p(R|θ) = log Σ_c p(c) exp(R_c/T) = logsumexp(log p(c) + R_c/T)
  w_c = p(c|θ,R) = softmax(log p(c) + R_c/T)

Note: Annealing β_k is applied externally: log π_k(θ) = log p0(θ) + β_k * log p(R|θ).
      β must NOT appear inside the mixture terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

from .specs import ModeSystemConfig

Array = Union[np.ndarray, Any]


def _backend(x: Array) -> str:
    if JAX_AVAILABLE and hasattr(x, "block_until_ready"):
        return "jax"
    return "numpy"


def compute_log_terms(
    rewards: Array,
    log_priors: Array,
    temperature: float,
    *,
    backend: Optional[str] = None,
) -> Array:
    """
    Compute unnormalized log-terms for mode mixture.

    log a_c = log p(c) + R_c / T

    Args:
        rewards: (M, C) or (C,) rewards R_c per mode
        log_priors: (C,) log p(c)
        temperature: T > 0 for Boltzmann
        backend: "jax" | "numpy" | None

    Returns:
        log_terms: same shape as rewards
    """
    T = max(float(temperature), 1e-8)
    if backend is None:
        backend = _backend(rewards)

    if backend == "jax" and JAX_AVAILABLE:
        rewards = jnp.asarray(rewards, dtype=jnp.float32)
        log_priors = jnp.asarray(log_priors, dtype=jnp.float32)
        return log_priors[None, :] + rewards / T if rewards.ndim > 1 else log_priors + rewards / T

    rewards = np.asarray(rewards, dtype=np.float32)
    log_priors = np.asarray(log_priors, dtype=np.float32)
    if rewards.ndim > 1:
        return log_priors[None, :] + rewards / T
    return log_priors + rewards / T


def compute_marginal_log_likelihood(
    log_terms: Array,
    *,
    axis: int = -1,
    backend: Optional[str] = None,
) -> Array:
    """
    Compute log p(R|θ) = log Σ_c exp(log a_c) = logsumexp(log_terms).

    Args:
        log_terms: (..., C) unnormalized log p(c) p(R|θ,c)
        axis: axis of mixture components
        backend: "jax" | "numpy" | None

    Returns:
        log p(R|θ) per proposal, shape = log_terms.shape with axis reduced
    """
    if backend is None:
        backend = _backend(log_terms)

    if backend == "jax" and JAX_AVAILABLE:
        return jax.scipy.special.logsumexp(log_terms, axis=axis)

    log_terms = np.asarray(log_terms, dtype=np.float64)
    x_max = np.max(log_terms, axis=axis, keepdims=True)
    shifted = log_terms - x_max
    log_sum = np.log(np.sum(np.exp(shifted), axis=axis) + 1e-12)
    return (np.squeeze(x_max, axis=axis) + log_sum).astype(np.float32)


def compute_responsibilities(
    log_terms: Array,
    *,
    axis: int = -1,
    backend: Optional[str] = None,
) -> Array:
    """
    Compute w_c = p(c|θ,R) = softmax(log_terms).

    w_c = exp(log a_c) / Σ_c' exp(log a_c')

    Args:
        log_terms: (..., C) log p(c) + R_c/T
        axis: axis of mixture components
        backend: "jax" | "numpy" | None

    Returns:
        responsibilities: (..., C) normalized, sum to 1 along axis
    """
    if backend is None:
        backend = _backend(log_terms)

    if backend == "jax" and JAX_AVAILABLE:
        return jnp.exp(jax.nn.log_softmax(log_terms, axis=axis))

    log_terms = np.asarray(log_terms, dtype=np.float64)
    x_max = np.max(log_terms, axis=axis, keepdims=True)
    exp_shifted = np.exp(log_terms - x_max)
    weights = exp_shifted / (np.sum(exp_shifted, axis=axis, keepdims=True) + 1e-12)
    return weights.astype(np.float32)


@dataclass
class ModeMarginalizerResult:
    """Result from mode marginalization."""

    log_terms: Array
    marginal_log_likelihood: Array
    responsibilities: Array
    extra: Dict[str, Any] = field(default_factory=dict)


class ModeMarginalizer:
    """
    Mode marginalization: theory-correct implementation.

    Provides:
    - log p(R|θ) = logsumexp(log p(c) + R_c/T)
    - w_c = responsibilities
    - Extensible for custom temperature schedules
    """

    def __init__(
        self,
        config: ModeSystemConfig,
        *,
        backend: str = "jax",
    ):
        self.config = config
        self.backend = backend
        self._log_priors = config.get_log_priors()

    @property
    def num_modes(self) -> int:
        return self.config.num_modes

    def __call__(
        self,
        rewards: Array,
        *,
        temperature: Optional[float] = None,
    ) -> ModeMarginalizerResult:
        """
        Compute marginal log-likelihood and responsibilities.

        Args:
            rewards: (M, C) rewards R_c for M proposals, C modes
            temperature: Override config temperature

        Returns:
            ModeMarginalizerResult with log_terms, marginal_log_likelihood, responsibilities
        """
        T = temperature if temperature is not None else self.config.reward_temperature
        log_terms = compute_log_terms(
            rewards,
            self._log_priors,
            T,
            backend=self.backend,
        )
        marginal_log = compute_marginal_log_likelihood(log_terms, axis=-1, backend=self.backend)
        w_c = compute_responsibilities(log_terms, axis=-1, backend=self.backend)
        return ModeMarginalizerResult(
            log_terms=log_terms,
            marginal_log_likelihood=marginal_log,
            responsibilities=w_c,
        )

    def log_likelihood(
        self,
        rewards: Array,
        *,
        temperature: Optional[float] = None,
    ) -> Array:
        """Convenience: return only log p(R|θ)."""
        return self(rewards, temperature=temperature).marginal_log_likelihood

    def responsibilities(
        self,
        rewards: Array,
        *,
        temperature: Optional[float] = None,
    ) -> Array:
        """Convenience: return only w_c."""
        return self(rewards, temperature=temperature).responsibilities
