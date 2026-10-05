"""
Mixture model primitives for mode-marginalization and importance sampling.

Provides numerically stable operations for:
- Responsibility weights (soft assignment to mixture components)
- Mixture log-probability
- Weighted score combination (for gradient of mixture w.r.t. parameters)

Used by inference with discrete latent modes, including contact and friction
regimes.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

Array = Union[np.ndarray, Any]


def _backend(x: Array) -> str:
    if JAX_AVAILABLE and hasattr(x, "block_until_ready"):
        return "jax"
    return "numpy"


def stable_log_softmax(
    logits: Array,
    axis: int = -1,
    *,
    backend: Optional[str] = None,
) -> Array:
    """
    Numerically stable log-softmax: log_softmax(x) = x - log(sum(exp(x - max(x)))).

    Args:
        logits: unnormalized log-probabilities
        axis: axis over which to normalize
        backend: "jax" | "numpy" | None

    Returns:
        log-softmax output (same shape as logits)
    """
    if backend is None:
        backend = _backend(logits)

    if backend == "jax" and JAX_AVAILABLE:
        return jax.nn.log_softmax(logits, axis=axis)
    logits = np.asarray(logits, dtype=np.float64)
    x_max = np.max(logits, axis=axis, keepdims=True)
    shifted = logits - x_max
    exp_shifted = np.exp(shifted)
    log_sum_exp = np.log(np.sum(exp_shifted, axis=axis, keepdims=True) + 1e-12)
    return (shifted - log_sum_exp).astype(np.float32)


def responsibilities(
    log_terms: Array,
    *,
    axis: int = -1,
    backend: Optional[str] = None,
) -> Array:
    """
    Compute responsibility weights from log-terms (unnormalized log-probs).

    w_c = exp(log_a_c) / sum_c' exp(log_a_c')
    with numerical stability via log-sum-exp.

    Args:
        log_terms: (..., K) unnormalized log-probabilities per component
        axis: axis along which components are indexed
        backend: "jax" | "numpy" | None

    Returns:
        weights: (..., K) normalized responsibilities, sum to 1 along axis
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


def mixture_logprob(
    log_terms: Array,
    *,
    axis: int = -1,
    backend: Optional[str] = None,
) -> Union[float, Any]:
    """
    Compute log of mixture normalizing constant: log(sum_c exp(log_a_c)).

    Args:
        log_terms: (..., K) unnormalized log-probabilities
        axis: axis of mixture components
        backend: "jax" | "numpy" | None

    Returns:
        log Z = log(sum exp(log_terms))
    """
    if backend is None:
        backend = _backend(log_terms)

    if backend == "jax" and JAX_AVAILABLE:
        return jax.scipy.special.logsumexp(log_terms, axis=axis)
    log_terms = np.asarray(log_terms, dtype=np.float64)
    x_max = np.max(log_terms, axis=axis, keepdims=True)
    shifted = log_terms - x_max
    log_sum = np.log(np.sum(np.exp(shifted), axis=axis) + 1e-12)
    return (np.squeeze(x_max) + log_sum).astype(np.float32)


def weighted_score(
    scores: Array,
    weights: Array,
    *,
    axis: int = 0,
    backend: Optional[str] = None,
) -> Array:
    """
    Combine per-component score vectors by responsibility weighting.

    ∇ log p_mix = sum_c w_c ∇ log p_c

    Args:
        scores: (K, ...) or (..., K, ...) gradient/score per component
        weights: (K,) or (..., K) responsibility weights
        axis: axis indexing components
        backend: "jax" | "numpy" | None

    Returns:
        weighted score (shape = scores shape with axis reduced)
    """
    if backend is None:
        backend = _backend(scores)

    if backend == "jax" and JAX_AVAILABLE:
        weights = jnp.asarray(weights, dtype=jnp.float32)
        scores = jnp.asarray(scores, dtype=jnp.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)

    # Ensure weights broadcast correctly: (K,) -> (K, 1, 1, ...) along axis
    ndim = scores.ndim
    w_shape = [1] * ndim
    w_shape[axis] = int(weights.shape[0] if weights.ndim >= 1 else 1)
    if backend == "jax" and JAX_AVAILABLE:
        w = jnp.reshape(weights, w_shape)
        return jnp.sum(w * scores, axis=axis)
    w = np.reshape(weights, w_shape)
    return np.sum(w * scores, axis=axis).astype(np.float32)
