"""
Low-rank correlated Gaussian covariance primitives.

Implements efficient O(pr² + r³) operations for Σ = AA' + σ²I where A ∈ ℝ^{p×r}
and r ≪ p, using the Woodbury matrix identity and matrix determinant lemma.

Used by likelihoods with structured low-rank nuisance variables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Union

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

Array = Union[np.ndarray, Any]  # numpy or JAX array


def _to_numpy(x: Array) -> np.ndarray:
    """Convert to numpy if JAX array."""
    if JAX_AVAILABLE and hasattr(x, "block_until_ready"):
        return np.asarray(x)
    return np.asarray(x)


def _backend(x: Array) -> str:
    """Infer backend from array."""
    if JAX_AVAILABLE and hasattr(x, "block_until_ready"):
        return "jax"
    return "numpy"


# -----------------------------------------------------------------------------
# Pure functions (backend-agnostic via duck typing)
# -----------------------------------------------------------------------------


def woodbury_solve(
    A: Array,
    sigma2: float,
    v: Array,
    *,
    backend: Optional[str] = None,
) -> Array:
    """
    Solve (AA' + σ²I) x = v using Woodbury identity.

    (AA' + σ²I)^{-1} = (1/σ²)I - (1/σ²)A (I_r + (1/σ²)A'A)^{-1} A' (1/σ²)

    Complexity: O(pr² + r³) instead of O(p³).

    Args:
        A: (p, r) low-rank basis matrix
        sigma2: scalar σ² > 0 (iid noise variance)
        v: (p,) or (p, b) right-hand side
        backend: "jax" | "numpy" | None (auto-detect from A)

    Returns:
        x: solution of same shape as v
    """
    if backend is None:
        backend = _backend(A)

    if backend == "jax" and JAX_AVAILABLE:
        return _woodbury_solve_jax(A, sigma2, v)
    return _woodbury_solve_numpy(A, sigma2, v)


def _woodbury_solve_numpy(A: np.ndarray, sigma2: float, v: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p, r = A.shape
    inv_sigma2 = 1.0 / sigma2

    # (I_r + (1/σ²)A'A)^{-1}
    AtA = A.T @ A
    M = np.eye(r, dtype=np.float64) + inv_sigma2 * AtA
    Minv = np.linalg.inv(M)

    # x = (1/σ²)v - (1/σ²)² A Minv A' v
    Atv = A.T @ v
    x = inv_sigma2 * v - (inv_sigma2 ** 2) * (A @ (Minv @ Atv))
    return x.astype(np.float32)


def _woodbury_solve_jax(A: Any, sigma2: float, v: Any) -> Any:
    A = jnp.asarray(A, dtype=jnp.float32)
    v = jnp.asarray(v, dtype=jnp.float32)
    p, r = A.shape
    inv_sigma2 = 1.0 / sigma2

    AtA = jnp.dot(A.T, A)
    M = jnp.eye(r, dtype=jnp.float32) + inv_sigma2 * AtA
    Minv = jnp.linalg.inv(M)

    Atv = jnp.dot(A.T, v)
    x = inv_sigma2 * v - (inv_sigma2 ** 2) * jnp.dot(A, jnp.dot(Minv, Atv))
    return jnp.asarray(x, dtype=jnp.float32)


def lowrank_logdet(
    A: Array,
    sigma2: float,
    p: int,
    *,
    backend: Optional[str] = None,
) -> float:
    """
    Compute log|Σ| = log|AA' + σ²I| using matrix determinant lemma.

    log|Σ| = p·log(σ²) + log|I_r + (1/σ²)A'A|

    Args:
        A: (p, r) low-rank basis
        sigma2: σ² > 0
        p: dimension of observation space (for iid term)
        backend: "jax" | "numpy" | None

    Returns:
        log|Σ| as Python float
    """
    if backend is None:
        backend = _backend(A)

    if backend == "jax" and JAX_AVAILABLE:
        ld = _lowrank_logdet_jax(A, sigma2, p)
        return float(np.asarray(ld))
    return _lowrank_logdet_numpy(A, sigma2, p)


def _lowrank_logdet_numpy(A: np.ndarray, sigma2: float, p: int) -> float:
    A = np.asarray(A, dtype=np.float64)
    r = A.shape[1]
    inv_sigma2 = 1.0 / sigma2
    AtA = A.T @ A
    M = np.eye(r, dtype=np.float64) + inv_sigma2 * AtA
    sign, logdet_M = np.linalg.slogdet(M)
    return p * np.log(sigma2) + float(logdet_M)


def _lowrank_logdet_jax(A: Any, sigma2: float, p: int) -> Any:
    A = jnp.asarray(A, dtype=jnp.float32)
    r = A.shape[1]
    inv_sigma2 = 1.0 / sigma2
    AtA = jnp.dot(A.T, A)
    M = jnp.eye(r, dtype=jnp.float32) + inv_sigma2 * AtA
    logdet_M = jnp.linalg.slogdet(M)[1]
    return p * jnp.log(sigma2) + logdet_M


def quad_form_lowrank(
    A: Array,
    sigma2: float,
    r: Array,
    *,
    backend: Optional[str] = None,
) -> Union[float, Any]:
    """
    Compute r' Σ^{-1} r for Σ = AA' + σ²I.

    Uses Woodbury: r' Σ^{-1} r = (1/σ²)||r||² - (1/σ²)² r'A (I + (1/σ²)A'A)^{-1} A'r

    Args:
        A: (p, r) low-rank basis
        sigma2: σ² > 0
        r: (p,) residual vector
        backend: "jax" | "numpy" | None

    Returns:
        scalar r' Σ^{-1} r
    """
    if backend is None:
        backend = _backend(A)

    if backend == "jax" and JAX_AVAILABLE:
        return _quad_form_lowrank_jax(A, sigma2, r)
    return _quad_form_lowrank_numpy(A, sigma2, r)


def _quad_form_lowrank_numpy(A: np.ndarray, sigma2: float, r: np.ndarray) -> float:
    x = _woodbury_solve_numpy(A, sigma2, r)
    return float(np.dot(r.ravel(), x.ravel()))


def _quad_form_lowrank_jax(A: Any, sigma2: float, r: Any) -> Any:
    x = _woodbury_solve_jax(A, sigma2, r)
    return jnp.dot(r.ravel(), x.ravel())


# -----------------------------------------------------------------------------
# Stateful helper (caches factorization for repeated solves)
# -----------------------------------------------------------------------------


@dataclass
class LowRankCovariance:
    """
    Cached low-rank covariance Σ = AA' + σ²I for repeated operations.

    Precomputes (I_r + (1/σ²)A'A)^{-1} so that solve/quad_form/logdet
    are O(pr) per call after O(r³) setup.

    Attributes:
        A: (p, r) low-rank basis
        sigma2: iid noise variance σ²
        p: observation dimension (for logdet)
        _Minv: cached (I + (1/σ²)A'A)^{-1} (internal)
    """

    A: Array
    sigma2: float
    p: int
    _Minv: Optional[Array] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.sigma2 <= 0:
            raise ValueError(f"sigma2 must be positive, got {self.sigma2}")
        self._ensure_Minv()

    def _ensure_Minv(self) -> None:
        if self._Minv is not None:
            return
        A = np.asarray(self.A, dtype=np.float32) if not JAX_AVAILABLE else jnp.asarray(self.A, dtype=jnp.float32)
        r = A.shape[1]
        inv_sigma2 = 1.0 / self.sigma2
        if JAX_AVAILABLE and hasattr(A, "block_until_ready"):
            AtA = jnp.dot(A.T, A)
            M = jnp.eye(r, dtype=jnp.float32) + inv_sigma2 * AtA
            self._Minv = jnp.linalg.inv(M)
        else:
            A = np.asarray(A)
            AtA = A.T @ A
            M = np.eye(r, dtype=np.float64) + inv_sigma2 * AtA
            self._Minv = np.linalg.inv(M)

    def solve(self, v: Array) -> Array:
        """Solve Σ x = v."""
        A = self.A
        sigma2 = self.sigma2
        Minv = self._Minv
        inv_sigma2 = 1.0 / sigma2

        if JAX_AVAILABLE and hasattr(A, "block_until_ready"):
            v = jnp.asarray(v, dtype=jnp.float32)
            Atv = jnp.dot(A.T, v)
            x = inv_sigma2 * v - (inv_sigma2 ** 2) * jnp.dot(A, jnp.dot(Minv, Atv))
            return jnp.asarray(x, dtype=jnp.float32)
        v = np.asarray(v, dtype=np.float32)
        Atv = A.T @ v
        x = inv_sigma2 * v - (inv_sigma2 ** 2) * (A @ (Minv @ Atv))
        return x.astype(np.float32)

    def quad_form(self, r: Array) -> Union[float, Any]:
        """Compute r' Σ^{-1} r."""
        x = self.solve(r)
        if JAX_AVAILABLE and hasattr(r, "block_until_ready"):
            return jnp.dot(jnp.ravel(r), jnp.ravel(x))
        return float(np.dot(np.ravel(r), np.ravel(x)))

    def logdet(self) -> float:
        """Compute log|Σ|."""
        return lowrank_logdet(self.A, self.sigma2, self.p, backend=_backend(self.A))

    def invalidate_cache(self) -> None:
        """Invalidate cached factorization (e.g., after A/sigma2 change)."""
        self._Minv = None
