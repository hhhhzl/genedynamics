"""
JAX manifold helpers shared by geometry-aware solvers.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg


def build_active_rows(
    a_time: jnp.ndarray,
    topk_active: int,
    eps_stab: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Build low-rank active geometry basis from time-wise proxy vectors.

    Returns:
      A_rows: normalized active rows, shape (K, act_dim)
      topu: top-k row norms, shape (K,)
      idx_top: top-k indices in time axis, shape (K,)
      active_mask_t: binary active mask over time, shape (H,)
      score_base: masked geometry proxy, shape (H, act_dim)
    """
    a_norm_t = jnp.linalg.norm(a_time, axis=-1)
    topu, idx_top = jax.lax.top_k(a_norm_t, topk_active)
    thresh_u = topu[-1]
    active_mask_t = (a_norm_t >= thresh_u).astype(jnp.float32)
    score_base = a_time * active_mask_t[:, None]
    A_rows = a_time[idx_top]
    A_rows = A_rows / (jnp.linalg.norm(A_rows, axis=-1, keepdims=True) + eps_stab)
    return A_rows, topu, idx_top, active_mask_t, score_base


def project_complement_batch(
    vectors: jnp.ndarray,
    A_rows: jnp.ndarray,
    system: jnp.ndarray,
) -> jnp.ndarray:
    """
    Project batch of vectors through I - A^T system^{-1} A using one factorization.

    Args:
      vectors: (B, D)
      A_rows: (K, D)
      system: (K, K), SPD system matrix.
    """
    Av = vectors @ jnp.transpose(A_rows)  # (B, K)
    chol = jnp.linalg.cholesky(system)

    def _solve_row(rhs: jnp.ndarray) -> jnp.ndarray:
        y = jsp_linalg.solve_triangular(chol, rhs, lower=True)
        return jsp_linalg.solve_triangular(jnp.transpose(chol), y, lower=False)

    y_batch = jax.vmap(_solve_row, in_axes=0)(Av)  # (B, K)
    return vectors - (y_batch @ A_rows)

