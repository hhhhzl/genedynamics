"""
JAX implementation of low-level geometry operations.

Migrated from ``genedynamics.solvers.common.manifold.jax_ops`` and
wrapped in a :class:`GeometryOps` subclass for registry integration.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from genedynamics.genemetry.ops.base import GeometryOps
from genedynamics.genemetry.registry import register_genemetry


# ------------------------------------------------------------------
# Standalone functions (usable without the registry)
# ------------------------------------------------------------------

def build_active_rows(
    a_time: jnp.ndarray,
    topk_active: int,
    eps_stab: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build low-rank active geometry basis from time-wise proxy vectors.

    Returns
    -------
    A_rows      : (K, D) normalized active rows
    topu        : (K,)   top-k row norms
    idx_top     : (K,)   top-k indices in time axis
    active_mask_t : (H,) binary active mask
    score_base  : (H, D) masked geometry proxy
    """
    a_norm_t = jnp.linalg.norm(a_time, axis=-1)
    topu, idx_top = jax.lax.top_k(a_norm_t, topk_active)
    thresh_u = topu[-1]
    active_mask_t = (a_norm_t >= thresh_u).astype(jnp.float32)
    score_base = a_time * active_mask_t[:, None]
    A_rows = a_time[idx_top]
    A_rows = A_rows / (
        jnp.linalg.norm(A_rows, axis=-1, keepdims=True) + eps_stab
    )
    return A_rows, topu, idx_top, active_mask_t, score_base


def project_complement_batch(
    vectors: jnp.ndarray,
    A_rows: jnp.ndarray,
    system: jnp.ndarray,
) -> jnp.ndarray:
    """Project batch of vectors through I - A^T system^{-1} A.

    Uses a single Cholesky factorization for the whole batch.

    Parameters
    ----------
    vectors : (B, D)
    A_rows  : (K, D)
    system  : (K, K), SPD
    """
    Av = vectors @ jnp.transpose(A_rows)          # (B, K)
    chol = jnp.linalg.cholesky(system)

    def _solve_row(rhs: jnp.ndarray) -> jnp.ndarray:
        y = jsp_linalg.solve_triangular(chol, rhs, lower=True)
        return jsp_linalg.solve_triangular(
            jnp.transpose(chol), y, lower=False
        )

    y_batch = jax.vmap(_solve_row, in_axes=0)(Av)  # (B, K)
    return vectors - (y_batch @ A_rows)


# ------------------------------------------------------------------
# Registry wrapper
# ------------------------------------------------------------------

@register_genemetry("ops", "geometry", "jax")
class GeometryOpsJax(GeometryOps):
    """JAX geometry operations (registered via genemetry registry)."""

    def build_active_rows(self, a_time, topk_active, eps_stab):
        return build_active_rows(a_time, topk_active, eps_stab)

    def project_complement_batch(self, vectors, A_rows, system):
        return project_complement_batch(vectors, A_rows, system)
