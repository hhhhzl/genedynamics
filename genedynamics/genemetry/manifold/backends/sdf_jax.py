"""
JAX implementation of the SDF constraint manifold.

Extracts and structures the geometry computation that was previously
inlined in ``twogo_jax.py`` (active-row selection, metric/tangent
system construction, complement projection).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from genedynamics.genemetry.base import ConstraintManifold
from genedynamics.genemetry.types import GeometryBundle
from genedynamics.genemetry.registry import register_genemetry
from genedynamics.genemetry.ops.backends.jax_ops import (
    build_active_rows,
    project_complement_batch,
)


@register_genemetry("manifold", "sdf", "jax")
class SdfManifoldJax(ConstraintManifold):
    """JAX SDF constraint manifold.

    Given per-timestep constraint proxy vectors (e.g. windowed /
    modulated SDF normals), computes:

    1. Top-k active constraint basis ``A_rows``  (K, D).
    2. Metric system ``metric_sys = I + A A^T + eps I``  (K, K).
    3. Tangent system ``tan_sys = A A^T + eps I``  (K, K).

    Projection uses a single Cholesky factorization shared across
    the vector batch.
    """

    def geometry(
        self,
        constraint_vectors: jnp.ndarray,
        topk_active: int,
        eps_stab: float,
    ) -> GeometryBundle:
        A_rows, topu, idx_top, active_mask_t, score_base = build_active_rows(
            constraint_vectors, topk_active, eps_stab
        )

        active_count = jnp.sum((topu > eps_stab).astype(jnp.float32))
        is_valid = active_count >= 1.0

        AAT = A_rows @ jnp.transpose(A_rows)
        I_k = jnp.eye(topk_active, dtype=jnp.float32)
        metric_sys = I_k + AAT + eps_stab * I_k
        tan_sys = AAT + eps_stab * I_k

        return GeometryBundle(
            A_rows=A_rows,
            topu=topu,
            active_count=active_count,
            is_valid=is_valid,
            metric_sys=metric_sys,
            tan_sys=tan_sys,
            raw_geometry=constraint_vectors,
            meta={
                "idx_top": idx_top,
                "active_mask_t": active_mask_t,
                "score_base": score_base,
            },
        )

    def project(
        self,
        vectors: jnp.ndarray,
        bundle: GeometryBundle,
        mode: str = "metric",
    ) -> jnp.ndarray:
        system = bundle.metric_sys if mode == "metric" else bundle.tan_sys
        return project_complement_batch(vectors, bundle.A_rows, system)
