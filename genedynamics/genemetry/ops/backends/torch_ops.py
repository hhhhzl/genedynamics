"""Torch implementations of the low-level constraint-geometry operations."""

from __future__ import annotations

import torch

from genedynamics.genemetry.ops.base import GeometryOps
from genedynamics.genemetry.registry import register_genemetry


def build_active_rows(
    a_time: torch.Tensor,
    topk_active: int,
    eps_stab: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match the JAX top-k active-row construction without leaving Torch."""

    if not isinstance(a_time, torch.Tensor) or a_time.ndim != 2:
        raise TypeError(
            "Torch constraint geometry must have shape [horizon, dimension]"
        )
    if not torch.isfinite(a_time).all():
        raise ValueError("Torch constraint geometry must be finite")
    if (
        isinstance(topk_active, bool)
        or not isinstance(topk_active, int)
        or not 1 <= topk_active <= a_time.shape[0]
    ):
        raise ValueError("Torch active-row count must be within the horizon")
    if eps_stab <= 0:
        raise ValueError("Torch geometry stabilization must be positive")

    a_norm_t = torch.linalg.vector_norm(a_time, dim=-1)
    topu, idx_top = torch.topk(a_norm_t, topk_active, largest=True, sorted=True)
    active_mask_t = (a_norm_t >= topu[-1]).to(dtype=a_time.dtype)
    score_base = a_time * active_mask_t[:, None]
    rows = a_time.index_select(0, idx_top)
    rows = rows / (
        torch.linalg.vector_norm(rows, dim=-1, keepdim=True) + float(eps_stab)
    )
    return rows, topu, idx_top, active_mask_t, score_base


def project_complement_batch(
    vectors: torch.Tensor,
    rows: torch.Tensor,
    system: torch.Tensor,
) -> torch.Tensor:
    """Project through ``I - A^T system^-1 A`` with one Cholesky solve."""

    if vectors.ndim != 2 or rows.ndim != 2 or system.ndim != 2:
        raise ValueError("Torch complement projection requires matrix inputs")
    if vectors.shape[1] != rows.shape[1] or system.shape != (
        rows.shape[0],
        rows.shape[0],
    ):
        raise ValueError("Torch complement projection shapes are inconsistent")
    if not all(torch.isfinite(value).all() for value in (vectors, rows, system)):
        raise ValueError("Torch complement projection inputs must be finite")
    products = vectors @ rows.transpose(0, 1)
    factor = torch.linalg.cholesky(system)
    solved = torch.cholesky_solve(products.transpose(0, 1), factor).transpose(0, 1)
    return vectors - solved @ rows


@register_genemetry("ops", "geometry", "torch")
class GeometryOpsTorch(GeometryOps):
    """Registered Torch geometry operations."""

    def build_active_rows(self, a_time, topk_active, eps_stab):
        return build_active_rows(a_time, topk_active, eps_stab)

    def project_complement_batch(self, vectors, rows, system):
        return project_complement_batch(vectors, rows, system)


__all__ = ["GeometryOpsTorch", "build_active_rows", "project_complement_batch"]
