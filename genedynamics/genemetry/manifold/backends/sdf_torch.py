"""Torch constraint-manifold implementation used by evaluator-bound 2GO."""

from __future__ import annotations

import torch

from genedynamics.genemetry.base import ConstraintManifold
from genedynamics.genemetry.ops.backends.torch_ops import (
    build_active_rows,
    project_complement_batch,
)
from genedynamics.genemetry.registry import register_genemetry
from genedynamics.genemetry.types import GeometryBundle


@register_genemetry("manifold", "sdf", "torch")
class SdfManifoldTorch(ConstraintManifold):
    """Construct and project through the same low-rank geometry as JAX 2GO."""

    def geometry(self, constraint_vectors, topk_active, eps_stab):
        rows, topu, idx_top, active_mask_t, score_base = build_active_rows(
            constraint_vectors, topk_active, eps_stab
        )
        active_count = (topu > float(eps_stab)).to(rows.dtype).sum()
        identity = torch.eye(topk_active, dtype=rows.dtype, device=rows.device)
        products = rows @ rows.transpose(0, 1)
        return GeometryBundle(
            A_rows=rows,
            topu=topu,
            active_count=active_count,
            is_valid=active_count >= 1,
            metric_sys=identity + products + float(eps_stab) * identity,
            tan_sys=products + float(eps_stab) * identity,
            raw_geometry=constraint_vectors,
            meta={
                "idx_top": idx_top,
                "active_mask_t": active_mask_t,
                "score_base": score_base,
            },
        )

    def project(self, vectors, bundle, mode="metric"):
        if mode not in {"metric", "tangent"}:
            raise ValueError("Torch manifold projection mode must be metric or tangent")
        system = bundle.metric_sys if mode == "metric" else bundle.tan_sys
        return project_complement_batch(vectors, bundle.A_rows, system)


__all__ = ["SdfManifoldTorch"]
