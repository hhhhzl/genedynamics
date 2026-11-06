"""Torch gather/retract/scatter probe pipeline for 2GO."""

from __future__ import annotations

import torch

from genedynamics.genemetry.base import ProbeSampler, RetractionOperator
from genedynamics.genemetry.registry import register_genemetry
from genedynamics.genemetry.types import ProbeResult


@register_genemetry("pipeline", "probe", "torch")
class ProbePipelineTorch(ProbeSampler):
    """Eager Torch implementation with the same tail/random mixture as JAX."""

    def __init__(
        self,
        tail_mix: float = 0.5,
        pool_ratio: float = 0.25,
        max_probes: int | None = None,
    ):
        if not 0.0 <= tail_mix <= 1.0 or not 0.0 < pool_ratio <= 1.0:
            raise ValueError("Torch probe mixture values are outside [0, 1]")
        self._tail_mix = float(tail_mix)
        self._pool_ratio = float(pool_ratio)
        self._max_probes = max_probes

    def sample_and_retract(
        self,
        trajectories,
        costs,
        retraction_op: RetractionOperator,
        state,
        retract_params,
        rng_keys,
    ):
        if (
            not isinstance(trajectories, torch.Tensor)
            or trajectories.ndim != 3
            or not isinstance(costs, torch.Tensor)
            or costs.shape != trajectories.shape[:1]
        ):
            raise ValueError("Torch probes require [N,H,D] trajectories and [N] costs")
        if not torch.isfinite(trajectories).all() or not torch.isfinite(costs).all():
            raise ValueError("Torch probe inputs must be finite")

        count = trajectories.shape[0]
        requested = retract_params.get("M_eff", count)
        if isinstance(requested, torch.Tensor):
            requested = int(requested.detach().cpu())
        maximum = count if self._max_probes is None else int(self._max_probes)
        effective = max(0, min(int(requested), count, maximum))
        if effective == 0:
            return ProbeResult(
                fixed_trajectories=trajectories,
                residual_geometry=torch.zeros_like(trajectories[0]),
                meta={
                    "bmask": torch.zeros(
                        count, dtype=trajectories.dtype, device=trajectories.device
                    ),
                    "n_probed": 0,
                    "idx_m": torch.empty(
                        0, dtype=torch.int64, device=trajectories.device
                    ),
                },
            )

        generator = (
            rng_keys.get("generator") if isinstance(rng_keys, dict) else rng_keys
        )
        pool = min(count, max(1, int(self._pool_ratio * count)))
        tail_count = min(effective, max(0, round(effective * self._tail_mix)))
        tail_pool = torch.topk(costs, pool, largest=True, sorted=True).indices
        tail_permutation = torch.randperm(
            pool, device=trajectories.device, generator=generator
        )
        random_permutation = torch.randperm(
            count, device=trajectories.device, generator=generator
        )
        indices = torch.cat(
            (
                tail_pool.index_select(0, tail_permutation[:tail_count]),
                random_permutation[: effective - tail_count],
            )
        )

        selected = trajectories.index_select(0, indices)
        repaired = []
        for trajectory in selected:
            repaired.append(
                retraction_op.retract(state, trajectory, retract_params).trajectory
            )
        repaired = torch.stack(repaired)
        fixed = trajectories.clone()
        mask = torch.zeros(count, dtype=trajectories.dtype, device=trajectories.device)
        # Preserve JAX scatter semantics when tail/random sampling selects a duplicate:
        # the later slot wins and the sample is counted once in the fixed batch.
        for index, value in zip(indices, repaired):
            fixed[index] = value
            mask[index] = 1
        residual = (repaired - selected).mean(dim=0)
        return ProbeResult(
            fixed_trajectories=fixed,
            residual_geometry=residual,
            meta={"bmask": mask, "n_probed": effective, "idx_m": indices},
        )


__all__ = ["ProbePipelineTorch"]
