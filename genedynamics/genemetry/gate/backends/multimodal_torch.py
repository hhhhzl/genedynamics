"""Torch parity implementation of 2GO's route-multimodality gate."""

from __future__ import annotations

import torch

from genedynamics.genemetry.base import GatePolicy
from genedynamics.genemetry.registry import register_genemetry
from genedynamics.genemetry.types import GateDecision


def _sorted_gap_proxy(projection: torch.Tensor) -> torch.Tensor:
    if projection.ndim != 1 or projection.numel() < 2:
        raise ValueError("Torch multimodality requires at least two samples")
    ordered = torch.sort(projection).values
    gaps = ordered[1:] - ordered[:-1]
    scale = ordered[-1] - ordered[0] + projection.new_tensor(1.0e-8)
    return (gaps.max() / scale).clamp(0.0, 1.0)


@register_genemetry("gate", "multimodal", "torch")
class MultimodalGateTorch(GatePolicy):
    """Use the same principal projection and sorted-gap proxy as JAX 2GO."""

    def __init__(
        self,
        *,
        multi_scale: float = 0.25,
        enable_local_gating: bool = True,
        **kwargs,
    ):
        del kwargs
        self._multi_scale = float(max(multi_scale, 1.0e-6))
        self._enable_local_gating = bool(enable_local_gating)

    def evaluate(self, samples, reference, window_mask, sigma_scale, theta):
        del sigma_scale
        if (
            not isinstance(samples, torch.Tensor)
            or samples.ndim != 3
            or samples.shape[0] < 2
        ):
            raise ValueError("Torch multimodality samples must have shape [N>=2,H,D]")
        if (
            reference.shape != samples.shape[1:]
            or window_mask.shape != samples.shape[1:2]
        ):
            raise ValueError("Torch multimodality reference or window shape differs")
        if not all(
            torch.isfinite(value).all() for value in (samples, reference, window_mask)
        ):
            raise ValueError("Torch multimodality inputs must be finite")

        count, horizon, _ = samples.shape
        quarter = max(horizon // 4, 1)
        masked = samples * window_mask[None, :, None]
        endpoints = masked[:, horizon - quarter :, :].reshape(count, -1)
        centered = endpoints - endpoints.mean(dim=0, keepdim=True)
        eps = samples.new_tensor(1.0e-8)
        direction = centered.std(dim=0, unbiased=False)
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(eps)
        projected = centered @ direction
        direction = centered.transpose(0, 1) @ projected
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(eps)
        projected = centered @ direction
        route_probability = _sorted_gap_proxy(projected)
        threshold = torch.as_tensor(theta, dtype=samples.dtype, device=samples.device)
        gamma = (route_probability > threshold).to(samples.dtype)
        if not self._enable_local_gating:
            gamma = torch.ones((), dtype=samples.dtype, device=samples.device)
        spread = (
            (samples - reference).std(dim=0, unbiased=False) * window_mask[:, None]
        ).mean()
        return GateDecision(
            gamma=gamma,
            meta={"pi_multi": route_probability, "spread": spread},
        )


__all__ = ["MultimodalGateTorch"]
