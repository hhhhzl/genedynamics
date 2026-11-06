"""Torch schedule overlay for source-owned Torch 2GO planning."""

from __future__ import annotations

import torch

from genedynamics.genemetry.base import ScheduleOverlayBase
from genedynamics.genemetry.registry import register_genemetry
from genedynamics.genemetry.schedule.config import OverlayConfig


def _tensor(value):
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


@register_genemetry("schedule", "overlay", "torch")
class ScheduleOverlayTorch(ScheduleOverlayBase):
    """Torch equivalent of the JAX/NumPy hardness and overlay chain."""

    def __init__(self, config: OverlayConfig):
        self._cfg = config

    def hardness(self, rho):
        value = _tensor(rho).clamp_min(0)
        denominator = value.new_tensor(self._cfg.rho_ref).log1p().clamp_min(1.0e-6)
        return (value.log1p() / denominator).clamp(0.0, 1.0)

    def constraint_overlay(self, margin, rho):
        c = self._cfg
        rho_value = _tensor(rho).clamp_min(0)
        margin_value = _tensor(margin).to(device=rho_value.device).clamp_min(0)
        kappa = c.kappa0 + c.kappa_rho_gain * rho_value
        delta = c.delta_margin_scale * margin_value
        delta = delta / (1.0 + c.delta_rho_gain * rho_value)
        return kappa, delta.clamp_min(c.delta_min)

    def diffusion_overlay(self, hardness, eta_base):
        c = self._cfg
        hard = _tensor(hardness).clamp(0.0, 1.0)
        one_minus = 1.0 - hard
        sigma = c.sigma_max * one_minus.pow(c.sigma_hardness_power)
        theta = c.theta_max - (c.theta_max - c.theta_min) * hard
        scale = c.eta_scale_min + (c.eta_scale_max - c.eta_scale_min) * one_minus.pow(
            c.eta_hardness_power
        )
        return sigma, theta, float(eta_base) * scale


__all__ = ["ScheduleOverlayTorch"]
