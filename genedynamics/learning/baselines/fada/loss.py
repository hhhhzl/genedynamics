# SPDX-License-Identifier: Apache-2.0
# Derived from LeCAR-Lab/FADA commit 290b3a9c5c0ebe2e9851b61350617544be05e9bc.
# Source: src/holosoma/holosoma/fada/planner_idm/loss_utils.py
# Changes: isolate native PyTorch core; replace Holosoma imports with local imports.
from __future__ import annotations

import torch
from torch.nn import functional as F

def weighted_horizon_mse(pred: torch.Tensor, target: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Exponentially-decayed MSE over [B, K, D] tensors.

    w_t = gamma^t (t=0,1,...,K-1); loss = sum_t(w_t * mse_t) / sum_t(w_t)
    where mse_t is mean over B and D dimensions.

    When gamma == 1.0, returns F.mse_loss(pred, target) exactly.
    """
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if gamma == 1.0:
        return F.mse_loss(pred, target)
    K = pred.shape[1]
    t = torch.arange(K, device=pred.device, dtype=pred.dtype)
    weights = gamma**t  # [K]
    mse_per_step = ((pred - target) ** 2).mean(dim=(0, 2))  # [K]
    return (weights * mse_per_step).sum() / weights.sum()

