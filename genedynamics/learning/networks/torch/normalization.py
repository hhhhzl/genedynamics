# Adapted from zita-ch/supermimic_vibed @ 0525a23a2dcfa61f8543f98b84c075fe66d9d286.
# Copyright (c) 2026 ETH Zurich, Chong Zhang. MIT; see LICENSE.supermimic_vibed.
"""Streaming normalization with explicit update/freeze lifecycle."""

import torch
from torch import nn


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``."""

    def __init__(self, dim: int, max_count: float = 200000000.0, eps: float = 0.01):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = (max_count, eps)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if self.count >= self.max_count:
            return
        n = x.shape[0]
        m, v = (x.mean(0), x.var(0, unbiased=False))
        tot = self.count + n
        delta = m - self.mean
        self.mean += delta * n / tot
        self.var = (
            self.var * self.count + v * n + delta**2 * self.count * n / tot
        ) / tot
        self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)
