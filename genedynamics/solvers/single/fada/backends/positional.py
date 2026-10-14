# SPDX-License-Identifier: Apache-2.0
# Derived from LeCAR-Lab/FADA commit 290b3a9c5c0ebe2e9851b61350617544be05e9bc.
# Source: src/holosoma/holosoma/fada/common/backbone.py
# Changes: isolate native PyTorch core; replace Holosoma imports with local imports.
from __future__ import annotations

import math
import torch
from torch import nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if max_len <= 0:
            raise ValueError("max_len must be positive")

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if seq_len > self.pe.shape[1]:
            raise ValueError(f"seq_len={seq_len} exceeds max_len={self.pe.shape[1]}")
        return self.pe[:, :seq_len, :].to(device=device, dtype=dtype)

