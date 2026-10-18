# Derived from Yifan Hou ACP (MIT), commit 74d68eca3bfb5cb59da131c6351d9e1c9b129ff6.
# Copyright (c) 2024 Yifan Hou@Stanford. See LICENSE and SOURCE.json.
import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, max_value=10000):
        super().__init__()
        self.dim = dim
        self.max_value = max_value

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.max_value) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
