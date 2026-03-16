import os
import sys
import math
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamePad(nn.Module):
    def __init__(self, size, dim, pad_mode, ceil_mode):
        super(SamePad, self).__init__()
        if pad_mode == 'replicate':
            self.pad = self.replicate_pad
        elif pad_mode == 'zero':
            self.pad = self.zero_pad
        else:
            raise ValueError('invalid pad mode: {}'.format(pad_mode))

        half_padding = math.floor(size / 2)
        if size % 2 == 0:
            base_padding = [half_padding - 1, half_padding] if ceil_mode else [half_padding, half_padding - 1]
        else:
            base_padding = [half_padding, half_padding]

        self.padding = base_padding * dim

    def zero_pad(self, x):
        return F.pad(x, self.padding, 'constant', 0)

    def replicate_pad(self, x):
        return F.pad(x, self.padding, 'replicate')

    def forward(self, x):
        return self.pad(x)


class _GaussianConvNd(nn.Module):
    def __init__(self, sigma, size, pad_mode, ceil_mode, conv, dim, dtype):
        super(_GaussianConvNd, self).__init__()
        self.sigma = sigma
        self.size = size
        self.conv = conv
        self.pad = SamePad(size, dim, pad_mode, ceil_mode)

        weight = torch.arange(-math.floor(size / 2), math.ceil(size / 2)).to(dtype)
        weight = torch.exp(-(weight / sigma)**2 / 2) / (sigma * math.sqrt(2 * math.pi))
        self.register_buffer('weight', weight.detach(), persistent=False)

        w_size_template = [1] * (dim + 2)
        self.w_sizes = []
        for d in range(dim):
            w_size = w_size_template.copy()
            w_size[d + 2] = size
            self.w_sizes.append(w_size)

    def forward(self, x):
        out = self.pad(x)
        in_channels = out.size(1)
        for w_size in self.w_sizes:
            expand_size = [-1] * len(w_size)
            expand_size[0] = in_channels
            out = self.conv(out, self.weight.view(w_size).expand(expand_size), groups=in_channels)
        return out


class GaussianConv1d(_GaussianConvNd):
    def __init__(self, sigma, size, pad_mode='replicate', ceil_mode=False, dtype=torch.double):
        super(GaussianConv1d, self).__init__(sigma, size, pad_mode, ceil_mode, F.conv1d, 1, dtype)


class GaussianConv2d(_GaussianConvNd):
    def __init__(self, sigma, size, pad_mode='replicate', ceil_mode=False, dtype=torch.double):
        super(GaussianConv2d, self).__init__(sigma, size, pad_mode, ceil_mode, F.conv2d, 2, dtype)


class GaussianConv3d(_GaussianConvNd):
    def __init__(self, sigma, size, pad_mode='replicate', ceil_mode=False, dtype=torch.double):
        super(GaussianConv3d, self).__init__(sigma, size, pad_mode, ceil_mode, F.conv3d, 3, dtype)


if __name__ == "__main__":
    gaussian_conv = GaussianConv2d(2, 59)
    t = torch.ones(1, 1, 224, 224)
    t = gaussian_conv(t)
    print(gaussian_conv.weight)
