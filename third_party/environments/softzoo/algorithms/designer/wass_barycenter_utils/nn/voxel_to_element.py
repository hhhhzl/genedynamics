import math
from itertools import product

import torch
import torch.autograd as autograd
import torch.nn as nn


class VoxelToElementBinaryFunction(autograd.Function):

    @staticmethod
    def forward(ctx, voxel, cooptimizer):
        ctx.save_for_backward(voxel)
        ctx.cooptimizer = cooptimizer
        elements = voxel.new_zeros(cooptimizer.rest_mesh_blob.num_elements)
        for k in cooptimizer.blob_to_fish_cell.keys():
            elements[k] = 1.0
        return elements

    @staticmethod
    def backward(ctx, dl_delements):
        voxel, = ctx.saved_tensors
        cooptimizer = ctx.cooptimizer
        dl_dvoxel = torch.zeros_like(voxel)
        cell_indices = cooptimizer.rest_mesh_blob.cell_indices
        for x, y, z in product(*[range(s) for s in voxel.size()]):
            element_id = cell_indices[x, y, z]
            if element_id == -1:
                raise ValueError('A blob should not contain -1 in cell indices')
            dl_dvoxel[x, y, z] = dl_delements[element_id]
        return dl_dvoxel, None


class VoxelToElementBinary(nn.Module):
    def __init__(self, cooptimizer, eps=1e-7):
        super().__init__()
        self.cooptimizer = cooptimizer
        self.eps = eps

    def forward(self, voxel):
        voxel = voxel.clamp(0, 1)
        return VoxelToElementBinaryFunction.apply(voxel, self.cooptimizer) + self.eps


class VoxelToElementSchlick(nn.Module):
    # https://arxiv.org/pdf/2010.09714.pdf
    def __init__(self, a, alpha=None, cooptimizer=None, eps=1e-7):
        super().__init__()
        self.a = a
        self.alpha = alpha
        self.eps = eps
        if cooptimizer is None or alpha is None:
            self.binary = None
        else:
            self.binary = VoxelToElementBinary(cooptimizer, 0.0)

    @staticmethod
    def bias(x, a):
        return x / (1 + (1 / a - 2) * (1 - x))

    @staticmethod
    def gain(x, a):
        left = 0.5 * VoxelToElementSchlick.bias(2 * x, a)
        right = 0.5 * (VoxelToElementSchlick.bias(2 * x - 1, 1 - a) + 1)
        return torch.where(x < 0.5, left, right)

    def forward(self, voxel):
        # clamp to [0, 1]
        elements = voxel.clamp(0, 1)
        # apply Schlick's Gain Function
        if self.binary is None:
            elements = VoxelToElementSchlick.gain(elements.view(-1), self.a) + self.eps
        else:
            elements = self.alpha * VoxelToElementSchlick.gain(
                elements, self.a).view(-1) + (1 - self.alpha) * self.binary(elements[0, 0]) + self.eps
        return elements


class VoxelToElementSoft(nn.Module):
    def __init__(self, sharp_coe=None):
        super().__init__()
        self.sharp_coe = sharp_coe # basically 1/2 or 1/3

    def sharpen(self, elements):
        if self.sharp_coe is None or self.sharp_coe == 1.0:
            return elements
        sharp_elements = []
        sharp_coe = self.sharp_coe
        for ele in elements:
            if ele >= 0:
                sharp_elements.append(ele ** sharp_coe)
            else:
                sharp_elements.append(-((-ele) ** sharp_coe))
        sharp_elements = torch.stack(sharp_elements)
        return sharp_elements

    def forward(self, voxel):

        # clamp to [0, 1]
        elements = voxel.clamp(0, 1).view(-1)

        # rescale to [-pi/2, pi/2]
        elements = (elements - 0.5) * math.pi

        # apply sine, rescale to [-1, 1]
        elements = torch.sin(elements)

        # sharpen
        elements = self.sharpen(elements)

        # rescale to [0, 1]
        elements = (elements + 1.0) * 0.5

        return elements


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    x = np.linspace(-1, 1, 100)

    y = x * np.pi * 0.5
    y = np.cbrt(np.sin(y))

    ax.plot(x, y)
    plt.show()
