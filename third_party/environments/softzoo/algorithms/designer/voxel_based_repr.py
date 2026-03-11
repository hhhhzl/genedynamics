import torch
import torch.nn as nn
import torch.optim as optim

from .base import Base


class VoxelBasedRepresentation(Base):
    def __init__(self,
                 env,
                 n_actuators,
                 voxel_resolution,
                 lr,
                 geometry_offset,
                 softness_offset,
                 device='cpu'):
        super(VoxelBasedRepresentation, self).__init__(env)

        voxel_grid_size = tuple(voxel_resolution)

        self.geometry = nn.Parameter(torch.ones(voxel_grid_size))
        self.softness = nn.Parameter(torch.ones(voxel_grid_size))
        self.actuator = nn.Parameter(torch.ones((n_actuators,) + voxel_grid_size))

        # Others
        self.device = torch.device(device)
        self.to(self.device)

        self.geometry_offset = geometry_offset
        self.softness_offset = softness_offset

        self.optim = optim.Adam(self.parameters(), lr=lr)

    def reset(self):
        self.out_cache = dict(geometry=None, softness=None, actuator=None)

    def forward(self, inp=None):
        geometry = torch.sigmoid(self.geometry) + self.geometry_offset
        softness = torch.sigmoid(self.softness) + self.softness_offset
        actuator = torch.softmax(self.actuator, dim=0)

        geom_thresh = 0.5

        geometry = geometry * (geometry > geom_thresh).detach() # NOTE: thresholding occupancy

        geo_mask_st = torch.clamp(torch.sign(geometry - geom_thresh), 0.) # NOTE: gradient flow from actuator and softness to shape
        softness = softness * geo_mask_st
        actuator = actuator * geo_mask_st

        self.out_cache['geometry'] = geometry
        self.out_cache['softness'] = softness
        self.out_cache['actuator'] = actuator

        # NOTE: must use clone here otherwise tensor may be modified in-place in sim
        design = {k: v.clone() for k, v in self.out_cache.items()}

        return design

