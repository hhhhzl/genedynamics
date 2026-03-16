import torch
import torch.nn as nn
import torch.optim as optim

from .base import Base


class LossLandscapeVBR(Base):
    def __init__(self,
                 env,
                 voxel_resolution,
                 grid_index,
                 value_range,
                 n_trials,
                 trial_type,
                 n_actuators, # dummy
                 lr=None, # dummy
                 device='cpu'):
        super(LossLandscapeVBR, self).__init__(env)

        voxel_grid_size = tuple(voxel_resolution)

        self.geometry = nn.Parameter(torch.ones(voxel_grid_size))
        self.softness = nn.Parameter(torch.ones(voxel_grid_size))
        self.actuator = nn.Parameter(torch.ones((n_actuators,) + voxel_grid_size))

        # Others
        self.device = torch.device(device)
        self.to(self.device)

        self.grid_index = grid_index
        self.trial_values = torch.linspace(*value_range, n_trials)
        self.n_trials = n_trials
        self.trial_type = trial_type

        self.optim = optim.Adam(self.parameters(), lr=lr) # dummy

    def reset(self):
        self.out_cache = dict(geometry=None, softness=None, actuator=None)

    def forward(self, inp=None):
        geometry = self.geometry
        softness = self.softness
        actuator = torch.clamp(self.actuator, min=0.)
        actuator = actuator / actuator.sum(0)

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

    def set_trial_idx(self, trial_i):
        val = self.trial_values[trial_i]
        if self.trial_type == 'geometry':
            if -1 in self.grid_index:
                self.geometry.data.fill_(val)
            else:
                self.geometry.data[self.grid_index[0], self.grid_index[1], self.grid_index[2]] = val
        elif self.trial_type == 'softness':
            if -1 in self.grid_index:
                self.softness.data.fill_(val)
            else:
                self.softness.data[self.grid_index[0], self.grid_index[1], self.grid_index[2]] = val
        elif self.trial_type == 'actuator':
            act_id = torch.floor(val).int()
            val -= act_id

            self.actuator.data[act_id, self.grid_index[0], self.grid_index[1], self.grid_index[2]] = val
            self.actuator.data[act_id+1, self.grid_index[0], self.grid_index[1], self.grid_index[2]] = 1 - val
        else:
            raise ValueError(f'Unrecognized trial type {self.trial_type}')

    def update(self, grad, retain_graph=False):
        raise NotImplementedError('Shouldn\'t be here')

    @property
    def total_trials(self):
        return self.n_trials
