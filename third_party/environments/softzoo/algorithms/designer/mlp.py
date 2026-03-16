import torch
import torch.nn as nn
import torch.optim as optim

from .base import Base
from .implicit_function_inputs import ImplicitFunctionInputs


class MLP(Base):
    def __init__(self,
                 env,
                 coord_input_names,
                 filters,
                 n_actuators,
                 activation,
                 lr,
                 geometry_offset,
                 softness_offset,
                 seed_meshes=[],
                 device='cpu'):
        super(MLP, self).__init__(env)

        self.coord_input_names = coord_input_names
        self.seed_meshes = seed_meshes
        self.input_obj = ImplicitFunctionInputs(self.env, self.coord_input_names, self.seed_meshes)
        self.input_obj.construct_data()

        # Base net
        base_net = []
        in_features = self.input_obj.dim
        for i in range(len(filters)):
            out_features = filters[i]
            base_net.append(nn.Linear(
                in_features=in_features,
                out_features=out_features,
            ))
            base_net.append(nn.BatchNorm1d(out_features)) # NOTE: otherwise may result in very similar output
            base_net.append(getattr(nn, activation)())
            in_features = out_features

        self.base_net = nn.Sequential(*base_net)

        # Heads for different output branch
        self.head_geometry = nn.Sequential(
            nn.Linear(out_features, 1),
            nn.Sigmoid(),
        )

        self.head_softness = nn.Sequential(
            nn.Linear(out_features, 1),
            nn.Sigmoid(),
        )

        self.head_actuator = nn.Sequential(
            nn.Linear(out_features, n_actuators),
            nn.Softmax(dim=1),
        )

        # Others
        self.device = torch.device(device)
        self.to(self.device)
        self.input_obj.data.to(self.device)

        self.geometry_offset = geometry_offset
        self.softness_offset = softness_offset

        self.optim = optim.Adam(self.parameters(), lr=lr)

    def reset(self):
        self.out_cache = dict(geometry=None, softness=None, actuator=None)

    def forward(self, inp=None):
        x = self.input_obj.data
        x = self.base_net(x)

        geometry = self.head_geometry(x)[:, 0] + self.geometry_offset
        softness = self.head_softness(x)[:, 0] + self.softness_offset
        actuator = self.head_actuator(x).permute(1, 0)

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
