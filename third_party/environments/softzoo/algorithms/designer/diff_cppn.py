import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base import Base
from .implicit_function_inputs import ImplicitFunctionInputs


def sigmoid_activation(x):
    return torch.sigmoid(5 * x)


def tanh_activation(x):
    return torch.tanh(2.5 * x)


def sin_activation(x):
    return torch.sin(x)


def gauss_activation(x):
    return torch.exp(-5.0 * x**2)


def relu_activation(x):
    return F.relu(x)


def elu_activation(x):
    return F.elu(x)


def lelu_activation(x):
    return F.lelu(x)


def selu_activation(x):
    return F.selu(x)


def softplus_activation(x):
    return F.softplus(x)


def identity_activation(x):
    return x


def clamped_activation(x):
    return torch.clamp(x, min=-1., max=1.)


def inv_activation(x):
    try:
        out = 1.0 / x
    except ArithmeticError:
        return 0.0
    else:
        return out


def log_activation(x):
    return torch.log(torch.clamp(x, min=1e-7))


def exp_activation(x):
    return torch.exp(torch.clamp(x, min=-60., max=60.))


def abs_activation(x):
    return torch.abs(x)


def hat_activation(x):
    return torch.clamp(1 - torch.abs(x), min=0.0)


def square_activation(x):
    return x ** 2


def cube_activation(x):
    return x ** 3


str_to_activation = {
    'sigmoid': sigmoid_activation,
    'tanh': tanh_activation,
    'sin': sin_activation,
    'gauss': gauss_activation,
    'relu': relu_activation,
    'elu': elu_activation,
    'lelu': lelu_activation,
    'selu': selu_activation,
    'softplus': softplus_activation,
    'identity': identity_activation,
    'clamped': clamped_activation,
    'inv': inv_activation,
    'log': log_activation,
    'exp': exp_activation,
    'abs': abs_activation,
    'hat': hat_activation,
    'square': square_activation,
    'cube': cube_activation,
}


class DiffCPPN(Base):
    def __init__(self,
                 env,
                 coord_input_names,
                 n_hiddens,
                 n_actuators,
                 lr,
                 geometry_offset,
                 softness_offset,
                 activation_repeat=1,
                 activation_options=['sin'],
                 seed_meshes=[],
                 aggregation_options=['sum'],
                 device='cpu'):
        super(DiffCPPN, self).__init__(env)

        assert len(aggregation_options) == 1 and aggregation_options[0] == 'sum', \
            'Only aggregation_options == [\'sum\'] is supported for now'

        self.coord_input_names = coord_input_names
        self.seed_meshes = seed_meshes
        self.input_obj = ImplicitFunctionInputs(self.env, self.coord_input_names, self.seed_meshes)
        self.input_obj.construct_data()

        n_inputs = self.input_obj.dim

        # Base net
        base_net = nn.ModuleList()
        in_features = n_inputs
        for i in range(n_hiddens):
            out_features = len(activation_options) * activation_repeat
            base_net.append(nn.Sequential(
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.BatchNorm1d(out_features), # NOTE: otherwise may result in very similar output
            ))
            in_features = out_features

        activations = []
        for act in activation_options:
            activations.extend([str_to_activation[act]] * activation_repeat)

        self.base_net = base_net
        self.activations = activations
        
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
        for layer in self.base_net:
            x = layer(x)
            
            act_x = []
            for act_i, act in enumerate(self.activations):
                act_x.append(act(x[:, act_i]))
            act_x = torch.stack(act_x, dim=1)
            
            x = act_x

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
