import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .base import Base


class SinWaveOpenLoop(Base):
    def __init__(self,
                 n_actuators,
                 n_sin_waves,
                 actuation_omega,
                 actuation_strength,
                 lr,
                 device):
        super(SinWaveOpenLoop, self).__init__()

        self.n_actuators = n_actuators
        self.n_sin_waves = n_sin_waves
        self.actuation_omega = actuation_omega
        self.actuation_strength = actuation_strength
        self.device = torch.device(device)

        self.weight = nn.Parameter(torch.zeros((self.n_actuators, self.n_sin_waves * len(self.actuation_omega)), requires_grad=True))
        self.bias = nn.Parameter(torch.zeros((self.n_actuators,), requires_grad=True))

        self.to(device)
        self.weight.data.normal_(0.0, 0.3)

        self.optim = optim.Adam([self.weight, self.bias], lr=lr)

    def reset(self):
        self.all_s = []
        self.all_act = []

    def forward(self, s, inp):
        inp = inp['time'].float()
        self.all_s.append(s)

        x = []
        for actuation_omega in self.actuation_omega:
            x.append(torch.sin(actuation_omega * inp + 2 * torch.pi / self.n_sin_waves * torch.arange(self.n_sin_waves)))
        x = torch.cat(x)
        x = x.to(self.device) 
        act = self.weight @ x
        act += self.bias
        act = torch.tanh(act) * self.actuation_strength
        self.all_act.append(act)

        return act.detach().clone() # NOTE: need clone here otherwise the tensor with the same "identifier"

    def update(self, grad, retain_graph=False):
        all_act = torch.stack(self.all_act)
        grad = torch.stack(grad)
        grad = grad.to(all_act) # make sure they are of the same type

        self.optim.zero_grad()
        all_act.backward(gradient=grad, retain_graph=retain_graph)
        self.optim.step()


class PureSinWaveOpenLoop(Base):
    def __init__(self,
                 n_actuators,
                 actuation_strength,
                 omega_mul,
                 lr,
                 device):
        super(PureSinWaveOpenLoop, self).__init__()

        self.n_actuators = n_actuators
        self.actuation_strength = actuation_strength
        self.omega_mul = omega_mul
        self.device = torch.device(device)

        self.amp = nn.Parameter(torch.ones((self.n_actuators,), requires_grad=True))
        self.omega = nn.Parameter(torch.ones((self.n_actuators,), requires_grad=True))
        self.psi = nn.Parameter(torch.zeros((self.n_actuators,), requires_grad=True))

        self.to(device)
        self.omega.data.normal_(0.0, 1.0)

        self.optim = optim.Adam([self.amp, self.omega, self.psi], lr=lr)

    def reset(self):
        self.all_s = []
        self.all_act = []

    def forward(self, s, inp):
        inp = inp['time'].float()
        self.all_s.append(s)
        omega = 2 * torch.pi * self.omega * self.omega_mul
        act = self.amp * torch.sin(omega * inp + self.psi) * self.actuation_strength
        self.all_act.append(act)

        return act.detach().clone() # NOTE: need clone here otherwise the tensor with the same "identifier"


class SinWaveClosedLoop(Base):
    def __init__(self,
                 obs_space,
                 obs_names,
                 hidden_filters,
                 activation,
                 n_actuators,
                 n_sin_waves,
                 actuation_omega,
                 actuation_strength,
                 lr,
                 device):
        super(SinWaveClosedLoop, self).__init__()

        in_features = 0
        for obs_name in obs_names:
            assert obs_name in obs_space.spaces.keys(), f'{obs_name} not in observation space'
            obs_shape = obs_space.spaces[obs_name].shape
            assert len(obs_shape) == 1, f'Only support 1-d observation space, not {obs_shape}'
            in_features += obs_shape[0]
        filters = [in_features] + hidden_filters

        net = []
        for i in range(len(filters) - 1):
            net.append(nn.Linear(in_features=filters[i], out_features=filters[i+1]))
            net.append(getattr(nn, activation)())
        self.net = nn.Sequential(*net)

        self.head_weight = nn.Sequential(nn.Linear(filters[-1], n_actuators * n_sin_waves * len(actuation_omega)))
        self.head_bias = nn.Sequential(nn.Linear(filters[-1], n_actuators))

        self.obs_names = obs_names
        self.n_actuators = n_actuators
        self.n_sin_waves = n_sin_waves
        self.actuation_omega = actuation_omega
        self.actuation_strength = actuation_strength
        self.device = torch.device(device)
        self.to(device)

        self.optim = optim.Adam(self.parameters(), lr=lr)

    def reset(self):
        self.all_s = []
        self.all_act = []

    def forward(self, s, inp):
        # construct input
        self.all_s.append(s)
        x = []
        for obs_name in self.obs_names:
            x.append(inp[obs_name].float())
        x = torch.cat(x)
        x = x.to(self.device)

        # generate weight and bias
        z = self.net(x)
        weight = self.head_weight(z)
        weight = weight.reshape(self.n_actuators, -1)
        bias = self.head_bias(z)

        # get actuation from sin wave
        time = inp['time'].float()
        x_sinwave = []
        for actuation_omega in self.actuation_omega:
            x_sinwave.append(torch.sin(actuation_omega * time + 2 * torch.pi / self.n_sin_waves * torch.arange(self.n_sin_waves)))
        x_sinwave = torch.cat(x_sinwave)
        x_sinwave = x_sinwave.to(self.device) 
        act = weight @ x_sinwave
        act += bias
        act = torch.tanh(act) * self.actuation_strength
        self.all_act.append(act)

        return act.detach().clone() # NOTE: need clone here otherwise the tensor with the same "identifier"
