import torch
import torch.nn as nn
import torch.optim as optim


from .base import Base


class MLP(Base):
    def __init__(self,
                 obs_space,
                 obs_names,
                 hidden_filters,
                 activation,
                 n_actuators,
                 actuation_strength,
                 lr,
                 device,
                 final_activation,
                 actuation_dim=1):
        super(MLP, self).__init__()

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

        self.head = [nn.Linear(filters[-1], n_actuators * actuation_dim)]
        if final_activation not in [None, 'None']:
            self.head.append(getattr(nn, final_activation)())
        self.head = nn.Sequential(*self.head)

        self.device = torch.device(device)
        self.to(self.device)

        self.obs_names = obs_names
        self.actuation_strength = actuation_strength
        self.actuation_dim = actuation_dim
        self.optim = optim.Adam(self.parameters(), lr=lr)

    def reset(self):
        self.all_s = []
        self.all_act = []

    def forward(self, s, inp):
        self.all_s.append(s)
        x = []
        for obs_name in self.obs_names:
            x.append(inp[obs_name].float())
        x = torch.cat(x)
        x = x.to(self.device)

        z = self.net(x)
        act = self.head(z) * self.actuation_strength
        act = act.reshape(-1, self.actuation_dim)
        act = act.squeeze(-1) # drop actuation dimension if it's 1
        self.all_act.append(act)

        return act.detach().clone() # NOTE: need clone here otherwise the tensor with the same "identifier"
