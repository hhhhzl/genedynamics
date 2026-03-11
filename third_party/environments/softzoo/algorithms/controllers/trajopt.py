import torch
import torch.nn as nn
import torch.optim as optim

from .base import Base


class TrajOpt(Base):
    def __init__(self,
                 n_actuators,
                 actuation_strength,
                 max_steps,
                 lr,
                 device,
                 actuation_dim=1,
                 activation='tanh'):
        super(TrajOpt, self).__init__()

        self.n_actuators = n_actuators
        self.actuation_strength = actuation_strength
        self.max_steps = max_steps
        self.device = torch.device(device)
        self.actuation_dim = actuation_dim
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'softmax':
            self.activation = lambda x: torch.softmax(x, dim=-1)
        elif activation == 'linear':
            self.activation = lambda x: torch.clamp(x, min=-self.actuation_strength, max=self.actuation_strength)
        else:
            raise ValueError(f'Unrecognized activation {activation}')

        self.actuation_logits = nn.Parameter(torch.zeros(self.max_steps, self.n_actuators, actuation_dim), requires_grad=True)
        self.to(self.device)
        self.actuation_logits.data.zero_() # NOTE: zero initialization

        self.optim = optim.Adam(self.parameters(), lr=lr)

    def reset(self):
        self.all_s = []
        self.all_act = []

    def forward(self, s, inp):
        act_i = len(self.all_s)
        assert act_i < self.max_steps
        act = self.activation(self.actuation_logits[act_i]) * self.actuation_strength
        act = act.squeeze(-1) # drop actuation dimension if it's 1

        self.all_s.append(s)
        self.all_act.append(act)

        return act.detach().clone()
