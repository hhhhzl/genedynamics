"""Torch branching-Q network used by the AdaptSim parameter-update policy.

Adapted from the MIT-licensed AdaptSim `branching_q_network.py` and
`branching_q.py`. The shared value/advantage decomposition and losses match
the upstream implementation; the network does not own task simulation.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class BranchingQNetwork(nn.Module):
    def __init__(self, input_dim: int, bins: Sequence[int], hidden: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        if input_dim <= 0 or not bins or any(n <= 0 for n in bins):
            raise ValueError("input_dim and all branch sizes must be positive")
        dims = (input_dim, *hidden)
        if len(dims) < 2 or any(n <= 0 for n in dims):
            raise ValueError("at least one positive hidden layer is required")
        layers: list[nn.Module] = []
        for left, right in zip(dims[:-1], dims[1:]):
            layers.extend((nn.Linear(left, right), nn.ELU()))
        self.model = nn.Sequential(*layers)
        self.value_head = nn.Linear(dims[-1], 1)
        self.adv_heads = nn.ModuleList(nn.Linear(dims[-1], size) for size in bins)
        self.bins = tuple(int(size) for size in bins)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        representation = self.model(state)
        value = self.value_head(representation).unsqueeze(2)
        if len(set(self.bins)) != 1:
            raise ValueError("this backend requires equal branch bin counts")
        advantages = torch.stack([head(representation) for head in self.adv_heads], dim=1)
        return value + advantages - advantages.mean(dim=2, keepdim=True)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.model(state))


class BranchingQBackendTorch:
    def __init__(
        self,
        input_dim: int,
        bins: Sequence[int],
        *,
        hidden: Sequence[int] = (128, 128),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        single_step: bool = False,
        device: str = "cpu",
    ) -> None:
        if not 0 <= gamma <= 1 or not 0 <= tau <= 1 or learning_rate <= 0:
            raise ValueError("invalid learning hyperparameters")
        self.device = torch.device(device)
        self.network = BranchingQNetwork(input_dim, bins, hidden).to(self.device)
        self.target = deepcopy(self.network)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.single_step = bool(single_step)

    def act(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.network(state.to(self.device)).argmax(dim=2)

    def train_batch(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor | None = None,
        done: torch.Tensor | None = None,
    ) -> float:
        state = state.to(self.device, dtype=torch.float32)
        action = action.to(self.device, dtype=torch.long)
        reward = reward.to(self.device, dtype=torch.float32).reshape(-1, 1)
        if state.ndim != 2 or action.shape != (len(state), len(self.network.bins)):
            raise ValueError("state/action batch shape mismatch")
        q = self.network(state).gather(2, action.unsqueeze(-1)).squeeze(-1)
        if self.single_step:
            target = reward.expand_as(q)
            loss = F.smooth_l1_loss(q, target)
        else:
            if next_state is None or done is None:
                raise ValueError("multi-step update requires next_state and done")
            next_state = next_state.to(self.device, dtype=torch.float32)
            done = done.to(self.device, dtype=torch.float32).reshape(-1, 1)
            with torch.no_grad():
                best_action = self.network(next_state).argmax(dim=2)
                next_q = self.target(next_state).gather(2, best_action.unsqueeze(-1)).squeeze(-1)
                next_value = next_q.mean(dim=1, keepdim=True)
                target = (reward + self.gamma * (1 - done) * next_value).expand_as(q)
            loss = F.mse_loss(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        if not self.single_step:
            with torch.no_grad():
                for target_param, source_param in zip(self.target.parameters(), self.network.parameters()):
                    target_param.lerp_(source_param, self.tau)
        return float(loss.detach().cpu())
