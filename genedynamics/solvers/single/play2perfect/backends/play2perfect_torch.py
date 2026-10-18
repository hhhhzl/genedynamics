"""Native recurrent Play2Perfect actor execution."""

from __future__ import annotations

from typing import Any

import torch


class Play2PerfectBackendTorch:
    """Run the released rl-games actor without owning Isaac Sim lifecycle."""

    def __init__(self, model: Any, *, observation_dim: int, device: str = "cpu"):
        self.device = torch.device("cuda" if device == "gpu" else device)
        self.model = model.to(self.device).eval()
        self.observation_dim = int(observation_dim)
        if self.observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        self.reset()

    def reset(self, batch_size: int | None = None) -> None:
        self.batch_size = None if batch_size is None else int(batch_size)
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.states = None
        if self.batch_size is not None:
            default = self.model.get_default_rnn_state()
            self.states = [
                torch.zeros((state.shape[0], self.batch_size, state.shape[2]), device=self.device)
                for state in default
            ]

    @torch.inference_mode()
    def act(self, observation: Any, *, coefficient: Any = 0.0):
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs.ndim == 1:
            obs = obs[None]
        if obs.shape[1:] != (self.observation_dim,):
            raise ValueError(f"expected [batch,{self.observation_dim}] native observation")
        if self.batch_size is None:
            self.reset(obs.shape[0])
        elif self.batch_size != obs.shape[0]:
            raise ValueError("batch changed; reset at the episode boundary")
        coef = torch.as_tensor(coefficient, dtype=obs.dtype, device=self.device)
        coef = coef.expand(obs.shape[0]).reshape(-1, 1)
        result = self.model({
            "is_train": False,
            "prev_actions": None,
            "obs": torch.cat((obs, coef), dim=-1),
            "rnn_states": self.states,
        })
        self.states = result["rnn_states"]
        return result["mus"]


__all__ = ["Play2PerfectBackendTorch"]
