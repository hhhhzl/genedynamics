from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass(frozen=True)
class LimitsStats:
    mins: np.ndarray
    maxs: np.ndarray


class LimitsNormalizer:
    """
    Maps [mins, maxs] to [-1, 1] and back.
    """

    def __init__(self, mins: np.ndarray, maxs: np.ndarray, eps: float = 1e-6):
        mins = np.asarray(mins, dtype=np.float32).reshape(-1)
        maxs = np.asarray(maxs, dtype=np.float32).reshape(-1)
        # avoid divide-by-zero for constant dims
        span = (maxs - mins)
        span = np.where(np.abs(span) < eps, 1.0, span).astype(np.float32)
        self.mins = mins
        self.maxs = mins + span

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x01 = (x - self.mins) / (self.maxs - self.mins)
        return 2.0 * x01 - 1.0

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x01 = (x + 1.0) / 2.0
        return x01 * (self.maxs - self.mins) + self.mins

    def normalize_torch(self, x: torch.Tensor) -> torch.Tensor:
        mins = torch.tensor(self.mins, dtype=torch.float32, device=x.device)
        maxs = torch.tensor(self.maxs, dtype=torch.float32, device=x.device)
        x01 = (x - mins) / (maxs - mins)
        return 2.0 * x01 - 1.0

    def unnormalize_torch(self, x: torch.Tensor) -> torch.Tensor:
        mins = torch.tensor(self.mins, dtype=torch.float32, device=x.device)
        maxs = torch.tensor(self.maxs, dtype=torch.float32, device=x.device)
        x01 = (x + 1.0) / 2.0
        return x01 * (maxs - mins) + mins


class PlanningNormalizer:
    """
    Minimal normalizer object compatible with SafeDiffuser Policy/diffusion hooks.
    """

    def __init__(self, obs_mins: np.ndarray, obs_maxs: np.ndarray, act_mins: np.ndarray, act_maxs: np.ndarray):
        self.normalizers: Dict[str, LimitsNormalizer] = {
            "observations": LimitsNormalizer(obs_mins, obs_maxs),
            "actions": LimitsNormalizer(act_mins, act_maxs),
        }
        self.observation_dim = int(np.asarray(obs_mins).size)
        self.action_dim = int(np.asarray(act_mins).size)

    def normalize(self, x: np.ndarray, key: str) -> np.ndarray:
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x: np.ndarray, key: str) -> np.ndarray:
        return self.normalizers[key].unnormalize(x)

    def normalize_torch(self, x: torch.Tensor, key: str) -> torch.Tensor:
        return self.normalizers[key].normalize_torch(x)

    def unnormalize_torch(self, x: torch.Tensor, key: str) -> torch.Tensor:
        return self.normalizers[key].unnormalize_torch(x)

