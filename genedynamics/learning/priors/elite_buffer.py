"""Elite buffer — top-K optimized control sequences -> `s_theta` training data.

Shared infrastructure: a solver pushes its
optimized `U` (and the conditioning obs + achieved reward) each replan; the
top-K by reward form the dataset for training the learned diffusion prior
`s_theta`. Plain numpy ring buffer with a reward-priority `sample` — no jax, no
solver coupling, import-safe everywhere.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class EliteBuffer:
    """Fixed-capacity reward-prioritized buffer of (obs_seq, act_seq, reward)."""

    def __init__(self, capacity: int = 4096, seed: int = 0):
        self.capacity = int(capacity)
        self._obs: list = []
        self._act: list = []
        self._rew: np.ndarray = np.empty((0,), np.float32)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._act)

    def add(self, obs_seq, act_seq, reward: float) -> None:
        """Add one trajectory; evict the lowest-reward entry when over capacity."""
        self._obs.append(np.asarray(obs_seq, np.float32))
        self._act.append(np.asarray(act_seq, np.float32))
        self._rew = np.append(self._rew, np.float32(reward))
        if len(self._act) > self.capacity:
            drop = int(np.argmin(self._rew))
            self._obs.pop(drop); self._act.pop(drop)
            self._rew = np.delete(self._rew, drop)

    def add_batch(self, obs_b, act_b, rew_b) -> None:
        for o, a, r in zip(obs_b, act_b, rew_b):
            self.add(o, a, float(r))

    def topk(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The k highest-reward (obs_seq, act_seq, reward) stacked."""
        if len(self) == 0:
            raise ValueError("EliteBuffer is empty")
        k = min(int(k), len(self))
        idx = np.argsort(self._rew)[::-1][:k]
        return (np.stack([self._obs[i] for i in idx]),
                np.stack([self._act[i] for i in idx]),
                self._rew[idx])

    def sample(self, k: int, *, elite_frac: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """k samples drawn from the top `elite_frac` of the buffer by reward
        (the `s_theta` training minibatch)."""
        if len(self) == 0:
            raise ValueError("EliteBuffer is empty")
        n_elite = max(1, int(len(self) * float(elite_frac)))
        elite_idx = np.argsort(self._rew)[::-1][:n_elite]
        pick = self._rng.choice(elite_idx, size=int(k), replace=len(elite_idx) < k)
        return (np.stack([self._obs[i] for i in pick]),
                np.stack([self._act[i] for i in pick]),
                self._rew[pick])


__all__ = ["EliteBuffer"]
