# SPDX-License-Identifier: Apache-2.0
# Derived from LeCAR-Lab/FADA commit 290b3a9c5c0ebe2e9851b61350617544be05e9bc.
# Source: src/holosoma/holosoma/fada/common/lora_utils.py
# Changes: isolate native PyTorch core; replace Holosoma imports with local imports.
from __future__ import annotations

from bisect import bisect_right
import dataclasses
import numpy as np
import torch

@dataclasses.dataclass
class Trajectory:
    obs: np.ndarray
    actions: np.ndarray
    current_command: np.ndarray
    source: str



class TrajectoryWindowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        trajectories: list[Trajectory],
        *,
        history_len: int,
        pred_horizon_k: int,
        include_metadata: bool = False,
    ):
        super().__init__()
        self.trajectories = trajectories
        self.history_len = int(history_len)
        self.pred_horizon_k = int(pred_horizon_k)
        self.include_metadata = bool(include_metadata)
        if self.history_len <= 0:
            raise ValueError("history_len must be positive")
        if self.pred_horizon_k <= 0:
            raise ValueError("pred_horizon_k must be positive")

        self.sample_counts = [max(0, int(traj.obs.shape[0]) - self.pred_horizon_k) for traj in trajectories]
        self.cumsum = np.cumsum(np.asarray(self.sample_counts, dtype=np.int64))
        self.total_samples = int(self.cumsum[-1]) if self.cumsum.size > 0 else 0
        if self.total_samples <= 0:
            raise RuntimeError("No valid windows in trajectory dataset. Collect longer trajectories or reduce K.")

    def __len__(self) -> int:
        return self.total_samples

    def _resolve(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= self.total_samples:
            raise IndexError(f"Index {index} out of range for dataset of size {self.total_samples}")
        traj_idx = int(bisect_right(self.cumsum, index))
        prev = int(self.cumsum[traj_idx - 1]) if traj_idx > 0 else 0
        local_t = int(index - prev)
        return traj_idx, local_t

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        traj_idx, t = self._resolve(index)
        traj = self.trajectories[traj_idx]
        obs = traj.obs
        actions = traj.actions
        current_command_seq = traj.current_command

        history_obs = np.zeros((self.history_len, obs.shape[1]), dtype=np.float32)
        history_act = np.zeros((self.history_len, actions.shape[1]), dtype=np.float32)

        start = max(0, t - self.history_len + 1)
        hist_indices = np.arange(start, t + 1, dtype=np.int64)
        offset = int(self.history_len - hist_indices.shape[0])

        history_obs[offset:] = obs[hist_indices]
        history_valid = np.zeros((self.history_len,), dtype=np.bool_)
        history_valid[offset:] = True

        prev_indices = hist_indices - 1
        valid_prev = prev_indices >= 0
        if np.any(valid_prev):
            valid_slots = np.flatnonzero(valid_prev)
            history_act[offset + valid_slots] = actions[prev_indices[valid_prev]]

        current_command = current_command_seq[t].astype(np.float32, copy=False)
        future_obs = obs[t + 1 : t + 1 + self.pred_horizon_k]
        target_actions = actions[t : t + self.pred_horizon_k]
        if future_obs.shape[0] != self.pred_horizon_k or target_actions.shape[0] != self.pred_horizon_k:
            raise RuntimeError(
                f"Unexpected target horizon at index {index}: "
                f"future_obs={future_obs.shape[0]} target_actions={target_actions.shape[0]} expected={self.pred_horizon_k}"
            )

        sample = {
            "history_obs": torch.from_numpy(history_obs),
            "history_act": torch.from_numpy(history_act),
            "current_command": torch.from_numpy(current_command),
            "history_valid_mask": torch.from_numpy(history_valid),
            "future_observations": torch.from_numpy(future_obs.astype(np.float32, copy=False)),
            "target_actions": torch.from_numpy(target_actions.astype(np.float32, copy=False)),
        }
        if self.include_metadata:
            sample["trajectory_index"] = torch.tensor(traj_idx, dtype=torch.int64)
            sample["window_index"] = torch.tensor(t, dtype=torch.int64)
            sample["source"] = traj.source
        return sample

