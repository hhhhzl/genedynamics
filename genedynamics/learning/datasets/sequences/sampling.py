"""Windows remain inside a single episode; padding and burn-in are explicit."""

import numpy as np
from .schema import SequenceBatch


class SequenceSampler:
    def __init__(self, episodes, seed=0):
        self.episodes = tuple(episodes)
        if not self.episodes:
            raise ValueError("Cannot sample an empty dataset")
        self.random = np.random.default_rng(seed)
        fields = set(self.episodes[0].arrays)
        if any(set(episode.arrays) != fields for episode in self.episodes):
            raise ValueError("Sequence fields must have a common schema")

    def sample(self, batch_size, length, burn_in=0):
        if batch_size < 1 or length < 1 or burn_in < 0:
            raise ValueError("Invalid sequence sampling dimensions")
        total = length + burn_in
        arrays, masks, identifiers, starts = {}, [], [], []
        for _ in range(batch_size):
            episode = self.episodes[int(self.random.integers(len(self.episodes)))]
            start = int(self.random.integers(len(episode)))
            indices = start + np.arange(total)
            valid = indices < len(episode)
            for name, value in episode.arrays.items():
                arrays.setdefault(name, []).append(value[np.minimum(indices, len(episode) - 1)])
            masks.append(valid)
            identifiers.append(episode.identifier)
            starts.append(start)
        valid = np.stack(masks)
        loss_mask = valid.copy()
        loss_mask[:, :burn_in] = False
        return SequenceBatch({name: np.stack(values) for name, values in arrays.items()},
                             valid, loss_mask, tuple(identifiers), np.asarray(starts))
