from __future__ import annotations

from collections import namedtuple
from typing import Any, Dict

import einops
import torch

from enerdynamics.solvers.single.safediffuser import diffuser_utils as utils

Trajectories = namedtuple("Trajectories", "actions observations")


class Policy:
    """
    Minimal SafeDiffuser policy (dpcc-style location).

    Expects:
      - diffusion_model(conditions) -> (sample, diffusion_paths)
      - normalizer.normalize / unnormalize
      - normalizer.action_dim
    """

    def __init__(self, diffusion_model: Any, normalizer: Any, *, projector: Any | None = None):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.projector = projector
        self.action_dim = int(getattr(normalizer, "action_dim"))

    @property
    def device(self):
        params = list(self.diffusion_model.parameters())
        return params[0].device if len(params) else torch.device("cpu")

    def _format_conditions(self, conditions: Dict[int, Any], batch_size: int):
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, "observations")
        conditions = utils.to_torch(conditions, dtype=torch.float32, device=self.device)
        conditions = utils.apply_dict(
            einops.repeat, conditions, "d -> repeat d", repeat=batch_size
        )
        return conditions

    def __call__(self, conditions: Dict[int, Any], *, batch_size: int = 1):
        conditions = self._format_conditions(conditions, batch_size)

        # Optional hooks used by some diffusion implementations.
        try:
            # Keep a handle to normalizer for safety correction inside diffusion.
            self.diffusion_model.normalizer = self.normalizer
            self.diffusion_model.norm_mins = self.normalizer.normalizers["observations"].mins
            self.diffusion_model.norm_maxs = self.normalizer.normalizers["observations"].maxs
        except Exception:
            pass

        sample, diffusion = self.diffusion_model(conditions, projector=self.projector)

        sample = utils.to_np(sample)
        diffusion = utils.to_np(diffusion)

        actions = sample[:, :, : self.action_dim]
        actions = self.normalizer.unnormalize(actions, "actions")
        action0 = actions[0, 0]

        normed_observations = sample[:, :, self.action_dim :]
        observations = self.normalizer.unnormalize(normed_observations, "observations")

        normed_diffusion = diffusion[:, :, :, self.action_dim :]
        diffusions = self.normalizer.unnormalize(normed_diffusion, "observations")

        trajectories = Trajectories(actions, observations)
        safe1 = getattr(self.diffusion_model, "safe1", 0)
        safe2 = getattr(self.diffusion_model, "safe2", 0)
        elbo = 0
        return action0, trajectories, diffusions, safe1, safe2, elbo

