from collections import namedtuple

import einops
import numpy as np
import torch

import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn
from diffuser.utils.arrays import to_device

Trajectories = namedtuple("Trajectories", "actions observations")


class Policy:

    def __init__(
        self,
        model,
        normalizer,
        scheduler=None,
        preprocess_fns=None,
        test_ret=0,
        projector=None,
        trajectory_selection="random",
        **sample_kwargs,
    ):
        self.model = model
        self.scheduler = scheduler
        self.normalizer = normalizer
        self.action_dim = model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns or [])
        self.test_ret = test_ret
        self.sample_kwargs = sample_kwargs

        self.inverse_dynamics = False
        self.projector = projector
        self.trajectory_selection = trajectory_selection
        self.prev_observations = None

    def __call__(
        self,
        conditions,
        batch_size=1,
        horizon=16,
        test_ret=None,
        constraints=None,
        disable_projection=False,
    ):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        test_ret = test_ret if test_ret is not None else self.test_ret
        returns = to_device(test_ret * torch.ones(batch_size, 1), "cuda")

        projector = self.projector if not disable_projection else None
        samples, infos = self.model(
            conditions,
            returns=returns,
            projector=projector,
            constraints=constraints,
            horizon=horizon,
            **self.sample_kwargs,
        )

        trajectories = utils.to_np(samples)
        if "diffusion" not in infos:
            normed_observations = trajectories[:, :, self.action_dim :]
            observations = self.normalizer.unnormalize(normed_observations, "observations")
        else:
            diffusion_trajectories = utils.to_np(infos["diffusion"])
            observations = self.normalizer.unnormalize(
                diffusion_trajectories[:, :, :, self.action_dim :], "observations"
            )

        if (
            self.trajectory_selection == "temporal_consistency"
            and not disable_projection
            and self.prev_observations is not None
        ):
            order = np.argsort(
                np.linalg.norm(
                    observations[:, :-1, :] - self.prev_observations[:, 1:, :], axis=(1, 2)
                )
            )
            which_trajectory = order[0]
            observations = observations[order]
        elif self.trajectory_selection == "minimum_projection_cost" and not disable_projection:
            costs_total = np.zeros(batch_size)
            for _, cost in infos["projection_costs"].items():
                costs_total += cost
            which_trajectory = np.argmin(costs_total)
        else:
            which_trajectory = 0
        self.prev_observations = np.repeat(np.expand_dims(observations[0], axis=0), batch_size, axis=0)

        actions = trajectories[:, :, : self.action_dim]
        actions = self.normalizer.unnormalize(actions, "actions")
        action = actions[which_trajectory, 0]

        trajectories = Trajectories(actions, observations)
        return action, trajectories

    @property
    def device(self):
        parameters = list(self.model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, "observations")
        conditions = utils.to_torch(conditions, dtype=torch.float32, device="cpu")
        conditions = utils.apply_dict(
            einops.repeat, conditions, "d -> repeat d", repeat=batch_size
        )
        return conditions

