from collections import namedtuple
import torch
import time
import einops
import numpy as np
import diffuser.utils as utils
from diffuser.models.helpers import apply_conditioning
from diffuser.utils.arrays import to_device
from diffuser.datasets.preprocessing import get_policy_preprocess_fn

Trajectories = namedtuple('Trajectories', 'actions observations')


class Policy:

    def __init__(self, model, normalizer, scheduler=None, preprocess_fns=[], test_ret=0, projector=None, 
                 trajectory_selection='random', **sample_kwargs):
        self.model = model
        self.scheduler = scheduler,   # 'DDPM' or 'DDIM'
        self.normalizer = normalizer
        self.action_dim = model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.test_ret = test_ret
        self.sample_kwargs = sample_kwargs

        # Inverse dynamics model
        if self.model.__class__.__name__ == 'GaussianInvDynDiffusion':
            self.inverse_dynamics = True
            self.inv_model = self.model.inv_model
            self.action_dim = 0
        else:
            self.inverse_dynamics = False

        # Projector
        self.projector = projector

        # Trajectory selection
        self.trajectory_selection = trajectory_selection        # 'random' or 'temporal_consistency' or 'minimum_projection_cost'

        # Previous observations
        self.prev_observations = None

    def __call__(self, conditions, batch_size=1, horizon=16, test_ret=None, constraints=None, disable_projection=False):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        test_ret = test_ret if test_ret is not None else self.test_ret
        returns = to_device(test_ret * torch.ones(batch_size, 1), 'cuda')

        # Use GaussianDiffusion model with DDPM
        projector = self.projector if not disable_projection else None
        samples, infos = self.model(conditions, returns=returns, projector=projector, constraints=constraints, horizon=horizon, **self.sample_kwargs)

        trajectories = utils.to_np(samples)

        ## extract observations [ batch_size x horizon x observation_dim ]
        if not 'diffusion' in infos:
            normed_observations = trajectories[:, :, self.action_dim:]
            observations = self.normalizer.unnormalize(normed_observations, 'observations')
        if 'diffusion' in infos:
            diffusion_trajectories = utils.to_np(infos['diffusion'])         # Shape: batch_size x T x horizon x transition_dim     
            observations = self.normalizer.unnormalize(diffusion_trajectories[:, :, :, self.action_dim:], 'observations')
        
        # Sort according to similarity with previous observations
        if self.trajectory_selection == 'temporal_consistency' and not disable_projection and self.prev_observations is not None:   # Temporal consistency
            order = np.argsort(np.linalg.norm(observations[:,:-1,:] - self.prev_observations[:,1:,:], axis=(1,2)))
            which_trajectory = order[0]
            observations = observations[order]
        elif self.trajectory_selection == 'minimum_projection_cost' and not disable_projection:                                     # Minimum projection cost
            costs_total = np.zeros(batch_size)
            for timestep, cost in infos['projection_costs'].items():
                costs_total += cost
            which_trajectory = np.argmin(costs_total)
        else:                                                                                                                       # Random selection
            which_trajectory = 0
        self.prev_observations = np.repeat(np.expand_dims(observations[0], axis=0), batch_size, axis=0)

        ## Extract or calculate action
        if self.inverse_dynamics:
            obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
            obs_comb = obs_comb.reshape(-1, 2*samples.shape[-1])
            actions = self.inv_model(obs_comb)
            actions = utils.to_np(actions)
            actions = self.normalizer.unnormalize(actions, 'actions')
            action = actions[which_trajectory]
        else:
            ## extract action [ batch_size x horizon x action_dim ]
            actions = trajectories[:, :, :self.action_dim]
            actions = self.normalizer.unnormalize(actions, 'actions')

            ## extract first action
            action = actions[which_trajectory, 0]

        trajectories = Trajectories(actions, observations)

        return action, trajectories

    @property
    def device(self):
        parameters = list(self.model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cpu')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions
    