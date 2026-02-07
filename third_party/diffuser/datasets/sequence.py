from collections import namedtuple
import numpy as np
import torch

from .d4rl import sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer
from .preprocessing import get_preprocess_fn

RewardBatch = namedtuple('RewardBatch', 'trajectories conditions returns')
Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64, normalizer='LimitsNormalizer', 
                 max_path_length=100, max_n_episodes=100000, termination_penalty=0, preprocess_fns=[],
                 use_padding=False, discount=0.99, returns_scale=100, include_returns=False):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding

        # Rewards
        self.returns_scale = returns_scale
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.include_returns = include_returns

        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            if i >= max_n_episodes:
                break
            fields.add_path(episode)
            # Don't pad with zeros, instead use the last observation
            if use_padding:
                path_length = fields['path_lengths'][i]
                fields['observations'][i, path_length:] = fields['observations'][i, path_length-1]
                fields['actions'][i, path_length:] = fields['actions'][i, path_length-1]

        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.get_goal_dim()
        self.pad_goals()
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        # return {0: observations[0], 'goal': observations}
        return {0: observations[0]}
        # return {}
    
    def get_goal_dim(self):
        '''
            Get the number of dimensions in the observations that are constant across the entire plan
        '''
        idx = np.argmax(self.fields.path_lengths > 1)

        # self.goal_dim = (self.fields.observations[idx, 0] == self.fields.observations[idx, 1]).sum()
        self.goal_dim = (self.fields.observations[idx].std(axis=0) == 0).sum()

    def pad_goals(self):
        '''
            Pad goals to be the same in the padded interval as during the episode
        '''
        for i in range(self.n_episodes):
            self.fields.observations[i, self.path_lengths[i]:, -self.goal_dim:] = self.fields.observations[i, self.path_lengths[i]-1, -self.goal_dim:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        # if self.fields.path_lengths[path_ind] < end:
        #     print(f'Warning: path length {self.fields.path_lengths[path_ind]} is less than horizon {end}')

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        trajectories = np.concatenate([actions, observations], axis=-1)
        conditions = self.get_conditions(observations)

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start:]
            # rewards = self.fields.rewards[path_ind, start:end]
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns/self.returns_scale], dtype=np.float32)      # Maybe better self.returns_scale = self.discounts.sum()?
            batch = RewardBatch(trajectories, conditions, returns)
            # batch = Batch(trajectories, conditions)
        else:
            batch = Batch(trajectories, conditions)
        return batch


class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }


class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]
        self.normed = False
        if normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True

    def _get_bounds(self):
        # print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.__getitem__(i).values.item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        # print('✓')
        return vmin, vmax

    def normalize_value(self, value):
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed = normed * 2 - 1
        return normed

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        if self.normed:
            value = self.normalize_value(value)
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch
