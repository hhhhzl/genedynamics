from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from attrdict import AttrDict
from functools import partial
import gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.policies import register_policy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)


class SinWaveBasis(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        n_sin_waves: int,
        actuation_omega: int,
        n_actuators: int,
        actuation_strength: float,
        device: th.device,
    ):
        super(SinWaveBasis, self).__init__()

        self.head_weight = nn.Sequential(nn.Linear(feature_dim, n_actuators * n_sin_waves * len(actuation_omega)))
        self.head_bias = nn.Sequential(nn.Linear(feature_dim, n_actuators))
        self.device = device
        self.to(device)

        self.n_actuators = n_actuators
        self.n_sin_waves = n_sin_waves
        self.actuation_omega = actuation_omega
        self.actuation_strength = actuation_strength

    def forward(self, features: th.Tensor, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        bsize = features.shape[0]
        weight = self.head_weight(features)
        weight = weight.reshape(bsize, self.n_actuators, -1)
        bias = self.head_bias(features)

        time = obs['time'].float()
        x_sinwave = []
        for actuation_omega in self.actuation_omega:
            x_sinwave.append(th.sin(actuation_omega * time + 2 * th.pi / self.n_sin_waves * th.arange(self.n_sin_waves)))
        x_sinwave = th.cat(x_sinwave, dim=-1)
        x_sinwave = x_sinwave.to(self.device)
        act = th.bmm(weight, x_sinwave[..., None])[..., 0]
        act += bias
        act = th.tanh(act) * self.actuation_strength

        return act


class CustomActorCriticPolicy(MultiInputActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        custom_config: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        self.custom_config = custom_config
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        assert isinstance(self.action_dist, DiagGaussianDistribution)

        act_dist_type = self.custom_config['act_dist_type']
        if act_dist_type == 'sin_wave_basis':
            n_sin_waves = self.custom_config.get('n_sin_waves', 4)
            actuation_omega = self.custom_config.get('actuation_omega', [30.])
            n_actuators = self.action_dist.action_dim
            actuation_strength = self.action_space.high[0]
            
            self.action_net = SinWaveBasis(latent_dim_pi, n_sin_waves, actuation_omega, n_actuators, actuation_strength, self.device)
            self.log_std = nn.Parameter(th.ones(self.action_dist.action_dim) * self.log_std_init, requires_grad=True)
        else:
            raise NotImplementedError(f"Unsupported act_dist_type {act_dist_type}")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, obs)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi, obs)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi, obs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, obs: th.Tensor) -> Distribution:
        mean_actions = self.action_net(latent_pi, obs)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")


register_policy('CustomActorCriticPolicy', CustomActorCriticPolicy)
