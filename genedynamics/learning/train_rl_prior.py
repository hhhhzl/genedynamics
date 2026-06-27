"""Train an RL policy prior with brax (PPO), then wrap it as an `RLPrior`.

The JAX RL prior integrates brax's training, so training
is brax-native and runs in the docker brax image. `train_rl_prior` returns the
trained `params` + the network `config`; `build_rl_prior` reconstructs an
`RLPrior` (jax/brax backend) from them — the SAME network config must be used for
inference, so it is carried alongside the params.
"""

from __future__ import annotations

import functools
from typing import Any, Dict, Sequence, Tuple


def train_rl_prior(
    env: Any,
    *,
    num_timesteps: int = 200_000,
    episode_length: int = 200,
    num_envs: int = 64,
    policy_hidden_layer_sizes: Sequence[int] = (32, 32, 32, 32),
    normalize_observations: bool = True,
    learning_rate: float = 3e-4,
    seed: int = 0,
    **ppo_kwargs: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """PPO-train on a brax env; return (params, config). config carries what
    `build_rl_prior` needs to rebuild the identical network for inference."""
    from brax.training.agents.ppo import train as ppo_train
    from brax.training.agents.ppo import networks as ppo_networks

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(policy_hidden_layer_sizes),
    )
    _, params, _ = ppo_train.train(
        environment=env,
        num_timesteps=int(num_timesteps),
        episode_length=int(episode_length),
        num_envs=int(num_envs),
        learning_rate=float(learning_rate),
        normalize_observations=bool(normalize_observations),
        network_factory=network_factory,
        seed=int(seed),
        **ppo_kwargs,
    )
    config = {
        "observation_size": int(env.observation_size),
        "action_size": int(env.action_size),
        "policy_hidden_layer_sizes": tuple(policy_hidden_layer_sizes),
        "normalize_observations": bool(normalize_observations),
    }
    return params, config


def build_rl_prior(params: Any, config: Dict[str, Any], *, n_warm_nodes: int = 5,
                   deterministic: bool = True):
    """Reconstruct an `RLPrior` (jax/brax backend) from trained params + config."""
    from genedynamics.learning.priors.rl import RLPrior
    return RLPrior(
        backend="jax",
        observation_size=config["observation_size"],
        action_size=config["action_size"],
        params=params,
        n_warm_nodes=int(n_warm_nodes),
        normalize_observations=config["normalize_observations"],
        policy_hidden_layer_sizes=config["policy_hidden_layer_sizes"],
        deterministic=deterministic,
    )


__all__ = ["train_rl_prior", "build_rl_prior"]
