"""Train a standalone RL POLICY (PPO / SAC) on a brax env + checkpoint I/O.

Sibling of ``train_rl_prior`` (which trains a warm-start prior for MDAC): this trains a
policy to be deployed as a STANDALONE closed-loop baseline controller (the engine for the
ATACOM / ISSA RL baselines, which add a projection layer at deploy). brax-native, runs in
the docker brax image; ``(params, config)`` are pickled to a checkpoint and rebuilt at
deploy via :class:`genedynamics.learning.priors.rl.backends.brax_jax.BraxRLPrior` (PPO) or
its SAC analog. The same network config MUST be used for inference, so it is saved alongside.
"""

from __future__ import annotations

import functools
import os
import pickle
from typing import Any, Dict, Sequence, Tuple


def train_rl_policy(
    env: Any,
    *,
    algo: str = "ppo",
    num_timesteps: int = 200_000,
    episode_length: int = 200,
    num_envs: int = 64,
    policy_hidden_layer_sizes: Sequence[int] = (32, 32, 32, 32),
    normalize_observations: bool = True,
    learning_rate: float = 3e-4,
    seed: int = 0,
    warmup_steps: int = 5_000,
    **train_kwargs: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """Train a brax PPO or SAC policy on ``env``; return (params, config). config carries
    what the deploy side needs to rebuild the identical inference network.

    ``warmup_steps`` (SAC only): the replay buffer is PREFILLED for this many env steps —
    stored, NO gradient updates — before learning starts (brax ``min_replay_size``; brax's
    default is 0 = NO warm-up, which destabilizes SAC). Ignored for on-policy PPO. Anything
    in ``train_kwargs`` overrides these defaults (incl. ``min_replay_size``)."""
    algo = str(algo).lower()
    hidden = tuple(policy_hidden_layer_sizes)
    if algo == "ppo":
        from brax.training.agents.ppo import train as _train
        from brax.training.agents.ppo import networks as _nets
        network_factory = functools.partial(_nets.make_ppo_networks,
                                             policy_hidden_layer_sizes=hidden)
        extra: Dict[str, Any] = {}
    elif algo == "sac":
        from brax.training.agents.sac import train as _train
        from brax.training.agents.sac import networks as _nets
        network_factory = functools.partial(_nets.make_sac_networks,
                                             hidden_layer_sizes=hidden)
        # SAC warm-up: prefill the replay buffer (store-only, no updates) before learning.
        extra = {"min_replay_size": int(warmup_steps)}
    else:
        raise ValueError(f"algo must be 'ppo' or 'sac', got {algo!r}")
    extra.update(train_kwargs)                          # explicit caller kwargs win

    _, params, _ = _train.train(
        environment=env,
        num_timesteps=int(num_timesteps),
        episode_length=int(episode_length),
        num_envs=int(num_envs),
        learning_rate=float(learning_rate),
        normalize_observations=bool(normalize_observations),
        network_factory=network_factory,
        seed=int(seed),
        **extra,
    )
    config = {
        "algo": algo,
        "observation_size": int(env.observation_size),
        "action_size": int(env.action_size),
        "policy_hidden_layer_sizes": hidden,
        "normalize_observations": bool(normalize_observations),
    }
    return params, config


def save_policy(path: str, params: Any, config: Dict[str, Any]) -> str:
    """Pickle (params, config) to ``path`` (params are jax pytrees of host arrays)."""
    import jax
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    host_params = jax.device_get(params)               # pull off-device for portable pickling
    with open(path, "wb") as f:
        pickle.dump({"params": host_params, "config": config}, f)
    return path


def load_policy(path: str) -> Tuple[Any, Dict[str, Any]]:
    with open(path, "rb") as f:
        blob = pickle.load(f)
    return blob["params"], blob["config"]


def build_policy_act(params: Any, config: Dict[str, Any], *, deterministic: bool = True):
    """Rebuild an ``act(obs, key) -> action`` inference fn from (params, config).

    PPO reuses ``BraxRLPrior``; SAC builds the analogous brax sac inference fn."""
    algo = config.get("algo", "ppo")
    if algo == "ppo":
        from genedynamics.learning.priors.rl.backends.brax_jax import BraxRLPrior
        prior = BraxRLPrior(
            observation_size=config["observation_size"],
            action_size=config["action_size"], params=params,
            normalize_observations=config["normalize_observations"],
            policy_hidden_layer_sizes=config["policy_hidden_layer_sizes"],
            deterministic=deterministic,
        )
        return lambda obs, key: prior.act(obs, key=key, deterministic=deterministic)
    # SAC inference via brax sac networks
    from brax.training.agents.sac import networks as sac_nets
    from brax.training.acme import running_statistics
    import jax.numpy as jnp
    preprocess = running_statistics.normalize if config["normalize_observations"] else (lambda x, y: x)
    nets = sac_nets.make_sac_networks(
        observation_size=config["observation_size"], action_size=config["action_size"],
        preprocess_observations_fn=preprocess,
        hidden_layer_sizes=config["policy_hidden_layer_sizes"],
    )
    make_policy = sac_nets.make_inference_fn(nets)
    policy = make_policy(params, deterministic=deterministic)
    return lambda obs, key: policy(jnp.asarray(obs), key)[0]


__all__ = ["train_rl_policy", "save_policy", "load_policy", "build_policy_act"]
