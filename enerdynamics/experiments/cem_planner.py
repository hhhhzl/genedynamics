from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from enerdynamics.control.edoc import make_energy, make_env


@dataclass
class CEMArgs:
    seed: int = 0
    env_name: str = "double_integrator_box"
    horizon: int = 80
    dt: float = 0.1
    num_samples: int = 512
    num_iterations: int = 6
    elite_frac: float = 0.1
    init_std: float = 0.5
    min_std: float = 0.05
    action_limit: float = 1.0
    verbose: bool = False


def _prepare_env(env_name: str, dt: float, horizon: int, action_limit: float):
    env = make_env(env_name)
    if hasattr(env, "dt"):
        env.dt = dt
    if hasattr(env, "horizon"):
        env.horizon = horizon
    if hasattr(env, "control_limit"):
        env.control_limit = action_limit

    for attr in ("jax_env_transition", "jax_transition"):
        if hasattr(env, attr):
            transition_fn = getattr(env, attr)
            break
    else:
        raise ValueError("Environment must provide a JAX transition function.")
    if not hasattr(env, "jax_cost"):
        raise ValueError("Environment must provide a JAX cost function.")

    transition_fn = jax.jit(transition_fn)
    cost_fn = jax.jit(env.jax_cost)
    energy_fn = make_energy(env_name)
    return env, energy_fn, transition_fn, cost_fn


def _build_rollout_functions(transition_fn, cost_fn):
    @jax.jit
    def rollout_rewards(state_init: jax.Array, actions: jax.Array):
        def step_fn(carry, action):
            next_state = transition_fn(carry, action)
            reward = -cost_fn(next_state)
            return next_state, reward

        _, rewards = jax.lax.scan(step_fn, state_init, actions)
        return rewards

    @jax.jit
    def rollout_states(state_init: jax.Array, actions: jax.Array):
        def step_fn(carry, action):
            next_state = transition_fn(carry, action)
            return next_state, next_state

        _, states = jax.lax.scan(step_fn, state_init, actions)
        return jnp.concatenate([state_init[None, :], states], axis=0)

    rollout_rewards_batch = jax.jit(jax.vmap(rollout_rewards, in_axes=(None, 0)))
    return rollout_rewards, rollout_rewards_batch, rollout_states


def _compute_energies_np(energy_fn, states: np.ndarray, actions: np.ndarray):
    energies = []
    info: Dict = {}
    for t in range(actions.shape[0]):
        energies.append(float(energy_fn.compute(states[t], actions[t], {"t": t, **info})))
    return np.asarray(energies, dtype=np.float32)


def run_cem(args: CEMArgs, initial_state: Optional[np.ndarray] = None):
    env, energy_fn, transition_fn, cost_fn = _prepare_env(
        args.env_name, args.dt, args.horizon, args.action_limit
    )
    rollout_rewards, rollout_rewards_batch, rollout_states = _build_rollout_functions(
        transition_fn, cost_fn
    )

    if initial_state is None:
        state_init_np, _ = env.reset()
    else:
        state_init_np = np.asarray(initial_state, dtype=np.float32)
    state_init = jnp.asarray(state_init_np, dtype=jnp.float32)

    act_dim = env.act_dim
    horizon = args.horizon
    control_limit = getattr(env, "control_limit", None)

    rng = jax.random.PRNGKey(args.seed)
    mean = jnp.zeros((horizon, act_dim), dtype=jnp.float32)
    std = jnp.full((horizon, act_dim), float(args.init_std), dtype=jnp.float32)
    min_std = float(args.min_std)

    elite_count = max(1, int(args.num_samples * args.elite_frac))
    reward_history = []

    best_actions = mean
    best_return = -jnp.inf

    for _ in range(max(1, int(args.num_iterations))):
        rng, sample_key = jax.random.split(rng)
        samples = jax.random.normal(
            sample_key, (args.num_samples, horizon, act_dim), dtype=jnp.float32
        ) * std[None, :, :] + mean[None, :, :]
        if control_limit is not None:
            samples = jnp.clip(samples, -control_limit, control_limit)

        rewards_seq = rollout_rewards_batch(state_init, samples)
        total_returns = jnp.sum(rewards_seq, axis=1)

        total_returns_np = np.asarray(total_returns)
        samples_np = np.asarray(samples)

        best_idx = int(np.argmax(total_returns_np))
        if total_returns_np[best_idx] > float(best_return):
            best_return = total_returns[best_idx]
            best_actions = samples[best_idx]

        elite_idx = np.argsort(total_returns_np)[-elite_count:]
        elites = samples_np[elite_idx]
        mean = jnp.asarray(np.mean(elites, axis=0), dtype=jnp.float32)
        std = jnp.asarray(np.clip(np.std(elites, axis=0), min_std, None), dtype=jnp.float32)

        reward_history.append(float(np.mean(total_returns_np)))

    if control_limit is not None:
        best_actions = jnp.clip(best_actions, -control_limit, control_limit)

    rewards_final = rollout_rewards(state_init, best_actions)
    states_final = rollout_states(state_init, best_actions)

    actions_np = np.asarray(best_actions, dtype=np.float32)
    states_np = np.asarray(states_final, dtype=np.float32)
    rewards_np = np.asarray(rewards_final, dtype=np.float32)
    energies_np = _compute_energies_np(energy_fn, states_np, actions_np)

    result = {
        "actions": actions_np,
        "states": states_np,
        "rewards": rewards_np,
        "total_reward": float(np.sum(rewards_np)),
        "mean_reward": float(np.mean(rewards_np)) if rewards_np.size > 0 else 0.0,
        "initial_state": state_init_np,
        "reward_history": np.asarray(reward_history, dtype=np.float32),
        "energies": energies_np,
    }
    if args.verbose:
        print(f"CEM total reward: {result['total_reward']:.3f}")
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser("CEM planner for enerdynamics environments.")
    parser.add_argument("--seed", type=int, default=CEMArgs.seed)
    parser.add_argument("--env_name", type=str, default=CEMArgs.env_name)
    parser.add_argument("--horizon", type=int, default=CEMArgs.horizon)
    parser.add_argument("--dt", type=float, default=CEMArgs.dt)
    parser.add_argument("--num_samples", type=int, default=CEMArgs.num_samples)
    parser.add_argument("--num_iterations", type=int, default=CEMArgs.num_iterations)
    parser.add_argument("--elite_frac", type=float, default=CEMArgs.elite_frac)
    parser.add_argument("--init_std", type=float, default=CEMArgs.init_std)
    parser.add_argument("--min_std", type=float, default=CEMArgs.min_std)
    parser.add_argument("--action_limit", type=float, default=CEMArgs.action_limit)
    parser.add_argument("--verbose", action="store_true", default=False)

    cli_args = parser.parse_args()
    args = CEMArgs(
        seed=cli_args.seed,
        env_name=cli_args.env_name,
        horizon=cli_args.horizon,
        dt=cli_args.dt,
        num_samples=cli_args.num_samples,
        num_iterations=cli_args.num_iterations,
        elite_frac=cli_args.elite_frac,
        init_std=cli_args.init_std,
        min_std=cli_args.min_std,
        action_limit=cli_args.action_limit,
        verbose=cli_args.verbose,
    )
    run_cem(args)


if __name__ == "__main__":
    main()
