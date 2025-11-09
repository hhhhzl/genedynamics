from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from enerdynamics.control.edoc import make_energy, make_env


@dataclass
class CEMArgs:
    seed: int = 0
    env_name: str = "double_integrator_box"
    horizon: int = 80
    dt: float = 0.1
    num_samples: int = 512
    num_iterations: int = 5
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
    energy_fn = make_energy(env_name)
    return env, energy_fn


def _rollout(env, energy_fn, state_init: np.ndarray, actions: np.ndarray):
    state = np.asarray(state_init, dtype=np.float32)
    act_seq = np.asarray(actions, dtype=np.float32)

    control_limit = getattr(env, "control_limit", None)
    if control_limit is not None:
        act_seq = np.clip(act_seq, -control_limit, control_limit)

    states = [state]
    rewards = []
    energies = []
    info: Dict = {}
    for t, act in enumerate(act_seq):
        next_state_pred = env.transition(state, act)
        ctx = {"t": t, **info}
        energies.append(float(energy_fn.compute(state, act, ctx)))
        next_state, cost, done, info_next = env.step(
            np.array(next_state_pred, dtype=np.float32),
            np.array(act, dtype=np.float32),
            t,
            info,
        )
        rewards.append(float(-cost))
        states.append(next_state)
        state = next_state
        info = info_next
        if done:
            break

    return (
        np.stack(states, axis=0),
        np.asarray(rewards, dtype=np.float32),
        np.asarray(energies, dtype=np.float32),
    )


def run_cem(args: CEMArgs, initial_state: Optional[np.ndarray] = None):
    rng = np.random.default_rng(args.seed)
    env, energy_fn = _prepare_env(args.env_name, args.dt, args.horizon, args.action_limit)

    if initial_state is None:
        state_init, _ = env.reset()
    else:
        state_init = np.asarray(initial_state, dtype=np.float32)

    act_dim = env.act_dim
    horizon = args.horizon
    control_limit = getattr(env, "control_limit", None)

    mean = np.zeros((horizon, act_dim), dtype=np.float32)
    std = np.full((horizon, act_dim), args.init_std, dtype=np.float32)

    elite_count = max(1, int(args.num_samples * args.elite_frac))
    reward_history = []

    best_actions = mean.copy()
    best_cost = np.inf

    for _ in range(args.num_iterations):
        samples = rng.normal(mean, std, size=(args.num_samples, horizon, act_dim)).astype(np.float32)
        if control_limit is not None:
            samples = np.clip(samples, -control_limit, control_limit)

        costs = np.zeros((args.num_samples,), dtype=np.float32)
        rewards_buffer = np.zeros((args.num_samples,), dtype=np.float32)
        for i in range(args.num_samples):
            _, rewards, _ = _rollout(env, energy_fn, state_init, samples[i])
            total_cost = -float(np.sum(rewards))
            costs[i] = total_cost
            rewards_buffer[i] = float(np.sum(rewards))
            if total_cost < best_cost:
                best_cost = total_cost
                best_actions = samples[i]

        elite_idx = np.argsort(costs)[:elite_count]
        elites = samples[elite_idx]
        mean = np.mean(elites, axis=0)
        std = np.std(elites, axis=0)
        std = np.clip(std, args.min_std, None)

        reward_history.append(float(np.mean(rewards_buffer)))

    if control_limit is not None:
        best_actions = np.clip(best_actions, -control_limit, control_limit)

    states, rewards, energies = _rollout(env, energy_fn, state_init, best_actions)
    result = {
        "actions": np.asarray(best_actions, dtype=np.float32),
        "states": states,
        "rewards": rewards,
        "total_reward": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)) if rewards.size > 0 else 0.0,
        "initial_state": np.asarray(state_init, dtype=np.float32),
        "reward_history": np.asarray(reward_history, dtype=np.float32),
        "energies": energies,
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

