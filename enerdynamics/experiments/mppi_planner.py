from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from enerdynamics.control.edoc import make_energy, make_env


@dataclass
class MPPIArgs:
    seed: int = 0
    env_name: str = "double_integrator_box"
    horizon: int = 80
    dt: float = 0.1
    num_samples: int = 512
    num_iterations: int = 5
    noise_sigma: float = 0.3
    lambda_: float = 1.0
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


def run_mppi(args: MPPIArgs, initial_state: Optional[np.ndarray] = None):
    rng = np.random.default_rng(args.seed)
    env, energy_fn = _prepare_env(args.env_name, args.dt, args.horizon, args.action_limit)

    if initial_state is None:
        state_init, _ = env.reset()
    else:
        state_init = np.asarray(initial_state, dtype=np.float32)

    act_dim = env.act_dim
    horizon = args.horizon
    control_limit = getattr(env, "control_limit", None)

    mean_actions = np.zeros((horizon, act_dim), dtype=np.float32)

    best_actions = mean_actions.copy()
    best_cost = np.inf
    reward_history = []

    for _ in range(args.num_iterations):
        noise = rng.normal(0.0, args.noise_sigma, size=(args.num_samples, horizon, act_dim)).astype(
            np.float32
        )
        candidates = mean_actions[None, :, :] + noise
        if control_limit is not None:
            candidates = np.clip(candidates, -control_limit, control_limit)

        costs = np.zeros((args.num_samples,), dtype=np.float32)
        rewards_buffer = np.zeros((args.num_samples,), dtype=np.float32)
        for i in range(args.num_samples):
            _, rewards, _ = _rollout(env, energy_fn, state_init, candidates[i])
            total_cost = -float(np.sum(rewards))
            costs[i] = total_cost
            rewards_buffer[i] = float(np.sum(rewards))
            if total_cost < best_cost:
                best_cost = total_cost
                best_actions = candidates[i]

        beta = np.min(costs)
        weights = np.exp(-(costs - beta) / max(args.lambda_, 1e-6))
        weights_sum = np.sum(weights) + 1e-8
        mean_actions = np.einsum("i,ijk->jk", weights, candidates) / weights_sum

        reward_history.append(float(np.mean(rewards_buffer)))

    if control_limit is not None:
        mean_actions = np.clip(mean_actions, -control_limit, control_limit)

    final_actions = mean_actions if best_cost == np.inf else best_actions
    states, rewards, energies = _rollout(env, energy_fn, state_init, final_actions)
    result = {
        "actions": np.asarray(final_actions, dtype=np.float32),
        "states": states,
        "rewards": rewards,
        "total_reward": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)) if rewards.size > 0 else 0.0,
        "initial_state": np.asarray(state_init, dtype=np.float32),
        "reward_history": np.asarray(reward_history, dtype=np.float32),
        "energies": energies,
    }
    if args.verbose:
        print(f"MPPI total reward: {result['total_reward']:.3f}")
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser("MPPI planner for enerdynamics environments.")
    parser.add_argument("--seed", type=int, default=MPPIArgs.seed)
    parser.add_argument("--env_name", type=str, default=MPPIArgs.env_name)
    parser.add_argument("--horizon", type=int, default=MPPIArgs.horizon)
    parser.add_argument("--dt", type=float, default=MPPIArgs.dt)
    parser.add_argument("--num_samples", type=int, default=MPPIArgs.num_samples)
    parser.add_argument("--num_iterations", type=int, default=MPPIArgs.num_iterations)
    parser.add_argument("--noise_sigma", type=float, default=MPPIArgs.noise_sigma)
    parser.add_argument("--lambda_", type=float, default=MPPIArgs.lambda_)
    parser.add_argument("--action_limit", type=float, default=MPPIArgs.action_limit)
    parser.add_argument("--verbose", action="store_true", default=False)

    cli_args = parser.parse_args()
    args = MPPIArgs(
        seed=cli_args.seed,
        env_name=cli_args.env_name,
        horizon=cli_args.horizon,
        dt=cli_args.dt,
        num_samples=cli_args.num_samples,
        num_iterations=cli_args.num_iterations,
        noise_sigma=cli_args.noise_sigma,
        lambda_=cli_args.lambda_,
        action_limit=cli_args.action_limit,
        verbose=cli_args.verbose,
    )
    run_mppi(args)


if __name__ == "__main__":
    main()

