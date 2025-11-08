from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from enerdynamics.control.edoc import make_energy
from enerdynamics.envs.double_integrator_box import DoubleIntegratorBoxEnv


class EnvParams(NamedTuple):
    dt: float
    p_max: float
    v_max: float
    target: float
    vel_weight: float
    control_limit: float


@dataclass
class DiffusionArgs:
    seed: int = 0
    horizon: int = 80
    dt: float = 0.1
    Nsample: int = 2048
    Ndiffuse: int = 100
    temp_sample: float = 0.1
    beta0: float = 1e-4
    betaT: float = 1e-2
    action_limit: float = 1.0
    verbose: bool = True


def transition(state: jax.Array, action: jax.Array, params: EnvParams) -> jax.Array:
    u = jnp.clip(action[..., 0], -params.control_limit, params.control_limit)
    p = state[..., 0]
    v = state[..., 1]
    v_next = v + params.dt * u
    p_next = p + params.dt * v_next
    p_next = jnp.clip(p_next, -params.p_max, params.p_max)
    v_next = jnp.clip(v_next, -params.v_max, params.v_max)
    return jnp.stack([p_next, v_next], axis=-1)


def state_cost(state: jax.Array, params: EnvParams) -> jax.Array:
    pos_err = (state[..., 0] - params.target) ** 2
    vel_err = params.vel_weight * (state[..., 1] ** 2)
    return pos_err + vel_err


@jax.jit
def rollout_rewards(state_init: jax.Array, actions: jax.Array, params: EnvParams) -> jax.Array:
    def step_fn(carry, action):
        next_state = transition(carry, action, params)
        reward = -state_cost(next_state, params)
        return next_state, reward

    _, rewards = jax.lax.scan(step_fn, state_init, actions)
    return rewards


@jax.jit
def rollout_states(state_init: jax.Array, actions: jax.Array, params: EnvParams) -> jax.Array:
    def step_fn(carry, action):
        next_state = transition(carry, action, params)
        return next_state, next_state

    _, states = jax.lax.scan(step_fn, state_init, actions)
    return jnp.concatenate([state_init[None, :], states], axis=0)


def run_diffusion(args: DiffusionArgs):
    env = DoubleIntegratorBoxEnv(
        dt=args.dt,
        horizon=args.horizon,
        control_limit=args.action_limit,
    )
    params = EnvParams(
        dt=env.dt,
        p_max=env.p_max,
        v_max=env.v_max,
        target=env.target,
        vel_weight=env.vel_weight,
        control_limit=env.control_limit,
    )
    energy_fn = make_energy("double_integrator_box")

    rng = jax.random.PRNGKey(args.seed)
    state_init_np, _ = env.reset(rng)
    state_init = jnp.asarray(state_init_np, dtype=jnp.float32)
    rng, diffuse_rng = jax.random.split(rng)

    horizon = args.horizon
    action_dim = env.act_dim
    Nsample = args.Nsample
    temp_sample = args.temp_sample
    control_limit = params.control_limit

    betas = jnp.linspace(args.beta0, args.betaT, args.Ndiffuse, dtype=jnp.float32)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1.0 - alphas_bar)
    diffusion_indices = jnp.arange(args.Ndiffuse - 1, 0, -1, dtype=jnp.int32)

    def reverse_diffuse(rng_in, Ybar_init):
        def body(carry, idx):
            rng_curr, Ybar_curr = carry
            rng_curr, noise_key = jax.random.split(rng_curr)

            Yi = Ybar_curr * jnp.sqrt(alphas_bar[idx])
            eps = jax.random.normal(
                noise_key, (Nsample, horizon, action_dim), dtype=jnp.float32
            )
            Y0s = eps * sigmas[idx] + Ybar_curr
            Y0s = jnp.clip(Y0s, -control_limit, control_limit)

            rews = jax.vmap(lambda acts: rollout_rewards(state_init, acts, params))(Y0s)
            rews_mean = jnp.mean(rews, axis=-1)

            rew_mean = jnp.mean(rews_mean)
            rew_std = jnp.std(rews_mean)
            rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)

            logp0 = (rews_mean - rew_mean) / (rew_std * temp_sample)
            weights = jax.nn.softmax(logp0)
            Ybar_weighted = jnp.einsum("n,nij->ij", weights, Y0s)

            score = (-Yi + jnp.sqrt(alphas_bar[idx]) * Ybar_weighted) / (
                1.0 - alphas_bar[idx]
            )
            Yim1 = (Yi + (1.0 - alphas_bar[idx]) * score) / jnp.sqrt(alphas[idx])
            Ybar_next = Yim1 / jnp.sqrt(alphas_bar[idx - 1])

            return (rng_curr, Ybar_next), jnp.mean(rews_mean)

        (rng_out, Ybar_final), reward_hist = jax.lax.scan(
            body, (rng_in, Ybar_init), diffusion_indices
        )
        return rng_out, Ybar_final, reward_hist

    reverse_diffuse_jit = jax.jit(reverse_diffuse)

    Ybar_init = jnp.zeros((horizon, action_dim), dtype=jnp.float32)
    diffuse_rng, Ybar_final, reward_history = reverse_diffuse_jit(diffuse_rng, Ybar_init)

    final_actions = jnp.clip(Ybar_final, -control_limit, control_limit)
    rewards = rollout_rewards(state_init, final_actions, params)
    states = rollout_states(state_init, final_actions, params)
    total_reward = rewards.sum()
    mean_reward = rewards.mean()

    energy_list = []
    for t in range(final_actions.shape[0]):
        state_t = states[t]
        action_t = final_actions[t]
        ctx = {"t": int(t)}
        E_val = energy_fn.compute(state_t, action_t, ctx)
        energy_list.append(E_val)
    energies = jnp.asarray(energy_list, dtype=jnp.float32)
    total_energy = energies.sum()
    final_energy = energies[-1] if energies.size > 0 else jnp.asarray(0.0, dtype=jnp.float32)

    result = {
        "actions": final_actions,
        "states": states,
        "rewards": rewards,
        "total_reward": total_reward,
        "mean_reward": mean_reward,
        "initial_state": state_init,
        "reward_history": reward_history[::-1],
        "energies": energies,
    }

    if args.verbose:
        print("initial state:", jnp.asarray(state_init))
        print("final state:", jnp.asarray(states[-1]))
        print("total reward:", float(total_reward))
        print("mean per-step reward:", float(mean_reward))
        print("total energy:", float(total_energy))
        print("final energy:", float(final_energy))

    return result


def main():
    parser = argparse.ArgumentParser("Diffusion planner for the double-integrator box environment.")
    parser.add_argument("--seed", type=int, default=DiffusionArgs.seed)
    parser.add_argument("--horizon", type=int, default=DiffusionArgs.horizon)
    parser.add_argument("--dt", type=float, default=DiffusionArgs.dt)
    parser.add_argument("--Nsample", type=int, default=DiffusionArgs.Nsample)
    parser.add_argument("--Ndiffuse", type=int, default=DiffusionArgs.Ndiffuse)
    parser.add_argument("--temp_sample", type=float, default=DiffusionArgs.temp_sample)
    parser.add_argument("--beta0", type=float, default=DiffusionArgs.beta0)
    parser.add_argument("--betaT", type=float, default=DiffusionArgs.betaT)
    parser.add_argument("--action_limit", type=float, default=DiffusionArgs.action_limit)
    parser.add_argument("--no_verbose", action="store_true")

    cli_args = parser.parse_args()
    diff_args = DiffusionArgs(
        seed=cli_args.seed,
        horizon=cli_args.horizon,
        dt=cli_args.dt,
        Nsample=cli_args.Nsample,
        Ndiffuse=cli_args.Ndiffuse,
        temp_sample=cli_args.temp_sample,
        beta0=cli_args.beta0,
        betaT=cli_args.betaT,
        action_limit=cli_args.action_limit,
        verbose=not cli_args.no_verbose,
    )
    run_diffusion(diff_args)


if __name__ == "__main__":
    main()

