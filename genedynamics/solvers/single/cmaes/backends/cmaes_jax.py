"""
JAX backend for CMA-ES.

Rollout machinery is identical to the CEM JAX backend (jitted transition + cost,
vmapped reward rollout). Only the update rule differs: a separable (diagonal-
covariance) CMA-ES over the flattened action sequence — rank-mu weighted mean
update + diagonal covariance adaptation + 1/5-rule step-size control.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.core.types import Trajectory, State


class CMAESBackendJax:
    """JAX implementation of (separable) CMA-ES over the action sequence."""

    def __init__(
        self,
        *,
        env_adapter,
        legacy_energy,
        horizon: int,
        dt: float,
        num_samples: int,
        num_iterations: int,
        elite_frac: float,
        sigma0: float,
        min_std: float,
        action_limit: float,
        seed: int = 0,
    ):
        self.env = env_adapter
        self.energy = legacy_energy
        self.horizon = horizon
        self.dt = dt
        self.popsize = int(num_samples)
        self.generations = int(num_iterations)
        self.elite_frac = float(elite_frac)
        self.sigma0 = float(sigma0)
        self.min_std = float(min_std)
        self.action_limit = action_limit
        self.seed = seed
        self.act_dim = self.env.act_dim
        self._build_jax_functions()

    def _build_jax_functions(self) -> None:
        def transition_fn(state, action):
            return self.env.jax_transition(state, action)

        def cost_fn(state, action, ctx):
            return self.energy.compute(state, action, ctx)

        self._transition_fn = jax.jit(transition_fn)
        self._cost_fn = jax.jit(cost_fn)

        def rollout_rewards(state_init, actions):
            def step_fn(carry, action):
                next_state = self._transition_fn(carry, action)
                reward = -self._cost_fn(next_state, action, {"t": 0})
                return next_state, reward

            _, rewards = jax.lax.scan(step_fn, state_init, actions)
            return rewards

        self._rollout_rewards_fn = jax.jit(rollout_rewards)
        self._rollout_rewards_batch_fn = jax.jit(jax.vmap(self._rollout_rewards_fn, in_axes=(None, 0)))

        def rollout_states(state_init, actions):
            def step_fn(carry, action):
                next_state = self._transition_fn(carry, action)
                return next_state, next_state

            _, states = jax.lax.scan(step_fn, state_init, actions)
            return jnp.concatenate([state_init[None, :], states], axis=0)

        self._rollout_states_fn = jax.jit(rollout_states)

    def _returns(self, x0_jnp, flat_pop):
        """flat_pop: (P, H*A) -> mode-averaged total returns (P,)."""
        P = flat_pop.shape[0]
        acts = jnp.asarray(flat_pop, dtype=jnp.float32).reshape(P, self.horizon, self.act_dim)
        if self.action_limit is not None:
            acts = jnp.clip(acts, -self.action_limit, self.action_limit)
        rewards_seq = self._rollout_rewards_batch_fn(x0_jnp, acts)
        return np.asarray(jnp.sum(rewards_seq, axis=1))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def plan(self, x0: State, rng_key: Optional[Any] = None) -> Dict[str, Any]:
        x0_jnp = jnp.asarray(x0, dtype=jnp.float32)
        dim = self.horizon * self.act_dim
        rng = np.random.RandomState(self.seed)

        mean = np.zeros(dim, dtype=np.float64)
        sigma = float(self.sigma0)
        C_diag = np.ones(dim, dtype=np.float64)

        mu = max(int(self.elite_frac * self.popsize), 1)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        c_c = 2.0 / (dim + 2.0)

        best_flat = mean.copy()
        best_return = -np.inf

        for _ in range(self.generations):
            z = rng.randn(self.popsize, dim)
            pop = mean[None, :] + sigma * np.sqrt(C_diag)[None, :] * z
            if self.action_limit is not None:
                pop = np.clip(pop, -self.action_limit, self.action_limit)

            returns = self._returns(x0_jnp, pop.astype(np.float32))   # (P,)

            gbest = int(np.argmax(returns))
            if returns[gbest] > best_return:
                best_return = float(returns[gbest])
                best_flat = pop[gbest].astype(np.float64).copy()

            rank = np.argsort(-returns)[:mu]
            sel_z = z[rank]
            mean = mean + sigma * np.sqrt(C_diag) * np.dot(weights, sel_z)
            C_diag = (1 - c_c) * C_diag + c_c * np.dot(weights, sel_z ** 2)
            # 1/5-rule-ish step-size control
            sr = float(np.mean(returns[rank] > np.median(returns)))
            sigma *= 1.05 if sr > 0.2 else 0.95
            sigma = float(np.clip(sigma, self.min_std, 2.0))

        best_actions = jnp.asarray(best_flat.reshape(self.horizon, self.act_dim), dtype=jnp.float32)
        if self.action_limit is not None:
            best_actions = jnp.clip(best_actions, -self.action_limit, self.action_limit)

        states = self._rollout_states_fn(x0_jnp, best_actions)
        rewards = self._rollout_rewards_fn(x0_jnp, best_actions)
        return {
            "actions": np.asarray(best_actions, dtype=np.float32),
            "states": np.asarray(states, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "total_reward": float(np.sum(np.asarray(rewards))),
            "initial_state": np.asarray(states, dtype=np.float32)[0],
        }

    def sample_trajectories(self, x0: State, n_samples: int, rng_key: Optional[Any] = None) -> List[Trajectory]:
        x0_jnp = jnp.asarray(x0, dtype=jnp.float32)
        rng = np.random.RandomState(self.seed)
        trajs = []
        for _ in range(n_samples):
            flat = self.sigma0 * rng.randn(self.horizon * self.act_dim)
            acts = jnp.asarray(flat.reshape(self.horizon, self.act_dim), dtype=jnp.float32)
            if self.action_limit is not None:
                acts = jnp.clip(acts, -self.action_limit, self.action_limit)
            states = self._rollout_states_fn(x0_jnp, acts)
            trajs.append(Trajectory(
                states=[np.asarray(s, dtype=np.float32) for s in states],
                actions=[np.asarray(a, dtype=np.float32) for a in acts],
            ))
        return trajs
