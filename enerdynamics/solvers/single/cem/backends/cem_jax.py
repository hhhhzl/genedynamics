"""
JAX backend implementation for CEM.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from enerdynamics.core.types import Trajectory, State


class CEMBackendJax:
    """JAX implementation of the CEM algorithm."""

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
        init_std: float,
        min_std: float,
        action_limit: float,
        seed: int = 0,
    ):
        self.env = env_adapter
        self.energy = legacy_energy
        self.horizon = horizon
        self.dt = dt
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.elite_frac = elite_frac
        self.init_std = init_std
        self.min_std = min_std
        self.action_limit = action_limit
        self.seed = seed
        self.act_dim = self.env.act_dim

        self._build_jax_functions()

    # ------------------------------------------------------------------ #
    # JAX helpers
    # ------------------------------------------------------------------ #
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
                ctx = {"t": 0}
                reward = -self._cost_fn(next_state, action, ctx)
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

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def plan(self, x0: State, rng_key: Optional[Any] = None) -> Dict[str, Any]:
        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.seed)

        x0_jnp = jnp.asarray(x0, dtype=jnp.float32)
        mean = jnp.zeros((self.horizon, self.act_dim), dtype=jnp.float32)
        std = jnp.full((self.horizon, self.act_dim), float(self.init_std), dtype=jnp.float32)
        min_std_val = float(self.min_std)

        elite_count = max(1, int(self.num_samples * self.elite_frac))
        best_actions = mean
        best_return = -jnp.inf
        rng = rng_key

        for _ in range(self.num_iterations):
            rng, sample_key = jax.random.split(rng)
            samples = (
                jax.random.normal(sample_key, (self.num_samples, self.horizon, self.act_dim), dtype=jnp.float32)
                * std[None, :, :]
                + mean[None, :, :]
            )

            if self.action_limit is not None:
                samples = jnp.clip(samples, -self.action_limit, self.action_limit)

            rewards_seq = self._rollout_rewards_batch_fn(x0_jnp, samples)
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
            std = jnp.asarray(np.clip(np.std(elites, axis=0), min_std_val, None), dtype=jnp.float32)

        if self.action_limit is not None:
            best_actions = jnp.clip(best_actions, -self.action_limit, self.action_limit)

        states = self._rollout_states_fn(x0_jnp, best_actions)
        rewards = self._rollout_rewards_fn(x0_jnp, best_actions)

        states_np = np.asarray(states)
        actions_np = np.asarray(best_actions)
        energy_vals = []
        for t in range(actions_np.shape[0]):
            ctx = {"t": int(t)}
            energy_vals.append(float(self.energy.compute(states_np[t], actions_np[t], ctx)))
        energies_np = np.asarray(energy_vals, dtype=np.float32)

        return {
            "actions": actions_np,
            "states": states_np,
            "rewards": np.asarray(rewards, dtype=np.float32),
            "total_reward": float(np.sum(rewards)),
            "mean_reward": float(np.mean(rewards)) if rewards.size > 0 else 0.0,
            "initial_state": states_np[0],
            "energies": energies_np,
        }

    def sample_trajectories(
        self,
        x0: State,
        n_samples: int,
        rng_key: Optional[Any] = None,
    ) -> List[Trajectory]:
        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.seed)

        x0_jnp = jnp.asarray(x0, dtype=jnp.float32)
        mean = jnp.zeros((self.horizon, self.act_dim), dtype=jnp.float32)
        std = jnp.full((self.horizon, self.act_dim), float(self.init_std), dtype=jnp.float32)

        trajectories = []
        rng = rng_key
        for _ in range(n_samples):
            rng, sample_key = jax.random.split(rng)
            sample = jax.random.normal(sample_key, (self.horizon, self.act_dim), dtype=jnp.float32) * std + mean
            if self.action_limit is not None:
                sample = jnp.clip(sample, -self.action_limit, self.action_limit)
            states = self._rollout_states_fn(x0_jnp, sample)
            states_list = [np.asarray(s, dtype=np.float32) for s in states]
            actions_list = [np.asarray(a, dtype=np.float32) for a in sample]
            trajectories.append(Trajectory(states=states_list, actions=actions_list))
        return trajectories

