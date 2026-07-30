"""
JAX backend implementation for MPPI.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.core.types import Trajectory, State


class MPPIBackendJax:
    """JAX implementation of the MPPI algorithm."""

    def __init__(
        self,
        *,
        env_adapter,
        legacy_energy,
        horizon: int,
        dt: float,
        num_samples: int,
        num_iterations: int,
        noise_sigma: float,
        lambda_: float,
        action_limit: float,
        terminal_energy_weight: float = 0.0,
        guide_weight: float = 0.0,
        position_extractor: Optional[Any] = None,
        position_dim: int = 2,
        seed: int = 0,
        noise_sampler: Optional[Any] = None,
    ):
        self.env = env_adapter
        self.energy = legacy_energy
        self.horizon = horizon
        self.dt = dt
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.noise_sigma = noise_sigma
        self.lambda_ = lambda_
        self.action_limit = action_limit
        self.terminal_energy_weight = float(terminal_energy_weight)
        self.guide_weight = float(guide_weight)
        self.position_extractor = position_extractor
        self.position_dim = int(position_dim)
        self.seed = seed
        self.act_dim = self.env.act_dim
        # Optional pluggable noise sampler. When None, the inline isotropic
        # Gaussian path below is used (zero behaviour change vs prior code).
        # See genedynamics.core.prob.NoiseSampler for the protocol.
        self.noise_sampler = noise_sampler

        self._build_jax_functions()

    def _draw_noise(self, key, shape, *, state=None):
        """Draw noise. Routes to ``self.noise_sampler`` if set, else inline N(0, σ²I)."""
        if self.noise_sampler is not None:
            return self.noise_sampler.sample(shape, key=key, state=state)
        return jax.random.normal(key, shape, dtype=jnp.float32) * self.noise_sigma

    def _build_jax_functions(self) -> None:
        def transition_fn(state, action):
            return self.env.jax_transition(state, action)

        def cost_fn(state, action, ctx):
            return self.energy.compute(state, action, ctx)

        self._transition_fn = jax.jit(transition_fn)
        self._cost_fn = jax.jit(cost_fn)

        def rollout_rewards(state_init, actions):
            target_raw = getattr(self.env, "target", None)
            if target_raw is None:
                target = jnp.zeros(self.position_dim, dtype=jnp.float32)
            else:
                if self.position_extractor is not None:
                    target_np = np.asarray(
                        self.position_extractor(target_raw), dtype=np.float32
                    ).reshape(-1)[: self.position_dim]
                else:
                    target_np = np.asarray(target_raw, dtype=np.float32).reshape(-1)[
                        : self.position_dim
                    ]
                target = jnp.asarray(target_np, dtype=jnp.float32)

            d0 = jnp.linalg.norm(
                state_init[: self.position_dim] - target[: self.position_dim]
            )
            t_star = float(self.horizon - 1)
            guide_weight = jnp.asarray(self.guide_weight, dtype=jnp.float32)
            time_indices = jnp.arange(actions.shape[0], dtype=jnp.float32)

            def step_fn(carry, action_and_t):
                action, time_step = action_and_t
                next_state = self._transition_fn(carry, action)
                ctx = {"t": time_step, "target_xy": target}
                reward = -self._cost_fn(next_state, action, ctx)
                expected_distance = d0 * jnp.maximum(
                    0.0, 1.0 - time_step / (t_star + 1e-6)
                )
                distance = jnp.linalg.norm(
                    next_state[: self.position_dim] - target[: self.position_dim]
                )
                guide_reward = -guide_weight * jnp.square(
                    distance - expected_distance
                )
                return next_state, reward + guide_reward

            final_state, rewards = jax.lax.scan(
                step_fn, state_init, (actions, time_indices)
            )
            if hasattr(self.env, "terminal_distance_jax"):
                terminal_distance = self.env.terminal_distance_jax(final_state)
            else:
                terminal_distance = jnp.linalg.norm(
                    final_state[: self.position_dim] - target[: self.position_dim]
                )
            terminal_reward = (
                -jnp.asarray(self.terminal_energy_weight, dtype=jnp.float32)
                * terminal_distance
            )
            rewards = rewards.at[-1].add(terminal_reward)
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

    def plan(self, x0: State, rng_key: Optional[Any] = None) -> Dict[str, Any]:
        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.seed)

        x0_jnp = jnp.asarray(x0, dtype=jnp.float32)
        mean_actions = jnp.zeros((self.horizon, self.act_dim), dtype=jnp.float32)
        best_actions = mean_actions
        best_return = float(
            np.sum(np.asarray(self._rollout_rewards_fn(x0_jnp, mean_actions)))
        )
        incumbent_return_history = []
        incumbent_action_history = []
        # Preserve the same per-update action population consumed by the shared
        # cost visualizer for diffusion baselines. Preallocation avoids holding
        # a Python list plus multiple full-size stacked copies (important for
        # the humanoid 1024 x 100 x 50 rollout budget).
        sampled_action_history = np.empty(
            (
                self.num_iterations,
                self.num_samples,
                self.horizon,
                self.act_dim,
            ),
            dtype=np.float32,
        )
        rng = rng_key

        for iteration_idx in range(self.num_iterations):
            rng, noise_key = jax.random.split(rng)
            noise = self._draw_noise(
                noise_key,
                (self.num_samples, self.horizon, self.act_dim),
                state=mean_actions,
            )

            candidates = mean_actions[None, :, :] + noise
            if self.action_limit is not None:
                candidates = jnp.clip(candidates, -self.action_limit, self.action_limit)

            rewards_seq = self._rollout_rewards_batch_fn(x0_jnp, candidates)
            total_returns = jnp.sum(rewards_seq, axis=1)

            total_returns_np = np.asarray(total_returns)
            candidates_np = np.asarray(candidates)
            sampled_action_history[iteration_idx] = candidates_np

            best_idx = int(np.argmax(total_returns_np))
            if total_returns_np[best_idx] > best_return:
                best_return = float(total_returns_np[best_idx])
                best_actions = candidates[best_idx]

            costs_np = -total_returns_np
            beta = np.min(costs_np)
            weights = np.exp(-(costs_np - beta) / max(self.lambda_, 1e-6))
            weights_sum = np.sum(weights) + 1e-8
            mean_actions = jnp.asarray(
                np.einsum("i,ijk->jk", weights, candidates_np) / weights_sum, dtype=jnp.float32
            )
            mean_return = float(
                np.sum(np.asarray(self._rollout_rewards_fn(x0_jnp, mean_actions)))
            )
            if mean_return > best_return:
                best_return = mean_return
                best_actions = mean_actions
            incumbent_return_history.append(best_return)
            incumbent_action_history.append(
                np.asarray(best_actions, dtype=np.float32)
            )

        if self.action_limit is not None:
            best_actions = jnp.clip(best_actions, -self.action_limit, self.action_limit)

        states = self._rollout_states_fn(x0_jnp, best_actions)
        rewards = self._rollout_rewards_fn(x0_jnp, best_actions)

        states_np = np.asarray(states)
        actions_np = np.asarray(best_actions)
        incumbent_returns_np = np.asarray(
            incumbent_return_history, dtype=np.float32
        )
        incumbent_actions_np = np.asarray(
            incumbent_action_history, dtype=np.float32
        )
        energy_vals = []
        for t in range(actions_np.shape[0]):
            ctx = {"t": int(t)}
            energy_vals.append(
                float(self.energy.compute(states_np[t + 1], actions_np[t], ctx))
            )
        energies_np = np.asarray(energy_vals, dtype=np.float32)

        return {
            "actions": actions_np,
            "states": states_np,
            "rewards": np.asarray(rewards, dtype=np.float32),
            "total_reward": float(np.sum(rewards)),
            "mean_reward": float(np.mean(rewards)) if rewards.size > 0 else 0.0,
            "initial_state": states_np[0],
            "energies": energies_np,
            # The shared experiment cost visualizer expects diffusion histories
            # in clean-to-noisy order and reverses them for plotting. MPPI
            # naturally records optimization iterations from initial-to-final,
            # so expose the incumbent histories in reverse here. This yields a
            # chronological 100 -> 1 update curve in cost.json while keeping
            # the final MPPI solution at index 0, matching the shared schema.
            "reward_history": incumbent_returns_np[::-1].copy(),
            "diffusion_actions_traj": incumbent_actions_np[::-1].copy(),
            # Shared schema is clean-to-noisy; experiment.py reverses it back
            # to chronological update order before evaluating every candidate
            # with the common task + violation Cost definition.
            "diffusion_sampled_actions": sampled_action_history[::-1],
            # Native chronological diagnostic for MPPI-specific consumers.
            "optimization_cost_history": (-incumbent_returns_np).copy(),
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
        mean_actions = jnp.zeros((self.horizon, self.act_dim), dtype=jnp.float32)

        trajectories = []
        rng = rng_key
        for _ in range(n_samples):
            rng, noise_key = jax.random.split(rng)
            noise = self._draw_noise(
                noise_key,
                (self.horizon, self.act_dim),
                state=mean_actions,
            )
            candidate = mean_actions + noise
            if self.action_limit is not None:
                candidate = jnp.clip(candidate, -self.action_limit, self.action_limit)
            states = self._rollout_states_fn(x0_jnp, candidate)
            states_list = [np.asarray(s, dtype=np.float32) for s in states]
            actions_list = [np.asarray(a, dtype=np.float32) for a in candidate]
            trajectories.append(Trajectory(states=states_list, actions=actions_list))
        return trajectories
