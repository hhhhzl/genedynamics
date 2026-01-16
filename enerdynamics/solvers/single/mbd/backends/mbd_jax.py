"""
JAX backend implementation for MBD.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from enerdynamics.core.types import Trajectory, State


class MBDBackendJax:
    """JAX implementation of multi-scale barrier diffusion."""

    def __init__(
        self,
        *,
        env_adapter,
        legacy_energy,
        horizon: int,
        dt: float,
        Nsample: int,
        Ndiffuse: int,
        temp_sample: float,
        beta0: float,
        betaT: float,
        action_limit: float,
        seed: int = 0,
        scheduler: Any = None,
        show_tqdm: bool = False,
    ):
        self.env = env_adapter
        self.energy = legacy_energy
        self.horizon = horizon
        self.dt = dt
        self.Nsample = Nsample
        self.Ndiffuse = Ndiffuse
        self.temp_sample = temp_sample
        self.beta0 = beta0
        self.betaT = betaT
        self.action_limit = action_limit
        self.seed = seed
        self.act_dim = self.env.act_dim
        self.scheduler = scheduler
        self.show_tqdm = bool(show_tqdm)

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

    def plan(self, x0: State, rng_key: Optional[Any] = None) -> Dict[str, Any]:
        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.seed)
        x0_jnp = jnp.asarray(x0, dtype=jnp.float32)

        rng, diffuse_rng = jax.random.split(rng_key)
        try:
            from enerdynamics.core.constraints.schedulers.utils import DiffusionNoiseSchedule

            betas_np = np.linspace(self.beta0, self.betaT, self.Ndiffuse, dtype=np.float32)
            betas = jnp.asarray(DiffusionNoiseSchedule.from_betas(betas_np).betas, dtype=jnp.float32)
        except Exception:
            betas = jnp.linspace(self.beta0, self.betaT, self.Ndiffuse, dtype=jnp.float32)
        alphas = 1.0 - betas
        alphas_bar = jnp.cumprod(alphas)
        sigmas = jnp.sqrt(1.0 - alphas_bar)
        diffusion_indices = jnp.arange(self.Ndiffuse - 1, 0, -1, dtype=jnp.int32)

        # Diffusion scheduler params (T_k schedule only; M_k kept static for JAX shape stability)
        T_k_list = []
        if self.scheduler is not None and hasattr(self.scheduler, "diffusion_schedulers"):
            try:
                ds_list = getattr(self.scheduler, "diffusion_schedulers", [])
                if ds_list:
                    ds = ds_list[0]
                    from enerdynamics.core.constraints.core.types import ScheduleState

                    total_steps = self.Ndiffuse - 1
                    for k in range(self.Ndiffuse):
                        params = ds.diffusion_params(ScheduleState(k=k, K=total_steps)) or {}
                        T_k_list.append(float(params.get("T_k", self.temp_sample)))
                    # block adaptive diffusion scheduler if backend not jax
                    if hasattr(ds, "update") and getattr(ds, "backend", None) != "jax":
                        raise NotImplementedError(
                            "Adaptive diffusion scheduler update with non-JAX backend is not supported in JAX mbd. "
                            "Provide a JAX-compatible adaptive diffusion scheduler or use fixed parameters."
                        )
            except Exception:
                T_k_list = []
        if not T_k_list:
            T_k_list = [self.temp_sample for _ in range(self.Ndiffuse)]
        T_k_arr = jnp.asarray(T_k_list, dtype=jnp.float32)

        def reverse_diffuse(rng_in, Ybar_init):
            def body(carry, idx):
                rng_curr, Ybar_curr = carry
                rng_curr, noise_key = jax.random.split(rng_curr)

                Yi = Ybar_curr * jnp.sqrt(alphas_bar[idx])
                eps = jax.random.normal(noise_key, (self.Nsample, self.horizon, self.act_dim), dtype=jnp.float32)
                Y0s = eps * sigmas[idx] + Ybar_curr
                Y0s = jnp.clip(Y0s, -self.action_limit, self.action_limit)

                rews = self._rollout_rewards_batch_fn(x0_jnp, Y0s)
                rews_mean = jnp.mean(rews, axis=-1)

                rew_mean = jnp.mean(rews_mean)
                rew_std = jnp.std(rews_mean)
                rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)

                logp0 = (rews_mean - rew_mean) / (rew_std * T_k_arr[idx])
                weights = jax.nn.softmax(logp0)
                Ybar_weighted = jnp.einsum("n,nij->ij", weights, Y0s)

                score = (-Yi + jnp.sqrt(alphas_bar[idx]) * Ybar_weighted) / (1.0 - alphas_bar[idx])
                Yim1 = (Yi + (1.0 - alphas_bar[idx]) * score) / jnp.sqrt(alphas[idx])
                Ybar_next = Yim1 / jnp.sqrt(alphas_bar[idx - 1])

                return (rng_curr, Ybar_next), (jnp.mean(rews_mean), Ybar_next, Y0s)

            (rng_out, Ybar_final), (reward_hist, Ybar_hist, Ysamples_hist) = jax.lax.scan(
                body, (rng_in, Ybar_init), diffusion_indices
            )
            reward_hist = reward_hist[::-1]
            Ybar_hist = Ybar_hist[::-1]
            Ysamples_hist = Ysamples_hist[::-1]
            return rng_out, Ybar_final, reward_hist, Ybar_hist, Ysamples_hist

        reverse_diffuse_jit = jax.jit(reverse_diffuse)

        Ybar_init = jnp.zeros((self.horizon, self.act_dim), dtype=jnp.float32)
        _, Ybar_final, reward_hist, actions_traj, sampled_traj = reverse_diffuse_jit(diffuse_rng, Ybar_init)

        final_actions = jnp.clip(Ybar_final, -self.action_limit, self.action_limit)
        states = self._rollout_states_fn(x0_jnp, final_actions)
        rewards = self._rollout_rewards_fn(x0_jnp, final_actions)

        states_np = np.asarray(states)
        actions_np = np.asarray(final_actions)
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
            "reward_history": np.asarray(reward_hist, dtype=np.float32),
            "diffusion_actions_traj": np.asarray(actions_traj, dtype=np.float32),
            "diffusion_sampled_actions": np.asarray(sampled_traj, dtype=np.float32),
        }

    def sample_trajectories(
        self,
        x0: State,
        n_samples: int,
        rng_key: Optional[Any] = None,
    ) -> List[Trajectory]:
        result = self.plan(x0, rng_key)
        traj = Trajectory(
            states=[np.asarray(s, dtype=np.float32) for s in result["states"]],
            actions=[np.asarray(a, dtype=np.float32) for a in result["actions"]],
            info=result,
        )
        return [traj for _ in range(max(1, n_samples))]

