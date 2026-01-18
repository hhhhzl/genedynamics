"""
JAX backend implementation for MDOC.

MDOC is defined as:
  MBD diffusion driver + ConstraintFilter

This file intentionally mirrors the structure of `solvers/single/mbd/backends/mbd_jax.py`,
but inserts a `constraint_filter.apply_actions_batch(...)` hook every diffusion step.

For now, ConstraintFilter can be a NoOp filter; closed-form / QP variants will be plugged in later.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from enerdynamics.core.types import Trajectory, State
from enerdynamics.core.constraints.core.types import ScheduleState, ScheduleParams
from enerdynamics.core.constraints.action_filters import ConstraintFilter, NoOpConstraintFilter


class MDOCBackendJax:
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
        constraint_filter: Optional[ConstraintFilter] = None,
        obstacles: Any = None,
    ):
        self.env = env_adapter
        self.energy = legacy_energy
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.Nsample = int(Nsample)
        self.Ndiffuse = int(Ndiffuse)
        self.temp_sample = float(temp_sample)
        self.beta0 = float(beta0)
        self.betaT = float(betaT)
        self.action_limit = float(action_limit)
        self.seed = int(seed)
        self.act_dim = int(getattr(self.env, "act_dim", 2))
        self.scheduler = scheduler
        self.show_tqdm = bool(show_tqdm)
        self.obstacles = obstacles
        self.constraint_filter = constraint_filter or NoOpConstraintFilter()

        self._build_jax_functions()

        # Precompute diffusion temperature schedule (T_k) if scheduler provides it.
        self._T_k_arr = None
        try:
            T_k_list = []
            if self.scheduler is not None and hasattr(self.scheduler, "diffusion_schedulers"):
                ds_list = getattr(self.scheduler, "diffusion_schedulers", [])
                if ds_list:
                    ds = ds_list[0]
                    for k in range(self.Ndiffuse):
                        st = ScheduleState(k=k, K=max(1, self.Ndiffuse - 1))
                        d = ds.diffusion_params(st) or {}
                        T_k_list.append(float(d.get("T_k", self.temp_sample)))
            if T_k_list:
                self._T_k_arr = jnp.asarray(T_k_list, dtype=jnp.float32)
        except Exception:
            self._T_k_arr = None

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

    def _constraint_params(self, step_k: int, total_steps: int) -> ScheduleParams:
        # Default fixed params if no scheduler: margin=0, rho=1
        if self.scheduler is None:
            return ScheduleParams(margin=0.0, rho=1.0)
        try:
            # CompositeScheduler in this repo typically has constraint_schedulers list
            if hasattr(self.scheduler, "constraint_schedulers"):
                cs_list = getattr(self.scheduler, "constraint_schedulers", [])
                if cs_list:
                    cs = cs_list[0]
                    st = ScheduleState(k=step_k, K=total_steps)
                    if hasattr(cs, "constraint_params"):
                        d = cs.constraint_params(st) or {}
                        return ScheduleParams(
                            margin=float(d.get("margin", 0.0)),
                            rho=float(d.get("rho", 1.0)),
                            qp_gate=bool(d.get("qp_gate", True)),
                            qp_prob=float(d.get("qp_prob", 1.0)),
                            topK=d.get("topK", None),
                        )
        except Exception:
            pass
        return ScheduleParams(margin=0.0, rho=1.0)

    def plan(self, x0: State, rng_key: Optional[Any] = None) -> Dict[str, Any]:
        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.seed)
        x0_jnp = jnp.asarray(x0, dtype=jnp.float32)

        rng, diffuse_rng = jax.random.split(rng_key)
        betas = jnp.linspace(self.beta0, self.betaT, self.Ndiffuse, dtype=jnp.float32)
        alphas = 1.0 - betas
        alphas_bar = jnp.cumprod(alphas)
        sigmas = jnp.sqrt(1.0 - alphas_bar)
        diffusion_indices = jnp.arange(self.Ndiffuse - 1, 0, -1, dtype=jnp.int32)
        total_steps = int(self.Ndiffuse - 1)

        # We keep M fixed for JAX shape stability. (Matches existing MBD jax code.)
        def reverse_diffuse(rng_in, Ybar_init):
            def body(carry, idx):
                rng_curr, Ybar_curr = carry
                rng_curr, noise_key = jax.random.split(rng_curr)

                Yi = Ybar_curr * jnp.sqrt(alphas_bar[idx])
                eps = jax.random.normal(noise_key, (self.Nsample, self.horizon, self.act_dim), dtype=jnp.float32)
                Y0s = eps * sigmas[idx] + Ybar_curr
                Y0s = jnp.clip(Y0s, -self.action_limit, self.action_limit)

                # Map diffusion index -> step_k (same convention as EDOC/MBD numpy backend)
                step_k = int((self.Ndiffuse - 1) - int(idx))
                sched_state = ScheduleState(k=step_k, K=total_steps)
                sched_params = self._constraint_params(step_k, total_steps)

                # ConstraintFilter hook: filter candidate action sequences
                Y0s_f = self.constraint_filter.apply_actions_batch(
                    x0_jnp,
                    Y0s,
                    env=self.env,
                    obstacles=self.obstacles,
                    schedule_state=sched_state,
                    schedule_params=sched_params,
                )

                rews = self._rollout_rewards_batch_fn(x0_jnp, Y0s_f)
                rews_mean = jnp.mean(rews, axis=-1)

                rew_mean = jnp.mean(rews_mean)
                rew_std = jnp.std(rews_mean)
                rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)

                T_k = self.temp_sample
                if self._T_k_arr is not None:
                    # idx is diffusion index; schedule uses k (0..K) where larger means earlier/noisier.
                    # We use step_k (aligned with ScheduleState) to index T_k.
                    T_k = self._T_k_arr[jnp.clip(step_k, 0, self._T_k_arr.shape[0] - 1)]
                logp0 = (rews_mean - rew_mean) / (rew_std * T_k)
                weights = jax.nn.softmax(logp0)
                Ybar_weighted = jnp.einsum("n,nij->ij", weights, Y0s_f)

                score = (-Yi + jnp.sqrt(alphas_bar[idx]) * Ybar_weighted) / (1.0 - alphas_bar[idx])
                Yim1 = (Yi + (1.0 - alphas_bar[idx]) * score) / jnp.sqrt(alphas[idx])
                Ybar_next = Yim1 / jnp.sqrt(alphas_bar[idx - 1])

                # Optional: filter the mean trajectory too (keeps MDOC definition consistent)
                Ybar_next = self.constraint_filter.apply_actions(
                    x0_jnp,
                    Ybar_next,
                    env=self.env,
                    obstacles=self.obstacles,
                    schedule_state=sched_state,
                    schedule_params=sched_params,
                )

                return (rng_curr, Ybar_next), (jnp.mean(rews_mean), Ybar_next, Y0s_f)

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


