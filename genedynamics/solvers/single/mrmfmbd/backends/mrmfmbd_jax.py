"""
JAX backend for MRMFMBD (Soft-robot mode marginalization + fidelity ladder).

Multi-Resolution, Multi-Fidelity Model-Based Diffusion:
- Fidelity ladder: early diffusion steps use coarse sim, late steps use fine
- MCSA: reward-weighted score ascent (ES gradient)
- Optional mode-marginalization over contact/friction regimes
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.core.types import Trajectory


class MRMFMBDBackendJax:
    """
    JAX implementation of MR-MF-MBD for soft robots.

    Diffusion over actions with fidelity-dependent rollout.
    Early steps: coarse sim. Late steps: fine sim.
    """

    def __init__(
        self,
        *,
        fidelity_simulator: Any,
        fidelity_ladder: Any,
        horizon: int = 80,
        Nsample: int = 64,
        Ndiffuse: int = 100,
        temp_sample: float = 0.5,
        beta0: float = 1e-4,
        betaT: float = 1e-2,
        action_limit: float = 1.0,
        action_extra_sigma: float = 0.0,
        seed: int = 0,
        scheduler: Any = None,
        show_tqdm: bool = False,
        mode_marginalizer: Optional[Any] = None,
        **kwargs: Any,
    ):
        self.fidelity_sim = fidelity_simulator
        self.fidelity_ladder = fidelity_ladder
        self.horizon = horizon
        self.Nsample = Nsample
        self.Ndiffuse = Ndiffuse
        self.temp_sample = temp_sample
        self.beta0 = float(beta0)
        self.betaT = float(betaT)
        self.action_limit = action_limit
        self.action_extra_sigma = float(action_extra_sigma)
        self.seed = seed
        self.scheduler = scheduler
        self.show_tqdm = show_tqdm
        self.mode_marginalizer = mode_marginalizer
        self.act_dim = self.fidelity_sim.act_dim()
        self.config = kwargs

        self._build_jax_functions()

    def _build_jax_functions(self) -> None:
        """Build JIT-compiled rollout and diffusion functions."""
        # Precompute fidelity level per reverse diffusion step index
        # Early reverse (high idx, noisy) -> coarse; late reverse (low idx, clean) -> fine
        K = self.Ndiffuse
        self._fidelity_levels = {
            int(idx): self.fidelity_ladder(K - 1 - int(idx))
            for idx in np.arange(K - 1, 0, -1)
        }

        def rollout_rewards_single(x0: jnp.ndarray, actions: jnp.ndarray, fidelity_level: int) -> jnp.ndarray:
            return self.fidelity_sim.rollout_rewards(x0, actions, fidelity_level)

        def rollout_states_single(x0: jnp.ndarray, actions: jnp.ndarray, fidelity_level: int) -> jnp.ndarray:
            return self.fidelity_sim.rollout_states(x0, actions, fidelity_level)

        # Batch rollout: vmap over actions, fidelity_level is static for JIT
        # We need separate functions per fidelity level or pass level as static
        # For now: use a single level (0) in JIT - multi-fidelity via Python loop
        self._rollout_rewards_l0 = jax.jit(
            lambda x0, actions: self.fidelity_sim.rollout_rewards(x0, actions, 0)
        )
        self._rollout_rewards_batch_l0 = jax.jit(
            jax.vmap(self._rollout_rewards_l0, in_axes=(None, 0))
        )
        self._rollout_states_l0 = jax.jit(
            lambda x0, actions: self.fidelity_sim.rollout_states(x0, actions, 0)
        )

        # Multi-fidelity: create per-level rollout (for levels 0,1,2)
        fs = self.fidelity_sim
        self._rollout_rewards_batch_by_level = {}
        for ell in set(self._fidelity_levels.values()):
            def _make_batch(level: int):
                def _rollout(x0, actions):
                    return jax.vmap(lambda a: fs.rollout_rewards(x0, a, level), in_axes=0)(actions)
                return jax.jit(_rollout)

            self._rollout_rewards_batch_by_level[ell] = _make_batch(ell)

        # Fallback: use level 0 for all when not in level map
        self._rollout_rewards_batch_fn = self._rollout_rewards_batch_l0
        self._rollout_rewards_fn = self._rollout_rewards_l0
        self._rollout_states_fn = self._rollout_states_l0

    def _get_rollout_batch_fn(self, fidelity_level: int):
        return self._rollout_rewards_batch_by_level.get(
            fidelity_level, self._rollout_rewards_batch_l0
        )

    def plan(self, x0: Any, rng_key: Optional[Any] = None) -> Dict[str, Any]:
        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.seed)
        x0_jnp = jnp.asarray(x0, dtype=jnp.float32)

        try:
            from genedynamics.core.constraints.schedulers.utils import DiffusionNoiseSchedule
            betas_np = np.linspace(self.beta0, self.betaT, self.Ndiffuse, dtype=np.float32)
            betas = jnp.asarray(DiffusionNoiseSchedule.from_betas(betas_np).betas, dtype=jnp.float32)
        except Exception:
            betas = jnp.linspace(self.beta0, self.betaT, self.Ndiffuse, dtype=jnp.float32)
        alphas = 1.0 - betas
        alphas_bar = jnp.cumprod(alphas)
        sigmas = jnp.sqrt(1.0 - alphas_bar)
        diffusion_indices = jnp.arange(self.Ndiffuse - 1, 0, -1, dtype=jnp.int32)

        denom = jnp.maximum(float(self.Ndiffuse - 1), 1.0)
        progress_inc_by_idx = 1.0 - (jnp.arange(self.Ndiffuse, dtype=jnp.float32) / denom)
        extra_sigmas_by_idx = self.action_extra_sigma * (1.0 - progress_inc_by_idx)

        T_k_list = []
        if self.scheduler is not None and hasattr(self.scheduler, "diffusion_schedulers"):
            try:
                ds_list = getattr(self.scheduler, "diffusion_schedulers", [])
                if ds_list:
                    from genedynamics.core.constraints.core.types import ScheduleState
                    ds = ds_list[0]
                    total_steps = self.Ndiffuse - 1
                    for k in range(self.Ndiffuse):
                        params = ds.diffusion_params(ScheduleState(k=k, K=total_steps)) or {}
                        T_k_list.append(float(params.get("T_k", self.temp_sample)))
            except Exception:
                T_k_list = []
        if not T_k_list:
            T_k_list = [self.temp_sample for _ in range(self.Ndiffuse)]
        T_k_arr = jnp.asarray(T_k_list, dtype=jnp.float32)

        fidelity_history = []

        # Use Python loop for multi-fidelity (rollout_batch_fn varies per step)
        rng_curr = rng_key
        Ybar_curr = jnp.zeros((self.horizon, self.act_dim), dtype=jnp.float32)
        reward_hist = []
        Ybar_hist = []
        Ysamples_hist = []

        for i in range(len(diffusion_indices)):
            idx = int(diffusion_indices[i])
            fid_level = self._fidelity_levels.get(idx, 0)
            fidelity_history.append(fid_level)

            rng_curr, noise_key, extra_key = jax.random.split(rng_curr, 3)
            rollout_batch_fn = self._get_rollout_batch_fn(fid_level)

            Yi = Ybar_curr * jnp.sqrt(alphas_bar[idx])
            eps = jax.random.normal(noise_key, (self.Nsample, self.horizon, self.act_dim), dtype=jnp.float32)
            Y0s = eps * sigmas[idx] + Ybar_curr
            Y0s = jnp.clip(Y0s, -self.action_limit, self.action_limit)

            rews = rollout_batch_fn(x0_jnp, Y0s)
            rews_mean = jnp.mean(rews, axis=-1)
            rew_mean = jnp.mean(rews_mean)
            rew_std = jnp.where(jnp.std(rews_mean) < 1e-4, 1.0, jnp.std(rews_mean))

            logp0 = (rews_mean - rew_mean) / (rew_std * T_k_arr[idx])
            weights = jax.nn.softmax(logp0)
            Ybar_weighted = jnp.einsum("n,nij->ij", weights, Y0s)

            score = (-Yi + jnp.sqrt(alphas_bar[idx]) * Ybar_weighted) / (1.0 - alphas_bar[idx])
            Yim1 = (Yi + (1.0 - alphas_bar[idx]) * score) / jnp.sqrt(alphas[idx])
            Ybar_next = Yim1 / jnp.sqrt(alphas_bar[idx - 1])

            extra_sigma = extra_sigmas_by_idx[idx]
            noise_extra = jax.random.normal(extra_key, (self.horizon, self.act_dim), dtype=jnp.float32)
            Ybar_next = Ybar_next + extra_sigma * noise_extra
            Ybar_next = jnp.clip(Ybar_next, -self.action_limit, self.action_limit)

            reward_hist.append(float(jnp.mean(rews_mean)))
            Ybar_hist.append(np.asarray(Ybar_curr))
            Ysamples_hist.append(np.asarray(Y0s))

            Ybar_curr = Ybar_next

        final_actions = jnp.clip(Ybar_curr, -self.action_limit, self.action_limit)
        states = self._rollout_states_fn(x0_jnp, final_actions)
        rewards = self._rollout_rewards_fn(x0_jnp, final_actions)

        states_np = np.asarray(states)
        actions_np = np.asarray(final_actions)
        rewards_np = np.asarray(rewards, dtype=np.float32)

        if self.show_tqdm:
            import tqdm
            tqdm.tqdm.write(f"MRMFMBD: total_reward={float(np.sum(rewards_np)):.4f}")

        return {
            "actions": actions_np,
            "states": states_np,
            "rewards": rewards_np,
            "total_reward": float(np.sum(rewards_np)),
            "mean_reward": float(np.mean(rewards_np)) if rewards_np.size > 0 else 0.0,
            "initial_state": states_np[0],
            "reward_history": np.asarray(reward_hist, dtype=np.float32),
            "fidelity_history": fidelity_history,
            "diffusion_actions_traj": np.array(Ybar_hist, dtype=np.float32) if Ybar_hist else None,
            "diffusion_sampled_actions": np.array(Ysamples_hist, dtype=np.float32) if Ysamples_hist else None,
            "candidate_states": [states_np],
            "candidate_actions": [actions_np],
            "candidate_costs": np.asarray([-float(np.sum(rewards_np))], dtype=np.float32),
            "best_idx": 0,
        }

    def sample_trajectories(
        self,
        x0: Any,
        n_samples: int,
        rng_key: Optional[Any] = None,
    ) -> List[Trajectory]:
        result = self.plan(x0, rng_key=rng_key)
        traj = Trajectory(
            states=[np.asarray(s, dtype=np.float32) for s in result["states"]],
            actions=[np.asarray(a, dtype=np.float32) for a in result["actions"]],
            info=result,
        )
        return [traj for _ in range(max(1, n_samples))]

    def plan_batch(self, x0: Any, keys: Any) -> List[Dict[str, Any]]:
        return [self.plan(x0, rng_key=keys[i]) for i in range(keys.shape[0])]
