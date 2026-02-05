"""
JAX backend implementation for MBD.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from enerdynamics.solvers.single.diffusion_adaptors import diverse_topk_modes

from enerdynamics.core.types import Trajectory, State


class MBDBackendJax:
    """JAX implementation of multi-scale barrier diffusion."""

    def __init__(
        self,
        *,
        solver: Any = None,
        env_adapter=None,
        legacy_energy=None,
        horizon: int = 80,
        dt: float = 0.1,
        Nsample: int = 64,
        Ndiffuse: int = 100,
        temp_sample: float = 0.5,
        beta0: float = 1e-4,
        betaT: float = 1e-2,
        action_limit: float = 1.0,
        seed: int = 0,
        scheduler: Any = None,
        show_tqdm: bool = False,
        **kwargs,
    ):
        # Support both old-style (direct params) and new-style (solver object) initialization
        if solver is not None:
            env_adapter = solver._env_adapter
            legacy_energy = solver._legacy_energy
            horizon = solver.horizon
            dt = solver.dt
            Nsample = solver.config["Nsample"]
            Ndiffuse = solver.config["Ndiffuse"]
            temp_sample = solver.config["temp_sample"]
            beta0 = solver.config["beta0"]
            betaT = solver.config["betaT"]
            action_limit = solver.config["action_limit"]
            seed = solver.seed
            scheduler = solver.config.get("scheduler")
            show_tqdm = solver.config.get("show_tqdm", False)
            action_extra_sigma = getattr(solver, "action_extra_sigma", solver.config.get("action_extra_sigma", 0.0))
            terminal_energy_weight = float(solver.config.get("terminal_energy_weight", 100.0))
        else:
            action_extra_sigma = kwargs.get("action_extra_sigma", 0.0)
            terminal_energy_weight = float(kwargs.get("terminal_energy_weight", 100.0))
        
        self.env = env_adapter
        self.terminal_energy_weight = terminal_energy_weight
        self.energy = legacy_energy
        self.horizon = horizon
        self.dt = dt
        self.Nsample = Nsample
        self.Ndiffuse = Ndiffuse
        self.temp_sample = temp_sample
        self.beta0 = beta0
        self.betaT = betaT
        self.action_limit = action_limit
        self.action_extra_sigma = float(action_extra_sigma)
        self.seed = seed
        self.act_dim = self.env.act_dim
        self.scheduler = scheduler
        self.show_tqdm = bool(show_tqdm)
        # Multi-mode support: number of candidate trajectories to return
        if solver is not None:
            self.num_modes = int(solver.config.get("num_modes", 1))
            self.diversity_eta = float(solver.config.get("diversity_eta", 1.0))
            self.diversity_topK_cand = int(solver.config.get("diversity_topK_cand", None) or (self.Nsample // 2))
            self.diversity_use_state = bool(solver.config.get("diversity_use_state", True))
        else:
            self.num_modes = 1
            self.diversity_eta = 1.0
            self.diversity_topK_cand = self.Nsample // 2
            self.diversity_use_state = True

        self._build_jax_functions()

    def _build_jax_functions(self) -> None:
        def transition_fn(state, action):
            return self.env.jax_transition(state, action)

        def cost_fn(state, action, ctx):
            return self.energy.compute(state, action, ctx)

        self._transition_fn = jax.jit(transition_fn)
        self._cost_fn = jax.jit(cost_fn)

        def rollout_rewards(state_init, actions):
            target_obj = getattr(self.env, "target", None)
            target = jnp.asarray(target_obj, dtype=jnp.float32) if target_obj is not None else jnp.zeros(2)

            def step_fn(carry, action):
                next_state = self._transition_fn(carry, action)
                ctx = {"t": 0}
                reward = -self._cost_fn(next_state, action, ctx)
                return next_state, reward

            final_state, rewards = jax.lax.scan(step_fn, state_init, actions)
            # Terminal cost: -terminal_weight * dist(final_state, target)
            terminal_dist = jnp.linalg.norm(final_state[0:2] - target[0:2])
            terminal_reward = -jnp.asarray(self.terminal_energy_weight, dtype=jnp.float32) * terminal_dist
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

        # Compute extra_sigmas_by_idx for additional randomness (decays over diffusion steps)
        denom = jnp.maximum(float(self.Ndiffuse - 1), 1.0)
        progress_inc_by_idx = 1.0 - (jnp.arange(self.Ndiffuse, dtype=jnp.float32) / denom)
        extra_sigmas_by_idx = self.action_extra_sigma * (1.0 - progress_inc_by_idx)

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
                rng_curr, noise_key, extra_key = jax.random.split(rng_curr, 3)

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

                # Add extra noise for diversity (decays over diffusion steps, aligned with ebmbd)
                extra_sigma = extra_sigmas_by_idx[idx]
                noise_extra = jax.random.normal(extra_key, (self.horizon, self.act_dim), dtype=jnp.float32)
                Ybar_next = Ybar_next + extra_sigma * noise_extra
                Ybar_next = jnp.clip(Ybar_next, -self.action_limit, self.action_limit)

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
    
    def plan_batch(self, x0: State, rng_keys: jnp.ndarray) -> list[Dict[str, Any]]:
        """
        Batch version of plan using jax.vmap for parallel execution.
        
        Args:
            x0: initial state
            rng_keys: (C,) array of PRNG keys
            
        Returns:
            List of C result dictionaries (same format as plan)
        """
        x0_jnp = jnp.asarray(x0, dtype=jnp.float32)
        
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
        
        # Compute extra_sigmas_by_idx for additional randomness (decays over diffusion steps)
        # Similar to ebmbd: early steps have more noise, later steps have less
        denom = jnp.maximum(float(self.Ndiffuse - 1), 1.0)
        progress_inc_by_idx = 1.0 - (jnp.arange(self.Ndiffuse, dtype=jnp.float32) / denom)
        extra_sigmas_by_idx = self.action_extra_sigma * (1.0 - progress_inc_by_idx)
        
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
            except Exception:
                T_k_list = []
        if not T_k_list:
            T_k_list = [self.temp_sample for _ in range(self.Ndiffuse)]
        T_k_arr = jnp.asarray(T_k_list, dtype=jnp.float32)
        
        def reverse_diffuse_core(rng_key):
            # Split rng inside core function (aligned with ebmbd)
            rng, _ = jax.random.split(rng_key)
            def body(carry, idx):
                rng_curr, Ybar_curr = carry
                rng_curr, noise_key, extra_key = jax.random.split(rng_curr, 3)
                
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
                
                # Add extra noise for diversity (decays over diffusion steps, aligned with ebmbd)
                extra_sigma = extra_sigmas_by_idx[idx]
                noise_extra = jax.random.normal(extra_key, (self.horizon, self.act_dim), dtype=jnp.float32)
                Ybar_next = Ybar_next + extra_sigma * noise_extra
                Ybar_next = jnp.clip(Ybar_next, -self.action_limit, self.action_limit)
                
                return (rng_curr, Ybar_next), (jnp.mean(rews_mean), Ybar_next, Y0s)
            
            Ybar_init = jnp.zeros((self.horizon, self.act_dim), dtype=jnp.float32)
            (rng_out, Ybar_final), (reward_hist, Ybar_hist, Ysamples_hist) = jax.lax.scan(
                body, (rng, Ybar_init), diffusion_indices
            )
            reward_hist = reward_hist[::-1]
            Ybar_hist = Ybar_hist[::-1]
            Ysamples_hist = Ysamples_hist[::-1]
            return Ybar_final, reward_hist, Ybar_hist, Ysamples_hist
        
        # Vmap over rng_keys (aligned with ebmbd: pass rng_keys directly, split inside core)
        reverse_diffuse_batch_jit = jax.jit(jax.vmap(reverse_diffuse_core, in_axes=0))
        Ybar_finals, reward_hists, actions_trajs, sampled_trajs = reverse_diffuse_batch_jit(rng_keys)
        
        # Batch post-processing: clip, rollout states and rewards
        final_actions_batch = jnp.clip(Ybar_finals, -self.action_limit, self.action_limit)  # (C, H, act_dim)
        states_batch = jax.vmap(self._rollout_states_fn, in_axes=(None, 0))(x0_jnp, final_actions_batch)  # (C, H+1, state_dim)
        rewards_batch = jax.vmap(self._rollout_rewards_fn, in_axes=(None, 0))(x0_jnp, final_actions_batch)  # (C, H)
        
        # Batch energy computation using _cost_fn (JAX-compatible)
        # states_batch: (C, H+1, state_dim), actions_batch: (C, H, act_dim)
        # For each trajectory, compute energy at each time step: state[t+1], action[t]
        # Use vmap to batch over both trajectories and time steps
        states_for_energy = states_batch[:, 1:, :]  # (C, H, state_dim) - skip initial state
        # Vmap over both batch dimension and time dimension
        def compute_energy_batch(state, action):
            return self._cost_fn(state, action, {"t": 0})  # ctx doesn't matter for JAX
        
        # Vmap over both (C, H) dimensions
        energies_batch = jax.vmap(jax.vmap(compute_energy_batch, in_axes=(0, 0)), in_axes=(0, 0))(
            states_for_energy, final_actions_batch
        )  # (C, H)
        
        # Convert to numpy
        states_batch_np = np.asarray(states_batch)  # (C, H+1, state_dim)
        actions_batch_np = np.asarray(final_actions_batch)  # (C, H, act_dim)
        rewards_batch_np = np.asarray(rewards_batch)  # (C, H)
        energies_batch_np = np.asarray(energies_batch)  # (C, H)
        
        # Batch compute costs and rewards
        total_costs = -np.sum(rewards_batch_np, axis=-1)  # (C,)
        total_rewards = np.sum(rewards_batch_np, axis=-1)  # (C,)
        mean_rewards = np.mean(rewards_batch_np, axis=-1)  # (C,)
        
        # Build results list (only this loop remains, all computation is batched)
        C = rng_keys.shape[0]
        results = []
        for i in range(C):
            results.append({
                "actions": actions_batch_np[i],
                "states": states_batch_np[i],
                "rewards": rewards_batch_np[i],
                "total_reward": float(total_rewards[i]),
                "mean_reward": float(mean_rewards[i]),
                "initial_state": states_batch_np[i, 0],
                "energies": energies_batch_np[i],
                "reward_history": np.asarray(reward_hists[i]),
                "diffusion_actions_traj": np.asarray(actions_trajs[i]),
                "diffusion_sampled_actions": np.asarray(sampled_trajs[i]),
                "candidate_states": [states_batch_np[i]],
                "candidate_actions": [actions_batch_np[i]],
                "candidate_costs": np.asarray([float(total_costs[i])], dtype=np.float32),
                "best_idx": 0,
                "rng": rng_keys[i],  # Aligned with ebmbd: include rng in result
            })
        
        return results

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
        total_cost_final = -float(np.sum(rewards))  # Cost = -reward (lower is better)

        # Multi-mode: collect candidate trajectories from final diffusion step
        candidate_states_list = []
        candidate_actions_list = []
        candidate_costs_list = []
        
        if self.num_modes > 1 and len(sampled_traj) > 0:
            # Strategy: sample from multiple diffusion steps to get diverse candidates
            num_steps_to_sample = min(3, len(sampled_traj))
            step_indices = np.linspace(0, len(sampled_traj) - 1, num_steps_to_sample, dtype=int)
            
            all_samples_list = []
            all_states_list = []
            all_costs_list = []
            
            for step_idx in step_indices:
                step_samples = sampled_traj[step_idx]  # (M, H, act_dim)
                M = step_samples.shape[0]
                
                # Rollout all samples to get rewards
                states_step = self._rollout_states_batch_fn(x0_jnp, step_samples)  # (M, H+1, state_dim)
                rewards_step = self._rollout_rewards_batch_fn(x0_jnp, step_samples)  # (M, H)
                total_rewards_step = np.sum(rewards_step, axis=-1)  # (M,)
                
                # Cost = -reward (lower is better)
                total_costs_step = -total_rewards_step  # (M,)
                
                all_samples_list.append(step_samples)
                all_states_list.append(states_step)
                all_costs_list.append(total_costs_step)
            
            # Concatenate samples from all steps
            all_samples = np.concatenate(all_samples_list, axis=0)  # (M_total, H, act_dim)
            all_states = np.concatenate(all_states_list, axis=0)  # (M_total, H+1, state_dim)
            all_costs = np.concatenate(all_costs_list, axis=0)  # (M_total,)
            
            # Use diverse top-K selection
            selected_indices, selected_costs = diverse_topk_modes(
                all_samples,
                all_states,
                all_costs,
                C=self.num_modes,
                topK_cand=self.diversity_topK_cand,
                eta=self.diversity_eta,
                use_state_features=self.diversity_use_state,
                feature_stride=4,
            )
            
            # Extract candidate trajectories
            # selected_indices are global indices, selected_costs are the corresponding costs
            for i, idx in enumerate(selected_indices):
                candidate_states_list.append(np.asarray(all_states[idx], dtype=np.float32))
                candidate_actions_list.append(np.asarray(all_samples[idx], dtype=np.float32))
                candidate_costs_list.append(float(selected_costs[i]))  # Use i, not idx, since selected_costs is already indexed
            
            # Find best index (lowest cost)
            best_idx = int(np.argmin(candidate_costs_list))
        else:
            # Single mode: just use final trajectory
            candidate_states_list = [np.asarray(states_np, dtype=np.float32)]
            candidate_actions_list = [np.asarray(actions_np, dtype=np.float32)]
            candidate_costs_list = [float(total_cost_final)]
            best_idx = 0

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
            # Multi-mode candidates
            "candidate_states": candidate_states_list,
            "candidate_actions": candidate_actions_list,
            "candidate_costs": np.asarray(candidate_costs_list, dtype=np.float32),
            "best_idx": best_idx,
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

