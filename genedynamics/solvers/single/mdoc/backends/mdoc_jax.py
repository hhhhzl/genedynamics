"""
JAX backend implementation for MDOC.

MDOC is defined as:
  MBD diffusion driver + ConstraintFilter

This file intentionally mirrors the structure of `solvers/single/mbd/backends/mbd_jax.py`,
but inserts a `constraint_filter.apply_actions_batch(...)` hook every diffusion step.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.solvers.single.diffusion_adaptors import diverse_topk_modes
from genedynamics.experiments.plugins.obstacles.d3il_avoiding_fixed import get_d3il_target_line_positions

from genedynamics.core.types import Trajectory, State
from genedynamics.core.constraints.core.types import ScheduleState
from genedynamics.core.constraints.action_filters import ConstraintFilter, NoOpConstraintFilter


class MDOCBackendJax:
    def __init__(
        self,
        *,
        solver: Any = None,
        env_adapter=None,
        legacy_energy=None,
        horizon: int = 64,
        dt: float = 0.05,
        Nsample: int = 256,
        Ndiffuse: int = 100,
        temp_sample: float = 0.3,
        beta0: float = 1e-4,
        betaT: float = 1e-2,
        action_limit: float = 1.0,
        seed: int = 0,
        scheduler: Any = None,
        show_tqdm: bool = False,
        constraint_filter: Optional[ConstraintFilter] = None,
        obstacles: Any = None,
        **kwargs: Any,
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
            constraint_filter = solver.constraint_filter
            obstacles = solver.obstacles
            # Extract CBF params from solver
            if hasattr(solver, "cbf_params"):
                kwargs.update(solver.cbf_params)
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
        # Multi-mode support: number of candidate trajectories to return
        if solver is not None:
            self.num_modes = int(solver.config.get("num_modes", 1))
            self.use_target_line = bool(solver.config.get("use_target_line", False))
            self.num_targets = int(solver.config.get("num_targets", 4))
            self.diversity_eta = float(solver.config.get("diversity_eta", 1.0))
            self.diversity_topK_cand = int(solver.config.get("diversity_topK_cand", None) or (self.Nsample // 2))
            self.diversity_use_state = bool(solver.config.get("diversity_use_state", True))
        else:
            self.num_modes = 1
            self.use_target_line = False
            self.num_targets = 4
            self.diversity_eta = 1.0
            self.diversity_topK_cand = self.Nsample // 2
            self.diversity_use_state = True

        # Extract fixed CBF params from kwargs
        self.cbf_tau = float(kwargs.get("cbf_tau", 0.005))
        self.cbf_eta = float(kwargs.get("cbf_eta", 1.5))
        self.cbf_margin_fixed = float(kwargs.get("cbf_margin", 0.1))
        self.base_beta = float(kwargs.get("base_beta", 0.05))
        
        # New: Support for terminal cost and guidance (aligned with mdoc.py)
        self.terminal_weight = float(kwargs.get("terminal_energy_weight", 100.0))
        self.guide_weight = float(kwargs.get("guide_weight", 20.0)) 

        self._build_jax_functions()

        # Precompute constraint schedule arrays (so we don't build Python objects inside JAX transforms).
        self._margin_arr = None
        self._rho_arr = None
        self._qp_gate_arr = None
        self._qp_prob_arr = None
        self._topK = -1
        try:
            margin_list: List[float] = []
            rho_list: List[float] = []
            qp_gate_list: List[bool] = []
            qp_prob_list: List[float] = []
            topK_val = None
            if self.scheduler is not None and hasattr(self.scheduler, "constraint_schedulers"):
                cs_list = getattr(self.scheduler, "constraint_schedulers", [])
                if cs_list:
                    cs = cs_list[0]
                    for k in range(self.Ndiffuse):
                        st = ScheduleState(k=k, K=max(1, self.Ndiffuse - 1))
                        d = cs.constraint_params(st) or {}
                        margin_list.append(float(d.get("margin", 0.0)))
                        rho_list.append(float(d.get("rho", 1.0)))
                        qp_gate_list.append(bool(d.get("qp_gate", True)))
                        qp_prob_list.append(float(d.get("qp_prob", 1.0)))
                        topK_val = d.get("topK", topK_val)
            if margin_list:
                self._margin_arr = jnp.asarray(margin_list, dtype=jnp.float32)
                self._rho_arr = jnp.asarray(rho_list, dtype=jnp.float32)
                self._qp_gate_arr = jnp.asarray(qp_gate_list, dtype=jnp.bool_)
                self._qp_prob_arr = jnp.asarray(qp_prob_list, dtype=jnp.float32)
            if topK_val is not None:
                self._topK = int(topK_val)
        except Exception:
            self._margin_arr = None
            self._rho_arr = None
            self._qp_gate_arr = None
            self._qp_prob_arr = None

        # Precompute diffusion temperature schedule (T_k) if scheduler provides it.
        self._T_k_arr = None
        try:
            T_k_list: List[float] = []
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
            target_obj = getattr(self.env, "target", None)
            target = jnp.asarray(target_obj, dtype=jnp.float32) if target_obj is not None else jnp.zeros(2)
            
            d0 = jnp.linalg.norm(state_init[0:2] - target[0:2])
            t_star = float(self.horizon - 1)
            
            def step_fn(carry, action_and_t):
                next_state, cum_reward = carry
                action, t = action_and_t
                
                next_state = self._transition_fn(next_state, action)
                ctx = {"t": t}
                
                # Standard running cost
                reward = -self._cost_fn(next_state, action, ctx)
                
                # Guidance cost (Time smoothness)
                d_hat = d0 * jnp.maximum(0.0, 1.0 - t / (t_star + 1e-6))
                dist_to_goal = jnp.linalg.norm(next_state[0:2] - target[0:2])
                guide_reward = -self.guide_weight * jnp.square(dist_to_goal - d_hat)
                
                step_total = reward + guide_reward
                return (next_state, cum_reward + step_total), step_total

            t_indices = jnp.arange(self.horizon, dtype=jnp.float32)
            (final_state, _), step_rewards = jax.lax.scan(
                step_fn, (state_init, 0.0), (actions, t_indices)
            )
            
            # Terminal reward
            terminal_dist = jnp.linalg.norm(final_state[0:2] - target[0:2])
            terminal_reward = -self.terminal_weight * terminal_dist
            
            # Add terminal reward to the last step for visualization consistency
            step_rewards = step_rewards.at[-1].add(terminal_reward)
            return step_rewards

        def rollout_rewards_with_target(state_init, actions, target):
            target = jnp.asarray(target, dtype=jnp.float32).reshape(-1)[:2]
            d0 = jnp.linalg.norm(state_init[0:2] - target[0:2])
            t_star = float(self.horizon - 1)

            def step_fn(carry, action_and_t):
                next_state, cum_reward = carry
                action, t = action_and_t
                next_state = self._transition_fn(next_state, action)
                ctx = {"t": t, "target_xy": target}
                reward = -self._cost_fn(next_state, action, ctx)
                d_hat = d0 * jnp.maximum(0.0, 1.0 - t / (t_star + 1e-6))
                dist_to_goal = jnp.linalg.norm(next_state[0:2] - target[0:2])
                guide_reward = -self.guide_weight * jnp.square(dist_to_goal - d_hat)
                step_total = reward + guide_reward
                return (next_state, cum_reward + step_total), step_total

            t_indices = jnp.arange(self.horizon, dtype=jnp.float32)
            (final_state, _), step_rewards = jax.lax.scan(
                step_fn, (state_init, 0.0), (actions, t_indices)
            )
            terminal_dist = jnp.linalg.norm(final_state[0:2] - target[0:2])
            terminal_reward = -self.terminal_weight * terminal_dist
            step_rewards = step_rewards.at[-1].add(terminal_reward)
            return step_rewards

        self._rollout_rewards_fn = jax.jit(rollout_rewards)
        self._rollout_rewards_batch_fn = jax.jit(jax.vmap(self._rollout_rewards_fn, in_axes=(None, 0)))
        self._rollout_rewards_with_target_fn = jax.jit(rollout_rewards_with_target)
        self._rollout_rewards_with_target_batch_fn = jax.jit(jax.vmap(self._rollout_rewards_with_target_fn, in_axes=(None, 0, None)))

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
        betas = jnp.linspace(self.beta0, self.betaT, self.Ndiffuse, dtype=jnp.float32)
        alphas = 1.0 - betas
        alphas_bar = jnp.cumprod(alphas)
        sigmas = jnp.sqrt(1.0 - alphas_bar)
        diffusion_indices = jnp.arange(self.Ndiffuse - 1, 0, -1, dtype=jnp.int32)
        total_steps = int(self.Ndiffuse - 1)
        total_steps_jnp = jnp.asarray(total_steps, dtype=jnp.int32)

        def reverse_diffuse(rng_in, Ybar_init):
            def body(carry, idx):
                rng_curr, Ybar_curr = carry
                rng_curr, noise_key = jax.random.split(rng_curr)

                # Generate candidates around current mean
                eps = jax.random.normal(noise_key, (self.Nsample, self.horizon, self.act_dim), dtype=jnp.float32)
                Y0s = eps * sigmas[idx] + Ybar_curr
                Y0s = jnp.clip(Y0s, -self.action_limit, self.action_limit)

                step_k = jnp.asarray((self.Ndiffuse - 1), dtype=jnp.int32) - idx

                sched_state = {"k": step_k, "K": total_steps_jnp}
                if self._margin_arr is not None:
                    kk = jnp.clip(step_k, 0, self._margin_arr.shape[0] - 1)
                    margin = self._margin_arr[kk]
                    rho = self._rho_arr[kk]
                    qp_gate = self._qp_gate_arr[kk]
                    qp_prob = self._qp_prob_arr[kk]
                else:
                    margin = jnp.asarray(0.0, dtype=jnp.float32)
                    rho = jnp.asarray(1.0, dtype=jnp.float32)
                    qp_gate = jnp.asarray(True, dtype=jnp.bool_)
                    qp_prob = jnp.asarray(1.0, dtype=jnp.float32)
                
                sched_params = {
                    "margin": margin, 
                    "rho": rho, 
                    "qp_gate": qp_gate, 
                    "qp_prob": qp_prob, 
                    "topK": jnp.asarray(self._topK, dtype=jnp.int32),
                    "cbf_tau": jnp.asarray(self.cbf_tau, dtype=jnp.float32),
                    "cbf_eta": jnp.asarray(self.cbf_eta, dtype=jnp.float32),
                    "cbf_margin": jnp.asarray(self.cbf_margin_fixed, dtype=jnp.float32),
                    "base_beta": jnp.asarray(self.base_beta, dtype=jnp.float32),
                }

                # Apply Filter per diffusion step (MDOC core)
                Y0s_f = self.constraint_filter.apply_actions_batch(
                    x0_jnp, Y0s, env=self.env, obstacles=self.obstacles,
                    schedule_state=sched_state, schedule_params=sched_params,
                )

                # Importance weighting
                rews_per_step = self._rollout_rewards_batch_fn(x0_jnp, Y0s_f)
                rews = jnp.sum(rews_per_step, axis=-1)
                
                rew_mean = jnp.mean(rews)
                rew_std = jnp.std(rews)
                rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)

                T_k = self.temp_sample
                if self._T_k_arr is not None:
                    kkT = jnp.clip(step_k, 0, self._T_k_arr.shape[0] - 1)
                    T_k = self._T_k_arr[kkT]

                logp0 = (rews - rew_mean) / (rew_std * T_k)
                weights = jax.nn.softmax(logp0)
                Ybar_weighted = jnp.einsum("n,nij->ij", weights, Y0s_f)

                # Refinement update
                Ybar_next = Ybar_weighted

                # Filter the mean too
                Ybar_next = self.constraint_filter.apply_actions(
                    x0_jnp, Ybar_next, env=self.env, obstacles=self.obstacles,
                    schedule_state=sched_state, schedule_params=sched_params,
                )

                return (rng_curr, Ybar_next), (jnp.mean(rews), Ybar_next, Y0s_f)

            (rng_out, Ybar_final), (reward_hist, Ybar_hist, Ysamples_hist) = jax.lax.scan(
                body, (rng_in, Ybar_init), diffusion_indices
            )
            return rng_out, Ybar_final, reward_hist[::-1], Ybar_hist[::-1], Ysamples_hist[::-1]

        reverse_diffuse_jit = jax.jit(reverse_diffuse)

        Ybar_init = jnp.zeros((self.horizon, self.act_dim), dtype=jnp.float32)
        _, Ybar_final, reward_hist, actions_traj, sampled_traj = reverse_diffuse_jit(diffuse_rng, Ybar_init)

        final_actions = jnp.clip(Ybar_final, -self.action_limit, self.action_limit)
        states = self._rollout_states_fn(x0_jnp, final_actions)
        rewards = self._rollout_rewards_fn(x0_jnp, final_actions)

        states_np = np.asarray(states)
        actions_np = np.asarray(final_actions)
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
        
        # Build standard result dict
        res = {
            "actions": actions_np,
            "states": states_np,
            "rewards": np.asarray(rewards, dtype=np.float32),
            "initial_state": states_np[0],
            "diffusion_rewards": np.asarray(reward_hist),
            "diffusion_actions_traj": np.asarray(actions_traj),
            "diffusion_sampled_actions": np.asarray(sampled_traj),
            # Multi-mode candidates
            "candidate_states": candidate_states_list,
            "candidate_actions": candidate_actions_list,
            "candidate_costs": np.asarray(candidate_costs_list, dtype=np.float32),
            "best_idx": best_idx,
        }
        return res
    
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
        C = int(rng_keys.shape[0])
        if getattr(self, "use_target_line", False) and C > 0:
            target_line = get_d3il_target_line_positions(getattr(self, "num_targets", 4))
            targets_per_mode = jnp.asarray(
                np.asarray([target_line[i % len(target_line)] for i in range(C)], dtype=np.float32),
                dtype=jnp.float32,
            )
        else:
            default_tgt = getattr(self.env, "target", None)
            default_tgt = np.asarray(default_tgt, dtype=np.float32).reshape(-1)[:2] if default_tgt is not None else np.zeros(2, dtype=np.float32)
            targets_per_mode = jnp.tile(jnp.asarray(default_tgt, dtype=jnp.float32), (C, 1))

        betas = jnp.linspace(self.beta0, self.betaT, self.Ndiffuse, dtype=jnp.float32)
        alphas = 1.0 - betas
        alphas_bar = jnp.cumprod(alphas)
        sigmas = jnp.sqrt(1.0 - alphas_bar)
        diffusion_indices = jnp.arange(self.Ndiffuse - 1, 0, -1, dtype=jnp.int32)
        total_steps = int(self.Ndiffuse - 1)
        total_steps_jnp = jnp.asarray(total_steps, dtype=jnp.int32)
        
        def reverse_diffuse_core(rng_key, target):
            # Split rng inside core function (aligned with ebmbd)
            rng, _ = jax.random.split(rng_key)
            def body(carry, idx):
                rng_curr, Ybar_curr = carry
                rng_curr, noise_key = jax.random.split(rng_curr)
                
                eps = jax.random.normal(noise_key, (self.Nsample, self.horizon, self.act_dim), dtype=jnp.float32)
                Y0s = eps * sigmas[idx] + Ybar_curr
                Y0s = jnp.clip(Y0s, -self.action_limit, self.action_limit)
                
                step_k = jnp.asarray((self.Ndiffuse - 1), dtype=jnp.int32) - idx
                sched_state = {"k": step_k, "K": total_steps_jnp}
                if self._margin_arr is not None:
                    kk = jnp.clip(step_k, 0, self._margin_arr.shape[0] - 1)
                    margin = self._margin_arr[kk]
                    rho = self._rho_arr[kk]
                    qp_gate = self._qp_gate_arr[kk]
                    qp_prob = self._qp_prob_arr[kk]
                else:
                    margin = jnp.asarray(0.0, dtype=jnp.float32)
                    rho = jnp.asarray(1.0, dtype=jnp.float32)
                    qp_gate = jnp.asarray(True, dtype=jnp.bool_)
                    qp_prob = jnp.asarray(1.0, dtype=jnp.float32)
                
                sched_params = {
                    "margin": margin,
                    "rho": rho,
                    "qp_gate": qp_gate,
                    "qp_prob": qp_prob,
                    "topK": jnp.asarray(self._topK, dtype=jnp.int32),
                    "cbf_tau": jnp.asarray(self.cbf_tau, dtype=jnp.float32),
                    "cbf_eta": jnp.asarray(self.cbf_eta, dtype=jnp.float32),
                    "cbf_margin": jnp.asarray(self.cbf_margin_fixed, dtype=jnp.float32),
                    "base_beta": jnp.asarray(self.base_beta, dtype=jnp.float32),
                }
                
                Y0s_f = self.constraint_filter.apply_actions_batch(
                    x0_jnp, Y0s, env=self.env, obstacles=self.obstacles,
                    schedule_state=sched_state, schedule_params=sched_params,
                )
                
                rews_per_step = self._rollout_rewards_with_target_batch_fn(x0_jnp, Y0s_f, target)
                rews = jnp.sum(rews_per_step, axis=-1)
                
                rew_mean = jnp.mean(rews)
                rew_std = jnp.std(rews)
                rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)
                
                T_k = self.temp_sample
                if self._T_k_arr is not None:
                    kkT = jnp.clip(step_k, 0, self._T_k_arr.shape[0] - 1)
                    T_k = self._T_k_arr[kkT]
                
                logp0 = (rews - rew_mean) / (rew_std * T_k)
                weights = jax.nn.softmax(logp0)
                Ybar_weighted = jnp.einsum("n,nij->ij", weights, Y0s_f)
                Ybar_next = Ybar_weighted
                Ybar_next = self.constraint_filter.apply_actions(
                    x0_jnp, Ybar_next, env=self.env, obstacles=self.obstacles,
                    schedule_state=sched_state, schedule_params=sched_params,
                )
                
                return (rng_curr, Ybar_next), (jnp.mean(rews), Ybar_next, Y0s_f)
            
            Ybar_init = jnp.zeros((self.horizon, self.act_dim), dtype=jnp.float32)
            (rng_out, Ybar_final), (reward_hist, Ybar_hist, Ysamples_hist) = jax.lax.scan(
                body, (rng, Ybar_init), diffusion_indices
            )
            return Ybar_final, reward_hist[::-1], Ybar_hist[::-1], Ysamples_hist[::-1]
        
        # Vmap over (rng_keys, targets_per_mode) for per-mode target support
        reverse_diffuse_batch_jit = jax.jit(jax.vmap(reverse_diffuse_core, in_axes=(0, 0)))
        Ybar_finals, reward_hists, actions_trajs, sampled_trajs = reverse_diffuse_batch_jit(rng_keys, targets_per_mode)
        
        # Batch post-processing: clip, rollout states and rewards (use per-mode target when use_target_line)
        final_actions_batch = jnp.clip(Ybar_finals, -self.action_limit, self.action_limit)  # (C, H, act_dim)
        states_batch = jax.vmap(self._rollout_states_fn, in_axes=(None, 0))(x0_jnp, final_actions_batch)  # (C, H+1, state_dim)
        rewards_batch = jax.vmap(self._rollout_rewards_with_target_fn, in_axes=(None, 0, 0))(x0_jnp, final_actions_batch, targets_per_mode)  # (C, H)
        
        # Batch energy computation using _cost_fn (JAX-compatible, per-mode target)
        states_for_energy = states_batch[:, 1:, :]  # (C, H, state_dim) - skip initial state
        def compute_energy_with_target(state, action, target):
            return self._cost_fn(state, action, {"t": 0, "target_xy": target})
        energies_batch = jax.vmap(
            jax.vmap(compute_energy_with_target, in_axes=(0, 0, None)),
            in_axes=(0, 0, 0),
        )(states_for_energy, final_actions_batch, targets_per_mode)  # (C, H)
        
        # Convert to numpy
        states_batch_np = np.asarray(states_batch)  # (C, H+1, state_dim)
        actions_batch_np = np.asarray(final_actions_batch)  # (C, H, act_dim)
        rewards_batch_np = np.asarray(rewards_batch)  # (C, H)
        energies_batch_np = np.asarray(energies_batch)  # (C, H)
        
        # Batch compute costs and rewards
        total_costs = -np.sum(rewards_batch_np, axis=-1)  # (C,)
        total_rewards = np.sum(rewards_batch_np, axis=-1)  # (C,)
        
        # Build results list (aligned with ebmbd format)
        C = rng_keys.shape[0]
        results = []
        for i in range(C):
            results.append({
                "actions": actions_batch_np[i],
                "states": states_batch_np[i],
                "rewards": rewards_batch_np[i],
                "initial_state": states_batch_np[i, 0],
                "reward_history": np.asarray(reward_hists[i]),  # Aligned with ebmbd: use "reward_history" instead of "diffusion_rewards"
                "diffusion_actions_traj": np.asarray(actions_trajs[i]),
                "diffusion_sampled_actions": np.asarray(sampled_trajs[i]),
                "candidate_states": [states_batch_np[i]],
                "candidate_actions": [actions_batch_np[i]],
                "candidate_costs": np.asarray([float(total_costs[i])], dtype=np.float32),
                "best_idx": 0,
                "rng": rng_keys[i],  # Aligned with ebmbd: include rng in result
            })
        
        return results

    def sample_trajectories(self, x0: State, n_samples: int, rng_key: Optional[Any] = None) -> List[Trajectory]:
        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.seed)
        res = self.plan(x0, rng_key)
        traj = Trajectory(
            states=[np.asarray(s, dtype=np.float32) for s in res["states"]],
            actions=[np.asarray(a, dtype=np.float32) for a in res["actions"]],
            info=res
        )
        return [traj] * n_samples
