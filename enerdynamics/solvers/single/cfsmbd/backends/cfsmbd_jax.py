"""
JAX backend implementation for CFS-MBD.

CFS-MBD is defined as:
  MBD diffusion driver + Augmented Lagrangian objective + CFS-based per-step QP projection

This file mirrors the structure of `mdoc/backends/mdoc_jax.py`,
but adds augmented Lagrangian penalty in the reward computation.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from enerdynamics.solvers.single.diffusion_adaptors import diverse_topk_modes

from enerdynamics.core.types import Trajectory, State
from enerdynamics.core.constraints.core.types import ScheduleState
from enerdynamics.core.constraints.action_filters import ConstraintFilter, NoOpConstraintFilter


class CFSMBDBackendJax:
    def __init__(
        self,
        *,
        solver: Any = None,
        env_adapter=None,
        legacy_energy=None,
        horizon: int = 64,
        dt: float = 0.05,
        Nsample: int = 4096,
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
        aug_lambda: float = 0.0,
        aug_rho: float = 1.0,
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
            aug_lambda = solver.config.get("aug_lambda", 0.0)
            aug_rho = solver.config.get("aug_rho", 1.0)
            action_extra_sigma = solver.config.get("action_extra_sigma", 0.0)
        else:
            action_extra_sigma = kwargs.get("action_extra_sigma", 0.0)
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
        self.aug_lambda = float(aug_lambda)
        self.aug_rho = float(aug_rho)
        self.action_extra_sigma = float(action_extra_sigma)
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

        # Resolve constraint scheduler (composite vs single)
        self._cs = None
        if self.scheduler is not None and hasattr(self.scheduler, "constraint_schedulers"):
            cs_list = getattr(self.scheduler, "constraint_schedulers", [])
            if cs_list:
                self._cs = cs_list[0]
        elif self.scheduler is not None:
            self._cs = self.scheduler

        self._use_jax_adaptive = (
            self._cs is not None
            and hasattr(self._cs, "jax_init_carry")
            and hasattr(self._cs, "jax_compute_params")
            and hasattr(self._cs, "jax_update")
        )

        # Precompute constraint schedule arrays (skip when using JAX adaptive).
        self._margin_arr = None
        self._rho_arr = None
        self._qp_gate_arr = None
        self._qp_prob_arr = None
        self._topK = -1
        if not self._use_jax_adaptive:
            try:
                margin_list: List[float] = []
                rho_list: List[float] = []
                qp_gate_list: List[bool] = []
                qp_prob_list: List[float] = []
                topK_val = None
                if self._cs is not None:
                    for k in range(self.Ndiffuse):
                        st = ScheduleState(k=k, K=max(1, self.Ndiffuse))
                        d = self._cs.constraint_params(st) or {}
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
                        st = ScheduleState(k=k, K=max(1, self.Ndiffuse))
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

        # Build SDF function for constraint evaluation (JAX-compatible)
        def sdf_fn(pos, clearance):
            """Compute SDF and constraint violation [g]_+."""
            if self.obstacles is None:
                return jnp.asarray(0.0, dtype=jnp.float32)
            
            # Try to use JAX-compatible SDF
            try:
                if hasattr(self.obstacles, "jax_sdf"):
                    sdf_val = self.obstacles.jax_sdf(pos)
                elif hasattr(self.obstacles, "sample_sdf_and_grad_2d"):
                    # Use texture-based SDF (faster)
                    sdf_result, _ = self.obstacles.sample_sdf_and_grad_2d(
                        pos[None, :], backend="jax"
                    )
                    sdf_val = sdf_result[0] if isinstance(sdf_result, (list, tuple)) else sdf_result
                    if isinstance(sdf_val, np.ndarray):
                        sdf_val = jnp.asarray(sdf_val)
                else:
                    # Fallback: return 0 (no constraint)
                    return jnp.asarray(0.0, dtype=jnp.float32)
                
                # Convert to scalar if needed
                if isinstance(sdf_val, jnp.ndarray) and sdf_val.size > 0:
                    sdf_val = sdf_val[0] if sdf_val.ndim > 0 else sdf_val
                sdf_val = jnp.asarray(sdf_val, dtype=jnp.float32)
                
                # g_t = clearance - sdf, [g_t]_+ = max(0, g_t)
                g_t = clearance - sdf_val
                g_plus = jnp.maximum(0.0, g_t)
                return g_plus
            except Exception:
                # If SDF computation fails, return 0 (no violation)
                return jnp.asarray(0.0, dtype=jnp.float32)

        def rollout_rewards_with_augmented(state_init, actions, clearance, aug_lambda, aug_rho):
            """
            Rollout rewards with augmented Lagrangian penalty.
            
            Returns total augmented reward (scalar).
            """
            target_obj = getattr(self.env, "target", None)
            target = jnp.asarray(target_obj, dtype=jnp.float32) if target_obj is not None else jnp.zeros(2)
            
            d0 = jnp.linalg.norm(state_init[0:2] - target[0:2])
            t_star = float(self.horizon - 1)
            
            def step_fn(carry, action_and_t):
                next_state, cum_reward, cum_g_plus = carry
                action, t = action_and_t
                
                next_state = self._transition_fn(next_state, action)
                ctx = {"t": t}
                
                # Standard running cost
                reward = -self._cost_fn(next_state, action, ctx)
                
                # Constraint violation [g]_+
                pos = next_state[0:2]  # single_2d: state is position
                g_plus = sdf_fn(pos, clearance)
                
                step_total = reward
                return (next_state, cum_reward + step_total, cum_g_plus + g_plus), (step_total, g_plus)

            t_indices = jnp.arange(self.horizon, dtype=jnp.float32)
            (final_state, total_reward, total_g_plus), (step_rewards, step_g_plus) = jax.lax.scan(
                step_fn, (state_init, 0.0, 0.0), (actions, t_indices)
            )
            
            # Terminal reward
            terminal_dist = jnp.linalg.norm(final_state[0:2] - target[0:2])
            terminal_reward = -100.0 * terminal_dist  # Fixed terminal weight for now
            
            # Augmented Lagrangian penalty (mean over horizon to avoid H-scale explosion)
            # J(τ) + λ mean_t [g]_+ + (ρ/2) mean_t [g]_+^2
            H = jnp.asarray(self.horizon, dtype=jnp.float32)
            dual_term = aug_lambda * (total_g_plus / H)
            penalty_term = (aug_rho / 2.0) * jnp.mean(step_g_plus ** 2)
            augmented_penalty = dual_term + penalty_term
            
            # Total reward = sum of step rewards + terminal - augmented penalty
            total_augmented_reward = total_reward + terminal_reward - augmented_penalty
            
            return total_augmented_reward

        def rollout_rewards_with_augmented_and_v(state_init, actions, clearance, aug_lambda, aug_rho):
            """Same as above but also returns v_n = sum_t [g]_+ for residual r_p."""
            target_obj = getattr(self.env, "target", None)
            target = jnp.asarray(target_obj, dtype=jnp.float32) if target_obj is not None else jnp.zeros(2)
            
            def step_fn(carry, action_and_t):
                next_state, cum_reward, cum_g_plus = carry
                action, t = action_and_t
                next_state = self._transition_fn(next_state, action)
                ctx = {"t": t}
                reward = -self._cost_fn(next_state, action, ctx)
                pos = next_state[0:2]
                g_plus = sdf_fn(pos, clearance)
                return (next_state, cum_reward + reward, cum_g_plus + g_plus), (reward, g_plus)

            t_indices = jnp.arange(self.horizon, dtype=jnp.float32)
            (final_state, total_reward, total_g_plus), (step_rewards, step_g_plus) = jax.lax.scan(
                step_fn, (state_init, 0.0, 0.0), (actions, t_indices)
            )
            terminal_dist = jnp.linalg.norm(final_state[0:2] - target[0:2])
            terminal_reward = -100.0 * terminal_dist
            H = jnp.asarray(self.horizon, dtype=jnp.float32)
            dual_term = aug_lambda * (total_g_plus / H)
            penalty_term = (aug_rho / 2.0) * jnp.mean(step_g_plus ** 2)
            total_augmented_reward = total_reward + terminal_reward - dual_term - penalty_term
            v_n = total_g_plus / H
            return total_augmented_reward, v_n

        def rollout_rewards(state_init, actions):
            """Rollout rewards per step (for visualization)."""
            target_obj = getattr(self.env, "target", None)
            target = jnp.asarray(target_obj, dtype=jnp.float32) if target_obj is not None else jnp.zeros(2)
            
            def step_fn(carry, action_and_t):
                next_state, cum_reward = carry
                action, t = action_and_t
                
                next_state = self._transition_fn(next_state, action)
                ctx = {"t": t}
                reward = -self._cost_fn(next_state, action, ctx)
                return (next_state, cum_reward + reward), reward

            t_indices = jnp.arange(self.horizon, dtype=jnp.float32)
            (final_state, _), step_rewards = jax.lax.scan(
                step_fn, (state_init, 0.0), (actions, t_indices)
            )
            
            # Terminal reward
            terminal_dist = jnp.linalg.norm(final_state[0:2] - target[0:2])
            terminal_reward = -100.0 * terminal_dist
            step_rewards = step_rewards.at[-1].add(terminal_reward)
            
            return step_rewards

        # JIT compile functions
        self._rollout_rewards_fn = jax.jit(rollout_rewards)
        
        # Batch version with augmented Lagrangian (takes clearance, aug_lambda, aug_rho as static)
        # Note: We need to make clearance, aug_lambda, aug_rho static args since they vary per diffusion step
        self._rollout_rewards_with_augmented_fn = rollout_rewards_with_augmented
        self._rollout_augmented_and_v_fn = rollout_rewards_with_augmented_and_v

        # ---------------------------------------------------------------------
        # Performance (architecture): pre-jit heavyweight subroutines so the
        # outer reverse-diffusion jit does NOT inline their full graphs.
        # This reduces compilation time and improves cache reuse.
        # ---------------------------------------------------------------------
        def filter_actions_batch(x0_in, actions_batch, sched_state, sched_params):
            return self.constraint_filter.apply_actions_batch(
                x0_in,
                actions_batch,
                env=self.env,
                obstacles=self.obstacles,
                schedule_state=sched_state,
                schedule_params=sched_params,
            )

        def filter_actions_single(x0_in, actions_single, sched_state, sched_params):
            return self.constraint_filter.apply_actions(
                x0_in,
                actions_single,
                env=self.env,
                obstacles=self.obstacles,
                schedule_state=sched_state,
                schedule_params=sched_params,
            )

        def augmented_rewards_batch(x0_in, actions_batch, clearance, aug_lambda, aug_rho):
            return jax.vmap(
                lambda actions_seq: rollout_rewards_with_augmented(x0_in, actions_seq, clearance, aug_lambda, aug_rho),
                in_axes=0,
            )(actions_batch)

        def augmented_and_v_batch(x0_in, actions_batch, clearance, aug_lambda, aug_rho):
            rews, v = jax.vmap(
                lambda actions_seq: rollout_rewards_with_augmented_and_v(x0_in, actions_seq, clearance, aug_lambda, aug_rho),
                in_axes=0,
            )(actions_batch)
            return rews, v

        self._filter_actions_batch_jit = jax.jit(filter_actions_batch)
        self._filter_actions_single_jit = jax.jit(filter_actions_single)
        self._augmented_rewards_batch_jit = jax.jit(augmented_rewards_batch)
        self._augmented_and_v_batch_jit = jax.jit(augmented_and_v_batch)

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
        # Reverse diffusion: 99 (noisiest) -> 0 (clean), 100 steps.
        diffusion_indices = jnp.arange(self.Ndiffuse - 1, -1, -1, dtype=jnp.int32)
        total_steps = int(self.Ndiffuse)
        total_steps_jnp = jnp.asarray(total_steps, dtype=jnp.int32)
        k_top = max(1, min(self.Nsample, int(math.ceil(0.1 * self.Nsample))))
        
        # Extra noise schedule (decays over diffusion steps, aligned with ebmbd)
        denom = jnp.maximum(float(self.Ndiffuse - 1), 1.0)
        progress_inc_by_idx = 1.0 - (jnp.arange(self.Ndiffuse, dtype=jnp.float32) / denom)
        extra_sigmas_by_idx = self.action_extra_sigma * (1.0 - progress_inc_by_idx)

        print("[CFS-MBD JAX] _use_jax_adaptive =", self._use_jax_adaptive)

        def reverse_diffuse(rng_in, Ybar_init):
            def body(carry, idx):
                rng_curr, Ybar_curr = carry
                # Need separate RNGs for diffusion noise and extra noise.
                rng_curr, noise_key, extra_key = jax.random.split(rng_curr, 3)

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
                    "topK": jnp.asarray(self._topK if self._topK >= 0 else 8, dtype=jnp.int32),
                }

                Y0s_f = self._filter_actions_batch_jit(x0_jnp, Y0s, sched_state, sched_params)

                def compute_augmented_reward(actions_seq):
                    return self._rollout_rewards_with_augmented_fn(
                        x0_jnp, actions_seq, margin,
                        jnp.asarray(self.aug_lambda, dtype=jnp.float32),
                        jnp.asarray(self.aug_rho, dtype=jnp.float32),
                    )

                rews = self._augmented_rewards_batch_jit(
                    x0_jnp,
                    Y0s_f,
                    margin,
                    jnp.asarray(self.aug_lambda, dtype=jnp.float32),
                    jnp.asarray(self.aug_rho, dtype=jnp.float32),
                )
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
                Ybar_next = self._filter_actions_single_jit(x0_jnp, Ybar_next, sched_state, sched_params)
                
                # Add extra noise for diversity (decays over diffusion steps, aligned with ebmbd)
                extra_sigma = extra_sigmas_by_idx[idx]
                noise_extra = jax.random.normal(extra_key, (self.horizon, self.act_dim), dtype=jnp.float32)
                Ybar_next = Ybar_next + extra_sigma * noise_extra
                Ybar_next = jnp.clip(Ybar_next, -self.action_limit, self.action_limit)

                return (rng_curr, Ybar_next), (jnp.mean(rews), Ybar_next, Y0s_f)

            (rng_out, Ybar_final), (reward_hist, Ybar_hist, Ysamples_hist) = jax.lax.scan(
                body, (rng_in, Ybar_init), diffusion_indices
            )
            return rng_out, Ybar_final, reward_hist[::-1], Ybar_hist[::-1], Ysamples_hist[::-1]

        def reverse_diffuse_adaptive(rng_in, Ybar_init, carry_sched_init):
            from enerdynamics.core.constraints.schedulers.ConstraintScheduler.almadaptive.backends.alm_adaptive_jax import quantile_90

            cs = self._cs
            filter_fn = self.constraint_filter
            
            # Extra noise schedule (decays over diffusion steps, aligned with ebmbd)
            denom_adaptive = jnp.maximum(float(self.Ndiffuse - 1), 1.0)
            progress_inc_by_idx_adaptive = 1.0 - (jnp.arange(self.Ndiffuse, dtype=jnp.float32) / denom_adaptive)
            extra_sigmas_by_idx_adaptive = self.action_extra_sigma * (1.0 - progress_inc_by_idx_adaptive)

            def body(carry, idx):
                rng_curr, Ybar_curr, carry_sched = carry
                rng_curr, noise_key, extra_key = jax.random.split(rng_curr, 3)

                eps = jax.random.normal(noise_key, (self.Nsample, self.horizon, self.act_dim), dtype=jnp.float32)
                Y0s = eps * sigmas[idx] + Ybar_curr
                Y0s = jnp.clip(Y0s, -self.action_limit, self.action_limit)

                step_k = jnp.asarray((self.Ndiffuse - 1), dtype=jnp.int32) - idx
                sched_state = {"k": step_k, "K": total_steps_jnp}

                params, rng_sched_next = cs.jax_compute_params(carry_sched, step_k, total_steps)
                do_qp = params["qp_gate"]
                margin = params["margin"]
                aug_lam = params["aug_lambda"]
                aug_rho = params["aug_rho"]

                sched_params = {
                    "margin": margin,
                    "rho": params["rho"],
                    "qp_gate": do_qp,
                    "qp_prob": params["qp_prob"],
                    "topK": params["topK"],
                    "eps": params["eps"],
                    "I_QP": params["I_QP"],
                }

                def apply_filter(ys):
                    return self._filter_actions_batch_jit(x0_jnp, ys, sched_state, sched_params)

                Y0s_f = jax.lax.cond(do_qp, apply_filter, lambda ys: ys, Y0s)

                def rollout_one(actions_seq):
                    return self._rollout_augmented_and_v_fn(
                        x0_jnp, actions_seq, margin, aug_lam, aug_rho,
                    )

                rews, v_batch = self._augmented_and_v_batch_jit(x0_jnp, Y0s_f, margin, aug_lam, aug_rho)
                r_p = quantile_90(v_batch, k_top)
                proj = jnp.mean(jnp.linalg.norm(Y0s_f - Y0s, axis=(1, 2)))
                feedback = {"r_p": r_p, "proj": proj}
                carry_sched_new = cs.jax_update(carry_sched, feedback, rng_sched_next)

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

                def filter_mean(_):
                    return self._filter_actions_single_jit(x0_jnp, Ybar_weighted, sched_state, sched_params)

                Ybar_next = jax.lax.cond(do_qp, filter_mean, lambda _: Ybar_weighted, operand=None)
                
                # Add extra noise for diversity (decays over diffusion steps, aligned with ebmbd)
                extra_sigma = extra_sigmas_by_idx_adaptive[idx]
                noise_extra = jax.random.normal(extra_key, (self.horizon, self.act_dim), dtype=jnp.float32)
                Ybar_next = Ybar_next + extra_sigma * noise_extra
                Ybar_next = jnp.clip(Ybar_next, -self.action_limit, self.action_limit)

                return (rng_curr, Ybar_next, carry_sched_new), (
                    jnp.mean(rews), Ybar_next, Y0s_f,
                    margin, params["rho"], params["topK"],
                    params["I_QP"], params["eps"], aug_lam, params["qp_prob"],
                )

            (rng_out, Ybar_final, _), (
                reward_hist, Ybar_hist, Ysamples_hist,
                margin_hist, rho_hist, topK_hist,
                I_hist, eps_hist, lam_hist, p_hist,
            ) = jax.lax.scan(body, (rng_in, Ybar_init, carry_sched_init), diffusion_indices)
            # Keep reward/Ybar/sampled reversed for downstream (index 0 = clean). Log arrays
            # stay in scan order: [0]=first iter (noisy, idx=99), [-1]=last (clean, idx=1).
            return (
                rng_out, Ybar_final,
                reward_hist[::-1], Ybar_hist[::-1], Ysamples_hist[::-1],
                margin_hist, rho_hist, topK_hist,
                I_hist, eps_hist, lam_hist, p_hist,
            )

        reverse_diffuse_jit = jax.jit(reverse_diffuse)

        Ybar_init = jnp.zeros((self.horizon, self.act_dim), dtype=jnp.float32)
        if self._use_jax_adaptive:
            rng_d, rng_sched = jax.random.split(diffuse_rng)
            carry_sched_init = self._cs.jax_init_carry(rng_sched)
            reverse_adaptive_jit = jax.jit(reverse_diffuse_adaptive)
            # Timing: compilation + run (best-effort; compile may be cached)
            t0 = time.perf_counter()
            try:
                compiled = reverse_adaptive_jit.lower(rng_d, Ybar_init, carry_sched_init).compile()
                t_compile = time.perf_counter() - t0
            except Exception:
                compiled = reverse_adaptive_jit
                t_compile = time.perf_counter() - t0
            t1 = time.perf_counter()
            _, Ybar_final, reward_hist, actions_traj, sampled_traj, margin_hist, rho_hist, topK_hist, I_hist, eps_hist, lam_hist, p_hist = compiled(
                rng_d, Ybar_init, carry_sched_init
            )
            # Ensure compute finished for accurate timing
            try:
                Ybar_final.block_until_ready()
            except Exception:
                pass
            t_run = time.perf_counter() - t1
            mh = np.asarray(margin_hist)
            rh = np.asarray(rho_hist)
            th = np.asarray(topK_hist, dtype=np.int32)
            ih = np.asarray(I_hist, dtype=np.int32)
            eh = np.asarray(eps_hist)
            lh = np.asarray(lam_hist)
            ph = np.asarray(p_hist)
            # Scan order: k=0 = first iter (noisy, idx=99), k=99 = last (clean, idx=0).
            # Print diffusion_k 99 -> 0 (noisy -> clean), 100 steps.
            for k in range(len(mh)):
                diffusion_k = (total_steps - 1) - k
                print(f"[CFS-MBD JAX adaptive] step {diffusion_k}: margin={mh[k]:.6f} rho={rh[k]:.6f} topK={int(th[k])} I={int(ih[k])} eps={eh[k]:.2e} lambda={lh[k]:.6f} p={ph[k]:.6f}")
            margin_vary = len(np.unique(np.round(mh, 6))) > 1
            rho_vary = len(np.unique(np.round(rh, 6))) > 1
            topK_vary = len(np.unique(th)) > 1
            I_vary = len(np.unique(ih)) > 1
            eps_vary = len(np.unique(np.round(eh, 8))) > 1
            lam_vary = len(np.unique(np.round(lh, 6))) > 1
            p_vary = len(np.unique(np.round(ph, 6))) > 1
            print(f"[CFS-MBD JAX adaptive] vary: margin={margin_vary} rho={rho_vary} topK={topK_vary} I={I_vary} eps={eps_vary} lambda={lam_vary} p={p_vary}")
        else:
            t0 = time.perf_counter()
            try:
                compiled = reverse_diffuse_jit.lower(diffuse_rng, Ybar_init).compile()
                t_compile = time.perf_counter() - t0
            except Exception:
                compiled = reverse_diffuse_jit
                t_compile = time.perf_counter() - t0
            t1 = time.perf_counter()
            _, Ybar_final, reward_hist, actions_traj, sampled_traj = compiled(diffuse_rng, Ybar_init)
            try:
                Ybar_final.block_until_ready()
            except Exception:
                pass
            t_run = time.perf_counter() - t1
            n_steps = self.Ndiffuse
            if self._margin_arr is not None and self._rho_arr is not None:
                mh = np.asarray(self._margin_arr)
                rh = np.asarray(self._rho_arr)
                topk_val = int(self._topK) if self._topK >= 0 else 8
                # Precompute: [0]=noisy (first iter), [99]=clean (last). Print diffusion_k 99->0, 100 steps.
                for k in range(min(n_steps, len(mh))):
                    diffusion_k = (total_steps - 1) - k
                    print(f"[CFS-MBD JAX precompute] step {diffusion_k}: margin={float(mh[k]):.6f} rho={float(rh[k]):.6f} topK={topk_val}")
                margin_vary = len(np.unique(np.round(mh[:n_steps], 6))) > 1 if n_steps <= len(mh) else False
                rho_vary = len(np.unique(np.round(rh[:n_steps], 6))) > 1 if n_steps <= len(rh) else False
                print(f"[CFS-MBD JAX precompute] margin/rho/topK vary across steps: margin={margin_vary} rho={rho_vary} topK=False")

        # Postprocess timing: rollout + device->host
        t_post0 = time.perf_counter()
        final_actions = jnp.clip(Ybar_final, -self.action_limit, self.action_limit)
        states = self._rollout_states_fn(x0_jnp, final_actions)
        try:
            states.block_until_ready()
        except Exception:
            pass
        t_states = time.perf_counter() - t_post0

        t_post1 = time.perf_counter()
        rewards = self._rollout_rewards_fn(x0_jnp, final_actions)
        try:
            rewards.block_until_ready()
        except Exception:
            pass
        t_rewards = time.perf_counter() - t_post1

        t_post2 = time.perf_counter()
        states_np = np.asarray(states)
        actions_np = np.asarray(final_actions)
        total_cost_final = -float(np.sum(np.asarray(rewards)))  # Cost = -reward (lower is better)
        t_d2h = time.perf_counter() - t_post2

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
            # Timing breakdown (single mode plan)
            "timing_plan_compile_s": float(t_compile),
            "timing_plan_run_s": float(t_run),
            "timing_post_rollout_states_s": float(t_states),
            "timing_post_rollout_rewards_s": float(t_rewards),
            "timing_post_device_to_host_s": float(t_d2h),
        }
        return res
    
    def plan_batch(self, x0: State, rng_keys: jnp.ndarray) -> list[Dict[str, Any]]:
        """
        Batch version of plan using jax.vmap for parallel execution (non-adaptive mode only).
        
        Args:
            x0: initial state
            rng_keys: (C,) array of PRNG keys
            
        Returns:
            List of C result dictionaries (same format as plan)
        """
        if self._use_jax_adaptive:
            # Keep existing API behavior for adaptive mode, but don't do expensive batching here.
            # Use `plan_batch_minimal` + a single `plan` for best mode (see solver) for performance.
            results = []
            for k in rng_keys:
                results.append(self.plan(x0, k))
            return results
        
        x0_jnp = jnp.asarray(x0, dtype=jnp.float32)
        betas = jnp.linspace(self.beta0, self.betaT, self.Ndiffuse, dtype=jnp.float32)
        alphas = 1.0 - betas
        alphas_bar = jnp.cumprod(alphas)
        sigmas = jnp.sqrt(1.0 - alphas_bar)
        diffusion_indices = jnp.arange(self.Ndiffuse - 1, -1, -1, dtype=jnp.int32)
        total_steps = int(self.Ndiffuse)
        total_steps_jnp = jnp.asarray(total_steps, dtype=jnp.int32)
        
        # Extra noise schedule (decays over diffusion steps, aligned with ebmbd)
        denom_batch = jnp.maximum(float(self.Ndiffuse - 1), 1.0)
        progress_inc_by_idx_batch = 1.0 - (jnp.arange(self.Ndiffuse, dtype=jnp.float32) / denom_batch)
        extra_sigmas_by_idx_batch = self.action_extra_sigma * (1.0 - progress_inc_by_idx_batch)
        
        def reverse_diffuse_core(rng_key):
            # Split rng inside core function (aligned with ebmbd)
            rng, _ = jax.random.split(rng_key)
            def body(carry, idx):
                rng_curr, Ybar_curr = carry
                rng_curr, noise_key, extra_key = jax.random.split(rng_curr, 3)
                
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
                    "topK": jnp.asarray(self._topK if self._topK >= 0 else 8, dtype=jnp.int32),
                }
                
                Y0s_f = self._filter_actions_batch_jit(x0_jnp, Y0s, sched_state, sched_params)
                
                def compute_augmented_reward(actions_seq):
                    return self._rollout_rewards_with_augmented_fn(
                        x0_jnp, actions_seq, margin,
                        jnp.asarray(self.aug_lambda, dtype=jnp.float32),
                        jnp.asarray(self.aug_rho, dtype=jnp.float32),
                    )
                
                rews = self._augmented_rewards_batch_jit(
                    x0_jnp,
                    Y0s_f,
                    margin,
                    jnp.asarray(self.aug_lambda, dtype=jnp.float32),
                    jnp.asarray(self.aug_rho, dtype=jnp.float32),
                )
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
                Ybar_next = self._filter_actions_single_jit(x0_jnp, Ybar_next, sched_state, sched_params)
                
                # Add extra noise for diversity (decays over diffusion steps, aligned with ebmbd)
                extra_sigma = extra_sigmas_by_idx_batch[idx]
                noise_extra = jax.random.normal(extra_key, (self.horizon, self.act_dim), dtype=jnp.float32)
                Ybar_next = Ybar_next + extra_sigma * noise_extra
                Ybar_next = jnp.clip(Ybar_next, -self.action_limit, self.action_limit)
                
                return (rng_curr, Ybar_next), (jnp.mean(rews), Ybar_next, Y0s_f)
            
            Ybar_init = jnp.zeros((self.horizon, self.act_dim), dtype=jnp.float32)
            (rng_out, Ybar_final), (reward_hist, Ybar_hist, Ysamples_hist) = jax.lax.scan(
                body, (rng, Ybar_init), diffusion_indices
            )
            return Ybar_final, reward_hist[::-1], Ybar_hist[::-1], Ysamples_hist[::-1]
        
        # Vmap over rng_keys (aligned with ebmbd: pass rng_keys directly, split inside core)
        reverse_diffuse_batch_jit = jax.jit(jax.vmap(reverse_diffuse_core, in_axes=0))
        Ybar_finals, reward_hists, actions_trajs, sampled_trajs = reverse_diffuse_batch_jit(rng_keys)
        
        # Batch post-processing: clip, rollout states and rewards (aligned with mbd batch processing)
        final_actions_batch = jnp.clip(Ybar_finals, -self.action_limit, self.action_limit)  # (C, H, act_dim)
        states_batch = jax.vmap(self._rollout_states_fn, in_axes=(None, 0))(x0_jnp, final_actions_batch)  # (C, H+1, state_dim)
        rewards_batch = jax.vmap(self._rollout_rewards_fn, in_axes=(None, 0))(x0_jnp, final_actions_batch)  # (C, H)
        
        # Convert to numpy
        states_batch_np = np.asarray(states_batch)  # (C, H+1, state_dim)
        actions_batch_np = np.asarray(final_actions_batch)  # (C, H, act_dim)
        rewards_batch_np = np.asarray(rewards_batch)  # (C, H)
        
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

    def plan_batch_minimal(self, x0: State, rng_keys: jnp.ndarray) -> Dict[str, Any]:
        """
        Fast batched multirun for selecting the best mode.

        Key property: **same theory/parameters**, but avoids producing per-mode diffusion histories
        (diffusion_actions_traj / diffusion_sampled_actions), which are huge and dominate runtime
        for C=50 on CPU.

        Returns a dict with:
        - candidate_states: List[np.ndarray] length C, each (H+1, state_dim)
        - candidate_actions: List[np.ndarray] length C, each (H, act_dim)
        - candidate_costs: np.ndarray (C,)
        """
        x0_jnp = jnp.asarray(x0, dtype=jnp.float32)
        rng_keys = jnp.asarray(rng_keys)
        if rng_keys.ndim != 2 or rng_keys.shape[-1] != 2:
            raise ValueError(f"rng_keys must have shape (C,2), got {rng_keys.shape}")

        # Common diffusion schedule (same as plan)
        betas = jnp.linspace(self.beta0, self.betaT, self.Ndiffuse, dtype=jnp.float32)
        alphas = 1.0 - betas
        alphas_bar = jnp.cumprod(alphas)
        sigmas = jnp.sqrt(1.0 - alphas_bar)
        diffusion_indices = jnp.arange(self.Ndiffuse - 1, -1, -1, dtype=jnp.int32)
        total_steps = int(self.Ndiffuse)
        total_steps_jnp = jnp.asarray(total_steps, dtype=jnp.int32)

        # Extra noise schedule
        denom = jnp.maximum(float(self.Ndiffuse - 1), 1.0)
        progress_inc_by_idx = 1.0 - (jnp.arange(self.Ndiffuse, dtype=jnp.float32) / denom)
        extra_sigmas_by_idx = self.action_extra_sigma * (1.0 - progress_inc_by_idx)

        def reverse_diffuse_minimal_nonadaptive_batch(rng_keys_batch: jnp.ndarray) -> jnp.ndarray:
            """
            Architecture optimization: fuse the mode dimension C into the solver, instead of vmapping
            an entire scan. This reduces overhead and lets XLA fuse more work.
            """
            C = rng_keys_batch.shape[0]
            rng0 = rng_keys_batch  # (C,2)
            Ybar0 = jnp.zeros((C, self.horizon, self.act_dim), dtype=jnp.float32)

            split3 = lambda k: jax.random.split(k, 3)  # (3,2)

            def body(carry, idx):
                rng_curr, Ybar_curr = carry  # rng_curr: (C,2), Ybar_curr: (C,H,D)

                keys3 = jax.vmap(split3, in_axes=0)(rng_curr)  # (C,3,2)
                rng_next = keys3[:, 0, :]
                noise_keys = keys3[:, 1, :]
                extra_keys = keys3[:, 2, :]

                # Sample eps for each mode (C independent keys)
                eps = jax.vmap(
                    lambda k: jax.random.normal(
                        k, (self.Nsample, self.horizon, self.act_dim), dtype=jnp.float32
                    ),
                    in_axes=0,
                )(noise_keys)  # (C, N, H, D)

                Y0s = eps * sigmas[idx] + Ybar_curr[:, None, :, :]  # (C, N, H, D)
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
                    "topK": jnp.asarray(self._topK if self._topK >= 0 else 8, dtype=jnp.int32),
                }

                # Filter all samples across all modes in one call (flatten C*N)
                Y0s_flat = Y0s.reshape((C * self.Nsample, self.horizon, self.act_dim))
                Y0s_f_flat = self._filter_actions_batch_jit(x0_jnp, Y0s_flat, sched_state, sched_params)
                Y0s_f = Y0s_f_flat.reshape((C, self.Nsample, self.horizon, self.act_dim))

                rews_flat = self._augmented_rewards_batch_jit(
                    x0_jnp,
                    Y0s_f_flat,
                    margin,
                    jnp.asarray(self.aug_lambda, dtype=jnp.float32),
                    jnp.asarray(self.aug_rho, dtype=jnp.float32),
                )  # (C*N,)
                rews = rews_flat.reshape((C, self.Nsample))
                rew_mean = jnp.mean(rews, axis=1, keepdims=True)
                rew_std = jnp.std(rews, axis=1, keepdims=True)
                rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)

                T_k = self.temp_sample
                if self._T_k_arr is not None:
                    kkT = jnp.clip(step_k, 0, self._T_k_arr.shape[0] - 1)
                    T_k = self._T_k_arr[kkT]

                logp0 = (rews - rew_mean) / (rew_std * T_k)
                weights = jax.nn.softmax(logp0, axis=1)  # (C,N)
                Ybar_weighted = jnp.einsum("cn,cnhd->chd", weights, Y0s_f)  # (C,H,D)

                # Project mean trajectories as a batch of size C
                Ybar_next = self._filter_actions_batch_jit(x0_jnp, Ybar_weighted, sched_state, sched_params)  # (C,H,D)

                # Extra noise per mode
                extra_sigma = extra_sigmas_by_idx[idx]
                noise_extra = jax.vmap(
                    lambda k: jax.random.normal(k, (self.horizon, self.act_dim), dtype=jnp.float32),
                    in_axes=0,
                )(extra_keys)  # (C,H,D)
                Ybar_next = jnp.clip(
                    Ybar_next + extra_sigma * noise_extra,
                    -self.action_limit,
                    self.action_limit,
                )
                return (rng_next, Ybar_next), None

            (rng_out, Ybar_final), _ = jax.lax.scan(body, (rng0, Ybar0), diffusion_indices)
            return Ybar_final  # (C,H,D)

        def reverse_diffuse_minimal_adaptive(rng_key):
            # Same adaptive logic as plan(), but minimal outputs.
            cs = self._cs
            rng_d, rng_sched = jax.random.split(rng_key)
            carry_sched = cs.jax_init_carry(rng_sched)
            rng = rng_d

            def body(carry, idx):
                rng_curr, Ybar_curr, carry_sched_curr = carry
                rng_curr, noise_key, extra_key = jax.random.split(rng_curr, 3)

                eps = jax.random.normal(noise_key, (self.Nsample, self.horizon, self.act_dim), dtype=jnp.float32)
                Y0s = eps * sigmas[idx] + Ybar_curr
                Y0s = jnp.clip(Y0s, -self.action_limit, self.action_limit)

                step_k = jnp.asarray((self.Ndiffuse - 1), dtype=jnp.int32) - idx
                sched_state = {"k": step_k, "K": total_steps_jnp}

                params, rng_sched_next = cs.jax_compute_params(carry_sched_curr, step_k, total_steps)
                do_qp = params["qp_gate"]
                margin = params["margin"]
                aug_lam = params["aug_lambda"]
                aug_rho = params["aug_rho"]
                sched_params = {
                    "margin": margin,
                    "rho": params["rho"],
                    "qp_gate": do_qp,
                    "qp_prob": params["qp_prob"],
                    "topK": params["topK"],
                    "eps": params["eps"],
                    "I_QP": params["I_QP"],
                }

                def apply_filter(ys):
                    return self._filter_actions_batch_jit(x0_jnp, ys, sched_state, sched_params)

                Y0s_f = jax.lax.cond(do_qp, apply_filter, lambda ys: ys, Y0s)

                def rollout_one(actions_seq):
                    return self._rollout_augmented_and_v_fn(x0_jnp, actions_seq, margin, aug_lam, aug_rho)

                rews, v_batch = self._augmented_and_v_batch_jit(x0_jnp, Y0s_f, margin, aug_lam, aug_rho)
                # feedback metrics (same as plan): r_p and proj
                from enerdynamics.core.constraints.schedulers.ConstraintScheduler.almadaptive.backends.alm_adaptive_jax import quantile_90
                k_top = max(1, min(self.Nsample, int(math.ceil(0.1 * self.Nsample))))
                r_p = quantile_90(v_batch, k_top)
                proj = jnp.mean(jnp.linalg.norm(Y0s_f - Y0s, axis=(1, 2)))
                carry_sched_new = cs.jax_update(carry_sched_curr, {"r_p": r_p, "proj": proj}, rng_sched_next)

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

                def filter_mean(_):
                    return self._filter_actions_single_jit(x0_jnp, Ybar_weighted, sched_state, sched_params)

                Ybar_next = jax.lax.cond(do_qp, filter_mean, lambda _: Ybar_weighted, operand=None)
                extra_sigma = extra_sigmas_by_idx[idx]
                noise_extra = jax.random.normal(extra_key, (self.horizon, self.act_dim), dtype=jnp.float32)
                Ybar_next = jnp.clip(Ybar_next + extra_sigma * noise_extra, -self.action_limit, self.action_limit)
                return (rng_curr, Ybar_next, carry_sched_new), None

            Ybar_init = jnp.zeros((self.horizon, self.act_dim), dtype=jnp.float32)
            (_, Ybar_final, _), _ = jax.lax.scan(body, (rng, Ybar_init, carry_sched), diffusion_indices)
            return Ybar_final

        def reverse_diffuse_minimal_adaptive_batch(rng_keys_batch: jnp.ndarray) -> jnp.ndarray:
            """
            Fused adaptive batch (C modes) with per-mode scheduler carry.
            Same logic, but avoids vmap(scan(...)) overhead on CPU.
            """
            cs = self._cs
            C = rng_keys_batch.shape[0]

            split2 = lambda k: jax.random.split(k, 2)  # (2,2)
            keys2 = jax.vmap(split2, in_axes=0)(rng_keys_batch)  # (C,2,2)
            rng0 = keys2[:, 0, :]
            rng_sched0 = keys2[:, 1, :]
            carry_sched0 = jax.vmap(cs.jax_init_carry, in_axes=0)(rng_sched0)
            Ybar0 = jnp.zeros((C, self.horizon, self.act_dim), dtype=jnp.float32)

            split3 = lambda k: jax.random.split(k, 3)  # (3,2)
            k_top = max(1, min(self.Nsample, int(math.ceil(0.1 * self.Nsample))))

            def body(carry, idx):
                rng_curr, Ybar_curr, carry_sched_curr = carry

                keys3 = jax.vmap(split3, in_axes=0)(rng_curr)  # (C,3,2)
                rng_next = keys3[:, 0, :]
                noise_keys = keys3[:, 1, :]
                extra_keys = keys3[:, 2, :]

                eps = jax.vmap(
                    lambda k: jax.random.normal(
                        k, (self.Nsample, self.horizon, self.act_dim), dtype=jnp.float32
                    ),
                    in_axes=0,
                )(noise_keys)  # (C,N,H,D)
                Y0s = eps * sigmas[idx] + Ybar_curr[:, None, :, :]
                Y0s = jnp.clip(Y0s, -self.action_limit, self.action_limit)

                step_k = jnp.asarray((self.Ndiffuse - 1), dtype=jnp.int32) - idx
                sched_state = {"k": step_k, "K": total_steps_jnp}

                # Compute scheduler params per mode
                def compute_params_one(carry_s):
                    return cs.jax_compute_params(carry_s, step_k, total_steps)

                params_batch, rng_sched_next = jax.vmap(compute_params_one, in_axes=0)(carry_sched_curr)
                do_qp = params_batch["qp_gate"]  # (C,)
                margin = params_batch["margin"]  # (C,)
                aug_lam = params_batch["aug_lambda"]
                aug_rho = params_batch["aug_rho"]

                sched_params = {
                    "margin": margin,
                    "rho": params_batch["rho"],
                    "qp_gate": do_qp,
                    "qp_prob": params_batch["qp_prob"],
                    "topK": params_batch["topK"],
                    "eps": params_batch["eps"],
                    "I_QP": params_batch["I_QP"],
                }

                # Apply filter to samples (flatten C*N). Respect per-mode qp_gate via masking.
                Y0s_flat = Y0s.reshape((C * self.Nsample, self.horizon, self.act_dim))

                def do_filter_all(y_flat):
                    return self._filter_actions_batch_jit(x0_jnp, y_flat, sched_state, sched_params)

                any_qp = jnp.any(do_qp)
                Y0s_f_flat = jax.lax.cond(any_qp, do_filter_all, lambda y: y, Y0s_flat)
                Y0s_f = Y0s_f_flat.reshape((C, self.Nsample, self.horizon, self.act_dim))
                Y0s_f = jnp.where(do_qp[:, None, None, None], Y0s_f, Y0s)

                # Rollout reward+v for all samples (flatten C*N), with per-mode (margin, lam, rho)
                def rollout_one(actions_seq, m, lam, rho):
                    return self._rollout_augmented_and_v_fn(x0_jnp, actions_seq, m, lam, rho)

                margin_rep = jnp.repeat(margin, self.Nsample)
                lam_rep = jnp.repeat(aug_lam, self.Nsample)
                rho_rep = jnp.repeat(aug_rho, self.Nsample)
                rews_flat, v_flat = jax.vmap(rollout_one, in_axes=(0, 0, 0, 0))(Y0s_f_flat, margin_rep, lam_rep, rho_rep)
                rews = rews_flat.reshape((C, self.Nsample))
                v_batch = v_flat.reshape((C, self.Nsample))

                from enerdynamics.core.constraints.schedulers.ConstraintScheduler.almadaptive.backends.alm_adaptive_jax import quantile_90
                r_p = jax.vmap(lambda v: quantile_90(v, k_top), in_axes=0)(v_batch)  # (C,)
                proj = jnp.mean(jnp.linalg.norm(Y0s_f - Y0s, axis=(2, 3)), axis=1)  # (C,)

                carry_sched_new = jax.vmap(
                    lambda carry_s, rp, pj, rng_s: cs.jax_update(carry_s, {"r_p": rp, "proj": pj}, rng_s),
                    in_axes=(0, 0, 0, 0),
                )(carry_sched_curr, r_p, proj, rng_sched_next)

                # Weighting per mode
                rew_mean = jnp.mean(rews, axis=1, keepdims=True)
                rew_std = jnp.std(rews, axis=1, keepdims=True)
                rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)
                T_k = self.temp_sample
                if self._T_k_arr is not None:
                    kkT = jnp.clip(step_k, 0, self._T_k_arr.shape[0] - 1)
                    T_k = self._T_k_arr[kkT]
                logp0 = (rews - rew_mean) / (rew_std * T_k)
                weights = jax.nn.softmax(logp0, axis=1)
                Ybar_weighted = jnp.einsum("cn,cnhd->chd", weights, Y0s_f)

                # Project mean trajectories (batch). Respect per-mode qp_gate via masking.
                def filter_mean_all(y):
                    return self._filter_actions_batch_jit(x0_jnp, y, sched_state, sched_params)

                Ybar_filt = jax.lax.cond(jnp.any(do_qp), filter_mean_all, lambda y: y, Ybar_weighted)
                Ybar_next = jnp.where(do_qp[:, None, None], Ybar_filt, Ybar_weighted)

                extra_sigma = extra_sigmas_by_idx[idx]
                noise_extra = jax.vmap(
                    lambda k: jax.random.normal(k, (self.horizon, self.act_dim), dtype=jnp.float32),
                    in_axes=0,
                )(extra_keys)
                Ybar_next = jnp.clip(Ybar_next + extra_sigma * noise_extra, -self.action_limit, self.action_limit)

                return (rng_next, Ybar_next, carry_sched_new), None

            (_, Ybar_final, _), _ = jax.lax.scan(body, (rng0, Ybar0, carry_sched0), diffusion_indices)
            return Ybar_final

        # Compile + run timing (batch-minimal)
        if self._use_jax_adaptive:
            jit_core = jax.jit(reverse_diffuse_minimal_adaptive_batch)
        else:
            jit_core = jax.jit(reverse_diffuse_minimal_nonadaptive_batch)

        t0 = time.perf_counter()
        try:
            compiled = jit_core.lower(rng_keys).compile()
            t_compile = time.perf_counter() - t0
        except Exception:
            compiled = jit_core
            t_compile = time.perf_counter() - t0

        t1 = time.perf_counter()
        Ybar_finals = compiled(rng_keys)  # (C,H,act_dim)
        try:
            Ybar_finals.block_until_ready()
        except Exception:
            pass
        t_run = time.perf_counter() - t1

        # Postprocess timing
        t2 = time.perf_counter()
        final_actions_batch = jnp.clip(Ybar_finals, -self.action_limit, self.action_limit)
        states_batch = jax.vmap(self._rollout_states_fn, in_axes=(None, 0))(x0_jnp, final_actions_batch)
        rewards_batch = jax.vmap(self._rollout_rewards_fn, in_axes=(None, 0))(x0_jnp, final_actions_batch)
        try:
            states_batch.block_until_ready()
            rewards_batch.block_until_ready()
        except Exception:
            pass
        t_post = time.perf_counter() - t2

        t3 = time.perf_counter()

        states_np = np.asarray(states_batch, dtype=np.float32)
        actions_np = np.asarray(final_actions_batch, dtype=np.float32)
        rewards_np = np.asarray(rewards_batch, dtype=np.float32)
        costs = -np.sum(rewards_np, axis=-1).astype(np.float32)
        t_d2h = time.perf_counter() - t3

        candidate_states = [states_np[i] for i in range(states_np.shape[0])]
        candidate_actions = [actions_np[i] for i in range(actions_np.shape[0])]
        res = {
            "candidate_states": candidate_states,
            "candidate_actions": candidate_actions,
            "candidate_costs": costs,
            # Timing breakdown (batch-minimal)
            "timing_batch_minimal_compile_s": float(t_compile),
            "timing_batch_minimal_run_s": float(t_run),
            "timing_batch_minimal_post_s": float(t_post),
            "timing_batch_minimal_device_to_host_s": float(t_d2h),
        }

        # ---------------------------------------------------------------------
        # Extra decomposition of `batch_minimal_run_s` (diagnostics):
        # Measure per-call cost of key components with representative shapes.
        # NOTE: these are microbenchmarks; to avoid failures on some JAX versions
        # we only time call execution (and optionally a second call to estimate
        # steady-state after compilation).
        # This does not change the algorithm output; it only adds timing fields.
        # ---------------------------------------------------------------------
        try:
            C = int(rng_keys.shape[0])
            N = int(self.Nsample)
            H = int(self.horizon)
            D = int(self.act_dim)
            total_steps = int(self.Ndiffuse)

            # Representative schedule (non-adaptive): take k=0 (same for fixed schedulers).
            sched_state_bench = {"k": jnp.asarray(0, dtype=jnp.int32), "K": jnp.asarray(max(1, total_steps), dtype=jnp.int32)}
            if self._margin_arr is not None and self._rho_arr is not None and self._qp_gate_arr is not None and self._qp_prob_arr is not None:
                margin_b = self._margin_arr[0]
                rho_b = self._rho_arr[0]
                qp_gate_b = self._qp_gate_arr[0]
                qp_prob_b = self._qp_prob_arr[0]
            else:
                margin_b = jnp.asarray(0.0, dtype=jnp.float32)
                rho_b = jnp.asarray(1.0, dtype=jnp.float32)
                qp_gate_b = jnp.asarray(True, dtype=jnp.bool_)
                qp_prob_b = jnp.asarray(1.0, dtype=jnp.float32)

            sched_params_bench = {
                "margin": margin_b,
                "rho": rho_b,
                "qp_gate": qp_gate_b,
                "qp_prob": qp_prob_b,
                "topK": jnp.asarray(self._topK if self._topK >= 0 else 8, dtype=jnp.int32),
            }

            # Dummy batches (same shapes as inside diffusion).
            key_b = jax.random.PRNGKey(0)
            key_b, k1, k2, k3 = jax.random.split(key_b, 4)
            Y0s_flat = jax.random.normal(k1, (C * N, H, D), dtype=jnp.float32)
            Ybar_weighted = jax.random.normal(k2, (C, H, D), dtype=jnp.float32)
            rews = jax.random.normal(k3, (C, N), dtype=jnp.float32)

            def time_call(fn, *args):
                t_a = time.perf_counter()
                out_a = fn(*args)
                try:
                    out_a.block_until_ready()
                except Exception:
                    pass
                t_first = time.perf_counter() - t_a
                # second call: try to estimate steady-state (after compile)
                t_b = time.perf_counter()
                out_b = fn(*args)
                try:
                    out_b.block_until_ready()
                except Exception:
                    pass
                t_second = time.perf_counter() - t_b
                return t_first, t_second, out_b

            # 1) Filter on samples: (C*N,H,D)
            t_f1, t_f2, Y0s_f = time_call(self._filter_actions_batch_jit, x0_jnp, Y0s_flat, sched_state_bench, sched_params_bench)

            # 2) Reward on samples: (C*N,H,D)  (use filtered samples to match inner-loop)
            aug_lam = jnp.asarray(self.aug_lambda, dtype=jnp.float32)
            aug_rho = jnp.asarray(self.aug_rho, dtype=jnp.float32)
            t_r1, t_r2, _ = time_call(self._augmented_rewards_batch_jit, x0_jnp, Y0s_f, margin_b, aug_lam, aug_rho)

            # 3) Weight update: softmax + einsum (C,N) x (C,N,H,D) -> (C,H,D)
            def weight_update(rews_in, ys_in):
                rew_mean = jnp.mean(rews_in, axis=1, keepdims=True)
                rew_std = jnp.std(rews_in, axis=1, keepdims=True)
                rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)
                logp0 = (rews_in - rew_mean) / (rew_std * jnp.asarray(self.temp_sample, dtype=jnp.float32))
                w = jax.nn.softmax(logp0, axis=1)
                return jnp.einsum("cn,cnhd->chd", w, ys_in)

            wjit = jax.jit(weight_update)
            ys_reshaped = Y0s_f.reshape((C, N, H, D))
            t_w1, t_w2, _ = time_call(wjit, rews, ys_reshaped)

            # 4) Filter on mean: (C,H,D)
            t_m1, t_m2, _ = time_call(self._filter_actions_batch_jit, x0_jnp, Ybar_weighted, sched_state_bench, sched_params_bench)

            res.update(
                {
                    "timing_bench_filter_samples_first_s": float(t_f1),
                    "timing_bench_filter_samples_second_s": float(t_f2),
                    "timing_bench_reward_samples_first_s": float(t_r1),
                    "timing_bench_reward_samples_second_s": float(t_r2),
                    "timing_bench_weight_update_first_s": float(t_w1),
                    "timing_bench_weight_update_second_s": float(t_w2),
                    "timing_bench_filter_mean_first_s": float(t_m1),
                    "timing_bench_filter_mean_second_s": float(t_m2),
                }
            )
            # Rough per-step run estimate (ignores extra-noise and RNG ops; good enough for bottlenecking).
            per_step_est = t_f2 + t_r2 + t_w2 + t_m2
            res["timing_bench_estimated_batch_minimal_run_s"] = float(per_step_est * float(total_steps))
        except Exception:
            # Diagnostics are best-effort; do not affect planning.
            pass

        return res

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
