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
import types
from typing import Any, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.solvers.single.diffusion_adaptors import diverse_topk_modes
from genedynamics.solvers.single.cfsmbd.backends import _batched_alm_adaptive as _batched_alm
from genedynamics.experiments.plugins.obstacles.d3il_avoiding_fixed import get_d3il_target_line_positions

from genedynamics.core.types import Trajectory, State
from genedynamics.core.task_spec import legacy_extract_position
from genedynamics.core.constraints.core.types import ScheduleState
from genedynamics.core.constraints.action_filters import ConstraintFilter, NoOpConstraintFilter


class CFSMBDBackendJax:
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
            self.use_target_line = bool(solver.config.get("use_target_line", False))
            self.num_targets = int(solver.config.get("num_targets", 4))
            self.diversity_eta = float(solver.config.get("diversity_eta", 1.0))
            self.diversity_topK_cand = int(solver.config.get("diversity_topK_cand", None) or (self.Nsample // 2))
            self.diversity_use_state = bool(solver.config.get("diversity_use_state", True))
            # Optional: use batched ALMAdaptive backend (init_carry / compute_params)
            # even when running a single mode (C=1). This makes it easy to later
            # switch to multi-mode without refactoring call sites.
            self.use_batched_alm_adaptive = bool(solver.config.get("use_batched_alm_adaptive", False))
            self.position_extractor = getattr(solver, "position_extractor", None) or legacy_extract_position
            self.position_dim = int(getattr(solver, "position_dim", 2))
        else:
            self.num_modes = 1
            self.use_target_line = False
            self.num_targets = 4
            self.diversity_eta = 1.0
            self.diversity_topK_cand = self.Nsample // 2
            self.diversity_use_state = True
            self.use_batched_alm_adaptive = bool(kwargs.get("use_batched_alm_adaptive", False))
            self.position_extractor = kwargs.get("position_extractor") or legacy_extract_position
            self.position_dim = int(kwargs.get("position_dim", 2))

        self._build_jax_functions()

        # Resolve constraint scheduler (composite vs single)
        self._cs = None
        if self.scheduler is not None and hasattr(self.scheduler, "constraint_schedulers"):
            cs_list = getattr(self.scheduler, "constraint_schedulers", [])
            if cs_list:
                self._cs = cs_list[0]
        elif self.scheduler is not None:
            self._cs = self.scheduler

        # Optional: route scheduler init/compute_params through batched backend.
        # We keep jax_update unchanged (single-mode path), so this is safe when rng is (2,).
        if self.use_batched_alm_adaptive and (self._cs is not None):
            # Patch instance methods (avoids global monkey-patching).
            def _jax_init_carry_batched(cs_self, rng):
                return _batched_alm.init_carry(
                    rng,
                    cs_self.lam0,
                    cs_self.rho0,
                    cs_self.p_max,
                    cs_self.nu0,
                    cs_self.topK_max,
                    cs_self.I_max,
                    cs_self.eps_min,
                    0.0,
                    0.0,
                    0,
                )

            def _jax_compute_params_batched(cs_self, carry, step_k, K: int):
                return _batched_alm.compute_params(
                    carry,
                    step_k,
                    K,
                    margin_base=cs_self.margin_base,
                    gamma=cs_self.gamma,
                    rho_max=cs_self.rho_max,
                    kappa=cs_self.kappa,
                    p_min=cs_self.p_min,
                    p_max=cs_self.p_max,
                    eps_min=cs_self.eps_min,
                    eps_max=cs_self.eps_max,
                    I_min=cs_self.I_min,
                    I_max=cs_self.I_max,
                    topK_min=cs_self.topK_min,
                    topK_max=cs_self.topK_max,
                    use_stochastic_gate=cs_self.use_stochastic_gate,
                    v_signal_mode=cs_self.v_signal_mode,
                    v_star=cs_self.v_star,
                    v_ema_beta=cs_self.v_ema_beta,
                    v_tau_hi=cs_self.v_tau_hi,
                    v_tau_lo=cs_self.v_tau_lo,
                    eta_p=cs_self.eta_p,
                    eta_p_plus=cs_self.eta_p_plus,
                    eta_p_minus=cs_self.eta_p_minus,
                    eta_c_p=cs_self.eta_c_p,
                    eta_topK=cs_self.eta_topK,
                    eta_topK_plus=cs_self.eta_topK_plus,
                    eta_topK_minus=cs_self.eta_topK_minus,
                    eta_c_topK=cs_self.eta_c_topK,
                    eta_I=cs_self.eta_I,
                    eta_I_plus=cs_self.eta_I_plus,
                    eta_I_minus=cs_self.eta_I_minus,
                    eta_c_I=cs_self.eta_c_I,
                    eta_eps=cs_self.eta_eps,
                    eta_eps_plus=cs_self.eta_eps_plus,
                    eta_eps_minus=cs_self.eta_eps_minus,
                    eta_c_eps=cs_self.eta_c_eps,
                    compute_budget_B=cs_self.compute_budget_B,
                    compute_budget_time_profile=getattr(cs_self, "compute_budget_time_profile", "constant"),
                    compute_budget_time_amp=getattr(cs_self, "compute_budget_time_amp", 0.0),
                    compute_budget_time_mu=getattr(cs_self, "compute_budget_time_mu", 0.5),
                    compute_budget_time_sigma=getattr(cs_self, "compute_budget_time_sigma", 0.2),
                    compute_cost_mode=cs_self.compute_cost_mode,
                    compute_cost_a0=cs_self.compute_cost_a0,
                    compute_cost_aK=cs_self.compute_cost_aK,
                    compute_cost_aI=cs_self.compute_cost_aI,
                    compute_cost_use_qp_gate=cs_self.compute_cost_use_qp_gate,
                )

            self._cs.jax_init_carry = types.MethodType(_jax_init_carry_batched, self._cs)
            self._cs.jax_compute_params = types.MethodType(_jax_compute_params_batched, self._cs)

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
        self._I_QP_arr = None
        self._eps_arr = None
        self._topK = -1
        if not self._use_jax_adaptive:
            try:
                margin_list: List[float] = []
                rho_list: List[float] = []
                qp_gate_list: List[bool] = []
                qp_prob_list: List[float] = []
                I_QP_list: List[int] = []
                eps_list: List[float] = []
                topK_val = None
                if self._cs is not None:
                    for k in range(self.Ndiffuse):
                        st = ScheduleState(k=k, K=max(1, self.Ndiffuse))
                        d = self._cs.constraint_params(st) or {}
                        margin_list.append(float(d.get("margin", 0.0)))
                        rho_list.append(float(d.get("rho", 1.0)))
                        qp_gate_list.append(bool(d.get("qp_gate", True)))
                        qp_prob_list.append(float(d.get("qp_prob", 1.0)))
                        I_QP_list.append(int(d.get("I_QP", d.get("cfs_outer_iters", 1))))
                        eps_list.append(float(d.get("eps", 1e-4)))
                        topK_val = d.get("topK", topK_val)
                if margin_list:
                    self._margin_arr = jnp.asarray(margin_list, dtype=jnp.float32)
                    self._rho_arr = jnp.asarray(rho_list, dtype=jnp.float32)
                    self._qp_gate_arr = jnp.asarray(qp_gate_list, dtype=jnp.bool_)
                    self._qp_prob_arr = jnp.asarray(qp_prob_list, dtype=jnp.float32)
                    self._I_QP_arr = jnp.asarray(I_QP_list, dtype=jnp.int32)
                    self._eps_arr = jnp.asarray(eps_list, dtype=jnp.float32)
                if topK_val is not None:
                    self._topK = int(topK_val)
            except Exception:
                self._margin_arr = None
                self._rho_arr = None
                self._qp_gate_arr = None
                self._qp_prob_arr = None
                self._I_QP_arr = None
                self._eps_arr = None

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
        # Match origin/main: origin/main's ObstacleManager has no jax_sdf, so sdf_fn uses
        # sample_sdf_and_grad_2d (texture). Genedynamics added jax_sdf for MJX; prefer texture
        # when available to preserve origin/main behavior (jax_sdf returns 1e5 for UnionObstacles).
        _use_sdf_texture = (
            self.obstacles is not None
            and hasattr(self.obstacles, "get_sdf_texture_2d")
            and self.obstacles.get_sdf_texture_2d() is not None
        )

        def sdf_fn(pos, clearance):
            """Compute SDF and constraint violation [g]_+."""
            if self.obstacles is None:
                return jnp.asarray(0.0, dtype=jnp.float32)
            
            # Prefer texture-based SDF for 2D (box2d level 7-10 use UnionObstacles without jax_sdf)
            try:
                if _use_sdf_texture and hasattr(self.obstacles, "sample_sdf_and_grad_2d"):
                    sdf_result, _ = self.obstacles.sample_sdf_and_grad_2d(
                        pos[None, :], backend="jax"
                    )
                    sdf_val = sdf_result[0] if isinstance(sdf_result, (list, tuple)) else sdf_result
                    if isinstance(sdf_val, np.ndarray):
                        sdf_val = jnp.asarray(sdf_val)
                elif hasattr(self.obstacles, "jax_sdf"):
                    sdf_val = self.obstacles.jax_sdf(pos)
                elif hasattr(self.obstacles, "sample_sdf_and_grad_2d"):
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

        def rollout_augmented_reward_and_v(state_init, actions, clearance, aug_lambda, aug_rho, target):
            """
            Rollout rewards with augmented Lagrangian penalty.
            target: (position_dim,) goal for terminal reward and ctx.target_xy for stage cost.

            Returns:
              - total_augmented_reward: scalar
              - v_n: scalar, max_t [g]_+ (used for risk residual r_k and violation stats v_k)
            """
            target = jnp.asarray(target, dtype=jnp.float32).reshape(-1)[: self.position_dim]

            def step_fn(carry, action_and_t):
                next_state, cum_reward, cum_g_plus, cum_g_plus_sq, max_g_plus = carry
                action, t = action_and_t
                
                next_state = self._transition_fn(next_state, action)
                ctx = {"t": t, "target_xy": target}
                
                # Standard running cost
                reward = -self._cost_fn(next_state, action, ctx)
                
                # Constraint violation [g]_+ (optional env-specific metric, e.g. foot SDF for quadrupeds)
                hook = getattr(self.env, "jax_cfs_alm_g_plus_from_state", None)
                if callable(hook):
                    g_plus = hook(next_state, clearance)
                else:
                    pos = next_state[: self.position_dim]
                    g_plus = sdf_fn(pos, clearance)

                return (
                    next_state,
                    cum_reward + reward,
                    cum_g_plus + g_plus,
                    cum_g_plus_sq + (g_plus * g_plus),
                    jnp.maximum(max_g_plus, g_plus),
                ), None

            t_indices = jnp.arange(self.horizon, dtype=jnp.float32)
            (final_state, total_reward, total_g_plus, total_g_plus_sq, max_g_plus), _ = jax.lax.scan(
                step_fn, (state_init, 0.0, 0.0, 0.0, 0.0), (actions, t_indices)
            )
            
            # Terminal reward
            if hasattr(self.env, "terminal_distance_jax"):
                terminal_dist = self.env.terminal_distance_jax(final_state)
            else:
                terminal_dist = jnp.linalg.norm(final_state[: self.position_dim] - target[: self.position_dim])
            terminal_weight = float(getattr(self.env, "terminal_reward_weight", 100.0))
            terminal_reward = -terminal_weight * terminal_dist
            
            # Augmented Lagrangian penalty (mean over horizon to avoid H-scale explosion)
            # J(τ) + λ mean_t [g]_+ + (ρ/2) mean_t [g]_+^2
            H = jnp.asarray(self.horizon, dtype=jnp.float32)
            dual_term = aug_lambda * (total_g_plus / H)
            penalty_term = (aug_rho / 2.0) * (total_g_plus_sq / H)
            augmented_penalty = dual_term + penalty_term
            
            # Total reward = sum of step rewards + terminal - augmented penalty
            total_augmented_reward = total_reward + terminal_reward - augmented_penalty

            # Violation magnitude (per-trajectory scalar).
            # Use mean rather than max to avoid rho spiking from a single
            # bad timestep in long-horizon humanoid trajectories.
            v_n = total_g_plus / H
            return total_augmented_reward, v_n

        def _default_target():
            t = getattr(self.env, "target", None)
            if t is None:
                return jnp.zeros(self.position_dim, dtype=jnp.float32)
            pos = np.asarray(self.position_extractor(t), dtype=np.float32).reshape(-1)
            return jnp.asarray(pos[: self.position_dim], dtype=jnp.float32)

        def rollout_rewards_with_augmented(state_init, actions, clearance, aug_lambda, aug_rho, target):
            total_aug, _ = rollout_augmented_reward_and_v(state_init, actions, clearance, aug_lambda, aug_rho, target)
            return total_aug

        def rollout_rewards_with_augmented_and_v(state_init, actions, clearance, aug_lambda, aug_rho, target):
            return rollout_augmented_reward_and_v(state_init, actions, clearance, aug_lambda, aug_rho, target)

        def rollout_rewards(state_init, actions, target):
            """Rollout rewards per step (for visualization). target: (position_dim,) goal."""
            target = jnp.asarray(target, dtype=jnp.float32).reshape(-1)[: self.position_dim]
            
            def step_fn(carry, action_and_t):
                next_state, cum_reward = carry
                action, t = action_and_t
                
                next_state = self._transition_fn(next_state, action)
                ctx = {"t": t, "target_xy": target}
                reward = -self._cost_fn(next_state, action, ctx)
                return (next_state, cum_reward + reward), reward

            t_indices = jnp.arange(self.horizon, dtype=jnp.float32)
            (final_state, _), step_rewards = jax.lax.scan(
                step_fn, (state_init, 0.0), (actions, t_indices)
            )
            
            # Terminal reward
            if hasattr(self.env, "terminal_distance_jax"):
                terminal_dist = self.env.terminal_distance_jax(final_state)
            else:
                terminal_dist = jnp.linalg.norm(final_state[: self.position_dim] - target[: self.position_dim])
            terminal_weight = float(getattr(self.env, "terminal_reward_weight", 100.0))
            terminal_reward = -terminal_weight * terminal_dist
            step_rewards = step_rewards.at[-1].add(terminal_reward)
            
            return step_rewards

        # JIT compile: rollout_rewards(state_init, actions, target)
        
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

        def augmented_rewards_batch(x0_in, actions_batch, clearance, aug_lambda, aug_rho, target):
            return jax.vmap(
                lambda actions_seq: rollout_rewards_with_augmented(x0_in, actions_seq, clearance, aug_lambda, aug_rho, target),
                in_axes=0,
            )(actions_batch)

        def augmented_and_v_batch(x0_in, actions_batch, clearance, aug_lambda, aug_rho, target):
            rews, v = jax.vmap(
                lambda actions_seq: rollout_rewards_with_augmented_and_v(x0_in, actions_seq, clearance, aug_lambda, aug_rho, target),
                in_axes=0,
            )(actions_batch)
            return rews, v

        # Fused helper: compute (rews, r_k, v_k stats) without exposing v_batch.
        from genedynamics.core.constraints.schedulers.ConstraintScheduler.almadaptive.backends.alm_adaptive_jax import (
            quantile_90,
        )
        _k_top_static = max(1, min(self.Nsample, int(math.ceil(0.1 * self.Nsample))))

        def augmented_rewards_and_rp_batch(x0_in, actions_batch, clearance, aug_lambda, aug_rho, target):
            rews, v = augmented_and_v_batch(x0_in, actions_batch, clearance, aug_lambda, aug_rho, target)
            # Risk residual: r_k = Q_0.9(v_1..v_M), with v_m = max_t [g]_+.
            r_p = quantile_90(v, _k_top_static)
            # Violation statistics (batch):
            # - v_k_mean: mean magnitude across samples
            # - v_k_rate: fraction of violating samples
            v_k_mean = jnp.mean(v)
            v_k_rate = jnp.mean(v > 0.0)
            return rews, r_p, v_k_rate, v_k_mean

        self._filter_actions_batch_jit = jax.jit(filter_actions_batch)
        self._filter_actions_single_jit = jax.jit(filter_actions_single)
        self._augmented_rewards_batch_jit = jax.jit(augmented_rewards_batch)
        self._augmented_and_v_batch_jit = jax.jit(augmented_and_v_batch)
        self._augmented_rewards_and_rp_batch_jit = jax.jit(augmented_rewards_and_rp_batch)

        def rollout_states(state_init, actions):
            def step_fn(carry, action):
                next_state = self._transition_fn(carry, action)
                return next_state, next_state

            _, states = jax.lax.scan(step_fn, state_init, actions)
            return jnp.concatenate([state_init[None, :], states], axis=0)

        self._rollout_states_fn = jax.jit(rollout_states)

        # Constraint normals in action space from SDF geometry (texture / jax sampler),
        # aligned with numpy 2GO refine: a_t[:, :pdim] = dt * (-∇sdf/||∇sdf||) on violated steps.
        pdim_geom = int(self.position_dim)
        H_geom = int(self.horizon)
        adim_geom = int(self.act_dim)
        dt_geom = float(self.dt)

        if self.obstacles is None:

            def constraint_geometry_time_single(x0_in, actions, clearance):
                del x0_in, actions, clearance
                return jnp.zeros((H_geom, adim_geom), dtype=jnp.float32)
        else:

            def constraint_geometry_time_single(x0_in, actions, clearance):
                states = rollout_states(x0_in, actions)
                pos = states[1:, :pdim_geom]
                sdf, grad = self.obstacles.sample_sdf_and_grad_2d(pos, backend="jax")
                sdf = jnp.asarray(sdf, dtype=jnp.float32).reshape(H_geom)
                grad = jnp.asarray(grad, dtype=jnp.float32)
                gdim = grad.shape[-1]
                grad = grad.reshape(H_geom, gdim)
                grad_p = grad[:, :pdim_geom]
                norms = jnp.linalg.norm(grad_p, axis=-1, keepdims=True) + jnp.asarray(1e-8, dtype=jnp.float32)
                n_unit = -grad_p / norms
                clr = jnp.asarray(clearance, dtype=jnp.float32)
                g_plus = jnp.maximum(jnp.asarray(0.0, dtype=jnp.float32), clr - sdf)
                mask = (g_plus > jnp.asarray(1e-7, dtype=jnp.float32)).astype(jnp.float32)
                a_slice = (jnp.asarray(dt_geom, dtype=jnp.float32) * n_unit) * mask[:, None]
                a_time = jnp.zeros((H_geom, adim_geom), dtype=jnp.float32)
                a_time = a_time.at[:, :pdim_geom].set(a_slice)
                return a_time

        self._constraint_geometry_time_jit = jax.jit(constraint_geometry_time_single)

        # Batch rollouts (used by multi-mode candidate extraction in `plan()`).
        def rollout_states_batch(state_init, actions_batch):
            return jax.vmap(lambda a: rollout_states(state_init, a), in_axes=0)(actions_batch)

        def rollout_rewards_batch(state_init, actions_batch, target):
            return jax.vmap(lambda a: rollout_rewards(state_init, a, target), in_axes=0)(actions_batch)

        self._default_target = _default_target()
        self._rollout_rewards_fn = jax.jit(lambda x0, a: rollout_rewards(x0, a, self._default_target))
        self._rollout_rewards_with_target_fn = jax.jit(rollout_rewards)
        self._rollout_states_batch_fn = jax.jit(rollout_states_batch)
        self._rollout_rewards_batch_fn = jax.jit(rollout_rewards_batch)

    def _run_adaptive_diffuse_single(
        self,
        x0_jnp: jnp.ndarray,
        rng_in: Any,
        Ybar_init: jnp.ndarray,
        carry_sched_init: Any,
        target: jnp.ndarray,
    ) -> tuple:
        """
        Single adaptive reverse-diffusion run. Used by plan() and vmap'd by plan_batch().
        Returns (rng_out, Ybar_final, reward_hist, actions_traj, sampled_traj,
                 margin_hist, rho_hist, topK_hist, I_hist, eps_hist, lam_hist,
                 p_hist, nu_hist, compute_cost_hist, r_hist, v_rate_hist, v_mean_hist).
        """
        betas = jnp.linspace(self.beta0, self.betaT, self.Ndiffuse, dtype=jnp.float32)
        alphas = 1.0 - betas
        alphas_bar = jnp.cumprod(alphas)
        sigmas = jnp.sqrt(1.0 - alphas_bar)
        diffusion_indices = jnp.arange(self.Ndiffuse - 1, -1, -1, dtype=jnp.int32)
        total_steps = int(self.Ndiffuse)
        total_steps_jnp = jnp.asarray(total_steps, dtype=jnp.int32)
        denom = jnp.maximum(float(self.Ndiffuse - 1), 1.0)
        progress_inc = 1.0 - (jnp.arange(self.Ndiffuse, dtype=jnp.float32) / denom)
        extra_sigmas_by_idx = self.action_extra_sigma * (1.0 - progress_inc)

        cs = self._cs

        def body(carry, idx):
            rng_curr, Ybar_curr, carry_sched = carry
            rng_curr, noise_key, extra_key, filter_key = jax.random.split(rng_curr, 4)

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
            nu = params["nu"]
            compute_cost_hat = params["compute_cost_hat"]

            sched_params = {
                "margin": margin,
                "rho": params["rho"],
                "qp_gate": do_qp,
                "qp_prob": params["qp_prob"],
                "topK": params["topK"],
                "eps": params["eps"],
                "I_QP": params["I_QP"],
                "rng_key": filter_key,
            }

            def apply_filter(ys):
                return self._filter_actions_batch_jit(x0_jnp, ys, sched_state, sched_params)

            Y0s_f = jax.lax.cond(do_qp, apply_filter, lambda ys: ys, Y0s)

            # Rewards from filtered trajectories (for diffusion weights)
            rews, _, _, _ = self._augmented_rewards_and_rp_batch_jit(
                x0_jnp, Y0s_f, margin, aug_lam, aug_rho, target
            )
            # Violation feedback from UNfiltered trajectories (drives adaptive scheduler)
            # Using Y0s ensures we see actual violation; Y0s_f would be ~0 after CFS projection
            _, r_p, v_k_rate, v_k_mean = self._augmented_rewards_and_rp_batch_jit(
                x0_jnp, Y0s, margin, aug_lam, aug_rho, target
            )
            proj = jnp.mean(jnp.linalg.norm(Y0s_f - Y0s, axis=(1, 2)))
            feedback = {
                "r_p": r_p,
                "v_k_rate": v_k_rate,
                "v_k_mean": v_k_mean,
                "proj": proj,
                "compute_cost_hat": compute_cost_hat,
                "compute_budget_B_eff": params["compute_budget_B_eff"],
            }
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

            extra_sigma = extra_sigmas_by_idx[idx]
            noise_extra = jax.random.normal(extra_key, (self.horizon, self.act_dim), dtype=jnp.float32)
            Ybar_next = Ybar_next + extra_sigma * noise_extra
            Ybar_next = jnp.clip(Ybar_next, -self.action_limit, self.action_limit)

            # reward_history: stage + terminal only (for convergence comparison across methods)
            rews_plain = self._rollout_rewards_batch_fn(x0_jnp, Y0s_f, target)  # (M, H)
            reward_stage_terminal = jnp.mean(jnp.sum(rews_plain, axis=-1))
            return (rng_curr, Ybar_next, carry_sched_new), (
                reward_stage_terminal, Ybar_next, Y0s_f,
                margin, params["rho"], params["topK"],
                params["I_QP"], params["eps"], aug_lam, params["qp_prob"],
                nu, compute_cost_hat, r_p, v_k_rate, v_k_mean,
            )

        (rng_out, Ybar_final, _), (
            reward_hist, Ybar_hist, Ysamples_hist,
            margin_hist, rho_hist, topK_hist,
            I_hist, eps_hist, lam_hist, p_hist, nu_hist, compute_cost_hist,
            r_hist, v_rate_hist, v_mean_hist,
        ) = jax.lax.scan(body, (rng_in, Ybar_init, carry_sched_init), diffusion_indices)
        return (
            rng_out, Ybar_final,
            reward_hist[::-1], Ybar_hist[::-1], Ysamples_hist[::-1],
            margin_hist, rho_hist, topK_hist,
            I_hist, eps_hist, lam_hist, p_hist, nu_hist, compute_cost_hist,
            r_hist, v_rate_hist, v_mean_hist,
        )

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

        def reverse_diffuse(rng_in, Ybar_init):
            def body(carry, idx):
                rng_curr, Ybar_curr = carry
                # Need separate RNGs for diffusion noise and extra noise.
                rng_curr, noise_key, extra_key, filter_key = jax.random.split(rng_curr, 4)

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
                    I_QP = self._I_QP_arr[kk] if self._I_QP_arr is not None else jnp.asarray(1, dtype=jnp.int32)
                    eps = self._eps_arr[kk] if self._eps_arr is not None else jnp.asarray(1e-4, dtype=jnp.float32)
                else:
                    margin = jnp.asarray(0.0, dtype=jnp.float32)
                    rho = jnp.asarray(1.0, dtype=jnp.float32)
                    qp_gate = jnp.asarray(True, dtype=jnp.bool_)
                    qp_prob = jnp.asarray(1.0, dtype=jnp.float32)
                    I_QP = jnp.asarray(1, dtype=jnp.int32)
                    eps = jnp.asarray(1e-4, dtype=jnp.float32)

                sched_params = {
                    "margin": margin,
                    "rho": rho,
                    "qp_gate": qp_gate,
                    "qp_prob": qp_prob,
                    "I_QP": I_QP,
                    "eps": eps,
                    "topK": jnp.asarray(self._topK if self._topK >= 0 else 8, dtype=jnp.int32),
                    "rng_key": filter_key,
                }

                Y0s_f = self._filter_actions_batch_jit(x0_jnp, Y0s, sched_state, sched_params)

                aug_lam = jnp.asarray(self.aug_lambda, dtype=jnp.float32)
                aug_rho = jnp.asarray(self.aug_rho, dtype=jnp.float32)
                rews, r_p, v_k_rate, v_k_mean = self._augmented_rewards_and_rp_batch_jit(
                    x0_jnp, Y0s_f, margin, aug_lam, aug_rho, self._default_target
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

                # reward_history: stage + terminal only (for convergence comparison across methods)
                rews_plain = self._rollout_rewards_batch_fn(x0_jnp, Y0s_f, self._default_target)  # (M, H)
                reward_stage_terminal = jnp.mean(jnp.sum(rews_plain, axis=-1))
                return (rng_curr, Ybar_next), (reward_stage_terminal, Ybar_next, Y0s_f, r_p, v_k_rate, v_k_mean)

            (rng_out, Ybar_final), (reward_hist, Ybar_hist, Ysamples_hist, r_hist, v_rate_hist, v_mean_hist) = jax.lax.scan(
                body, (rng_in, Ybar_init), diffusion_indices
            )
            return rng_out, Ybar_final, reward_hist[::-1], Ybar_hist[::-1], Ysamples_hist[::-1], r_hist, v_rate_hist, v_mean_hist

        reverse_diffuse_jit = jax.jit(reverse_diffuse)

        Ybar_init = jnp.zeros((self.horizon, self.act_dim), dtype=jnp.float32)
        if self._use_jax_adaptive:
            rng_d, rng_sched = jax.random.split(diffuse_rng)
            carry_sched_init = self._cs.jax_init_carry(rng_sched)
            reverse_adaptive_jit = jax.jit(self._run_adaptive_diffuse_single)
            # Timing: compilation + run (best-effort; compile may be cached)
            t0 = time.perf_counter()
            try:
                compiled = reverse_adaptive_jit.lower(x0_jnp, rng_d, Ybar_init, carry_sched_init, self._default_target).compile()
                t_compile = time.perf_counter() - t0
            except Exception:
                compiled = reverse_adaptive_jit
                t_compile = time.perf_counter() - t0
            t1 = time.perf_counter()
            _, Ybar_final, reward_hist, actions_traj, sampled_traj, margin_hist, rho_hist, topK_hist, I_hist, eps_hist, lam_hist, p_hist, nu_hist, compute_cost_hist, r_hist, v_rate_hist, v_mean_hist = compiled(
                x0_jnp, rng_d, Ybar_init, carry_sched_init, self._default_target
            )
            # Ensure compute finished for accurate timing
            try:
                Ybar_final.block_until_ready()
            except Exception:
                pass
            t_run = time.perf_counter() - t1
            I_QP_hist = I_hist
            lambda_hist = lam_hist
        else:
            t0 = time.perf_counter()
            try:
                compiled = reverse_diffuse_jit.lower(diffuse_rng, Ybar_init).compile()
                t_compile = time.perf_counter() - t0
            except Exception:
                compiled = reverse_diffuse_jit
                t_compile = time.perf_counter() - t0
            t1 = time.perf_counter()
            _, Ybar_final, reward_hist, actions_traj, sampled_traj, r_hist, v_rate_hist, v_mean_hist = compiled(diffuse_rng, Ybar_init)
            try:
                Ybar_final.block_until_ready()
            except Exception:
                pass
            t_run = time.perf_counter() - t1
            # Non-adaptive (fixed): build schedule arrays for logging only (unchanged from before).
            K = self.Ndiffuse
            rho_hist = np.full(K, float(self.aug_rho), dtype=np.float32)
            p_hist = np.asarray(self._qp_prob_arr[:K]).flatten().astype(np.float32) if (self._qp_prob_arr is not None and self._qp_prob_arr.size >= K) else np.full(K, 1.0, dtype=np.float32)
            topK_val = int(self._topK) if self._topK >= 0 else 8
            topK_hist = np.full(K, float(topK_val), dtype=np.float32)
            I_QP_hist = np.asarray(self._I_QP_arr[:K]).flatten().astype(np.float32) if (self._I_QP_arr is not None and self._I_QP_arr.size >= K) else np.full(K, 1.0, dtype=np.float32)
            eps_hist = np.asarray(self._eps_arr[:K]).flatten().astype(np.float32) if (self._eps_arr is not None and self._eps_arr.size >= K) else np.full(K, 1e-4, dtype=np.float32)
            lambda_hist = np.full(K, float(self.aug_lambda), dtype=np.float32)
            # Logging only (no effect on planning): c_k proxy and ν=0 for fixed strategy.
            _gate = self._qp_gate_arr
            qp_on = bool(np.all(np.asarray(_gate)[:K])) if (_gate is not None and getattr(_gate, "size", 0) >= K) else True
            if qp_on:
                compute_cost_hist = (p_hist * topK_hist * I_QP_hist).astype(np.float32)
            else:
                compute_cost_hist = np.zeros(K, dtype=np.float32)
            nu_hist = np.zeros(K, dtype=np.float32)

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
            "r_hist": np.asarray(r_hist, dtype=np.float32),
            "v_rate_hist": np.asarray(v_rate_hist, dtype=np.float32),
            "v_mean_hist": np.asarray(v_mean_hist, dtype=np.float32),
            "rho_hist": np.asarray(rho_hist, dtype=np.float32),
            "topK_hist": np.asarray(topK_hist, dtype=np.float32),
            "I_QP_hist": np.asarray(I_QP_hist, dtype=np.float32),
            "eps_hist": np.asarray(eps_hist, dtype=np.float32),
            "lambda_hist": np.asarray(lambda_hist, dtype=np.float32),
            "p_hist": np.asarray(p_hist, dtype=np.float32),
            "nu_hist": np.asarray(nu_hist, dtype=np.float32),
            "compute_cost_hist": np.asarray(compute_cost_hist, dtype=np.float32),
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
        x0_jnp = jnp.asarray(x0, dtype=jnp.float32)
        C = int(rng_keys.shape[0])
        if getattr(self, "use_target_line", False) and C > 0:
            target_line = get_d3il_target_line_positions(getattr(self, "num_targets", 4))
            targets_per_mode = jnp.asarray(
                np.asarray([target_line[i % len(target_line)] for i in range(C)], dtype=np.float32),
                dtype=jnp.float32,
            )
        else:
            default_tgt = np.asarray(self._default_target, dtype=np.float32).reshape(-1)[: self.position_dim]
            targets_per_mode = jnp.tile(jnp.asarray(default_tgt, dtype=jnp.float32), (C, 1))
        if self._use_jax_adaptive:
            # True JAX batch: vmap over _run_adaptive_diffuse_single (same CFS filter path as plan()).
            # vmap(jax.random.split)(rng_keys) returns shape (C, 2, 2), not two arrays; slice to get (C, 2) each.
            split_keys = jax.vmap(jax.random.split)(rng_keys)  # (C, 2, 2)
            rng_d_all = split_keys[:, 0, :]    # (C, 2)
            rng_sched_all = split_keys[:, 1, :]  # (C, 2)
            carry_sched_inits = jax.vmap(self._cs.jax_init_carry)(rng_sched_all)
            Ybar_init = jnp.zeros((self.horizon, self.act_dim), dtype=jnp.float32)
            batched_fn = jax.jit(
                jax.vmap(self._run_adaptive_diffuse_single, in_axes=(None, 0, None, 0, 0))
            )
            batched = batched_fn(x0_jnp, rng_d_all, Ybar_init, carry_sched_inits, targets_per_mode)
            # batched: (rng_out, Ybar_final, reward_hist, actions_traj, sampled_traj, margin_hist, rho_hist, topK_hist, I_hist, eps_hist, lam_hist, p_hist, nu_hist, compute_cost_hist, r_hist, v_rate_hist, v_mean_hist)
            Ybar_finals = batched[1]
            reward_hists = batched[2]
            actions_trajs = batched[3]
            sampled_trajs = batched[4]
            rho_hists = batched[6]
            topK_hists = batched[7]
            I_QP_hists = batched[8]
            eps_hists = batched[9]
            lam_hists = batched[10]
            p_hists = batched[11]
            nu_hists = batched[12]
            compute_cost_hists = batched[13]
            r_hists = batched[14]
            v_rate_hists = batched[15]
            v_mean_hists = batched[16]
            final_actions_batch = jnp.clip(Ybar_finals, -self.action_limit, self.action_limit)
            states_batch = jax.vmap(self._rollout_states_fn, in_axes=(None, 0))(
                x0_jnp, final_actions_batch
            )
            rewards_batch = jax.vmap(self._rollout_rewards_with_target_fn, in_axes=(None, 0, 0))(
                x0_jnp, final_actions_batch, targets_per_mode
            )
            states_batch_np = np.asarray(states_batch)
            actions_batch_np = np.asarray(final_actions_batch)
            rewards_batch_np = np.asarray(rewards_batch)
            total_costs = -np.sum(rewards_batch_np, axis=-1)
            results = []
            for i in range(C):
                results.append({
                    "actions": actions_batch_np[i],
                    "states": states_batch_np[i],
                    "rewards": rewards_batch_np[i],
                    "initial_state": states_batch_np[i, 0],
                    "reward_history": np.asarray(reward_hists[i]),
                    "diffusion_rewards": np.asarray(reward_hists[i]),
                    "diffusion_actions_traj": np.asarray(actions_trajs[i]),
                    "diffusion_sampled_actions": np.asarray(sampled_trajs[i]),
                    "r_hist": np.asarray(r_hists[i], dtype=np.float32),
                    "v_rate_hist": np.asarray(v_rate_hists[i], dtype=np.float32),
                    "v_mean_hist": np.asarray(v_mean_hists[i], dtype=np.float32),
                    "rho_hist": np.asarray(rho_hists[i], dtype=np.float32),
                    "topK_hist": np.asarray(topK_hists[i], dtype=np.float32),
                    "I_QP_hist": np.asarray(I_QP_hists[i], dtype=np.float32),
                    "eps_hist": np.asarray(eps_hists[i], dtype=np.float32),
                    "lambda_hist": np.asarray(lam_hists[i], dtype=np.float32),
                    "p_hist": np.asarray(p_hists[i], dtype=np.float32),
                    "nu_hist": np.asarray(nu_hists[i], dtype=np.float32),
                    "compute_cost_hist": np.asarray(compute_cost_hists[i], dtype=np.float32),
                    "candidate_states": [states_batch_np[i]],
                    "candidate_actions": [actions_batch_np[i]],
                    "candidate_costs": np.asarray([float(total_costs[i])], dtype=np.float32),
                    "best_idx": 0,
                    "rng": rng_keys[i],
                })
            return results

        # Non-adaptive: precomputed schedule
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
        
        def reverse_diffuse_core(rng_key, target):
            # Split rng inside core function (aligned with ebmbd)
            rng, _ = jax.random.split(rng_key)
            def body(carry, idx):
                rng_curr, Ybar_curr = carry
                rng_curr, noise_key, extra_key, filter_key = jax.random.split(rng_curr, 4)
                
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
                    I_QP = self._I_QP_arr[kk] if self._I_QP_arr is not None else jnp.asarray(1, dtype=jnp.int32)
                    eps = self._eps_arr[kk] if self._eps_arr is not None else jnp.asarray(1e-4, dtype=jnp.float32)
                else:
                    margin = jnp.asarray(0.0, dtype=jnp.float32)
                    rho = jnp.asarray(1.0, dtype=jnp.float32)
                    qp_gate = jnp.asarray(True, dtype=jnp.bool_)
                    qp_prob = jnp.asarray(1.0, dtype=jnp.float32)
                    I_QP = jnp.asarray(1, dtype=jnp.int32)
                    eps = jnp.asarray(1e-4, dtype=jnp.float32)
                
                sched_params = {
                    "margin": margin,
                    "rho": rho,
                    "qp_gate": qp_gate,
                    "qp_prob": qp_prob,
                    "I_QP": I_QP,
                    "eps": eps,
                    "topK": jnp.asarray(self._topK if self._topK >= 0 else 8, dtype=jnp.int32),
                    "rng_key": filter_key,
                }

                # Respect qp_gate in non-adaptive mode (JAX scalar/tracer).
                # If qp_gate is False, skip the (expensive) QP filter entirely.
                def _apply_filter_batch(ys):
                    return self._filter_actions_batch_jit(x0_jnp, ys, sched_state, sched_params)

                Y0s_f = jax.lax.cond(qp_gate, _apply_filter_batch, lambda ys: ys, Y0s)
                
                aug_lam = jnp.asarray(self.aug_lambda, dtype=jnp.float32)
                aug_rho = jnp.asarray(self.aug_rho, dtype=jnp.float32)
                rews, r_p, v_k_rate, v_k_mean = self._augmented_rewards_and_rp_batch_jit(
                    x0_jnp, Y0s_f, margin, aug_lam, aug_rho, target
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

                def _apply_filter_single(_):
                    return self._filter_actions_single_jit(x0_jnp, Ybar_next, sched_state, sched_params)

                Ybar_next = jax.lax.cond(qp_gate, _apply_filter_single, lambda _: Ybar_next, operand=None)
                
                # Add extra noise for diversity (decays over diffusion steps, aligned with ebmbd)
                extra_sigma = extra_sigmas_by_idx_batch[idx]
                noise_extra = jax.random.normal(extra_key, (self.horizon, self.act_dim), dtype=jnp.float32)
                Ybar_next = Ybar_next + extra_sigma * noise_extra
                Ybar_next = jnp.clip(Ybar_next, -self.action_limit, self.action_limit)
                
                return (rng_curr, Ybar_next), (jnp.mean(rews), Ybar_next, Y0s_f, r_p, v_k_rate, v_k_mean)
            
            Ybar_init = jnp.zeros((self.horizon, self.act_dim), dtype=jnp.float32)
            (rng_out, Ybar_final), (reward_hist, Ybar_hist, Ysamples_hist, r_hist, v_rate_hist, v_mean_hist) = jax.lax.scan(
                body, (rng, Ybar_init), diffusion_indices
            )
            return Ybar_final, reward_hist[::-1], Ybar_hist[::-1], Ysamples_hist[::-1], r_hist, v_rate_hist, v_mean_hist
        
        # Vmap over (rng_keys, targets) for per-mode targets (Option B)
        reverse_diffuse_batch_jit = jax.jit(jax.vmap(reverse_diffuse_core, in_axes=(0, 0)))
        Ybar_finals, reward_hists, actions_trajs, sampled_trajs, r_hists, v_rate_hists, v_mean_hists = reverse_diffuse_batch_jit(rng_keys, targets_per_mode)
        
        # Batch post-processing: clip, rollout states and rewards (aligned with mbd batch processing)
        final_actions_batch = jnp.clip(Ybar_finals, -self.action_limit, self.action_limit)  # (C, H, act_dim)
        states_batch = jax.vmap(self._rollout_states_fn, in_axes=(None, 0))(x0_jnp, final_actions_batch)  # (C, H+1, state_dim)
        rewards_batch = jax.vmap(self._rollout_rewards_with_target_fn, in_axes=(None, 0, 0))(x0_jnp, final_actions_batch, targets_per_mode)  # (C, H)
        
        # Convert to numpy
        states_batch_np = np.asarray(states_batch)  # (C, H+1, state_dim)
        actions_batch_np = np.asarray(final_actions_batch)  # (C, H, act_dim)
        rewards_batch_np = np.asarray(rewards_batch)  # (C, H)
        
        # Batch compute costs and rewards
        total_costs = -np.sum(rewards_batch_np, axis=-1)  # (C,)
        total_rewards = np.sum(rewards_batch_np, axis=-1)  # (C,)
        
        # Non-adaptive (fixed): same schedule + logging-only c_k/nu as in plan().
        K = self.Ndiffuse
        rho_hist = np.full(K, float(self.aug_rho), dtype=np.float32)
        p_hist = np.asarray(self._qp_prob_arr[:K]).flatten().astype(np.float32) if (self._qp_prob_arr is not None and self._qp_prob_arr.size >= K) else np.full(K, 1.0, dtype=np.float32)
        topK_val = int(self._topK) if self._topK >= 0 else 8
        topK_hist = np.full(K, float(topK_val), dtype=np.float32)
        I_QP_hist = np.asarray(self._I_QP_arr[:K]).flatten().astype(np.float32) if (self._I_QP_arr is not None and self._I_QP_arr.size >= K) else np.full(K, 1.0, dtype=np.float32)
        eps_hist = np.asarray(self._eps_arr[:K]).flatten().astype(np.float32) if (self._eps_arr is not None and self._eps_arr.size >= K) else np.full(K, 1e-4, dtype=np.float32)
        lambda_hist = np.full(K, float(self.aug_lambda), dtype=np.float32)
        _gate = self._qp_gate_arr
        qp_on = bool(np.all(np.asarray(_gate)[:K])) if (_gate is not None and getattr(_gate, "size", 0) >= K) else True
        if qp_on:
            compute_cost_hist = (p_hist * topK_hist * I_QP_hist).astype(np.float32)
        else:
            compute_cost_hist = np.zeros(K, dtype=np.float32)
        nu_hist = np.zeros(K, dtype=np.float32)

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
                "diffusion_rewards": np.asarray(reward_hists[i]),
                "diffusion_actions_traj": np.asarray(actions_trajs[i]),
                "diffusion_sampled_actions": np.asarray(sampled_trajs[i]),
                "r_hist": np.asarray(r_hists[i], dtype=np.float32),
                "v_rate_hist": np.asarray(v_rate_hists[i], dtype=np.float32),
                "v_mean_hist": np.asarray(v_mean_hists[i], dtype=np.float32),
                "rho_hist": rho_hist,
                "topK_hist": topK_hist,
                "I_QP_hist": I_QP_hist,
                "eps_hist": eps_hist,
                "lambda_hist": lambda_hist,
                "p_hist": p_hist,
                "nu_hist": nu_hist,
                "compute_cost_hist": compute_cost_hist,
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
