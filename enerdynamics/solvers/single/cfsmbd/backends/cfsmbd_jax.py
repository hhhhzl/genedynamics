"""
JAX backend implementation for CFS-MBD.

CFS-MBD is defined as:
  MBD diffusion driver + Augmented Lagrangian objective + CFS-based per-step QP projection

This file mirrors the structure of `mdoc/backends/mdoc_jax.py`,
but adds augmented Lagrangian penalty in the reward computation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from enerdynamics.core.types import Trajectory, State
from enerdynamics.core.constraints.core.types import ScheduleState
from enerdynamics.core.constraints.action_filters import ConstraintFilter, NoOpConstraintFilter


class CFSMBDBackendJax:
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
        aug_lambda: float = 0.0,
        aug_rho: float = 1.0,
        **kwargs: Any,
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
        self.aug_lambda = float(aug_lambda)
        self.aug_rho = float(aug_rho)

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
            
            # Augmented Lagrangian penalty
            # J(τ) + λ^T [g]_+ + (ρ/2) ||[g]_+||^2
            dual_term = aug_lambda * total_g_plus
            penalty_term = (aug_rho / 2.0) * jnp.sum(step_g_plus ** 2)
            augmented_penalty = dual_term + penalty_term
            
            # Total reward = sum of step rewards + terminal - augmented penalty
            total_augmented_reward = total_reward + terminal_reward - augmented_penalty
            
            return total_augmented_reward

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
                }

                # Apply Filter per diffusion step (CFS-MBD core)
                Y0s_f = self.constraint_filter.apply_actions_batch(
                    x0_jnp, Y0s, env=self.env, obstacles=self.obstacles,
                    schedule_state=sched_state, schedule_params=sched_params,
                )

                # Compute rewards with augmented Lagrangian
                # We need to compute augmented reward for each sample
                def compute_augmented_reward(actions_seq):
                    return self._rollout_rewards_with_augmented_fn(
                        x0_jnp, actions_seq, margin, 
                        jnp.asarray(self.aug_lambda, dtype=jnp.float32),
                        jnp.asarray(self.aug_rho, dtype=jnp.float32)
                    )
                
                # Vectorize over batch
                rews = jax.vmap(compute_augmented_reward)(Y0s_f)
                
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
        
        # Build standard result dict
        res = {
            "actions": actions_np,
            "states": states_np,
            "rewards": np.asarray(rewards, dtype=np.float32),
            "initial_state": states_np[0],
            "diffusion_rewards": np.asarray(reward_hist),
            "diffusion_actions_traj": np.asarray(actions_traj),
            "diffusion_sampled_actions": np.asarray(sampled_traj),
        }
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
