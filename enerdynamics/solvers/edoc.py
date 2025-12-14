"""
EDOC (Energy-Driven Optimal Control) solver implementation.

EDOC uses diffusion-based trajectory optimization with:
- A-MCSA (Adaptive Multi-scale Constraint Satisfaction Algorithm)
- Multi-scale barrier functions
- ADM (Alternating Direction Method) for constraint handling

This implementation is compatible with the new unified architecture.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Tuple, List

import numpy as np
import jax
import jax.numpy as jnp
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Dummy tqdm if not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

from enerdynamics.core.solvers import SamplingSolver
from enerdynamics.core.dynamics import DynamicsModel, DynamicsToEnvAdapter, EnvDynamicsAdapter
from enerdynamics.core.energy import EnergyFunctional, LegacyEnergyFunctional
from enerdynamics.core.backends import Backend, JaxBackend
from enerdynamics.core.types import State, Action, Trajectory
from enerdynamics.core.metrics import euclidean_metric_inv
from enerdynamics.core.constraints.base import project_box
from enerdynamics.core.constraints import ConstraintManager
from enerdynamics.core.integrators import langevin_step
from enerdynamics.envs.factories import make_env, make_energy


# ============================================================================
# EDOC Planner (Core Implementation)
# ============================================================================

class EDOCPlanner:
    """
    EDOC planner implementation.
    
    This is the core EDOC algorithm. It can be used directly or wrapped
    in the EDOCSolver class for the unified interface.
    """

    def __init__(
        self,
        env,
        energy: LegacyEnergyFunctional,
        horizon: int,
        dt: float,
        noise_std: float = 0.05,
        state_box: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        action_space: bool = True,
        diffusion_mode: str = "reverse",
        action_diffuse_steps: int = 100,
        action_beta0: float = 1e-4,
        action_betaT: float = 1e-2,
        action_temp: float = 0.1,
        action_extra_sigma: float = 0.0,
        action_stage_ratio: float = 0.5,
        action_score_mode: str = "reward",
        action_nsample: int = 256,
        use_antithetic: bool = False,
        dyn_loss_coeff: float = 1.0,
        dyn_loss_mode: str = "terminal",
        np_random_seed: Optional[int] = None,
        # ===== UX / profiling =====
        show_tqdm: bool = True,
        tqdm_chunk_len: int = 10,
        # ===== Constraint system integration =====
        constraint_manager: Optional[ConstraintManager] = None,
        lambda_energy: float = 1.0,  # Energy scaling in E_soft = (1/λ)J + S
        use_constraint_in_scoring: bool = True,  # Include soft constraints in scoring
    ):
        if action_score_mode not in {"reward", "energy", "learned"}:
            raise ValueError(f"Unknown action_score_mode {action_score_mode}")
        if dyn_loss_mode not in {"terminal", "trajectory"}:
            raise ValueError(f"Unknown dyn_loss_mode {dyn_loss_mode}")

        self.use_antithetic = bool(use_antithetic)
        self.env = env
        self.energy = energy
        self.horizon = horizon
        self.dt = dt
        self.noise_std = noise_std
        self.state_box = state_box
        self.action_space = action_space
        self.diffusion_mode = diffusion_mode
        self.action_diffuse_steps = action_diffuse_steps
        self.action_beta0 = action_beta0
        self.action_betaT = action_betaT
        self.action_temp = action_temp
        self.action_extra_sigma = action_extra_sigma
        self.action_stage_ratio = np.clip(action_stage_ratio, 0.0, 1.0)
        self.action_score_mode = action_score_mode
        self.action_nsample = max(1, int(action_nsample))
        self.dyn_loss_coeff = float(max(0.0, dyn_loss_coeff))
        self.dyn_loss_mode = dyn_loss_mode
        self._np_rng = np.random.default_rng(np_random_seed)
        self.show_tqdm = bool(show_tqdm)
        
        # ===== Constraint system =====
        self.constraint_manager = constraint_manager
        self.lambda_energy = float(lambda_energy) if lambda_energy > 0 else 1.0
        self.use_constraint_in_scoring = bool(use_constraint_in_scoring)
        self._rollout_states_energy_fn = None
        self._rollout_env_states_fn = None
        self._rollout_energy_fn = None
        self._batch_rollout_energy_fn = None
        self._reward_mean_fn = None
        self._batch_reward_mean_fn = None
        self._reverse_diffuse_jit = None
        self._reverse_diffuse_chunk_jit = None
        self._reverse_diffuse_chunk_len = int(max(1, tqdm_chunk_len))
        # Jacobian of model transition w.r.t. action (for tracking projected state trajectories)
        self._jac_model_u_fn = None
        # Optional JAX action safety filter (CBF-style) from ConstraintManager
        self._jax_action_filter = None

        # use JAX to automatically compute energy gradient
        def energy_x(x, u, ctx):
            return self.energy.compute(x, u, ctx)

        self.grad_energy_x = jax.jit(jax.grad(energy_x, argnums=0))

        if self.action_space:
            if not hasattr(self.env, "jax_transition"):
                raise ValueError("Environment must provide jax_transition when using action_space diffusion.")
            self.jax_transition = jax.jit(self.env.jax_transition)
            if hasattr(self.env, "jax_model_transition"):
                self.jax_model_transition = jax.jit(self.env.jax_model_transition)
            else:
                self.jax_model_transition = self.jax_transition
            self._time_index = jnp.arange(self.horizon, dtype=jnp.int32)

            energy_fn = self.energy
            transition_fn = self.jax_model_transition
            time_index = self._time_index
            dyn_coeff = self.dyn_loss_coeff
            dyn_mode = self.dyn_loss_mode

            if hasattr(self.env, "jax_env_transition"):
                env_transition_fn = jax.jit(self.env.jax_env_transition)
            else:
                env_transition_fn = transition_fn

            # For mapping projected state trajectories back to actions (tracking),
            # we need the Jacobian of the model transition w.r.t. action.
            # This is a lightweight SCP/Gauss-Newton step used only when hard projection edits states.
            self._jac_model_u_fn = jax.jit(jax.jacrev(transition_fn, argnums=1))

            # Get optional action filter from constraint manager (JAX callable)
            if self.constraint_manager is not None:
                self._jax_action_filter = self.constraint_manager.get_jax_action_filter()
            else:
                self._jax_action_filter = None

            if self._jax_action_filter is None:
                def apply_action_filter(s, act, hard_clearance, hard_enabled):
                    return act
            else:
                def apply_action_filter(s, act, hard_clearance, hard_enabled):
                    return self._jax_action_filter(s, act, hard_clearance, hard_enabled)

            def rollout_states_and_energy(state, actions, hard_clearance, hard_enabled):
                def body(carry, inputs):
                    s = carry
                    t, act = inputs
                    ctx = {"t": t}
                    act_safe = apply_action_filter(s, act, hard_clearance, hard_enabled)
                    e_val = energy_fn.compute(s, act_safe, ctx)
                    s_next = transition_fn(s, act_safe)
                    return s_next, (s_next, e_val)

                _, (states_seq, energy_seq) = jax.lax.scan(
                    body, state, (time_index, actions)
                )
                states_full = jnp.concatenate([state[None, :], states_seq], axis=0)
                return states_full, energy_seq

            def rollout_env_states(state, actions, hard_clearance, hard_enabled):
                def body(carry, inputs):
                    s = carry
                    _, act = inputs
                    act_safe = apply_action_filter(s, act, hard_clearance, hard_enabled)
                    s_next = env_transition_fn(s, act_safe)
                    return s_next, s_next

                _, states_seq = jax.lax.scan(
                    body, state, (time_index, actions)
                )
                states_full = jnp.concatenate([state[None, :], states_seq], axis=0)
                return states_full

            def total_energy(state, actions, hard_clearance, hard_enabled):
                states_full, energy_seq = rollout_states_and_energy(state, actions, hard_clearance, hard_enabled)
                total = jnp.sum(energy_seq)
                if dyn_coeff > 0.0:
                    env_states_full = rollout_env_states(state, actions, hard_clearance, hard_enabled)
                    if dyn_mode == "terminal":
                        diff = states_full[-1] - env_states_full[-1]
                        dyn_loss = jnp.sum(diff * diff)
                    else:
                        dyn_loss = jnp.sum((states_full - env_states_full) ** 2)
                    total = total + dyn_coeff * dyn_loss
                return total

            reward_cost_fn = getattr(self.env, "jax_cost", None)

            if reward_cost_fn is not None:

                def mean_reward(state, actions, hard_clearance, hard_enabled):
                    def body(carry, inputs):
                        s = carry
                        _, act = inputs
                        act_safe = apply_action_filter(s, act, hard_clearance, hard_enabled)
                        s_next = env_transition_fn(s, act_safe)
                        reward = -reward_cost_fn(s_next)
                        return s_next, reward

                    _, reward_seq = jax.lax.scan(
                        body, state, (time_index, actions)
                    )
                    return jnp.mean(reward_seq)

                self._reward_mean_fn = jax.jit(mean_reward)
                self._batch_reward_mean_fn = jax.jit(
                    jax.vmap(self._reward_mean_fn, in_axes=(None, 0, None, None))
                )

            self._rollout_states_energy_fn = jax.jit(rollout_states_and_energy)
            self._rollout_env_states_fn = jax.jit(rollout_env_states)
            self._rollout_energy_fn = jax.jit(total_energy)
            self._batch_rollout_energy_fn = jax.jit(
                jax.vmap(self._rollout_energy_fn, in_axes=(None, 0, None, None))
            )
            self._init_reverse_diffuse_jit()

    def _init_reverse_diffuse_jit(self):
        """Initialize JIT-compiled reverse diffusion function."""
        self._reverse_diffuse_jit = None
        self._reverse_diffuse_chunk_jit = None
        if not self.action_space:
            return
        mode = self.action_score_mode.lower()
        if mode == "learned":
            return
        # If we have a state-space feasibility operator (e.g., CFSProjection), we must run
        # the Python diffusion loop to apply per-step projection + tracking. Keep JIT for the
        # fast action-filter path only.
        if self.constraint_manager is not None and getattr(self.constraint_manager, "feasibility_operator", None) is not None:
            return
        if mode == "reward" and self._batch_reward_mean_fn is None:
            return
        if mode == "energy" and self._batch_rollout_energy_fn is None:
            return

        horizon = int(self.horizon)
        if horizon <= 0:
            return
        action_dim = int(self.env.act_dim)
        Ndiffuse = int(self.action_diffuse_steps)
        if Ndiffuse <= 1:
            return
        num_particles = int(self.action_nsample)
        if num_particles <= 0:
            return

        temp = float(self.action_temp if self.action_temp > 1e-6 else 1e-6)
        beta0 = float(self.action_beta0)
        betaT = float(self.action_betaT)
        extra_sigma0 = float(self.action_extra_sigma)
        stage_switch = max(1, int(self.action_stage_ratio * (Ndiffuse - 1)))

        control_limit = getattr(self.env, "control_limit", None)
        clip_actions = control_limit is not None
        if clip_actions:
            control_limit = float(control_limit)

        if mode == "reward":
            score_fn_base = self._batch_reward_mean_fn
            def score_fn(state_init, actions, hard_clearance, hard_enabled):
                return score_fn_base(state_init, actions, hard_clearance, hard_enabled)
            def transform_scores(values):
                return values
        else:
            score_fn_base = self._batch_rollout_energy_fn
            def score_fn(state_init, actions, hard_clearance, hard_enabled):
                return score_fn_base(state_init, actions, hard_clearance, hard_enabled)
            def transform_scores(values):
                return -values

        betas = jnp.linspace(beta0, betaT, Ndiffuse, dtype=jnp.float32)
        alphas = 1.0 - betas
        alphas_bar = jnp.cumprod(alphas)
        sigmas = jnp.sqrt(1.0 - alphas_bar)
        extra_sigmas = jnp.linspace(extra_sigma0, 0.0, Ndiffuse, dtype=jnp.float32)
        diffusion_indices = jnp.arange(Ndiffuse - 1, 0, -1, dtype=jnp.int32)
        stage_switch_val = jnp.int32(stage_switch)
        temp_val = jnp.float32(temp)
        num_particles_f = jnp.array(num_particles, dtype=jnp.float32)
        uniform_logw = jnp.full(
            (num_particles,),
            -jnp.log(num_particles_f),
            dtype=jnp.float32,
        )

        # Optional JAX soft obstacle penalty (MDOC-style, using SDFTexture2D if available)
        use_soft_jax = (
            bool(self.use_constraint_in_scoring)
            and (self.constraint_manager is not None)
            and self.constraint_manager.has_soft()
        )

        soft_sampler = None
        if use_soft_jax and self.constraint_manager is not None:
            try:
                from enerdynamics.core.constraints.obstacle_constraints import ObstacleSoftConstraint
                for c in self.constraint_manager.soft_constraints:
                    if isinstance(c, ObstacleSoftConstraint):
                        tex = c.obstacles.get_sdf_texture_2d()
                        if tex is not None:
                            tex_j = tex.to_jax()
                            x_min = float(tex.x_min)
                            y_min = float(tex.y_min)
                            res = float(tex.res)
                            H_tex = int(tex.H)
                            W_tex = int(tex.W)

                            def _sample_sdf(p2):
                                p2 = jnp.asarray(p2, dtype=jnp.float32)
                                shp = p2.shape[:-1]
                                pts = p2.reshape((-1, 2))
                                ix = (pts[:, 0] - x_min) / res
                                iy = (pts[:, 1] - y_min) / res
                                ix = jnp.clip(ix, 0.0, W_tex - 1.0)
                                iy = jnp.clip(iy, 0.0, H_tex - 1.0)
                                x0 = jnp.floor(ix).astype(jnp.int32)
                                y0 = jnp.floor(iy).astype(jnp.int32)
                                x1 = jnp.minimum(x0 + 1, W_tex - 1)
                                y1 = jnp.minimum(y0 + 1, H_tex - 1)
                                wx = (ix - x0).astype(jnp.float32)
                                wy = (iy - y0).astype(jnp.float32)
                                v00 = tex_j[0, y0, x0]
                                v10 = tex_j[0, y0, x1]
                                v01 = tex_j[0, y1, x0]
                                v11 = tex_j[0, y1, x1]
                                v0 = v00 * (1.0 - wx) + v10 * wx
                                v1 = v01 * (1.0 - wx) + v11 * wx
                                v = v0 * (1.0 - wy) + v1 * wy
                                return v.reshape(shp)

                            soft_sampler = _sample_sdf
                            soft_alpha_default = float(getattr(c, "alpha", 1.0))
                            soft_beta_default = float(getattr(c, "beta", 10.0))
                            break
            except Exception:
                soft_sampler = None

        lambda_energy = jnp.float32(self.lambda_energy)

        def reverse_diffuse(
            rng_key,
            state_init,
            hard_clearance_by_idx,
            hard_enabled_by_idx,
            soft_alpha_by_idx,
            soft_beta_by_idx,
        ):
            rng_key, init_key = jax.random.split(rng_key)
            Ybar_init = jax.random.normal(
                init_key, (horizon, action_dim), dtype=jnp.float32
            )

            def body(carry, idx):
                rng_curr, Ybar_curr = carry
                rng_next, noise_key, extra_key = jax.random.split(rng_curr, 3)

                sqrt_alpha_bar_i = jnp.sqrt(alphas_bar[idx])
                sigma_i = sigmas[idx]
                Yi = Ybar_curr * sqrt_alpha_bar_i

                # Scheduled hard parameters for this diffusion index
                hard_clearance = hard_clearance_by_idx[idx]
                hard_enabled = hard_enabled_by_idx[idx]

                # Antithetic sampling (static shapes; num_particles known at compile time)
                if self.use_antithetic and num_particles > 1:
                    half = num_particles // 2
                    has_extra = num_particles % 2
                    sample_count = half + has_extra
                    eps_core = jax.random.normal(
                        noise_key, (sample_count, horizon, action_dim), dtype=jnp.float32
                    )
                    eps_half = eps_core[:half]
                    Y_pos = Ybar_curr[None, ...] + sigma_i * eps_half
                    Y_neg = Ybar_curr[None, ...] - sigma_i * eps_half
                    if has_extra:
                        eps_extra = eps_core[-1:]
                        Y_extra = Ybar_curr[None, ...] + sigma_i * eps_extra
                        Y0s = jnp.concatenate([Y_pos, Y_neg, Y_extra], axis=0)
                    else:
                        Y0s = jnp.concatenate([Y_pos, Y_neg], axis=0)
                else:
                    eps = jax.random.normal(
                        noise_key, (num_particles, horizon, action_dim), dtype=jnp.float32
                    )
                    Y0s = eps * sigma_i + Ybar_curr

                if clip_actions:
                    Y0s = jnp.clip(Y0s, -control_limit, control_limit)

                use_uniform = idx >= stage_switch_val

                def score_branch(actions):
                    raw_scores = score_fn(state_init, actions, hard_clearance, hard_enabled)
                    scores = transform_scores(raw_scores)

                    # Optional soft obstacle penalty (approx) using SDF texture
                    if soft_sampler is not None:
                        alpha = soft_alpha_by_idx[idx]
                        beta = soft_beta_by_idx[idx]

                        def one_penalty(action_seq):
                            def p_body(s, act):
                                # Keep penalty consistent with rollout: apply action filter if enabled.
                                act_safe = act
                                if self._jax_action_filter is not None:
                                    act_safe = self._jax_action_filter(s, act, hard_clearance, hard_enabled)
                                s_next = self.jax_transition(s, act_safe)
                                p = s_next[0:2]
                                sdf_val = soft_sampler(p)
                                return s_next, jnp.exp(-beta * sdf_val)
                            _, vals = jax.lax.scan(p_body, state_init, action_seq)
                            return alpha * jnp.sum(vals)

                        soft_pen = jax.vmap(one_penalty)(actions)
                        scores = (scores / lambda_energy) - soft_pen

                    score_mean = jnp.mean(scores)
                    score_std = jnp.std(scores)
                    score_std = jnp.where(score_std < 1e-4, 1.0, score_std)
                    logw = (scores - score_mean) / (score_std * temp_val)
                    logw = logw - jnp.max(logw)
                    reward_val = jnp.mean(scores)
                    return logw, reward_val

                logw, reward_val = jax.lax.cond(
                    use_uniform,
                    lambda _: (uniform_logw, jnp.nan),
                    lambda acts: score_branch(acts),
                    Y0s,
                )
                weights = jax.nn.softmax(logw)

                Ybar_weighted = jnp.tensordot(weights, Y0s, axes=([0], [0]))

                one_minus_alpha_bar = 1.0 - alphas_bar[idx]
                score_val = (-Yi + sqrt_alpha_bar_i * Ybar_weighted) / one_minus_alpha_bar
                Yim1 = (Yi + one_minus_alpha_bar * score_val) / jnp.sqrt(alphas[idx])
                sqrt_alpha_bar_prev = jnp.sqrt(alphas_bar[idx - 1])
                Ybar_next = Yim1 / sqrt_alpha_bar_prev

                extra_sigma = extra_sigmas[idx]

                def add_noise(y_in):
                    noise = jax.random.normal(
                        extra_key, (horizon, action_dim), dtype=jnp.float32
                    )
                    return y_in + extra_sigma * noise

                Ybar_next = jax.lax.cond(
                    jnp.abs(extra_sigma) > 0.0,
                    add_noise,
                    lambda y_in: y_in,
                    Ybar_next,
                )

                return (rng_next, Ybar_next), (reward_val, Ybar_next, Y0s)

            (rng_out, Ybar_final), (reward_hist, Ybar_hist, Ysamples_hist) = jax.lax.scan(
                body, (rng_key, Ybar_init), diffusion_indices
            )
            reward_hist = reward_hist[::-1]
            Ybar_hist = Ybar_hist[::-1]
            Ysamples_hist = Ysamples_hist[::-1]
            if clip_actions:
                Ybar_hist = jnp.clip(Ybar_hist, -control_limit, control_limit)
                Ysamples_hist = jnp.clip(Ysamples_hist, -control_limit, control_limit)
            return rng_out, Ybar_final, reward_hist, Ybar_hist, Ysamples_hist

        self._reverse_diffuse_jit = jax.jit(reverse_diffuse)

        # Chunked variant for a real-time tqdm progress bar without de-optimizing the inner loop:
        # Each chunk runs a small lax.scan under JIT; Python updates the bar once per chunk.
        chunk_len = int(self._reverse_diffuse_chunk_len)
        if chunk_len < 1:
            chunk_len = 1

        def reverse_diffuse_chunk(
            rng_key,
            state_init,
            Ybar_curr,
            idx_chunk,
            valid_chunk,
            hard_clearance_by_idx,
            hard_enabled_by_idx,
            soft_alpha_by_idx,
            soft_beta_by_idx,
        ):
            def step_one(carry, inputs):
                rng_curr, Ybar_in = carry
                idx, is_valid = inputs

                def do_step(args):
                    rng_curr_, Ybar_in_, idx_ = args
                    rng_next, noise_key, extra_key = jax.random.split(rng_curr_, 3)

                    sqrt_alpha_bar_i = jnp.sqrt(alphas_bar[idx_])
                    sigma_i = sigmas[idx_]
                    Yi = Ybar_in_ * sqrt_alpha_bar_i

                    hard_clearance = hard_clearance_by_idx[idx_]
                    hard_enabled = hard_enabled_by_idx[idx_]

                    if self.use_antithetic and num_particles > 1:
                        half = num_particles // 2
                        has_extra = num_particles % 2
                        sample_count = half + has_extra
                        eps_core = jax.random.normal(
                            noise_key, (sample_count, horizon, action_dim), dtype=jnp.float32
                        )
                        eps_half = eps_core[:half]
                        Y_pos = Ybar_in_[None, ...] + sigma_i * eps_half
                        Y_neg = Ybar_in_[None, ...] - sigma_i * eps_half
                        if has_extra:
                            eps_extra = eps_core[-1:]
                            Y_extra = Ybar_in_[None, ...] + sigma_i * eps_extra
                            Y0s_ = jnp.concatenate([Y_pos, Y_neg, Y_extra], axis=0)
                        else:
                            Y0s_ = jnp.concatenate([Y_pos, Y_neg], axis=0)
                    else:
                        eps = jax.random.normal(
                            noise_key, (num_particles, horizon, action_dim), dtype=jnp.float32
                        )
                        Y0s_ = eps * sigma_i + Ybar_in_

                    if clip_actions:
                        Y0s_ = jnp.clip(Y0s_, -control_limit, control_limit)

                    use_uniform = idx_ >= stage_switch_val

                    def score_branch(actions):
                        raw_scores = score_fn(state_init, actions, hard_clearance, hard_enabled)
                        scores = transform_scores(raw_scores)
                        if soft_sampler is not None:
                            alpha = soft_alpha_by_idx[idx_]
                            beta = soft_beta_by_idx[idx_]

                            def one_penalty(action_seq):
                                def p_body(s, act):
                                    act_safe = act
                                    if self._jax_action_filter is not None:
                                        act_safe = self._jax_action_filter(s, act, hard_clearance, hard_enabled)
                                    s_next = self.jax_transition(s, act_safe)
                                    p = s_next[0:2]
                                    sdf_val = soft_sampler(p)
                                    return s_next, jnp.exp(-beta * sdf_val)
                                _, vals = jax.lax.scan(p_body, state_init, action_seq)
                                return alpha * jnp.sum(vals)

                            soft_pen = jax.vmap(one_penalty)(actions)
                            scores = (scores / lambda_energy) - soft_pen

                        score_mean = jnp.mean(scores)
                        score_std = jnp.std(scores)
                        score_std = jnp.where(score_std < 1e-4, 1.0, score_std)
                        logw = (scores - score_mean) / (score_std * temp_val)
                        logw = logw - jnp.max(logw)
                        reward_val_ = jnp.mean(scores)
                        return logw, reward_val_

                    logw, reward_val_ = jax.lax.cond(
                        use_uniform,
                        lambda _: (uniform_logw, jnp.nan),
                        lambda acts: score_branch(acts),
                        Y0s_,
                    )
                    weights = jax.nn.softmax(logw)
                    Ybar_weighted = jnp.tensordot(weights, Y0s_, axes=([0], [0]))

                    one_minus_alpha_bar = 1.0 - alphas_bar[idx_]
                    score_val = (-Yi + sqrt_alpha_bar_i * Ybar_weighted) / one_minus_alpha_bar
                    Yim1 = (Yi + one_minus_alpha_bar * score_val) / jnp.sqrt(alphas[idx_])
                    sqrt_alpha_bar_prev = jnp.sqrt(alphas_bar[idx_ - 1])
                    Ybar_next = Yim1 / sqrt_alpha_bar_prev

                    extra_sigma = extra_sigmas[idx_]
                    def add_noise(y_in):
                        noise = jax.random.normal(
                            extra_key, (horizon, action_dim), dtype=jnp.float32
                        )
                        return y_in + extra_sigma * noise
                    Ybar_next = jax.lax.cond(
                        jnp.abs(extra_sigma) > 0.0,
                        add_noise,
                        lambda y_in: y_in,
                        Ybar_next,
                    )

                    return (rng_next, Ybar_next), (reward_val_, Ybar_next, Y0s_)

                def do_noop(args):
                    rng_curr_, Ybar_in_, _ = args
                    zeros_Y0s = jnp.zeros((num_particles, horizon, action_dim), dtype=jnp.float32)
                    return (rng_curr_, Ybar_in_), (jnp.nan, Ybar_in_, zeros_Y0s)

                (rng_out, Ybar_out), outs = jax.lax.cond(
                    is_valid,
                    do_step,
                    do_noop,
                    (rng_curr, Ybar_in, idx),
                )
                return (rng_out, Ybar_out), outs

            (rng_out, Ybar_out), (reward_chunk, Ybar_chunk, Ysamples_chunk) = jax.lax.scan(
                step_one, (rng_key, Ybar_curr), (idx_chunk, valid_chunk)
            )
            return rng_out, Ybar_out, reward_chunk, Ybar_chunk, Ysamples_chunk

        self._reverse_diffuse_chunk_jit = jax.jit(reverse_diffuse_chunk)

    # ------------------- main interface -------------------
    def plan(self, rng: jax.Array) -> Dict[str, Any]:
        """Main planning interface. Always uses single particle mode (n_particles removed)."""
        try:
            x0, info = self.env.reset(rng)
        except TypeError:
            x0, info = self.env.reset()
        
        if self.action_space:
            if self.diffusion_mode == "reverse":
                return self._run_action_reverse(x0, info, rng)
            return self._run_action_space(x0, info, rng)
        
        # Default to single particle mode (n_particles removed)
        return self._run_single(x0, info)

    # ------------------- single particle -------------------
    def _run_single(self, x, info):
        """Single particle mode (default)."""
        xs = [x]
        Es = []
        term_list = []
        rewards = []

        for t in range(self.horizon):
            u = jnp.zeros((self.env.act_dim,), dtype=jnp.float32)
            ctx = {"t": t, **info}

            E_val = self.energy.compute(x, u, ctx)
            grad_x = self.grad_energy_x(x, u, ctx)

            G_inv = euclidean_metric_inv(x)
            x_new = langevin_step(x, grad_x, G_inv, self.dt, self.noise_std)

            # constraint projection
            if self.state_box is not None:
                low, high = self.state_box
                x_new = project_box(x_new, low, high)

            # really step the environment (to record cost etc.)
            x_env, cost, done, info_n = self.env.step(
                np.array(x_new, dtype=np.float32),
                np.array(u, dtype=np.float32),
                t,
                info,
            )
            reward = -cost

            xs.append(x_env)
            Es.append(E_val)
            term_list.append(self.energy.breakdown(x, u, ctx))
            rewards.append(reward)

            x, info = x_env, info_n
            if done:
                break

        return {
            "states": jnp.stack(xs, axis=0),
            "energies": jnp.stack(Es, axis=0),
            "terms": term_list,
            "rewards": jnp.stack(rewards, axis=0),
            "initial_state": xs[0],
        }

    # ------------------- action sequence diffusion -------------------
    def _run_action_space(self, x0, info, rng):
        """Action space optimization using gradient descent."""
        act_dim = self.env.act_dim
        actions = jnp.zeros((self.horizon, act_dim), dtype=jnp.float32)
        x0_jnp = jnp.asarray(x0, dtype=jnp.float32)

        energy_from_actions = lambda seq: self._rollout_energy_fn(x0_jnp, seq, 0.0, False)
        energy_and_grad = jax.jit(jax.value_and_grad(energy_from_actions))

        energy_hist = []
        for _ in range(self.horizon):
            E_val, grad_val = energy_and_grad(actions)
            energy_hist.append(float(E_val))

            actions_np = np.asarray(actions, dtype=np.float32)
            grad_np = np.asarray(grad_val, dtype=np.float32)

            flat_actions = actions_np.reshape(-1)
            flat_grad = grad_np.reshape(-1)
            G_inv = np.eye(flat_actions.size, dtype=np.float32)
            updated_flat = langevin_step(flat_actions, flat_grad, G_inv, self.dt, self.noise_std)

            updated = updated_flat.reshape(self.horizon, act_dim)

            if hasattr(self.env, "control_limit") and self.env.control_limit is not None:
                limit = float(self.env.control_limit)
                updated = np.clip(updated, -limit, limit)

            actions = jnp.asarray(updated, dtype=jnp.float32)

        x = np.array(x0, dtype=np.float32)
        xs = [x]
        rewards = []
        terms = []
        energies = []
        info_curr = info

        actions_np_final = np.asarray(actions, dtype=np.float32)

        for t in range(self.horizon):
            u = actions_np_final[t]
            ctx = {"t": t, **info_curr}
            E_val = self.energy.compute(x, u, ctx)
            energies.append(E_val)
            terms.append(self.energy.breakdown(x, u, ctx))

            x_pred = self.env.transition(x, u)
            x_env, cost, done, info_next = self.env.step(
                np.array(x_pred, dtype=np.float32),
                np.array(u, dtype=np.float32),
                t,
                info_curr,
            )

            rewards.append(-cost)
            xs.append(x_env)

            x = x_env
            info_curr = info_next
            if done:
                break

        return {
            "states": jnp.stack(xs, axis=0),
            "energies": jnp.stack(np.array(energies, dtype=np.float32), axis=0),
            "terms": terms,
            "rewards": jnp.stack(np.array(rewards, dtype=np.float32), axis=0),
            "actions": jnp.asarray(actions_np_final, dtype=jnp.float32),
            "initial_state": xs[0],
            "energy_iterations": jnp.asarray(np.array(energy_hist, dtype=np.float32)),
        }

    def _simulate_rewards_numpy(self, state, actions):
        """Simulate rewards using numpy (fallback)."""
        x = np.array(state, dtype=np.float32)
        rewards = []
        for u in np.asarray(actions, dtype=np.float32):
            x = self.env.transition(x, u)
            rewards.append(-self.env.cost(x))
        return np.array(rewards, dtype=np.float32)

    def _simulate_energy_numpy(self, state, actions, info):
        """Simulate energy using numpy (fallback)."""
        x = np.array(state, dtype=np.float32)
        info_curr = info
        total_energy = 0.0
        for t, u in enumerate(np.asarray(actions, dtype=np.float32)):
            ctx = {"t": t, **info_curr}
            total_energy += float(self.energy.compute(x, u, ctx))
            x = self.env.transition(x, u)
        return total_energy

    def _actions_to_trajectory(self, x0, actions):
        """
        Convert actions array to Trajectory for constraint evaluation.
        
        Args:
            x0: Initial state
            actions: Action sequence (horizon, act_dim) or single action (act_dim)
            
        Returns:
            Trajectory object
        """
        actions_array = np.asarray(actions, dtype=np.float32)
        if actions_array.ndim == 1:
            # Single action, create single-step trajectory
            actions_array = actions_array[None, :]
        
        states = [np.asarray(x0, dtype=np.float32)]
        x = np.asarray(x0, dtype=np.float32)
        
        actions_list = []
        for act in actions_array:
            actions_list.append(act)
            x_next = self.env.transition(x, act)
            states.append(x_next)
            x = x_next
        
        return Trajectory(states=states, actions=actions_list)
    
    def _extract_actions_from_trajectory(self, trajectory: Trajectory):
        """
        Extract actions array from trajectory.
        
        Args:
            trajectory: Trajectory object
            
        Returns:
            Actions array (horizon, act_dim)
        """
        if not trajectory.actions:
            return np.zeros((self.horizon, self.env.act_dim), dtype=np.float32)
        return np.stack([np.asarray(act, dtype=np.float32) for act in trajectory.actions], axis=0)

    def _track_actions_to_projected_states(
        self,
        x0,
        actions_init,
        target_states,
        *,
        gn_iters: int = 2,
        reg: float = 1e-3,
    ) -> np.ndarray:
        """
        Map a projected state trajectory back to an executable action sequence.

        Hard feasibility operators like CFS typically edit *states*.
        EDOC runs diffusion in *action space*, so if we don't re-solve actions,
        the projected trajectory won't affect Ybar (actions stay unchanged).

        We use a lightweight Gauss-Newton/SCP step per timestep:
            u <- u + (B^T B + reg I)^{-1} B^T (x_target - f(x,u))
        where B = ∂f/∂u at the current (x,u).
        """
        if not self.action_space or self._jac_model_u_fn is None:
            return np.asarray(actions_init, dtype=np.float32)

        actions = np.asarray(actions_init, dtype=np.float32).copy()
        if actions.ndim != 2:
            return np.asarray(actions_init, dtype=np.float32)

        if not target_states or len(target_states) < 2:
            return actions

        horizon = actions.shape[0]
        T = min(horizon, len(target_states) - 1)
        if T <= 0:
            return actions

        limit = getattr(self.env, "control_limit", None)
        limit_val = float(limit) if limit is not None else None

        x = jnp.asarray(x0, dtype=jnp.float32)
        for t in range(T):
            u = jnp.asarray(actions[t], dtype=jnp.float32)
            x_tgt = jnp.asarray(target_states[t + 1], dtype=jnp.float32)

            for _ in range(max(1, int(gn_iters))):
                x_pred = self.jax_model_transition(x, u)
                B = self._jac_model_u_fn(x, u)  # (state_dim, act_dim)
                Bt = jnp.swapaxes(B, -2, -1)   # (act_dim, state_dim)
                BtB = Bt @ B
                act_dim = int(BtB.shape[-1])
                BtB_reg = BtB + (float(reg) * jnp.eye(act_dim, dtype=jnp.float32))
                rhs = Bt @ (x_tgt - x_pred)
                delta = jnp.linalg.solve(BtB_reg, rhs)
                u = u + delta
                if limit_val is not None:
                    u = jnp.clip(u, -limit_val, limit_val)

            # Advance using updated action
            x = self.jax_model_transition(x, u)
            actions[t] = np.asarray(u, dtype=np.float32)

        return actions

    def _score_particles(
        self, 
        x0, 
        info, 
        batch_actions,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ):
        """Score action particles."""
        mode = self.action_score_mode.lower()
        x0_jnp = jnp.asarray(x0, dtype=jnp.float32)
        batch_actions_jnp = jnp.asarray(batch_actions, dtype=jnp.float32)

        if mode == "reward":
            if self._batch_reward_mean_fn is not None:
                hard_enabled = False
                hard_clearance = 0.0
                if self.constraint_manager and self.constraint_manager.action_filter_operator is not None:
                    hard_enabled = True
                if self.constraint_manager and self.constraint_manager.schedule_manager is not None:
                    hard_enabled = bool(self.constraint_manager.schedule_manager.is_hard_active(step, total_steps))
                    hard_clearance = float(
                        self.constraint_manager.schedule_manager.get_hard_clearance(
                            default=0.0, step=step, total_steps=total_steps
                        )
                    )
                rewards_mean = self._batch_reward_mean_fn(x0_jnp, batch_actions_jnp, hard_clearance, hard_enabled)
                scores = np.asarray(rewards_mean, dtype=np.float32)
            else:
                num = batch_actions.shape[0]
                scores = np.zeros((num,), dtype=np.float32)
                for idx in range(num):
                    rewards = self._simulate_rewards_numpy(x0, batch_actions[idx])
                    scores[idx] = float(np.mean(rewards))
        elif mode == "energy":
            if self._batch_rollout_energy_fn is not None:
                hard_enabled = False
                hard_clearance = 0.0
                if self.constraint_manager and self.constraint_manager.action_filter_operator is not None:
                    hard_enabled = True
                if self.constraint_manager and self.constraint_manager.schedule_manager is not None:
                    hard_enabled = bool(self.constraint_manager.schedule_manager.is_hard_active(step, total_steps))
                    hard_clearance = float(
                        self.constraint_manager.schedule_manager.get_hard_clearance(
                            default=0.0, step=step, total_steps=total_steps
                        )
                    )
                energy_vals = self._batch_rollout_energy_fn(x0_jnp, batch_actions_jnp, hard_clearance, hard_enabled)
                scores = -np.asarray(energy_vals, dtype=np.float32)
            else:
                num = batch_actions.shape[0]
                scores = np.zeros((num,), dtype=np.float32)
                for idx in range(num):
                    energy_val = float(
                        self._rollout_energy_fn(
                            x0_jnp,
                            jnp.asarray(batch_actions[idx], dtype=jnp.float32),
                            0.0,
                            False,
                        )
                    )
                    scores[idx] = -energy_val
        elif mode == "learned":
            scores = np.zeros((batch_actions.shape[0],), dtype=np.float32)
        else:
            raise ValueError(f"Unknown action_score_mode {self.action_score_mode}")
        
        # Add soft constraint penalties if enabled
        if self.use_constraint_in_scoring and self.constraint_manager and self.constraint_manager.has_soft():
            # Scale energy/reward by lambda_energy (E_soft = (1/λ)J + S)
            scores = scores / self.lambda_energy
            
            # Batch compute soft constraint penalties (optimized)
            num = batch_actions.shape[0]
            trajectories = [self._actions_to_trajectory(x0, batch_actions[idx]) for idx in range(num)]
            soft_penalties = self.constraint_manager.compute_soft_energy_batch(
                trajectories, step=step, total_steps=total_steps
            )
            scores = scores - soft_penalties  # Subtract penalty (higher penalty = lower score)
        
        return scores

    def _add_extra_noise(self, actions_array, sigma):
        """Add extra noise to actions."""
        if sigma <= 0.0:
            return actions_array
        noise = self._np_rng.standard_normal(size=actions_array.shape).astype(np.float32)
        return actions_array + sigma * noise

    def _run_action_reverse(self, x0, info, rng):
        """Reverse diffusion in action space."""
        act_dim = self.env.act_dim
        horizon = self.horizon
        Ndiffuse = self.action_diffuse_steps
        beta0 = self.action_beta0
        betaT = self.action_betaT
        temp = self.action_temp
        control_limit = getattr(self.env, "control_limit", None)
        num_particles = max(1, int(self.action_nsample))
        temp_eps = temp if temp > 1e-6 else 1e-6

        x0_jnp = jnp.asarray(x0, dtype=jnp.float32)
        reward_history_arr = None
        diffusion_actions_traj_arr = None
        diffusion_samples_traj_arr = None

        if self._reverse_diffuse_jit is not None:
            # Precompute scheduled hard/soft parameters as arrays indexed by diffusion idx (0..Ndiffuse-1),
            # so the reverse diffusion loop can remain pure JAX.
            hard_clearance_by_idx = jnp.zeros((Ndiffuse,), dtype=jnp.float32)
            hard_enabled_by_idx = jnp.zeros((Ndiffuse,), dtype=jnp.bool_)
            soft_alpha_by_idx = jnp.zeros((Ndiffuse,), dtype=jnp.float32)
            soft_beta_by_idx = jnp.zeros((Ndiffuse,), dtype=jnp.float32)

            # Resolve soft defaults if an ObstacleSoftConstraint exists (for JAX soft penalty)
            soft_alpha_default = 0.0
            soft_beta_default = 10.0
            if self.constraint_manager is not None and self.constraint_manager.has_soft():
                try:
                    from enerdynamics.core.constraints.obstacle_constraints import ObstacleSoftConstraint
                    for c in self.constraint_manager.soft_constraints:
                        if isinstance(c, ObstacleSoftConstraint):
                            soft_alpha_default = float(getattr(c, "alpha", 1.0))
                            soft_beta_default = float(getattr(c, "beta", 10.0))
                            break
                except Exception:
                    pass

            if self.constraint_manager is not None and self.constraint_manager.schedule_manager is not None:
                sched = self.constraint_manager.schedule_manager
                total_steps = max(1, Ndiffuse - 2)
                hard_filter_exists = self.constraint_manager.action_filter_operator is not None
                hc = []
                he = []
                sa = []
                sb = []
                for idx in range(Ndiffuse):
                    # Map diffusion index (idx) to schedule step in reverse_mode:
                    # diffusion runs i = Ndiffuse-1 (most noisy) -> 1 (most clean)
                    # schedule expects step = 0 (most noisy) -> total_steps (most clean)
                    # so: step_k = (Ndiffuse - 1) - idx, clamped to [0, total_steps]
                    step_k = int((Ndiffuse - 1) - idx)
                    if step_k < 0:
                        step_k = 0
                    if step_k > total_steps:
                        step_k = total_steps
                    if hard_filter_exists:
                        he.append(bool(sched.is_hard_active(step_k, total_steps)))
                        hc.append(float(sched.get_hard_clearance(default=0.0, step=step_k, total_steps=total_steps)))
                    else:
                        he.append(False)
                        hc.append(0.0)
                    # Soft schedules (alpha/beta) are used only if a JAX soft penalty exists
                    sa.append(float(sched.get_soft_alpha(default=soft_alpha_default, step=step_k, total_steps=total_steps)))
                    sb.append(float(sched.get_soft_beta(default=soft_beta_default, step=step_k, total_steps=total_steps)))

                hard_clearance_by_idx = jnp.asarray(np.asarray(hc, dtype=np.float32), dtype=jnp.float32)
                hard_enabled_by_idx = jnp.asarray(np.asarray(he, dtype=bool))
                soft_alpha_by_idx = jnp.asarray(np.asarray(sa, dtype=np.float32), dtype=jnp.float32)
                soft_beta_by_idx = jnp.asarray(np.asarray(sb, dtype=np.float32), dtype=jnp.float32)
            else:
                # No schedule: enable hard filter if present, with clearance=0; soft uses defaults.
                if self.constraint_manager is not None and self.constraint_manager.action_filter_operator is not None:
                    hard_enabled_by_idx = jnp.ones((Ndiffuse,), dtype=jnp.bool_)
                soft_alpha_by_idx = jnp.full((Ndiffuse,), jnp.float32(soft_alpha_default), dtype=jnp.float32)
                soft_beta_by_idx = jnp.full((Ndiffuse,), jnp.float32(soft_beta_default), dtype=jnp.float32)

            # If enabled, run in JITted chunks to show a real-time progress bar
            # without de-optimizing the inner loop.
            if self.show_tqdm and HAS_TQDM and self._reverse_diffuse_chunk_jit is not None:
                chunk_len = int(getattr(self, "_reverse_diffuse_chunk_len", 10))
                if chunk_len < 1:
                    chunk_len = 1
                total_steps = Ndiffuse - 1
                diffusion_indices = np.arange(Ndiffuse - 1, 0, -1, dtype=np.int32)

                rng, init_key = jax.random.split(rng)
                Ybar = jax.random.normal(init_key, (horizon, act_dim), dtype=jnp.float32)

                reward_chunks = []
                ybar_chunks = []
                ysamp_chunks = []
                pbar = tqdm(total=total_steps, desc="EDOC Diffusion", unit="step", leave=False)
                try:
                    for start in range(0, total_steps, chunk_len):
                        sub = diffusion_indices[start:start + chunk_len]
                        valid = np.ones((len(sub),), dtype=bool)
                        if len(sub) < chunk_len:
                            pad_n = chunk_len - len(sub)
                            sub = np.pad(sub, (0, pad_n), mode="constant", constant_values=1)
                            valid = np.pad(valid, (0, pad_n), mode="constant", constant_values=False)
                        idx_chunk = jnp.asarray(sub, dtype=jnp.int32)
                        valid_chunk = jnp.asarray(valid, dtype=jnp.bool_)
                        rng, Ybar, r_c, y_c, s_c = self._reverse_diffuse_chunk_jit(
                            rng,
                            x0_jnp,
                            Ybar,
                            idx_chunk,
                            valid_chunk,
                            hard_clearance_by_idx,
                            hard_enabled_by_idx,
                            soft_alpha_by_idx,
                            soft_beta_by_idx,
                        )
                        reward_chunks.append(r_c)
                        ybar_chunks.append(y_c)
                        ysamp_chunks.append(s_c)
                        pbar.update(int(np.sum(valid)))
                finally:
                    pbar.close()

                reward_hist = jnp.concatenate(reward_chunks, axis=0)[:total_steps]
                Ybar_hist = jnp.concatenate(ybar_chunks, axis=0)[:total_steps]
                Ysamples_hist = jnp.concatenate(ysamp_chunks, axis=0)[:total_steps]

                # Match the full-jit function output convention (reverse to early→late)
                reward_hist_jnp = reward_hist[::-1]
                traj_jnp = Ybar_hist[::-1]
                samples_jnp = Ysamples_hist[::-1]
                actions_jnp = Ybar
            else:
                rng, diffuse_key = jax.random.split(rng)
                _, actions_jnp, reward_hist_jnp, traj_jnp, samples_jnp = self._reverse_diffuse_jit(
                    diffuse_key,
                    x0_jnp,
                    hard_clearance_by_idx,
                    hard_enabled_by_idx,
                    soft_alpha_by_idx,
                    soft_beta_by_idx,
                )
            actions_np_final = np.asarray(actions_jnp, dtype=np.float32)
            reward_history_arr = jnp.asarray(reward_hist_jnp, dtype=jnp.float32)
            diffusion_actions_traj_arr = jnp.asarray(traj_jnp, dtype=jnp.float32)
            diffusion_samples_traj_arr = jnp.asarray(samples_jnp, dtype=jnp.float32)
        else:
            betas = np.linspace(beta0, betaT, Ndiffuse, dtype=np.float32)
            alphas = 1.0 - betas
            alphas_bar = np.cumprod(alphas, axis=0)
            sigmas = np.sqrt(1.0 - alphas_bar)
            extra_sigmas = np.linspace(self.action_extra_sigma, 0.0, Ndiffuse, dtype=np.float32)

            rng, init_key = jax.random.split(rng)
            Ybar = np.array(
                jax.random.normal(init_key, (horizon, act_dim), dtype=jnp.float32),
                dtype=np.float32,
            )

            reward_history = []
            Ybar_history = []
            Ysamples_history = []
            stage_switch = max(1, int(self.action_stage_ratio * (Ndiffuse - 1)))

            # Create progress bar for reverse diffusion
            diffusion_iter = range(Ndiffuse - 1, 0, -1)
            if HAS_TQDM:
                diffusion_iter = tqdm(diffusion_iter, desc="EDOC Diffusion", unit="step", 
                                     total=Ndiffuse-1, leave=False)

            for i in diffusion_iter:
                Yi = Ybar * np.sqrt(alphas_bar[i])

                rng, sample_key = jax.random.split(rng)
                sigma_i = float(sigmas[i])
                if self.use_antithetic and num_particles > 0:
                    half = num_particles // 2
                    has_extra = num_particles % 2
                    sample_count = half + has_extra
                    if sample_count > 0:
                        eps_core = np.array(
                            jax.random.normal(
                                sample_key, (sample_count, horizon, act_dim), dtype=jnp.float32
                            ),
                            dtype=np.float32,
                        )
                    else:
                        eps_core = np.zeros((0, horizon, act_dim), dtype=np.float32)

                    Y_candidates = []
                    if half > 0:
                        eps_half = eps_core[:half]
                        Y_candidates.append(Ybar + sigma_i * eps_half)
                        Y_candidates.append(Ybar - sigma_i * eps_half)
                    if has_extra:
                        eps_extra = eps_core[-1:]
                        Y_candidates.append(Ybar + sigma_i * eps_extra)

                    if not Y_candidates:
                        Y0s = np.broadcast_to(Ybar, (1, horizon, act_dim)).copy()
                    elif len(Y_candidates) == 1:
                        Y0s = Y_candidates[0]
                    else:
                        Y0s = np.concatenate(Y_candidates, axis=0)
                else:
                    if num_particles > 0:
                        eps = np.array(
                            jax.random.normal(
                                sample_key, (num_particles, horizon, act_dim), dtype=jnp.float32
                            ),
                            dtype=np.float32,
                        )
                        Y0s = Ybar + sigma_i * eps
                    else:
                        Y0s = np.broadcast_to(Ybar, (1, horizon, act_dim)).copy()

                if control_limit is not None:
                    limit = float(control_limit)
                    Y0s = np.clip(Y0s, -limit, limit)

                # Convert diffusion index to constraint step
                # i goes from Ndiffuse-1 (initial noise) to 1 (near final)
                # For reverse_mode constraint scheduling: step=0 is initial, step=total_steps is final
                constraint_step = (Ndiffuse - 1) - i  # i=Ndiffuse-1 -> 0, i=1 -> Ndiffuse-2
                constraint_total_steps = Ndiffuse - 2  # Total steps for constraints
                
                use_uniform = i >= stage_switch and self.action_score_mode != "learned"
                if use_uniform:
                    weights = np.full((num_particles,), 1.0 / num_particles, dtype=np.float32)
                    reward_history.append(np.nan)
                elif self.action_score_mode == "energy":
                    actions_batch = jnp.asarray(Y0s, dtype=jnp.float32)
                    hard_enabled = False
                    hard_clearance = 0.0
                    if self.constraint_manager and self.constraint_manager.action_filter_operator is not None:
                        hard_enabled = True
                    if self.constraint_manager and self.constraint_manager.schedule_manager is not None:
                        hard_enabled = bool(self.constraint_manager.schedule_manager.is_hard_active(constraint_step, constraint_total_steps))
                        hard_clearance = float(
                            self.constraint_manager.schedule_manager.get_hard_clearance(
                                default=0.0, step=constraint_step, total_steps=constraint_total_steps
                            )
                        )
                    energy_vals = np.asarray(self._batch_rollout_energy_fn(x0_jnp, actions_batch, hard_clearance, hard_enabled))
                    
                    # Add soft constraint penalties if enabled
                    if self.use_constraint_in_scoring and self.constraint_manager and self.constraint_manager.has_soft():
                        # Scale energy by lambda_energy
                        energy_vals = energy_vals / self.lambda_energy
                        
                        # Batch compute soft constraint penalties (optimized)
                        # Instead of looping over each particle, batch process all trajectories
                        trajectories = [self._actions_to_trajectory(x0, Y0s[idx]) for idx in range(len(Y0s))]
                        soft_penalties = self.constraint_manager.compute_soft_energy_batch(
                            trajectories, step=constraint_step, total_steps=constraint_total_steps
                        )
                        energy_vals = energy_vals + soft_penalties
                    
                    energy_mean = float(np.mean(energy_vals))
                    energy_std = float(np.std(energy_vals))
                    if energy_std < 1e-4:
                        weights = np.full((num_particles,), 1.0 / num_particles, dtype=np.float32)
                    else:
                        denom = energy_std * temp_eps
                        logw = -(energy_vals - energy_mean) / denom
                        logw = logw - np.max(logw)
                        weights = np.exp(logw).astype(np.float64)
                        weights = weights / np.sum(weights)
                    reward_history.append(float(-np.mean(energy_vals)))
                else:
                    if self.action_score_mode == "learned":
                        scores = np.zeros((num_particles,), dtype=np.float32)
                    else:
                        scores = self._score_particles(
                            x0, info, Y0s, 
                            step=constraint_step, 
                            total_steps=constraint_total_steps
                        )

                    score_std = float(scores.std())
                    if score_std < 1e-4:
                        weights = np.full((num_particles,), 1.0 / num_particles, dtype=np.float32)
                    else:
                        score_mean = float(scores.mean())
                        denom = score_std * temp_eps
                        logp0 = (scores - score_mean) / denom
                        logp0 = logp0 - np.max(logp0)
                        weights = np.exp(logp0)
                        weights = weights / np.sum(weights)
                    reward_history.append(float(np.mean(scores)))
                
                weights = np.asarray(weights, dtype=np.float32)

                Ybar_weighted = np.tensordot(weights, Y0s, axes=([0], [0]))

                score = (-Yi + np.sqrt(alphas_bar[i]) * Ybar_weighted) / (1.0 - alphas_bar[i])
                Yim1 = (Yi + (1.0 - alphas_bar[i]) * score) / np.sqrt(alphas[i])
                Ybar = Yim1 / np.sqrt(alphas_bar[i - 1])
                Ybar = self._add_extra_noise(Ybar, extra_sigmas[i])
                
                # ===== Apply hard constraint projection (state-space feasibility operator, e.g., CFS) =====
                if self.constraint_manager and self.constraint_manager.has_hard():
                    trajectory = self._actions_to_trajectory(x0, Ybar)
                    projected_trajectory = self.constraint_manager.project_hard(
                        trajectory,
                        step=constraint_step,
                        total_steps=constraint_total_steps,
                    )

                    # If projection edits states, map back to actions (tracking).
                    states_changed = True
                    try:
                        traj_states = np.asarray(trajectory.states, dtype=np.float32)
                        proj_states = np.asarray(projected_trajectory.states, dtype=np.float32)
                        if traj_states.shape == proj_states.shape:
                            max_delta = float(np.max(np.linalg.norm(traj_states - proj_states, axis=-1)))
                            states_changed = max_delta > 1e-6
                    except Exception:
                        states_changed = True

                    if states_changed:
                        Ybar = self._track_actions_to_projected_states(
                            x0, Ybar, projected_trajectory.states
                        )
                    else:
                        Ybar = self._extract_actions_from_trajectory(projected_trajectory)

                if control_limit is not None:
                    limit = float(control_limit)
                    Ybar_history.append(np.clip(Ybar, -limit, limit))
                    Ysamples_history.append(np.clip(Y0s, -limit, limit))
                else:
                    Ybar_history.append(np.array(Ybar, copy=True))
                    Ysamples_history.append(np.array(Y0s, copy=True))

            # Final hard constraint projection (step 0, final state)
            if self.constraint_manager and self.constraint_manager.has_hard():
                trajectory = self._actions_to_trajectory(x0, Ybar)
                projected_trajectory = self.constraint_manager.project_hard(
                    trajectory, step=0, total_steps=constraint_total_steps
                )
                Ybar = self._track_actions_to_projected_states(
                    x0, Ybar, projected_trajectory.states
                )
            
            if control_limit is not None:
                limit = float(control_limit)
                actions_final = np.clip(Ybar, -limit, limit)
            else:
                actions_final = Ybar

            actions_np_final = np.asarray(actions_final, dtype=np.float32)
            reward_history_arr = jnp.asarray(np.array(reward_history[::-1], dtype=np.float32))
            diffusion_actions_traj_arr = jnp.asarray(np.array(Ybar_history[::-1], dtype=np.float32))
            diffusion_samples_traj_arr = jnp.asarray(np.array(Ysamples_history[::-1], dtype=np.float32))

        # JAX rollout for final trajectory (GPU/JIT-friendly).
        # We intentionally skip env.step/info/terms here for speed; experiments mainly use states/energies/rewards.
        actions_jnp_final = jnp.asarray(actions_np_final, dtype=jnp.float32)
        reward_cost_fn = getattr(self.env, "jax_cost", None)

        def rollout_final(state0, actions_seq):
            def body(s, inputs):
                t, u = inputs
                ctx = {"t": t}
                e_val = self.energy.compute(s, u, ctx)
                s_next = self.jax_transition(s, u)
                if reward_cost_fn is None:
                    r = jnp.float32(0.0)
                else:
                    r = -reward_cost_fn(s_next)
                return s_next, (s_next, e_val, r)

            t_idx = jnp.arange(actions_seq.shape[0], dtype=jnp.int32)
            _, (states_seq, energies_seq, rewards_seq) = jax.lax.scan(body, state0, (t_idx, actions_seq))
            states_full = jnp.concatenate([state0[None, :], states_seq], axis=0)
            return states_full, energies_seq, rewards_seq

        states_full, energies_arr, rewards_arr = jax.jit(rollout_final)(x0_jnp, actions_jnp_final)

        return {
            "states": states_full,
            "energies": energies_arr,
            "terms": [],
            "rewards": rewards_arr,
            "actions": jnp.asarray(actions_np_final, dtype=jnp.float32),
            "initial_state": states_full[0],
            "reward_history": reward_history_arr if reward_history_arr is not None else jnp.asarray([], dtype=jnp.float32),
            "diffusion_actions_traj": diffusion_actions_traj_arr if diffusion_actions_traj_arr is not None else jnp.asarray([], dtype=jnp.float32),
            "diffusion_sampled_actions": diffusion_samples_traj_arr if diffusion_samples_traj_arr is not None else jnp.asarray([], dtype=jnp.float32),
        }


# ============================================================================
# Energy Adapter for Legacy Planners
# ============================================================================

class EnergyToLegacyAdapter:
    """
    Adapter that converts a new EnergyFunctional to LegacyEnergyFunctional.
    
    This is a temporary bridge until EDOCPlanner is fully refactored.
    The adapter creates a point-wise energy function that approximates
    the trajectory-based energy by evaluating single-step trajectories.
    """
    
    def __init__(self, energy: EnergyFunctional, dynamics: DynamicsModel):
        """
        Initialize adapter.
        
        Args:
            energy: New EnergyFunctional
            dynamics: Dynamics model (for trajectory rollout)
        """
        from enerdynamics.core.energy import EnergyTerm
        
        self.energy = energy
        self.dynamics = dynamics
        
        # Create a legacy energy that evaluates trajectories point-wise
        def trajectory_energy(x, u, ctx):
            """
            Evaluate energy at a single (state, action) pair.
            
            This creates a single-step trajectory and evaluates it using
            the new EnergyFunctional interface.
            """
            try:
                # Create a single-step trajectory
                x_next = self.dynamics.step(x, u)
                states = [x, x_next]
                actions = [u]
                traj = Trajectory(states=states, actions=actions)
                
                # Evaluate using new energy functional
                # This gives us the energy for this single step
                total_energy = self.energy.total_energy(traj)
                
                # Return as a scalar (JAX array)
                if isinstance(total_energy, (jnp.ndarray, np.ndarray)):
                    return float(total_energy)
                return float(total_energy)
            except Exception as e:
                # Fallback: return a default value
                # In production, you might want to log this
                return 0.0
        
        self._legacy_energy = LegacyEnergyFunctional({
            "total": EnergyTerm(trajectory_energy, 1.0)
        })
    
    def __call__(self, *args, **kwargs):
        """Delegate to legacy energy."""
        return self._legacy_energy(*args, **kwargs)
    
    def compute(self, *args, **kwargs):
        """Delegate to legacy energy."""
        return self._legacy_energy.compute(*args, **kwargs)
    
    def breakdown(self, *args, **kwargs):
        """Delegate to legacy energy."""
        return self._legacy_energy.breakdown(*args, **kwargs)
    
    @property
    def legacy_energy(self):
        """Get the legacy energy functional."""
        return self._legacy_energy


# ============================================================================
# EDOC Solver (Unified Interface)
# ============================================================================

class EDOCSolver(SamplingSolver):
    """
    EDOC solver that follows the new unified Solver interface.
    
    This wraps EDOCPlanner to make it compatible with the unified architecture.
    EDOC uses diffusion-based optimization, so it directly optimizes rather than
    sampling and selecting.
    """
    
    def __init__(
        self,
        dynamics: DynamicsModel,
        energy: EnergyFunctional,
        backend: Backend,
        horizon: int = 80,
        dt: float = 0.1,
        noise_std: float = 0.05,
        action_space: bool = True,
        diffusion_mode: str = "reverse",
        action_diffuse_steps: int = 100,
        action_beta0: float = 1e-4,
        action_betaT: float = 1e-2,
        action_temp: float = 0.1,
        action_extra_sigma: float = 0.0,
        action_stage_ratio: float = 1.0,
        action_score_mode: str = "energy",
        action_nsample: int = 128,
        use_antithetic: bool = True,
        dyn_loss_coeff: float = 1.0,
        dyn_loss_mode: str = "trajectory",
        state_box: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        seed: int = 0,
        **kwargs
    ):
        """
        Initialize EDOC solver.
        
        Args:
            dynamics: Dynamics model
            energy: Energy functional
            backend: Computational backend (must be JAX for EDOC)
            horizon: Planning horizon
            dt: Time step
            noise_std: Noise standard deviation for Langevin diffusion
            action_space: If True, optimize in action space; if False, optimize in state space
            diffusion_mode: "forward" or "reverse" diffusion
            action_diffuse_steps: Number of diffusion steps
            action_beta0: Initial noise level for diffusion
            action_betaT: Final noise level for diffusion
            action_temp: Temperature for sampling
            action_extra_sigma: Extra noise variance
            action_stage_ratio: Ratio for multi-stage diffusion
            action_score_mode: "energy", "reward", or "learned" scoring mode
            action_nsample: Number of action samples per diffusion step
            use_antithetic: Whether to use antithetic sampling
            dyn_loss_coeff: Coefficient for dynamics mismatch loss
            dyn_loss_mode: "terminal" or "trajectory" dynamics loss
            state_box: Optional state box constraints (low, high)
            seed: Random seed
            **kwargs: Additional EDOC-specific parameters
        """
        super().__init__(dynamics, energy, backend, **kwargs)
        
        # EDOC currently requires JAX backend
        if backend.name != "jax":
            raise ValueError(
                f"EDOC solver requires JAX backend, got {backend.name}. "
                "Use backend=get_backend('jax') when creating the solver."
            )
        
        # Store configuration
        self.horizon = horizon
        self.dt = dt
        self.seed = seed
        self.config.update({
            "noise_std": noise_std,
            "action_space": action_space,
            "diffusion_mode": diffusion_mode,
            "action_diffuse_steps": action_diffuse_steps,
            "action_beta0": action_beta0,
            "action_betaT": action_betaT,
            "action_temp": action_temp,
            "action_extra_sigma": action_extra_sigma,
            "action_stage_ratio": action_stage_ratio,
            "action_score_mode": action_score_mode,
            "action_nsample": action_nsample,
            "use_antithetic": use_antithetic,
            "dyn_loss_coeff": dyn_loss_coeff,
            "dyn_loss_mode": dyn_loss_mode,
        })
        
        # Create adapters
        self._env_adapter = DynamicsToEnvAdapter(dynamics, dt)
        
        # Convert energy to legacy format if needed
        if isinstance(energy, LegacyEnergyFunctional):
            self._legacy_energy = energy
        else:
            self._legacy_energy = EnergyToLegacyAdapter(energy, dynamics).legacy_energy
        
        # Create EDOCPlanner (lazy initialization)
        self._edoc_planner = None
        self._state_box = state_box
    
    def _get_edoc_planner(self):
        """Get or create EDOCPlanner instance."""
        if self._edoc_planner is None:
            self._edoc_planner = EDOCPlanner(
                env=self._env_adapter,
                energy=self._legacy_energy,
                horizon=self.horizon,
                dt=self.dt,
                noise_std=self.config["noise_std"],
                state_box=self._state_box,
                action_space=self.config["action_space"],
                diffusion_mode=self.config["diffusion_mode"],
                action_diffuse_steps=self.config["action_diffuse_steps"],
                action_beta0=self.config["action_beta0"],
                action_betaT=self.config["action_betaT"],
                action_temp=self.config["action_temp"],
                action_extra_sigma=self.config["action_extra_sigma"],
                action_stage_ratio=self.config["action_stage_ratio"],
                action_score_mode=self.config["action_score_mode"],
                action_nsample=self.config["action_nsample"],
                use_antithetic=self.config["use_antithetic"],
                dyn_loss_coeff=self.config["dyn_loss_coeff"],
                dyn_loss_mode=self.config["dyn_loss_mode"],
                np_random_seed=self.seed,
            )
        return self._edoc_planner
    
    def sample_trajectories(
        self,
        x0: State,
        horizon: int,
        n_samples: int,
        **kwargs
    ) -> List[Trajectory]:
        """
        Sample candidate trajectories using EDOC diffusion process.
        
        This method uses EDOC's diffusion to generate multiple candidate trajectories.
        Note: EDOC typically optimizes directly, so this is mainly for visualization/debugging.
        
        Args:
            x0: Initial state
            horizon: Planning horizon
            n_samples: Number of trajectories to sample
            **kwargs: Additional sampling parameters
            
        Returns:
            List of candidate trajectories
        """
        planner = self._get_edoc_planner()
        rng = jax.random.PRNGKey(self.seed)
        
        # Extract state data
        x0_data = np.asarray(x0, dtype=np.float32) if not isinstance(x0, jnp.ndarray) else x0
        
        trajectories = []
        for i in range(n_samples):
            # Use different random keys for each sample
            rng, sample_key = jax.random.split(rng)
            
            # Set initial state in environment adapter
            self._env_adapter._current_state = x0_data
            
            # Run EDOC planning
            result = planner.plan(sample_key)
            
            # Convert result to Trajectory
            states = result.get("states", [])
            actions = result.get("actions", None)
            
            if actions is not None:
                # Convert to list of actions
                actions_list = [actions[i] for i in range(len(actions))]
            else:
                # Generate actions from state transitions
                actions_list = []
                for j in range(len(states) - 1):
                    # Infer action from state transition
                    # This is approximate
                    actions_list.append(np.zeros(self._env_adapter.act_dim, dtype=np.float32))
            
            # Convert states to list
            states_list = [states[i] for i in range(len(states))]
            
            traj = Trajectory(states=states_list, actions=actions_list)
            trajectories.append(traj)
        
        return trajectories
    
    def solve(
        self,
        x0: State,
        horizon: int,
        **kwargs
    ) -> Trajectory:
        """
        Solve for optimal trajectory using EDOC.
        
        This method uses EDOCPlanner's diffusion-based optimization to find
        the optimal trajectory.
        
        Args:
            x0: Initial state
            horizon: Planning horizon (overrides constructor value if different)
            **kwargs: Additional solver parameters (e.g., rng_key for random seed)
            
        Returns:
            Optimized trajectory
        """
        planner = self._get_edoc_planner()
        
        # Use provided horizon or default
        if horizon != self.horizon:
            # Recreate planner with new horizon
            self.horizon = horizon
            self._edoc_planner = None
            planner = self._get_edoc_planner()
        
        # Get random key
        rng_key = kwargs.get("rng_key", None)
        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.seed)
        
        # Extract state data
        x0_data = np.asarray(x0, dtype=np.float32) if not isinstance(x0, jnp.ndarray) else x0
        
        # Set initial state in environment adapter
        self._env_adapter._current_state = x0_data
        
        # Run EDOC planning
        result = planner.plan(rng_key)
        
        # Convert result to Trajectory
        states = result.get("states", [])
        actions = result.get("actions", None)
        
        if actions is not None:
            # Convert JAX array to list of actions
            if isinstance(actions, jnp.ndarray):
                actions_list = [np.asarray(actions[i], dtype=np.float32) for i in range(len(actions))]
            else:
                actions_list = [actions[i] for i in range(len(actions))]
        else:
            # Generate dummy actions if not available
            actions_list = [np.zeros(self._env_adapter.act_dim, dtype=np.float32) 
                          for _ in range(len(states) - 1)]
        
        # Convert states to list
        if isinstance(states, jnp.ndarray):
            states_list = [np.asarray(states[i], dtype=np.float32) for i in range(len(states))]
        else:
            states_list = [states[i] for i in range(len(states))]
        
        # Create trajectory
        traj = Trajectory(
            states=states_list,
            actions=actions_list,
            info=result
        )
        
        return traj


# ============================================================================
# Main Entry Point (for backward compatibility)
# ============================================================================

def run_edoc(args):
    """
    Run EDOC planner (main entry point for backward compatibility).
    
    Args:
        args: EDOC configuration arguments (EDOCArgs from configs)
        
    Returns:
        Dictionary with planning results
    """
    rng = jax.random.PRNGKey(args.seed)
    np_seed = args.np_random_seed if args.np_random_seed is not None else args.seed
    if np_seed is not None:
        np.random.seed(np_seed)

    # env & energy
    env = make_env(args.env_name)
    energy = make_energy(args.env_name)

    state_box = None
    if args.use_state_box:
        low = jnp.array([args.state_low, args.state_low], dtype=jnp.float32)
        high = jnp.array([args.state_high, args.state_high], dtype=jnp.float32)
        state_box = (low, high)

    planner = EDOCPlanner(
        env=env,
        energy=energy,
        horizon=args.horizon,
        dt=args.dt,
        noise_std=args.noise_std,
        state_box=state_box,
        action_space=args.action_space,
        diffusion_mode=args.diffusion_mode,
        action_diffuse_steps=args.action_diffuse_steps,
        action_beta0=args.action_beta0,
        action_betaT=args.action_betaT,
        action_temp=args.action_temp,
        action_extra_sigma=args.action_extra_sigma,
        action_stage_ratio=args.action_stage_ratio,
        action_score_mode=args.action_score_mode,
        action_nsample=args.action_nsample,
        use_antithetic=args.use_antithetic,
        dyn_loss_coeff=args.dyn_loss_coeff,
        dyn_loss_mode=args.dyn_loss_mode,
        np_random_seed=np_seed,
    )

    out = planner.plan(rng)

    if args.verbose:
        if not out:
            print("Planner returned no output.")
        else:
            energies = out.get("energies")
            if energies is not None:
                print("energies:", energies)
            print("initial state:", out.get("initial_state"))
            states = out.get("states")
            if states is not None:
                print("final state:", states[-1])
            rewards = out.get("rewards")
            if rewards is not None:
                print("rewards:", rewards)
                print("total reward:", float(jnp.sum(rewards)))
            if "actions" in out:
                print("actions:", out["actions"])
            if "energy_iterations" in out:
                print("energy Iter history:", out["energy_iterations"])
            if "reward_history" in out:
                print("reward history:", out["reward_history"])

    return out


# ============================================================================
# CLI (for backward compatibility)
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("EDOC Planner")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--np_random_seed", type=int, default=None)
    parser.add_argument("--env_name", type=str, default="double_integrator_box")
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--use_state_box", action="store_true", default=True)
    parser.add_argument("--state_low", type=float, default=-2.0)
    parser.add_argument("--state_high", type=float, default=2.0)
    parser.add_argument("--action_space", action="store_true", default=True)
    parser.add_argument("--diffusion_mode", type=str, default="reverse")
    parser.add_argument("--action_diffuse_steps", type=int, default=100)
    parser.add_argument("--action_beta0", type=float, default=1e-4)
    parser.add_argument("--action_betaT", type=float, default=1e-2)
    parser.add_argument("--action_temp", type=float, default=0.1)
    parser.add_argument("--action_extra_sigma", type=float, default=0.01)
    parser.add_argument("--action_stage_ratio", type=float, default=1.0)
    parser.add_argument("--action_score_mode", type=str, default="reward")
    parser.add_argument("--action_nsample", type=int, default=256)
    parser.add_argument("--no_antithetic", action="store_true", default=False, help="Disable antithetic pairing when sampling action trajectories.")
    parser.add_argument("--dyn_loss_coeff", type=float, default=0.0)
    parser.add_argument("--dyn_loss_mode", type=str, default="terminal", choices=["terminal", "trajectory"])
    parser.add_argument("--verbose", action="store_true", default=True)

    cli_args = parser.parse_args()
    # Import EDOCArgs from configs
    from configs.double_integrator_box.edoc import EDOCArgs
    args = EDOCArgs(
        seed=cli_args.seed,
        np_random_seed=cli_args.np_random_seed,
        env_name=cli_args.env_name,
        horizon=cli_args.horizon,
        dt=cli_args.dt,
        noise_std=cli_args.noise_std,
        use_state_box=cli_args.use_state_box,
        state_low=cli_args.state_low,
        state_high=cli_args.state_high,
        action_space=cli_args.action_space,
        diffusion_mode=cli_args.diffusion_mode,
        action_diffuse_steps=cli_args.action_diffuse_steps,
        action_beta0=cli_args.action_beta0,
        action_betaT=cli_args.action_betaT,
        action_temp=cli_args.action_temp,
        action_extra_sigma=cli_args.action_extra_sigma,
        action_stage_ratio=cli_args.action_stage_ratio,
        action_score_mode=cli_args.action_score_mode,
        action_nsample=cli_args.action_nsample,
        use_antithetic=not cli_args.no_antithetic,
        dyn_loss_coeff=cli_args.dyn_loss_coeff,
        dyn_loss_mode=cli_args.dyn_loss_mode,
        verbose=cli_args.verbose,
    )

    run_edoc(args)
