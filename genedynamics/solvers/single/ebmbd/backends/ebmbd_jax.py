"""
JAX backend for EB-MBD (Emerging-Barrier Model-Based Diffusion).

It mirrors the lightweight reverse-diffusion loop in `MBDSolver`, but adds
emerging barrier costs (log barrier with scheduled offset) for constraint
handling. The design intentionally avoids any Trajectory/Python work inside
the traced diffusion loop to keep performance close to the original EB-MBD
paper implementation.
"""

from typing import Any, Dict, Tuple
import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.solvers.single.diffusion_adaptors import diverse_topk_modes
from genedynamics.experiments.plugins.obstacles.d3il_avoiding_fixed import get_d3il_target_line_positions

try:
    from genedynamics.core.registry.edoc_backends import register_edoc_backend
except ImportError:
    # Fallback decorator if registry is unavailable
    def register_edoc_backend(name):
        def deco(cls):
            return cls
        return deco


@register_edoc_backend("ebmbd_jax")
class EBMBDBackendJax:
    """EB-MBD JAX backend."""

    @staticmethod
    def _flatten_obstacles(obs: Any) -> list:
        """
        Flatten obstacle manager / composite obstacles into a list of primitives.

        Supports:
        - ObstacleManager with `.obstacles`
        - UnionObstacle / composite obstacles with `.obstacles`
        - Single primitive obstacle objects
        """
        if obs is None:
            return []
        # ObstacleManager or composite obstacle
        if hasattr(obs, "obstacles"):
            flat = []
            try:
                for child in list(getattr(obs, "obstacles", [])):
                    flat.extend(EBMBDBackendJax._flatten_obstacles(child))
            except Exception:
                pass
            return flat
        # Primitive
        return [obs]

    def __init__(self, solver: Any):
        # Grab solver configs
        self.env = solver._env_adapter
        self.energy = solver._legacy_energy
        self.horizon = solver.horizon
        self.Nsample = solver.Nsample
        self.Ndiffuse = solver.Ndiffuse
        self.temp = solver.temp_sample
        self.beta0 = solver.beta0
        self.betaT = solver.betaT
        self.action_limit = solver.action_limit
        self.action_extra_sigma = getattr(solver, "action_extra_sigma", 0.0)
        self.mu = solver.mu
        self.alpha = solver.alpha
        self.bound = solver.bound
        self.use_min_over_time = solver.use_min_over_time
        self.terminal_energy_weight = float(getattr(solver, "terminal_energy_weight", 0.0))
        self.obstacles = getattr(solver, "_obstacles", None)
        self.obstacle_config = getattr(solver, "_obstacle_config", {}) or {}
        self.robot_radius = float(self.obstacle_config.get("robot_radius", 0.05))
        self.scheduler = getattr(solver, "scheduler", None)
        self.show_tqdm = bool(getattr(solver, "show_tqdm", False))
        # Multi-mode support: number of candidate trajectories to return
        self.num_modes = int(getattr(solver, "num_modes", 1))
        self.use_target_line = bool(getattr(solver, "use_target_line", False))
        self.num_targets = int(getattr(solver, "num_targets", 4))
        # Diversity selection parameters
        self.diversity_eta = float(getattr(solver, "diversity_eta", 1.0))
        self.diversity_topK_cand = int(getattr(solver, "diversity_topK_cand", None) or (self.Nsample // 2))
        self.diversity_use_state = bool(getattr(solver, "diversity_use_state", True))

        # Allow constraint scheduler to override barrier params (emerging_barrier)
        if self.scheduler is not None and hasattr(self.scheduler, "constraint_schedulers"):
            try:
                cs_list = getattr(self.scheduler, "constraint_schedulers", [])
                if cs_list:
                    cs = cs_list[0]
                    try:
                        from genedynamics.core.constraints.core.types import ScheduleState
                        params_cs = cs.constraint_params(ScheduleState(k=0, K=max(self.Ndiffuse - 1, 1))) or {}
                    except Exception:
                        params_cs = cs.constraint_params(None) if hasattr(cs, "constraint_params") else {}
                    self.mu = float(params_cs.get("mu", self.mu))
                    self.alpha = float(params_cs.get("alpha", self.alpha))
                    self.bound = float(params_cs.get("bound", self.bound))
                    self.use_min_over_time = bool(params_cs.get("use_min_over_time", self.use_min_over_time))
                    self.terminal_energy_weight = float(
                        params_cs.get("terminal_energy_weight", self.terminal_energy_weight)
                    )
                    # If constraint scheduler is non-JAX and claims adaptive update, block
                    if hasattr(cs, "update") and getattr(cs, "backend", None) != "jax":
                        raise NotImplementedError(
                            "Adaptive emerging barrier update with non-JAX backend is not supported in JAX ebmbd. "
                            "Provide a JAX-compatible adaptive scheduler or use fixed parameters."
                        )
            except Exception:
                pass

        # Allow diffusion scheduler to override M_k / T_k / betas / Ndiffuse
        if self.scheduler is not None and hasattr(self.scheduler, "diffusion_schedulers"):
            try:
                ds_list = getattr(self.scheduler, "diffusion_schedulers", [])
                if ds_list:
                    ds = ds_list[0]
                    from genedynamics.core.constraints.core.types import ScheduleState

                    params = ds.diffusion_params(ScheduleState(k=0, K=max(self.Ndiffuse - 1, 1))) or {}
                    if "M_k" in params:
                        self.Nsample = int(params["M_k"])
                    if "T_k" in params:
                        self.temp = float(params["T_k"])
                    if "Ndiffuse" in params:
                        self.Ndiffuse = int(params["Ndiffuse"])
                    if "beta0" in params:
                        self.beta0 = float(params["beta0"])
                    if "betaT" in params:
                        self.betaT = float(params["betaT"])
                    # If diffusion scheduler is adaptive but non-JAX, block on JAX backend
                    if hasattr(ds, "update") and getattr(ds, "backend", None) != "jax":
                        raise NotImplementedError(
                            "Adaptive diffusion scheduler update with non-JAX backend is not supported in JAX ebmbd. "
                            "Provide a JAX-compatible adaptive diffusion scheduler or use fixed parameters."
                        )
            except Exception:
                pass

        # JAX primitives
        if not hasattr(self.env, "jax_transition"):
            raise ValueError("Environment must expose `jax_transition` for EB-MBD JAX backend.")

        self._transition_fn = jax.jit(self.env.jax_transition)
        self._cost_fn = jax.jit(self._build_cost_fn())
        self._rollout_states_fn = jax.jit(self._build_rollout_states_fn())
        self._rollout_states_batch_fn = jax.jit(jax.vmap(self._rollout_states_fn, in_axes=(None, 0)))
        self._rollout_rewards_fn = jax.jit(self._build_rollout_rewards_fn())
        self._rollout_rewards_batch_fn = jax.jit(jax.vmap(self._rollout_rewards_fn, in_axes=(None, 0)))
        self._rollout_rewards_with_target_fn = jax.jit(self._build_rollout_rewards_with_target_fn())
        self._rollout_total_cost_fn = jax.jit(self._build_rollout_total_cost_fn())
        self._rollout_total_cost_batch_fn = jax.jit(jax.vmap(self._rollout_total_cost_fn, in_axes=(None, 0)))
        _rollout_total_cost_with_target_fn = self._build_rollout_total_cost_with_target_fn()
        self._rollout_total_cost_with_target_batch_fn = jax.jit(
            jax.vmap(_rollout_total_cost_with_target_fn, in_axes=(None, 0, None))
        )

        # SDF backends (box and/or obstacle SDF texture)
        self._box_sdf_fn = jax.jit(self.env.jax_sdf) if hasattr(self.env, "jax_sdf") else None
        self._obs_sdf_fn = None
        if self.obstacles is not None:
            # Preferred: exact geometric SDF via each primitive's jax_sdf, union = min
            prims = self._flatten_obstacles(self.obstacles)
            prims = [p for p in prims if hasattr(p, "jax_sdf")]
            if len(prims) > 0:
                def _obs_sdf_exact(points: jnp.ndarray) -> jnp.ndarray:
                    sdfs = [p.jax_sdf(points) for p in prims]
                    return jnp.min(jnp.stack(sdfs, axis=0), axis=0)
                self._obs_sdf_fn = jax.jit(_obs_sdf_exact)
            else:
                # Fallback: approximate SDF texture (fast, but can be inaccurate)
                tex = getattr(self.obstacles, "_sdf_texture_2d", None)
                if tex is not None:
                    def _obs_sdf_tex(points: jnp.ndarray) -> jnp.ndarray:
                        sdf, _ = tex.sample(points, backend="jax")
                        return sdf
                    self._obs_sdf_fn = jax.jit(_obs_sdf_tex)

        if self._box_sdf_fn is None and self._obs_sdf_fn is None:
            raise ValueError("Environment/obstacle must expose a JAX SDF: either env.jax_sdf or obstacles SDF texture.")

    # ------------------------------------------------------------------ #
    # Builders
    # ------------------------------------------------------------------ #
    def _build_cost_fn(self):
        def cost(state, action, ctx):
            return self.energy.compute(state, action, ctx)
        return cost

    def _build_rollout_states_fn(self):
        def rollout_states(state_init, actions):
            def step_fn(carry, action):
                next_state = self._transition_fn(carry, action)
                return next_state, next_state

            _, states = jax.lax.scan(step_fn, state_init, actions)
            return jnp.concatenate([state_init[None, :], states], axis=0)

        return rollout_states

    def _build_rollout_rewards_fn(self):
        def rollout_rewards(state_init, actions):
            t_idxs = jnp.arange(actions.shape[0], dtype=jnp.int32)

            def step_fn(carry, inp):
                action, t = inp
                next_state = self._transition_fn(carry, action)
                ctx = {"t": t}
                reward = -self._cost_fn(next_state, action, ctx)
                return next_state, reward

            _, rewards = jax.lax.scan(step_fn, state_init, (actions, t_idxs))
            return rewards

        return rollout_rewards

    def _build_rollout_total_cost_fn(self):
        """
        Paper-aligned total cost:
          - Stage costs for actions[:-1] (scan)
          - Last action only advances to terminal state (no stage cost)
          - Terminal cost on that terminal state, weighted by terminal_energy_weight
        """
        terminal_w = jnp.asarray(self.terminal_energy_weight, dtype=jnp.float32)
        act_dim = self.env.act_dim

        def rollout_total_cost(state_init, actions):
            H = actions.shape[0]

            def do_stage(_):
                def step_fn(carry, act):
                    nxt = self._transition_fn(carry, act)
                    c = self._cost_fn(nxt, act, {"t": 0})
                    return nxt, c

                terminal_state, costs = jax.lax.scan(step_fn, state_init, actions[:-1])
                total_stage = jnp.sum(costs)
                # advance once with last action for terminal state
                terminal_state = self._transition_fn(terminal_state, actions[-1])
                zero_u = jnp.zeros((act_dim,), dtype=jnp.float32)
                terminal_c = self._cost_fn(terminal_state, zero_u, {"t": H})
                total = total_stage + terminal_w * terminal_c
                # For logging: mean reward over stage steps (avoid div0)
                rews_mean = -total_stage / jnp.maximum(float(H - 1), 1.0)
                return total, rews_mean

            def only_terminal(_):
                terminal_state = self._transition_fn(state_init, actions[0])
                zero_u = jnp.zeros((act_dim,), dtype=jnp.float32)
                terminal_c = self._cost_fn(terminal_state, zero_u, {"t": 1})
                total = terminal_w * terminal_c
                rews_mean = jnp.asarray(0.0, dtype=jnp.float32)
                return total, rews_mean

            total, rews_mean = jax.lax.cond(H > 1, do_stage, only_terminal, operand=None)
            return total, rews_mean

        return rollout_total_cost

    def _build_rollout_total_cost_with_target_fn(self):
        """Total cost with per-mode target passed in ctx."""
        terminal_w = jnp.asarray(self.terminal_energy_weight, dtype=jnp.float32)
        act_dim = self.env.act_dim

        def rollout_total_cost_with_target(state_init, actions, target):
            target = jnp.asarray(target, dtype=jnp.float32).reshape(-1)[:2]
            ctx_base = {"target_xy": target}
            H = actions.shape[0]

            def do_stage(_):
                def step_fn(carry, act):
                    nxt = self._transition_fn(carry, act)
                    c = self._cost_fn(nxt, act, {**ctx_base, "t": 0})
                    return nxt, c

                terminal_state, costs = jax.lax.scan(step_fn, state_init, actions[:-1])
                total_stage = jnp.sum(costs)
                terminal_state = self._transition_fn(terminal_state, actions[-1])
                zero_u = jnp.zeros((act_dim,), dtype=jnp.float32)
                terminal_c = self._cost_fn(terminal_state, zero_u, {**ctx_base, "t": H})
                total = total_stage + terminal_w * terminal_c
                rews_mean = -total_stage / jnp.maximum(float(H - 1), 1.0)
                return total, rews_mean

            def only_terminal(_):
                terminal_state = self._transition_fn(state_init, actions[0])
                zero_u = jnp.zeros((act_dim,), dtype=jnp.float32)
                terminal_c = self._cost_fn(terminal_state, zero_u, {**ctx_base, "t": 1})
                total = terminal_w * terminal_c
                rews_mean = jnp.asarray(0.0, dtype=jnp.float32)
                return total, rews_mean

            return jax.lax.cond(H > 1, do_stage, only_terminal, operand=None)

        return rollout_total_cost_with_target

    def _build_rollout_rewards_with_target_fn(self):
        def rollout_rewards_with_target(state_init, actions, target):
            target = jnp.asarray(target, dtype=jnp.float32).reshape(-1)[:2]
            t_idxs = jnp.arange(actions.shape[0], dtype=jnp.int32)

            def step_fn(carry, inp):
                action, t = inp
                next_state = self._transition_fn(carry, action)
                ctx = {"t": t, "target_xy": target}
                reward = -self._cost_fn(next_state, action, ctx)
                return next_state, reward

            _, rewards = jax.lax.scan(step_fn, state_init, (actions, t_idxs))
            return rewards

        return rollout_rewards_with_target

    # ------------------------------------------------------------------ #
    # Core reverse diffusion
    # ------------------------------------------------------------------ #
    def reverse_diffuse_batch(self, rng_keys: jnp.ndarray, state_init: np.ndarray) -> list[Dict[str, Any]]:
        """
        Batch version of reverse_diffuse using jax.vmap for parallel execution.
        
        Args:
            rng_keys: (C,) array of PRNG keys
            state_init: initial state
            
        Returns:
            List of C result dictionaries (same format as reverse_diffuse)
        """
        # Prepare shared schedule arrays (same as reverse_diffuse)
        horizon = self.horizon
        act_dim = self.env.act_dim
        Ndiffuse = self.Ndiffuse
        
        try:
            from genedynamics.core.constraints.schedulers.utils import DiffusionNoiseSchedule
            ds = None
            if self.scheduler is not None and hasattr(self.scheduler, "diffusion_schedulers"):
                ds_list = getattr(self.scheduler, "diffusion_schedulers", [])
                if ds_list:
                    ds = ds_list[0]
            beta0 = self.beta0
            betaT = self.betaT
            Ndiffuse_override = Ndiffuse
            if ds is not None:
                from genedynamics.core.constraints.core.types import ScheduleState
                try:
                    params = ds.diffusion_params(ScheduleState(k=0, K=max(Ndiffuse - 1, 1))) or {}
                    beta0 = float(params.get("beta0", beta0))
                    betaT = float(params.get("betaT", betaT))
                    Ndiffuse_override = int(params.get("Ndiffuse", Ndiffuse_override))
                except Exception:
                    pass
            betas_np = np.linspace(beta0, betaT, Ndiffuse_override, dtype=np.float32)
            betas = jnp.asarray(DiffusionNoiseSchedule.from_betas(betas_np).betas, dtype=jnp.float32)
            Ndiffuse = int(betas.shape[0])
        except Exception:
            betas = jnp.linspace(self.beta0, self.betaT, Ndiffuse, dtype=np.float32)
        
        alphas = 1.0 - betas
        alphas_bar = jnp.cumprod(alphas)
        sigmas = jnp.sqrt(1.0 - alphas_bar)
        diffusion_indices = jnp.arange(Ndiffuse - 1, 0, -1, dtype=jnp.int32)
        
        denom = jnp.maximum(float(Ndiffuse - 1), 1.0)
        progress_inc_by_idx = 1.0 - (jnp.arange(Ndiffuse, dtype=jnp.float32) / denom)
        offset_by_idx = self.bound * (1.0 - (progress_inc_by_idx ** self.alpha))
        extra_sigmas_by_idx = self.action_extra_sigma * (1.0 - progress_inc_by_idx)
        
        x0_jnp = jnp.asarray(state_init, dtype=jnp.float32)
        temp_eps = jnp.maximum(self.temp, 1e-6)

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

        # Core diffusion function (pure JAX, can be vmapped)
        def reverse_diffuse_core(rng_key, target):
            rng, _ = jax.random.split(rng_key)
            Ybar0 = jnp.zeros((horizon, act_dim), dtype=jnp.float32)
            
            def step_one(carry, idx):
                rng_curr, Ybar_curr = carry
                rng_curr, eps_key, extra_key = jax.random.split(rng_curr, 3)
                
                sqrt_alpha_bar_i = jnp.sqrt(alphas_bar[idx])
                sigma_i = sigmas[idx]
                Yi = Ybar_curr * sqrt_alpha_bar_i
                
                eps = jax.random.normal(eps_key, (self.Nsample, horizon, act_dim), dtype=jnp.float32)
                Y0s = Ybar_curr[None, :] + sigma_i * eps
                limit = self.action_limit
                Y0s = jnp.clip(Y0s, -limit, limit)
                
                states_batch = self._rollout_states_batch_fn(x0_jnp, Y0s)
                total_stage_terminal_cost, rews_mean = self._rollout_total_cost_with_target_batch_fn(x0_jnp, Y0s, target)
                
                min_sdf = self._compute_min_sdf_batch(states_batch)
                offset = offset_by_idx[idx]
                z_raw = min_sdf + offset
                infeasible = z_raw <= 0.0
                z = jnp.maximum(z_raw, 1e-6)
                z = jnp.minimum(z, 1.0)
                barrier = jnp.where(infeasible, jnp.inf, -self.mu * jnp.log(z))
                
                total_cost = total_stage_terminal_cost + barrier
                scores = -total_cost
                scores = jnp.where(jnp.isfinite(scores), scores, -1e9)
                logw = scores / temp_eps
                logw = logw - jnp.max(logw)
                weights = jax.nn.softmax(logw)
                weights = jnp.where(
                    jnp.any(jnp.logical_not(jnp.isfinite(weights))),
                    jnp.ones_like(weights) / weights.size,
                    weights,
                )
                
                Ybar_weighted = jnp.tensordot(weights, Y0s, axes=([0], [0]))
                
                one_minus_alpha_bar = 1.0 - alphas_bar[idx]
                score_val = (-Yi + sqrt_alpha_bar_i * Ybar_weighted) / one_minus_alpha_bar
                Yim1 = (Yi + one_minus_alpha_bar * score_val) / jnp.sqrt(alphas[idx])
                sqrt_alpha_bar_prev = jnp.sqrt(alphas_bar[idx - 1])
                Ybar_next = Yim1 / sqrt_alpha_bar_prev
                
                extra_sigma = extra_sigmas_by_idx[idx]
                noise_extra = jax.random.normal(extra_key, (horizon, act_dim), dtype=jnp.float32)
                Ybar_next = Ybar_next + extra_sigma * noise_extra
                Ybar_next = jnp.clip(Ybar_next, -limit, limit)
                # reward_history: stage + terminal only (for convergence comparison across methods)
                reward_val = jnp.mean(-total_stage_terminal_cost)
                return (rng_curr, Ybar_next), (reward_val, Ybar_next, Y0s)
            
            (rng_out, Ybar_final), (reward_hist, Ybar_hist, Ysamples_hist) = jax.lax.scan(
                step_one, (rng, Ybar0), diffusion_indices
            )
            
            reward_hist = reward_hist[::-1]
            Ybar_hist = Ybar_hist[::-1]
            Ysamples_hist = Ysamples_hist[::-1]
            
            return Ybar_final, reward_hist, Ybar_hist, Ysamples_hist
        
        # Vmap the core function over (rng_keys, targets_per_mode)
        reverse_diffuse_batch_jit = jax.jit(jax.vmap(reverse_diffuse_core, in_axes=(0, 0)))
        Ybar_finals, reward_hists, Ybar_hists, Ysamples_hists = reverse_diffuse_batch_jit(rng_keys, targets_per_mode)
        
        # Batch post-processing: clip, rollout states and rewards (use per-mode target when use_target_line)
        final_actions_batch = jnp.clip(Ybar_finals, -self.action_limit, self.action_limit)  # (C, H, act_dim)
        states_batch = jax.vmap(self._rollout_states_fn, in_axes=(None, 0))(x0_jnp, final_actions_batch)  # (C, H+1, state_dim)
        rewards_batch = jax.vmap(self._rollout_rewards_with_target_fn, in_axes=(None, 0, 0))(x0_jnp, final_actions_batch, targets_per_mode)  # (C, H)
        
        # Convert to numpy
        states_batch_np = np.asarray(states_batch)  # (C, H+1, state_dim)
        actions_batch_np = np.asarray(final_actions_batch)  # (C, H, act_dim)
        rewards_batch_np = np.asarray(rewards_batch)  # (C, H)
        energies_batch_np = -rewards_batch_np  # (C, H)
        
        # Batch compute costs
        total_costs = np.sum(energies_batch_np, axis=-1)  # (C,)
        
        # Build results list (only this loop remains, all computation is batched)
        C = rng_keys.shape[0]
        results = []
        for i in range(C):
            results.append({
                "states": states_batch_np[i],
                "actions": actions_batch_np[i],
                "rewards": rewards_batch_np[i],
                "energies": energies_batch_np[i],
                "reward_history": np.asarray(reward_hists[i]),
                "diffusion_actions_traj": np.asarray(Ybar_hists[i]),
                "diffusion_sampled_actions": np.asarray(Ysamples_hists[i]),
                "scheduler_params_history": [],
                "rng": rng_keys[i],
                "candidate_states": [states_batch_np[i]],
                "candidate_actions": [actions_batch_np[i]],
                "candidate_costs": np.asarray([float(total_costs[i])], dtype=np.float32),
                "best_idx": 0,
            })
        
        return results
    
    def reverse_diffuse(self, rng_key: Any, state_init: np.ndarray) -> Dict[str, Any]:
        horizon = self.horizon
        act_dim = self.env.act_dim
        Ndiffuse = self.Ndiffuse

        # Diffusion schedule (allow scheduler override)
        try:
            from genedynamics.core.constraints.schedulers.utils import DiffusionNoiseSchedule
            ds = None
            if self.scheduler is not None and hasattr(self.scheduler, "diffusion_schedulers"):
                ds_list = getattr(self.scheduler, "diffusion_schedulers", [])
                if ds_list:
                    ds = ds_list[0]
            beta0 = self.beta0
            betaT = self.betaT
            Ndiffuse_override = Ndiffuse
            if ds is not None:
                from genedynamics.core.constraints.core.types import ScheduleState

                try:
                    params = ds.diffusion_params(ScheduleState(k=0, K=max(Ndiffuse - 1, 1))) or {}
                    beta0 = float(params.get("beta0", beta0))
                    betaT = float(params.get("betaT", betaT))
                    Ndiffuse_override = int(params.get("Ndiffuse", Ndiffuse_override))
                except Exception:
                    pass
            betas_np = np.linspace(beta0, betaT, Ndiffuse_override, dtype=np.float32)
            betas = jnp.asarray(DiffusionNoiseSchedule.from_betas(betas_np).betas, dtype=jnp.float32)
            Ndiffuse = int(betas.shape[0])
        except Exception:
            betas = jnp.linspace(self.beta0, self.betaT, Ndiffuse, dtype=jnp.float32)

        # Compute schedule arrays (shared for both branches)
        alphas = 1.0 - betas
        alphas_bar = jnp.cumprod(alphas)
        sigmas = jnp.sqrt(1.0 - alphas_bar)
        diffusion_indices = jnp.arange(Ndiffuse - 1, 0, -1, dtype=jnp.int32)

        # Schedules must be aligned with diffusion index `idx`:
        # - idx = Ndiffuse-1: initial (most noisy)
        # - idx = 0:          final (least noisy)
        # We want:
        # - emerging barrier offset: start loose (bound) then tighten -> 0
        # - extra action noise: start high then decay -> 0
        denom = jnp.maximum(float(Ndiffuse - 1), 1.0)
        progress_inc_by_idx = 1.0 - (jnp.arange(Ndiffuse, dtype=jnp.float32) / denom)  # idx high -> 0, idx low -> 1
        offset_by_idx = self.bound * (1.0 - (progress_inc_by_idx ** self.alpha))        # idx high -> bound, idx low -> 0
        extra_sigmas_by_idx = self.action_extra_sigma * (1.0 - progress_inc_by_idx)     # idx high -> action_extra_sigma, idx low -> 0

        x0_jnp = jnp.asarray(state_init, dtype=jnp.float32)

        # Initialize Ybar
        rng, _ = jax.random.split(rng_key)
        # Match MBD/EDOC convention: start from 0-mean (noise is injected by diffusion schedule)
        Ybar0 = jnp.zeros((horizon, act_dim), dtype=jnp.float32)

        temp_eps = jnp.maximum(self.temp, 1e-6)

        def step_one(carry, idx):
            rng_curr, Ybar_curr = carry
            rng_curr, eps_key, extra_key = jax.random.split(rng_curr, 3)

            sqrt_alpha_bar_i = jnp.sqrt(alphas_bar[idx])
            sigma_i = sigmas[idx]
            Yi = Ybar_curr * sqrt_alpha_bar_i

            # Sample actions
            eps = jax.random.normal(eps_key, (self.Nsample, horizon, act_dim), dtype=jnp.float32)
            Y0s = Ybar_curr[None, :] + sigma_i * eps

            # Clip actions
            limit = self.action_limit
            Y0s = jnp.clip(Y0s, -limit, limit)

            # Rollout states & rewards
            states_batch = self._rollout_states_batch_fn(x0_jnp, Y0s)  # (M,H+1,state_dim)
            rewards = self._rollout_rewards_batch_fn(x0_jnp, Y0s)      # (M,H)
            # Paper-aligned total cost and logging reward (stage-only mean)
            total_stage_terminal_cost, rews_mean = self._rollout_total_cost_batch_fn(x0_jnp, Y0s)  # (M,), (M,)

            # Barrier cost
            min_sdf = self._compute_min_sdf_batch(states_batch)        # (M,)
            offset = offset_by_idx[idx]
            z_raw = min_sdf + offset
            infeasible = z_raw <= 0.0
            # IMPORTANT (EDOC alignment): do NOT "reward" large clearance.
            # Plain -mu*log(z) becomes negative when z>1 and can dominate task reward,
            # pushing trajectories away from target just to maximize clearance.
            # We cap z above at 1.0 so barrier is always non-negative and only
            # activates near the constraint boundary.
            z = jnp.maximum(z_raw, 1e-6)
            z = jnp.minimum(z, 1.0)
            barrier = jnp.where(infeasible, jnp.inf, -self.mu * jnp.log(z))

            # Total cost = (sum stage + terminal_weight * terminal) + barrier
            total_cost = total_stage_terminal_cost + barrier
            # Direct softmax on negative cost (EDOC-like energy mode): logw = -cost / temp
            scores = -total_cost
            scores = jnp.where(jnp.isfinite(scores), scores, -1e9)
            logw = scores / temp_eps
            logw = logw - jnp.max(logw)
            weights = jax.nn.softmax(logw)
            weights = jnp.where(
                jnp.any(jnp.logical_not(jnp.isfinite(weights))),
                jnp.ones_like(weights) / weights.size,
                weights,
            )

            # Weighted average
            Ybar_weighted = jnp.tensordot(weights, Y0s, axes=([0], [0]))

            # Reverse update
            one_minus_alpha_bar = 1.0 - alphas_bar[idx]
            score_val = (-Yi + sqrt_alpha_bar_i * Ybar_weighted) / one_minus_alpha_bar
            Yim1 = (Yi + one_minus_alpha_bar * score_val) / jnp.sqrt(alphas[idx])
            sqrt_alpha_bar_prev = jnp.sqrt(alphas_bar[idx - 1])
            Ybar_next = Yim1 / sqrt_alpha_bar_prev

            # Add extra noise (decays to 0)
            extra_sigma = extra_sigmas_by_idx[idx]
            noise_extra = jax.random.normal(extra_key, (horizon, act_dim), dtype=jnp.float32)
            Ybar_next = Ybar_next + extra_sigma * noise_extra

            # Clip
            Ybar_next = jnp.clip(Ybar_next, -limit, limit)

            # reward_history: stage + terminal only (for convergence comparison across methods)
            reward_val = jnp.mean(-total_stage_terminal_cost)
            return (rng_curr, Ybar_next), (reward_val, Ybar_next, Y0s)

        (rng_out, Ybar_final), (reward_hist, Ybar_hist, Ysamples_hist) = jax.lax.scan(
            step_one, (rng, Ybar0), diffusion_indices
        )

        # Match visualization convention: index 0 = final (least noisy), largest index = initial (most noisy)
        reward_hist = reward_hist[::-1]
        Ybar_hist = Ybar_hist[::-1]
        Ysamples_hist = Ysamples_hist[::-1]

        # Final rollout
        final_states = self._rollout_states_fn(x0_jnp, Ybar_final)
        rewards_final = self._rollout_rewards_fn(x0_jnp, Ybar_final)  # (H,)
        energies_final = -rewards_final  # per-step stage energy = cost
        total_cost_final = np.sum(energies_final)

        # Multi-mode: collect candidate trajectories from final diffusion step
        candidate_states_list = []
        candidate_actions_list = []
        candidate_costs_list = []
        
        if self.num_modes > 1 and len(Ysamples_hist) > 0:
            # Strategy: sample from multiple diffusion steps to get diverse candidates
            # Use steps with moderate noise (not too early, not too late)
            num_steps_to_sample = min(3, len(Ysamples_hist))
            step_indices = np.linspace(0, len(Ysamples_hist) - 1, num_steps_to_sample, dtype=int)
            
            all_samples_list = []
            all_states_list = []
            all_costs_list = []
            
            for step_idx in step_indices:
                step_samples = Ysamples_hist[step_idx]  # (M, H, act_dim)
                M = step_samples.shape[0]
                
                # Rollout all samples to get costs
                states_step = self._rollout_states_batch_fn(x0_jnp, step_samples)  # (M, H+1, state_dim)
                rewards_step = self._rollout_rewards_batch_fn(x0_jnp, step_samples)  # (M, H)
                total_stage_terminal_cost_step, _ = self._rollout_total_cost_batch_fn(x0_jnp, step_samples)  # (M,)
                
                # Compute barrier costs for all samples
                min_sdf_step = self._compute_min_sdf_batch(states_step)  # (M,)
                # Use appropriate offset for this step (approximate, since we're sampling from history)
                # For earlier steps, use larger offset; for later steps, use smaller offset
                step_progress = float(step_idx) / max(len(Ysamples_hist) - 1, 1)
                # Reverse progress: step_idx=0 (final) -> progress=0, step_idx=last -> progress=1
                step_progress_rev = 1.0 - step_progress
                offset_step = self.bound * (1.0 - (step_progress_rev ** self.alpha))
                z_raw_step = min_sdf_step + offset_step
                infeasible_step = z_raw_step <= 0.0
                z_step = jnp.maximum(z_raw_step, 1e-6)
                z_step = jnp.minimum(z_step, 1.0)
                barrier_step = jnp.where(infeasible_step, jnp.inf, -self.mu * jnp.log(z_step))
                
                # Total cost for each sample
                total_costs_step = total_stage_terminal_cost_step + barrier_step  # (M,)
                
                all_samples_list.append(step_samples)
                all_states_list.append(states_step)
                all_costs_list.append(np.asarray(total_costs_step, dtype=np.float32))
            
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
            candidate_states_list = [np.asarray(final_states, dtype=np.float32)]
            candidate_actions_list = [np.asarray(Ybar_final, dtype=np.float32)]
            candidate_costs_list = [float(total_cost_final)]
            best_idx = 0

        return {
            "states": np.asarray(final_states, dtype=np.float32),  # Best trajectory (for backward compatibility)
            "actions": np.asarray(Ybar_final, dtype=np.float32),
            "rewards": np.asarray(rewards_final, dtype=np.float32),
            "energies": np.asarray(energies_final, dtype=np.float32),
            "reward_history": np.asarray(reward_hist, dtype=np.float32),
            "diffusion_actions_traj": np.asarray(Ybar_hist, dtype=np.float32),
            "diffusion_sampled_actions": np.asarray(Ysamples_hist, dtype=np.float32),
            "scheduler_params_history": [],
            "rng": rng_out,
            # Multi-mode candidates
            "candidate_states": candidate_states_list,
            "candidate_actions": candidate_actions_list,
            "candidate_costs": np.asarray(candidate_costs_list, dtype=np.float32),
            "best_idx": best_idx,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _compute_min_sdf_batch(self, states_batch: jnp.ndarray) -> jnp.ndarray:
        """
        states_batch: (M, H+1, state_dim)
        returns (M,) min sdf over time or terminal sdf
        """
        positions = states_batch[..., :2]  # assume first 2 dims are position
        sdf_terms = []
        if self._box_sdf_fn is not None:
            sdf_terms.append(self._box_sdf_fn(positions))  # (M, H+1)
        if self._obs_sdf_fn is not None:
            # obstacle SDF is positive outside obstacle; keep a robot-radius margin
            sdf_terms.append(self._obs_sdf_fn(positions) - self.robot_radius)  # (M, H+1)
        sdf_vals = sdf_terms[0] if len(sdf_terms) == 1 else jnp.min(jnp.stack(sdf_terms, axis=0), axis=0)
        if self.use_min_over_time:
            return jnp.min(sdf_vals, axis=-1)
        return sdf_vals[:, -1]

