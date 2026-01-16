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

try:
    from enerdynamics.core.registry.edoc_backends import register_edoc_backend
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

        # Allow constraint scheduler to override barrier params (emerging_barrier)
        if self.scheduler is not None and hasattr(self.scheduler, "constraint_schedulers"):
            try:
                cs_list = getattr(self.scheduler, "constraint_schedulers", [])
                if cs_list:
                    cs = cs_list[0]
                    try:
                        from enerdynamics.core.constraints.core.types import ScheduleState
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
                    from enerdynamics.core.constraints.core.types import ScheduleState

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
        self._rollout_total_cost_fn = jax.jit(self._build_rollout_total_cost_fn())
        self._rollout_total_cost_batch_fn = jax.jit(jax.vmap(self._rollout_total_cost_fn, in_axes=(None, 0)))

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

    # ------------------------------------------------------------------ #
    # Core reverse diffusion
    # ------------------------------------------------------------------ #
    def reverse_diffuse(self, rng_key: Any, state_init: np.ndarray) -> Dict[str, Any]:
        horizon = self.horizon
        act_dim = self.env.act_dim
        Ndiffuse = self.Ndiffuse

        # Diffusion schedule (allow scheduler override)
        try:
            from enerdynamics.core.constraints.schedulers.utils import DiffusionNoiseSchedule
            ds = None
            if self.scheduler is not None and hasattr(self.scheduler, "diffusion_schedulers"):
                ds_list = getattr(self.scheduler, "diffusion_schedulers", [])
                if ds_list:
                    ds = ds_list[0]
            beta0 = self.beta0
            betaT = self.betaT
            Ndiffuse_override = Ndiffuse
            if ds is not None:
                from enerdynamics.core.constraints.core.types import ScheduleState

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

            # For visualization, keep a per-step-average reward scalar (easier to compare across horizons)
            reward_val = jnp.mean(rews_mean)
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

        return {
            "states": np.asarray(final_states, dtype=np.float32),
            "actions": np.asarray(Ybar_final, dtype=np.float32),
            "rewards": np.asarray(rewards_final, dtype=np.float32),
            "energies": np.asarray(energies_final, dtype=np.float32),
            "reward_history": np.asarray(reward_hist, dtype=np.float32),
            "diffusion_actions_traj": np.asarray(Ybar_hist, dtype=np.float32),
            "diffusion_sampled_actions": np.asarray(Ysamples_hist, dtype=np.float32),
            "scheduler_params_history": [],
            "rng": rng_out,
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

