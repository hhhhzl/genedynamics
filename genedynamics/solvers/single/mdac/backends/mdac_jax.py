"""MDAC JAX backend — high-performance parallel reverse-diffusion kernel.

Written in the SAME high-performance pattern as `mdoc_jax` / `twogo_jax` /
`cfsmbd_jax`: a `jax.jit`-ed `jax.lax.scan` reverse-diffuse loop, with the
candidate rollout `jax.vmap`-ed over the `Nsample` axis, per-step schedule values
precomputed, and the MDAC upgrades injected through the SAME optional seams those
solvers use — NOT hand-rolled math:

  * `noise_sampler`       — `core/prob` seam (candidate noise);     None => N(0,1)
  * `transport`           — `transport/backends` DDPM/DDIM/FM/Adaptive seam;
                            None => VERBATIM weighted mean (== DIAL)
  * augmented-Lagrangian  — the cfsmbd / mdcoas AL pattern (`aug_lambda/aug_rho` +
                            the `[g]_+` penalty folded into the rollout reward,
                            cfsmbd `rollout_augmented_reward_and_v`), read from the
                            env's `constraint_residual` hook; empty residual => plain
  * `constraint_filter`   — `core/constraints` action-space ConstraintFilter (the
                            mdoc mechanism — NOT where iALM lives); None/NoOp => off
  * geometry              — `genemetry` `SdfManifold.geometry/project` (metric +
                            tangent projection) + `CfsRetraction`, scheduled by
                            `ScheduleOverlay` (like twogo); inactive when the env
                            exposes no constraint-geometry fn

NB on attribution: soft-feasibility / iALM is the **cfsmbd (mdcoas)** mechanism
(augmented reward), reused here on the brax substrate; **mdoc**'s constraint
handling is the separate action-space `ConstraintFilter` (the `constraint_filter`
seam). Earlier docs that said "iALM like mdoc" were wrong.

The rollout substrate is DIAL's (brax `PipelineEnv` via the shared
`solvers/common/env_rollout`, node-spline parametrisation) because MDAC is
DIAL-derived. With every seam off / NoOp / None, the `body` reduces *operation
for operation* to `dial_jax.reverse_once(update_form='weighted_mean')`, so MDAC
is byte-identical to DIAL on unconstrained tasks (the regression gate).

The ONE genuinely-new primitive (position-stiffness SPD `K=exp(S)`) lives
UPSTREAM in `genedynamics/core/control/stiffness.py`; this backend only consumes
it (via `action_size`). Everything else is reuse.

A `rollout_fn` may be injected for CPU tests (no mjx), exactly as `dial_jax` does.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.solvers.single.dial.backends.dial_jax import (
    make_sigma_control,
    make_traj_diffuse_factors,
)
from genedynamics.solvers.single.dial.spline import NodeSpline

RolloutFn = Callable[[Any, jnp.ndarray, Any], jnp.ndarray]  # (state, us, t0)->rews


class MdacBackendJax:
    """JAX backend: parallel reverse-diffusion kernel + WarmStartPlanner."""

    def __init__(
        self,
        solver: Any = None,
        *,
        nu: Optional[int] = None,
        rollout_fn: Optional[RolloutFn] = None,
        step_fn: Optional[Callable[[Any, Any], Any]] = None,
        Hnode: int = 4,
        Hsample: int = 16,
        Nsample: int = 2048,
        temp_sample: float = 0.06,
        horizon_diffuse_factor: float = 0.9,
        traj_diffuse_factor: float = 0.5,
        sigma_scale: float = 1.0,
        action_limit: float = 1.0,
        ctrl_dt: float = 0.02,
        Ndiffuse: int = 2,
        Ndiffuse_init: int = 10,
        beta0: float = 1e-4,
        betaT: float = 1e-2,
        aug_lambda: float = 0.0,
        aug_rho: float = 1.0,
        use_soft_feasibility: bool = True,
        seed: int = 0,
        # --- optional seams (None => verbatim DIAL) ---
        noise_sampler: Any = None,
        transport: Any = None,
        constraint_filter: Any = None,
        obstacles: Any = None,
        geometry_gate_fn: Any = None,
    ) -> None:
        cfg = {}
        if solver is not None:
            cfg = solver.config
            nu = nu if nu is not None else int(solver.nu)
            Hnode = int(cfg.get("Hnode", Hnode))
            Hsample = int(cfg.get("Hsample", Hsample))
            Nsample = int(cfg.get("Nsample", Nsample))
            temp_sample = float(cfg.get("temp_sample", temp_sample))
            horizon_diffuse_factor = float(cfg.get("horizon_diffuse_factor", horizon_diffuse_factor))
            traj_diffuse_factor = float(cfg.get("traj_diffuse_factor", traj_diffuse_factor))
            sigma_scale = float(cfg.get("sigma_scale", sigma_scale))
            action_limit = float(cfg.get("action_limit", action_limit))
            ctrl_dt = float(cfg.get("ctrl_dt", ctrl_dt))
            Ndiffuse = int(cfg.get("Ndiffuse", Ndiffuse))
            Ndiffuse_init = int(cfg.get("Ndiffuse_init", Ndiffuse_init))
            beta0 = float(cfg.get("beta0", beta0))
            betaT = float(cfg.get("betaT", betaT))
            aug_lambda = float(cfg.get("aug_lambda", aug_lambda))
            aug_rho = float(cfg.get("aug_rho", aug_rho))
            flags = getattr(solver, "flags", None)
            use_soft_feasibility = bool(getattr(flags, "use_soft_feasibility", use_soft_feasibility))
            seed = int(getattr(solver, "seed", seed))
            rollout_fn = rollout_fn or getattr(solver, "_rollout_fn", None)
            step_fn = step_fn or getattr(solver, "_step_fn", None)
            noise_sampler = noise_sampler if noise_sampler is not None else getattr(solver, "noise_sampler", None)
            transport = transport if transport is not None else getattr(solver, "transport", None)
            constraint_filter = constraint_filter if constraint_filter is not None else getattr(solver, "constraint_filter", None)
            obstacles = obstacles if obstacles is not None else getattr(solver, "obstacles", None)
            geometry_gate_fn = (
                geometry_gate_fn
                if geometry_gate_fn is not None
                else getattr(solver, "geometry_gate_fn", None)
            )

        if nu is None:
            raise ValueError("MdacBackendJax requires nu (action dim).")

        self.nu = int(nu)
        self.Hnode, self.Hsample = int(Hnode), int(Hsample)
        self.Nsample = int(Nsample)
        self.temp_sample = float(temp_sample)
        self.traj_diffuse_factor = float(traj_diffuse_factor)
        self.action_limit = float(action_limit)
        self.Ndiffuse = int(Ndiffuse)
        self.Ndiffuse_init = int(Ndiffuse_init)
        self.beta0, self.betaT = float(beta0), float(betaT)
        self.aug_lambda, self.aug_rho = float(aug_lambda), float(aug_rho)
        self.use_soft_feasibility = bool(use_soft_feasibility)
        self.seed = int(seed)

        self.spline = NodeSpline.build(self.Hnode, self.Hsample, ctrl_dt)
        self.N2U = self.spline.N2U
        self.sigma_control = make_sigma_control(self.Hnode, horizon_diffuse_factor, sigma_scale)

        # --- seams (None / NoOp => verbatim DIAL) ---
        self.noise_sampler = noise_sampler
        self.transport = transport
        from genedynamics.core.constraints.action_filters import NoOpConstraintFilter
        self.constraint_filter = constraint_filter or NoOpConstraintFilter()
        self.obstacles = obstacles
        self._env = getattr(solver, "dynamics", None)

        # --- geometry seam (genemetry reuse, like 2GO): active only when the env
        # / solver supplies a constraint-geometry fn `(state, Ybar_nodes, t0) ->
        # a_geom (Hnode+1, nu)` (analogous to 2GO's _constraint_geometry_time_jit).
        # Absent => the manifold path is skipped and MDAC == DIAL. The
        # metric tangent projection + retraction are REUSED from genemetry
        # (SdfManifold / CfsRetraction), never hand-rolled.
        self.geometry_fn = getattr(solver, "geometry_fn", None) if solver is not None else None
        self.manifold = None
        if self.geometry_fn is not None:
            cfg = getattr(solver, "config", {}) if solver is not None else {}
            # node space has only Hnode+1 rows; top_k needs k <= rows.
            self.topk_active = min(int(cfg.get("mdac_topk_active", 8)), self.Hnode + 1)
            self.eps_stab = float(cfg.get("mdac_eps_stab", 1e-6))
            self.geom_gain = float(cfg.get("mdac_geom_gain", 1.0))
            from genedynamics.genemetry.manifold.sdf import SdfManifold
            self.manifold = SdfManifold(backend="jax")
        # Retraction is an independent seam. In particular, mdac_no_tangent keeps
        # the CLEAN-manifold retraction active, making it a true one-flag ablation.
        self.retraction = (
            getattr(solver, "retraction", None) if solver is not None else None
        )
        # Optional task-owned realization gate. It returns an action-space vector
        # plus named component diagnostics; this backend remains task-agnostic.
        self.geometry_gate_fn = geometry_gate_fn
        self.prepare_state_fn = (
            getattr(solver, "prepare_state_fn", None)
            if solver is not None else None
        )
        self._prepare_state_jit = (
            jax.jit(self.prepare_state_fn)
            if self.prepare_state_fn is not None else None
        )

        # --- prior seam (genedynamics/learning/priors): horizon proposal,
        # incumbent candidate, local trust region, and do-no-harm acceptance.
        # None => every branch below is skipped and DIAL parity is preserved.
        self.prior = getattr(solver, "prior", None) if solver is not None else None
        self.prior_lambda_shift = (
            float(getattr(solver, "prior_lambda_shift", 0.5)) if solver is not None else 0.5
        )
        self.risk_fn = getattr(solver, "risk_fn", None) if solver is not None else None
        self.prior_include_incumbent = bool(
            getattr(solver, "prior_include_incumbent", True)
        )
        self.prior_trust_radius = float(
            getattr(solver, "prior_trust_radius", 0.5)
        )
        self.prior_improvement_epsilon = float(
            getattr(solver, "prior_improvement_epsilon", 0.0)
        )
        self.prior_risk_tolerance = jnp.asarray(
            getattr(solver, "prior_risk_tolerance", (0.0, 0.0, 0.0, 0.0)),
            dtype=jnp.float32,
        )
        self.prior_acceptance = bool(
            getattr(solver, "prior_acceptance", True)
        )
        flags = getattr(solver, "flags", None)
        self.use_rl_prior = bool(getattr(flags, "use_rl_prior", True))
        # coupled annealing (sigma_k down, rho_k up, kappa_k up). Default OFF (no
        # flags / mock => byte-identical DIAL); "mdac" method => flags turn it on.
        self.use_adaptive_schedule = bool(getattr(flags, "use_adaptive_schedule", False))

        # rollout / step: injected (CPU tests) or built from the env (brax/mjx).
        # _augmented is set True only when the AL-augmented brax rollout is built.
        self._augmented = False
        self._rollout_fn = rollout_fn or (self._build_rollout_from_solver(solver) if solver else None)
        self._step_fn = step_fn or self._build_step_from_solver(solver)
        if self._rollout_fn is None:
            raise ValueError("MdacBackendJax needs a rollout_fn (inject one or construct via a solver).")

        # transport needs a diffusion alphabar schedule; DIAL (transport=None) doesn't.
        betas = jnp.linspace(self.beta0, self.betaT, max(2, self.Ndiffuse_init), dtype=jnp.float32)
        self._alphas = 1.0 - betas
        self._alphas_bar = jnp.cumprod(self._alphas)

        # --- adaptive schedule overlay (genemetry, built EXACTLY like 2GO,
        # twogo_jax.py:130-134). Per reverse step we feed the overlay a (margin,
        # rho) read off the CURRENT iterate's constraint state and take kappa to
        # modulate the geometry — same as 2GO's `overlay.compute(margin, rho_k,
        # eta).kappa`. The sampling sigma stays the base DIAL schedule and aug_rho
        # stays constant (2GO does NOT route the schedule into either: it samples
        # with a separate `sigmas[idx]` and keeps `aug_rho_const`). Because margin/
        # rho track the runtime feasibility (which rises and falls), the schedule
        # is NON-MONOTONIC across reverse steps. All 1.0 when off => byte-identical.
        _cfg = getattr(solver, "config", {}) if solver is not None else {}
        self._constraint_fn = getattr(self._env, "manifold_residual", None)
        self._overlay = None
        self._kappa_ref = 1.0
        if self.use_adaptive_schedule:
            from genedynamics.genemetry.schedule.overlay import ScheduleOverlay
            from genedynamics.genemetry.schedule.config import resolve_overlay_config
            self._overlay_cfg = resolve_overlay_config(
                _cfg.get("mdac_overlay", None), rho_ref_default=max(self.aug_rho, 1.0))
            self._overlay = ScheduleOverlay(config=self._overlay_cfg, backend="jax")
            kap_ref, _ = self._overlay.constraint_overlay(
                jnp.asarray(0.0, jnp.float32), jnp.asarray(self.aug_rho, jnp.float32))
            self._kappa_ref = float(np.asarray(kap_ref))

        # Receding execution calls replan at every real step. Jitting a stable
        # bound function here avoids creating a fresh lax.scan executable for
        # every call; only the init/steady schedule shapes compile separately.
        self._replan_scan_jit = jax.jit(self._replan_scan)

    @property
    def _prior_active(self):
        """Static-at-trace switch; also supports prior injection in unit tests."""
        return self.prior is not None and self.use_rl_prior

    def _kappa_mult(self, state, Ybar_curr):
        """Geometry kappa multiplier for this reverse step (2GO overlay usage).
        margin/rho are read off the CURRENT iterate's clean-state constraint
        (no mjx): rho rises with the current infeasibility, so via the genemetry
        `constraint_overlay(margin, rho) -> kappa` the geometry tracking tightens
        when infeasible and relaxes when feasible — NON-MONOTONIC across steps.
        Returns 1.0 (no-op) when adaptive is off or the env has no constraint."""
        if self._overlay is None or self._constraint_fn is None:
            return jnp.float32(1.0)
        c = self._constraint_fn(state, Ybar_curr)             # clean-state residual (no mjx)
        if c.size == 0:
            return jnp.float32(1.0)
        viol = jnp.mean(jnp.abs(c))                           # current infeasibility (>=0)
        margin = -viol                                        # signed feasibility margin
        rho = self.aug_rho * (1.0 + viol)                     # AL hardness rises with violation
        kappa, _ = self._overlay.constraint_overlay(margin, rho)
        return kappa / (self._kappa_ref + 1e-9)

    # --- rollout / step builders (reuse the shared brax layer; mjx-gated) -----
    def _build_rollout_from_solver(self, solver: Any) -> Optional[RolloutFn]:
        from genedynamics.solvers.common.env_rollout import (
            is_brax_env,
            build_brax_rollout,
            build_brax_rollout_augmented,
        )
        env = getattr(solver, "dynamics", None)
        if is_brax_env(env):
            # cfsmbd/mdcoas AL: augment the brax reward with the soft-feasibility
            # penalty from env.constraint_residual. The no-op default residual =>
            # 0 penalty => byte-identical to the plain rollout (== DIAL).
            if self.use_soft_feasibility:
                self._augmented = True                       # rollout takes aug params at call time
                return build_brax_rollout_augmented(env)
            return build_brax_rollout(env)
        adapter = getattr(solver, "_env_adapter", None)
        energy = getattr(solver, "_legacy_energy", None)
        if adapter is None or energy is None:
            return None
        trans = adapter.jax_transition

        def rollout_one(x0, us, t0):
            ts = jnp.asarray(t0, jnp.float32) + jnp.arange(us.shape[0], dtype=jnp.float32)

            def f(s, u_t):
                u, t = u_t
                s2 = trans(s, u)
                return s2, -energy.compute(s2, u, {"t": t})

            _, rews = jax.lax.scan(f, x0, (us, ts))
            return rews

        return jax.jit(jax.vmap(rollout_one, in_axes=(None, 0, None)))

    def _build_step_from_solver(self, solver: Any):
        if solver is None:
            return None
        from genedynamics.solvers.common.env_rollout import is_brax_env, build_brax_step
        env = getattr(solver, "dynamics", None)
        if is_brax_env(env):
            return build_brax_step(env)
        adapter = getattr(solver, "_env_adapter", None)
        return adapter.jax_transition if adapter is not None else None

    def _draw_noise(self, key, shape, state=None):
        """Unit-variance candidate noise; routes to the `noise_sampler` seam.
        Default branch matches `dial_jax.reverse_once` EXACTLY (no hardcoded
        dtype) so byte-identity to DIAL holds under any jax dtype config (x64)."""
        if self.noise_sampler is not None:
            return self.noise_sampler.sample(shape, key=key, state=state)
        return jax.random.normal(key, shape)

    # --- the parallel reverse step (one diffusion step), node space ----------
    def _project_to_prior(self, nodes, U_rl):
        """Project node plans into a normalized Euclidean prior trust ball."""
        if not self._prior_active or self.prior_trust_radius <= 0.0:
            return nodes
        delta = nodes - U_rl
        axes = tuple(range(delta.ndim - 2, delta.ndim))
        rms = jnp.sqrt(jnp.mean(delta * delta, axis=axes, keepdims=True))
        scale = jnp.minimum(
            1.0, self.prior_trust_radius / (rms + 1e-8)
        )
        return U_rl + scale * delta

    def _reverse_step(
        self, state, rng, Ybar_curr, noise_scale, k, t0, U_rl,
    ):
        """One reverse-diffusion step. With every seam off this is, op-for-op,
        `dial_jax.reverse_once(update_form='weighted_mean')`."""
        Hn1, nu = Ybar_curr.shape
        # idx_init counts down so the transport abar schedule denoises late->clean.
        idx_init = jnp.clip(jnp.asarray(self.Ndiffuse_init - 1, jnp.int32) - k,
                            0, self._alphas_bar.shape[0] - 1)
        rng, sub = jax.random.split(rng)
        _, y_rng = jax.random.split(sub)                          # DIAL split pattern
        # cast to the carry dtype so the replan scan carry stays type-consistent
        # (and byte-identical to DIAL) regardless of the jax x64 config.
        eps = self._draw_noise(y_rng, (self.Nsample, Hn1, nu), state=Ybar_curr).astype(Ybar_curr.dtype)
        Y0s = eps * noise_scale[None, :, None] + Ybar_curr[None]  # base DIAL sampling schedule
        Y0s = Y0s.at[:, 0].set(Ybar_curr[0])                      # pin node-0
        if self._prior_active:
            Y0s = self._project_to_prior(Y0s, U_rl)
            if self.prior_include_incumbent:
                # Raw RL proposal is never overwritten: append it independently
                # of the shifted/refined incumbent.
                Y0s = jnp.concatenate(
                    [Y0s, U_rl[None], Ybar_curr[None]], axis=0
                )
            else:
                Y0s = jnp.concatenate([Y0s, Ybar_curr[None]], axis=0)
        else:
            Y0s = jnp.concatenate([Y0s, Ybar_curr[None]], axis=0)
        Y0s = jnp.clip(Y0s, -self.action_limit, self.action_limit)

        us = jnp.einsum("hn,bnu->bhu", self.N2U, Y0s)             # node -> dense

        # --- constraint_filter seam (action-space, like mdoc): NoOp => identity ---
        # Guarded so the DIAL path adds ZERO ops (byte-identity); active on flat-
        # state constrained envs (corridor / stepping) where the filter is defined.
        from genedynamics.core.constraints.action_filters import NoOpConstraintFilter
        if not isinstance(self.constraint_filter, NoOpConstraintFilter):
            us = self.constraint_filter.apply_actions_batch(
                state, us, env=self._env, obstacles=self.obstacles,
                schedule_state={"k": idx_init, "K": self.Ndiffuse_init},
                schedule_params={},
            )

        # AL-augmented rollout takes (aug_lambda, aug_rho) at call time. aug_rho is
        # kept CONSTANT (2GO uses a constant aug_rho_const; the schedule modulates
        # geometry, not the AL penalty). Plain/injected rollout keeps the
        # (state, us, t0) signature => byte-identical DIAL.
        if self._augmented:
            rewss = self._rollout_fn(
                state, us, t0, self.aug_lambda, self.aug_rho
            )
        else:
            rewss = self._rollout_fn(state, us, t0)
        rew_incumbent = rewss[-1].mean()
        rews = rewss.mean(axis=-1)

        std = rews.std()
        std = jnp.where(std < 1e-6, 1.0, std)
        logp0 = (rews - rew_incumbent) / (std * self.temp_sample)
        weights = jax.nn.softmax(logp0)
        Ybar_weighted = jnp.einsum("n,nij->ij", weights, Y0s)     # DIAL weighted mean

        gate_diag = None
        gate_action = None
        if self.geometry_gate_fn is not None:
            gate_diag = self.geometry_gate_fn(state, Ybar_curr, t0)
            gate_action = jnp.clip(
                jnp.asarray(gate_diag["action"], Ybar_curr.dtype), 0.0, 1.0
            )[None, :]

        # --- geometry seam (genemetry reuse): metric tangent projection of the
        # update direction + optional CFS retraction. Skipped (=> DIAL) when no
        # geometry_fn. SdfManifold / CfsRetraction are REUSED, not re-implemented.
        if self.geometry_fn is not None:
            a_geom = self.geometry_fn(state, Ybar_curr, t0)       # (Hnode+1, nu)
            bundle = self.manifold.geometry(a_geom, self.topk_active, self.eps_stab)
            u_dir = Ybar_weighted - Ybar_curr                     # node-space update dir
            u_proj = self.manifold.project(u_dir, bundle, mode="metric")
            # adaptive-schedule kappa (genemetry overlay, 2GO usage): scales the
            # geometry tracking by the current feasibility -> NON-MONOTONIC; 1.0
            # when adaptive off => unchanged.
            kappa_mult = self._kappa_mult(state, Ybar_curr)
            projected_dir = (self.geom_gain * kappa_mult) * u_proj
            if gate_action is not None:
                shaped_dir = u_dir + gate_action * (projected_dir - u_dir)
            else:
                shaped_dir = projected_dir
            Ybar_weighted = Ybar_curr + shaped_dir
        if self.retraction is not None:
            Ybar_retracted = self.retraction.retract(
                state, Ybar_weighted,
                {
                    "sched_state": {"k": idx_init, "K": self.Ndiffuse_init},
                    "sched_params": {"t0": t0},
                },
            ).trajectory
            if gate_action is not None:
                Ybar_weighted = (
                    Ybar_weighted
                    + gate_action * (Ybar_retracted - Ybar_weighted)
                )
            else:
                Ybar_weighted = Ybar_retracted
        # --- transport seam: None => verbatim DIAL; else DDPM/DDIM/FM/Adaptive ---
        if self.transport is None:
            Ybar_next = Ybar_weighted
        else:
            abar_k = self._alphas_bar[idx_init]
            Yi = Ybar_curr * jnp.sqrt(abar_k)
            eps_k = (Yi - jnp.sqrt(abar_k) * Ybar_weighted) / jnp.sqrt(
                jnp.maximum(1.0 - abar_k, 1e-8)
            )
            Ybar_next = self.transport.step(
                tau_k=Yi, tau1_k=Ybar_weighted, eps_k=eps_k, score_g=None,
                sched={
                    "abar_k": abar_k,
                    "alpha_k": self._alphas[idx_init],
                    "abar_km1": self._alphas_bar[jnp.maximum(idx_init - 1, 0)],
                },
            )
        if self._prior_active:
            Ybar_next = self._project_to_prior(Ybar_next, U_rl)
        info = {"rews": rews, "mean_reward": rews.mean(), "weights_max": weights.max()}
        if gate_diag is not None:
            info.update({
                f"gate_{name}": gate_diag[name]
                for name in ("scalar", "path", "normal", "stiffness", "force")
            })
        return rng, Ybar_next, info

    # --- WarmStartPlanner protocol (consumed by the receding-horizon bridge) --
    def init_plan_var(self) -> jnp.ndarray:
        return jnp.zeros((self.Hnode + 1, self.nu), dtype=jnp.float32)

    def make_schedule(self, n_diffuse: int) -> jnp.ndarray:
        return make_traj_diffuse_factors(self.sigma_control, self.traj_diffuse_factor, int(n_diffuse))

    def _replan_scan(self, state, warm_start, schedule, rng, t0, U_rl) -> jnp.ndarray:
        n = schedule.shape[0]

        def body(carry, k):
            rng_c, Y = carry
            # k = reverse-step index; _reverse_step derives idx_init + the coupled-
            # annealing multipliers from it.
            rng_c, Y, _ = self._reverse_step(
                state,
                rng_c,
                Y,
                schedule[k],
                k,
                t0,
                U_rl,
            )
            return (rng_c, Y), None

        (rng_out, Y), _ = jax.lax.scan(body, (rng, warm_start), jnp.arange(n))
        return Y

    def _rollout_node_candidates(self, state, nodes, t0):
        dense = self.spline.node2u_batch(nodes)
        if self._augmented:
            rewards = self._rollout_fn(
                state, dense, t0, self.aug_lambda, self.aug_rho
            )
        else:
            rewards = self._rollout_fn(state, dense, t0)
        return dense, rewards

    def _accept_refinement(self, state, U_rl, refined, t0):
        candidates = jnp.stack([U_rl, refined], axis=0)
        dense, rewards = self._rollout_node_candidates(
            state, candidates, t0
        )
        scores = jnp.mean(rewards, axis=-1)
        improvement = scores[1] - scores[0]
        predicted_ok = improvement > self.prior_improvement_epsilon

        if self.risk_fn is not None:
            risks = jax.vmap(lambda us: self.risk_fn(state, us))(dense)
            tolerance = self.prior_risk_tolerance
            risk_ok = jnp.all(risks[1] <= risks[0] + tolerance)
            # A refined proposal with any predicted force-limit violation is
            # never accepted, even when the RL fallback is itself imperfect.
            hard_force_ok = risks[1, 0] <= 1e-8
        else:
            risks = jnp.zeros((2, 0), dtype=refined.dtype)
            risk_ok = jnp.asarray(True)
            hard_force_ok = jnp.asarray(True)

        accepted = predicted_ok & risk_ok & hard_force_ok
        if not self.prior_acceptance:
            accepted = jnp.asarray(True)
        selected = jnp.where(accepted, refined, U_rl)
        info = {
            "prior_accepted": accepted.astype(jnp.float32),
            "prior_predicted_improvement": improvement,
            "prior_score_rl": scores[0],
            "prior_score_refined": scores[1],
            "prior_risk_rl": risks[0],
            "prior_risk_refined": risks[1],
            "prior_risk_ok": risk_ok.astype(jnp.float32),
            "prior_force_veto": (~hard_force_ok).astype(jnp.float32),
        }
        return selected, info

    def replan_with_info(self, state, warm_start, schedule, rng, t0=0.0):
        t0 = jnp.asarray(t0, jnp.float32)
        if self._prior_active:
            lam = self.prior_lambda_shift
            U_rl = jnp.asarray(self.prior.warm_start(state), warm_start.dtype)
            warm_start = lam * warm_start + (1.0 - lam) * U_rl
            warm_start = self._project_to_prior(warm_start, U_rl)
        else:
            # Dummy of the right static shape; all prior branches are Python
            # static false, preserving the legacy candidate set and arithmetic.
            U_rl = warm_start
        if self._prepare_state_jit is not None:
            # The measured response is frozen around the actual incumbent, not
            # around an arbitrary shifted/RL mixture.
            response_center = U_rl if self._prior_active else warm_start
            state = self._prepare_state_jit(state, response_center, t0)
        refined = self._replan_scan_jit(
            state, warm_start, schedule, rng, t0, U_rl
        )
        if self._prior_active:
            return self._accept_refinement(state, U_rl, refined, t0)
        return refined, {}

    def replan(self, state, warm_start, schedule, rng, t0=0.0) -> jnp.ndarray:
        return self.replan_with_info(
            state, warm_start, schedule, rng, t0=t0
        )[0]

    def first_action(self, plan_var) -> jnp.ndarray:
        return self.spline.node2u(plan_var)[0]

    def shift(self, plan_var) -> jnp.ndarray:
        return self.spline.shift_nodes(plan_var)

    # --- unified backend interface (single-shot, parallel) -------------------
    def plan(self, x0: Any, rng_key: Optional[Any] = None) -> Dict[str, Any]:
        """Single-shot full-horizon plan (Ndiffuse_init reverse steps from cold),
        the whole reverse-diffuse jit-ed; candidate rollout vmap-ed over Nsample."""
        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.seed)
        replan_jit = jax.jit(self.replan_with_info)
        Y, prior_info = replan_jit(
            x0,
            self.init_plan_var(),
            self.make_schedule(self.Ndiffuse_init),
            rng_key,
        )
        us = self.spline.node2u(Y)
        _, reward_batch = self._rollout_node_candidates(
            x0, Y[None], jnp.float32(0.0)
        )
        rews = np.asarray(reward_batch[0], dtype=np.float32)
        states = self._rollout_states(x0, us)
        out = {
            "actions": np.asarray(us, dtype=np.float32),
            "states": states,
            "rewards": rews,
            "total_reward": float(np.sum(rews)),
            "mean_reward": float(np.mean(rews)) if rews.size else 0.0,
        }
        out.update({
            key: np.asarray(value)
            for key, value in prior_info.items()
        })
        return out

    def plan_batch(self, x0: Any, keys: Any) -> List[Dict[str, Any]]:
        return [self.plan(x0, rng_key=k) for k in keys]

    def sample_trajectories(self, x0: Any, n_samples: int, rng_key: Optional[Any] = None):
        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.seed)
        keys = jax.random.split(rng_key, max(1, int(n_samples)))
        return self.plan_batch(x0, keys)

    def _rollout_states(self, x0: Any, us: jnp.ndarray) -> np.ndarray:
        if self._step_fn is None:
            return np.asarray(x0, dtype=np.float32)[None]

        def f(s, u):
            s2 = self._step_fn(s, u)
            return s2, s2

        _, states = jax.lax.scan(f, x0, us)
        x0a = jnp.asarray(x0)
        return np.asarray(jnp.concatenate([x0a[None], states], axis=0), dtype=np.float32)


__all__ = ["MdacBackendJax"]
