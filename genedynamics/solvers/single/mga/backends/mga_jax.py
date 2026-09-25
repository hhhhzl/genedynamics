"""MGA JAX backend — high-performance parallel reverse-diffusion kernel.

Written in the SAME high-performance pattern as `mdoc_jax` / `twogo_jax` /
`cfsmbd_jax`: a `jax.jit`-ed `jax.lax.scan` reverse-diffuse loop, with the
candidate rollout `jax.vmap`-ed over the `Nsample` axis, per-step schedule values
precomputed, and the MGA upgrades injected through the SAME optional seams those
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
`solvers/common/env_rollout`, node-spline parametrisation) because MGA is
DIAL-derived. With every seam off / NoOp / None, the `body` reduces *operation
for operation* to `dial_jax.reverse_once(update_form='weighted_mean')`, so MGA
is byte-identical to DIAL on unconstrained tasks (the regression gate).

The ONE genuinely-new primitive (position-stiffness SPD `K=exp(S)`) lives
UPSTREAM in `genedynamics/core/control/stiffness.py`; this backend only consumes
it (via `action_size`). Everything else is reuse.

A `rollout_fn` may be injected for CPU tests (no mjx), exactly as `dial_jax` does.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.core.types import ExecutionRejected
from genedynamics.solvers.single.dial.backends.dial_jax import (
    make_sigma_control,
    make_traj_diffuse_factors,
)
from genedynamics.solvers.single.dial.spline import NodeSpline

RolloutFn = Callable[[Any, jnp.ndarray, Any], jnp.ndarray]  # (state, us, t0)->rews


class MgaBackendJax:
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
        candidate_rollout_batch_size: Optional[int] = None,
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
        stepwise_diffusion: bool = False,
        stepwise_acceptance: bool = False,
        lazy_emergency_scoring: bool = False,
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
            candidate_rollout_batch_size = cfg.get(
                "candidate_rollout_batch_size",
                candidate_rollout_batch_size,
            )
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
            stepwise_diffusion = bool(
                cfg.get("stepwise_diffusion", stepwise_diffusion)
            )
            stepwise_acceptance = bool(
                cfg.get("stepwise_acceptance", stepwise_acceptance)
            )
            lazy_emergency_scoring = bool(
                cfg.get("lazy_emergency_scoring", lazy_emergency_scoring)
            )
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
            raise ValueError("MgaBackendJax requires nu (action dim).")

        self.nu = int(nu)
        self.Hnode, self.Hsample = int(Hnode), int(Hsample)
        self.Nsample = int(Nsample)
        self.candidate_rollout_batch_size = (
            None
            if candidate_rollout_batch_size is None
            else int(candidate_rollout_batch_size)
        )
        self.temp_sample = float(temp_sample)
        self.traj_diffuse_factor = float(traj_diffuse_factor)
        self.action_limit = float(action_limit)
        self.Ndiffuse = int(Ndiffuse)
        self.Ndiffuse_init = int(Ndiffuse_init)
        self.beta0, self.betaT = float(beta0), float(betaT)
        self.aug_lambda, self.aug_rho = float(aug_lambda), float(aug_rho)
        self.use_soft_feasibility = bool(use_soft_feasibility)
        self.stepwise_diffusion = bool(stepwise_diffusion)
        self.stepwise_acceptance = bool(stepwise_acceptance)
        self.lazy_emergency_scoring = bool(lazy_emergency_scoring)
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
        # Absent => the manifold path is skipped and MGA == DIAL. The
        # metric tangent projection + retraction are REUSED from genemetry
        # (SdfManifold / CfsRetraction), never hand-rolled.
        self.geometry_fn = getattr(solver, "geometry_fn", None) if solver is not None else None
        self.manifold = None
        if self.geometry_fn is not None:
            cfg = getattr(solver, "config", {}) if solver is not None else {}
            # node space has only Hnode+1 rows; top_k needs k <= rows.
            self.topk_active = min(int(cfg.get("mga_topk_active", 8)), self.Hnode + 1)
            self.eps_stab = float(cfg.get("mga_eps_stab", 1e-6))
            self.geom_gain = float(cfg.get("mga_geom_gain", 1.0))
            from genedynamics.genemetry.manifold.sdf import SdfManifold
            self.manifold = SdfManifold(backend="jax")
        # Retraction is an independent seam. In particular, mga_no_tangent keeps
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
        self.candidate_projection_fn = (
            getattr(solver, "candidate_projection_fn", None)
            if solver is not None else None
        )
        self._candidate_projection_jit = (
            jax.jit(self.candidate_projection_fn)
            if self.candidate_projection_fn is not None else None
        )

        # --- prior seam (genedynamics/learning/priors): horizon proposal,
        # incumbent candidate, local trust region, and do-no-harm acceptance.
        # None => every branch below is skipped and DIAL parity is preserved.
        self.prior = getattr(solver, "prior", None) if solver is not None else None
        self.atacom_prior = (
            getattr(solver, "atacom_prior", None) if solver is not None else None
        )
        self.prior_lambda_shift = (
            float(getattr(solver, "prior_lambda_shift", 0.5)) if solver is not None else 0.5
        )
        self.risk_fn = getattr(solver, "risk_fn", None) if solver is not None else None
        self.score_risk_fn = getattr(self._env, "sequence_score_risk", None)
        self.emergency_score_risk_fn = getattr(
            self._env, "emergency_sequence_score_risk", None
        )
        self.risk_safe_fn = getattr(self._env, "sequence_risk_is_safe", None)
        self.risk_compare_fn = getattr(
            self._env, "sequence_risk_is_no_worse", None
        )
        # Every decision hook stays on the nominal task model.  Hidden OOD
        # geometry/dynamics belong only to the physical transition; observable
        # wrench/contact evidence is already carried by the execution state.
        # Calling methods on execution_env here would disclose the true hole
        # frame, clearance and friction to the controller.
        self._task_contract_env = self._env
        self.emergency_plan_fn = getattr(
            self._task_contract_env, "emergency_plan", None
        )
        self.emergency_plans_fn = getattr(
            self._task_contract_env, "emergency_plans", None
        )
        self.normal_recovery_plan_fn = getattr(
            self._task_contract_env, "normal_recovery_plan", None
        )
        self.normal_recovery_plans_fn = getattr(
            self._task_contract_env, "normal_recovery_plans", None
        )
        self.normal_rescue_plans_fn = getattr(
            self._task_contract_env, "normal_rescue_plans", None
        )
        self.prior_applicable_fn = getattr(
            self._task_contract_env, "mga_prior_applicable", None
        )
        self._prior_applicable_jit = (
            jax.jit(self.prior_applicable_fn)
            if self.prior_applicable_fn is not None else None
        )
        self.emergency_active_fn = getattr(
            self._task_contract_env, "emergency_plan_is_active", None
        )
        self.emergency_override_fn = getattr(
            self._task_contract_env, "emergency_plan_should_override", None
        )
        self.prior_include_incumbent = bool(
            getattr(solver, "prior_include_incumbent", True)
        )
        self.prior_trust_radius = float(
            getattr(solver, "prior_trust_radius", 0.5)
        )
        self.prior_stochastic_samples = int(
            getattr(solver, "prior_stochastic_samples", 0)
        )
        self.prior_atacom_samples = int(
            getattr(solver, "prior_atacom_samples", 0)
        )
        self.prior_union_trust = bool(
            getattr(solver, "prior_union_trust", False)
        )
        self.prior_mode = str(getattr(solver, "prior_mode", "guided"))
        if self.prior_mode not in {"guided", "warm_start", "additive"}:
            raise ValueError(
                "prior_mode must be 'guided', 'warm_start', or 'additive'"
            )
        self.prior_atacom_incumbent = bool(
            getattr(solver, "prior_atacom_incumbent", False)
        )
        self.prior_atacom_default = bool(
            getattr(solver, "prior_atacom_default", False)
        )
        self.prior_atacom_strict_risk = bool(
            getattr(solver, "prior_atacom_strict_risk", False)
        )
        if (
            self.prior_stochastic_samples < 0
            or self.prior_atacom_samples < 0
            or self.prior_stochastic_samples + self.prior_atacom_samples
            >= self.Nsample
        ):
            raise ValueError(
                "structured prior samples must be nonnegative and sum to less "
                f"than Nsample; got rl={self.prior_stochastic_samples}, "
                f"atacom={self.prior_atacom_samples}, Nsample={self.Nsample}"
            )
        if self.prior_atacom_samples and self.atacom_prior is None:
            raise ValueError("prior_atacom_samples requires atacom_prior")
        if self.prior_atacom_incumbent and self.atacom_prior is None:
            raise ValueError("prior_atacom_incumbent requires atacom_prior")
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
        self.prior_fallback_mode = str(
            getattr(solver, "prior_fallback_mode", "rl")
        )
        self.receding_shift_mode = str(
            getattr(solver, "receding_shift_mode", "legacy")
        )
        if self.receding_shift_mode not in {
            "legacy", "zero", "terminal_hold", "certified_terminal_hold",
        }:
            raise ValueError(
                "receding_shift_mode must be legacy, zero, terminal_hold, "
                "or certified_terminal_hold"
            )
        if self.prior_fallback_mode not in {"rl", "receding_incumbent"}:
            raise ValueError(
                "prior_fallback_mode must be 'rl' or 'receding_incumbent', got "
                f"{self.prior_fallback_mode!r}"
            )
        # An environment capability alone must not alter ordinary baselines or
        # acceptance-disabled MGA. Only a route that can select an executable
        # emergency opts into explicit modes.
        self._execution_context = None
        self._model_execution_context = None
        self._host_emergency_score_risk_fn = None
        self._normal_recovery_ready_fn = None
        self._host_emergency_prewarmed = False
        self.execution_step = None
        self._committed_shift_mode = 0
        # Provenance for the warm start carried across one real receding step.
        # A task-owned NORMAL recovery can intentionally lie outside a narrow
        # learned/Gaussian proposal tube.  Ordinary sampled/refined plans must
        # not inherit that authority merely because they are also incumbents.
        self._committed_shift_task_recovery = False
        self._pending_shift_task_recovery = False
        self._configure_execution_context()
        self.reliability_model = getattr(solver, "reliability_model", None)
        # These are observed execution-state signals (tracking lag, queued
        # command and measured wrench), not nominal rollout predictions.  Use
        # the task frame that generated the observation and training labels.
        self.reliability_feature_fn = getattr(
            self._task_contract_env, "reliability_features", None
        )
        # Hybrid contact tasks can expose a candidate-horizon feature contract.
        # Legacy tasks (including the frozen surface-scanning benchmark) keep
        # the original first-action contract exactly.
        self.reliability_sequence_feature_fn = getattr(
            self._task_contract_env, "reliability_features_sequence", None
        )
        if (
            self.reliability_model is not None
            and self.reliability_feature_fn is None
            and self.reliability_sequence_feature_fn is None
        ):
            raise ValueError(
                "learned reliability requires env.reliability_features or "
                "env.reliability_features_sequence"
            )
        self.reliability_risk_tolerance = jnp.asarray(
            getattr(solver, "reliability_risk_tolerance", (0.0, 0.0, 0.0, 0.0)),
            dtype=jnp.float32,
        )
        self.reliability_force_limit = float(
            getattr(solver, "reliability_force_limit", 0.25)
        )
        self.reliability_deformation_limit = float(
            getattr(solver, "reliability_deformation_limit", float("inf"))
        )
        raw_reliability_hard_limits = getattr(
            solver, "reliability_hard_limits", None
        )
        self.reliability_hard_limits = (
            None
            if raw_reliability_hard_limits is None
            else jnp.asarray(raw_reliability_hard_limits, jnp.float32)
        )
        self.reliability_support_mode = str(
            getattr(solver, "reliability_support_mode", "joint")
        )
        if self.reliability_support_mode not in {"joint", "state"}:
            raise ValueError(
                "reliability_support_mode must be 'joint' or 'state'"
            )
        self.reliability_ood_policy = str(
            getattr(solver, "reliability_ood_policy", "veto")
        )
        if self.reliability_ood_policy not in {"veto", "model_based"}:
            raise ValueError(
                "reliability_ood_policy must be 'veto' or 'model_based'"
            )
        # A model-based OOD policy may use learned confidence only after the
        # checkpoint has been independently validated and promoted.  The
        # frozen H1 development artifact is intentionally marked
        # ``promotion_eligible: false``; keeping its predictions as a hard
        # veto would turn false negatives into a permanent zero-progress
        # controller.  We still compute and log its predictions, but let the
        # task-owned physical certificate own acceptance until promotion.
        reliability_metadata = (
            getattr(self.reliability_model, "metadata", {}) or {}
        )
        self.reliability_promotion_eligible = bool(
            reliability_metadata.get("performance_validated", False)
            and reliability_metadata.get("promotion_eligible", False)
        )
        # An unpromoted checkpoint may become authoritative only in the
        # explicitly guarded PegInsert paired-validation run.  The method
        # plugin rejects this override outside development, so canonical
        # deployment still requires promotion metadata.
        self.reliability_validation_authoritative = bool(
            getattr(solver, "reliability_validation_authoritative", False)
        )
        self.reliability_gate_authoritative = bool(
            self.reliability_ood_policy == "veto"
            or self.reliability_promotion_eligible
            or self.reliability_validation_authoritative
        )
        flags = getattr(solver, "flags", None)
        self.use_rl_prior = bool(getattr(flags, "use_rl_prior", True))
        self._gate_controllability_only = bool(
            getattr(flags, "use_controllability_geometry", False)
        )
        # coupled annealing (sigma_k down, rho_k up, kappa_k up). Default OFF (no
        # flags / mock => byte-identical DIAL); "mga" method => flags turn it on.
        self.use_adaptive_schedule = bool(getattr(flags, "use_adaptive_schedule", False))

        # rollout / step: injected (CPU tests) or built from the env (brax/mjx).
        # _augmented is set True only when the AL-augmented brax rollout is built.
        self._augmented = False
        self._rollout_fn = rollout_fn or (self._build_rollout_from_solver(solver) if solver else None)
        self._step_fn = step_fn or self._build_step_from_solver(solver)
        if self._rollout_fn is None:
            raise ValueError("MgaBackendJax needs a rollout_fn (inject one or construct via a solver).")

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
                _cfg.get("mga_overlay", None), rho_ref_default=max(self.aug_rho, 1.0))
            self._overlay = ScheduleOverlay(config=self._overlay_cfg, backend="jax")
            kap_ref, _ = self._overlay.constraint_overlay(
                jnp.asarray(0.0, jnp.float32), jnp.asarray(self.aug_rho, jnp.float32))
            self._kappa_ref = float(np.asarray(kap_ref))

        # Receding execution calls replan at every real step. Jitting a stable
        # bound function here avoids creating a fresh lax.scan executable for
        # every call; only the init/steady schedule shapes compile separately.
        self._replan_scan_jit = jax.jit(self._replan_scan)
        self._replan_scan_structured_jit = jax.jit(
            self._replan_scan_structured
        )
        # Acceptance contains a task-owned horizon ``lax.scan``.  Leaving the
        # vmap/scan expression in eager Python causes a fresh transformed
        # function to be constructed on every MPC step, so CPU XLA retains an
        # episode's worth of acceptance executables.  Keep one stable bound JIT
        # exactly as for the reverse scan.  This changes no score, risk, or gate
        # arithmetic; it only makes the compilation/cache boundary explicit.
        self._accept_refinement_jit = (
            self._accept_refinement
            if self.stepwise_acceptance
            else jax.jit(self._accept_refinement)
        )
        self._normal_candidate_scores_jit = None
        self._emergency_candidate_scores_jit = None
        if self.stepwise_acceptance and self.score_risk_fn is not None:
            # Keep measured state as an explicit dynamic argument.  Creating a
            # fresh lambda that closes over each receding state makes JAX see a
            # new constant and compile one H-step MJX scorer per control step.
            self._normal_candidate_scores_jit = jax.jit(
                self._normal_candidate_scores
            )
            if self.emergency_score_risk_fn is not None:
                self._emergency_candidate_scores_jit = jax.jit(
                    self._emergency_candidate_scores
                )
        self._select_additive_prior_jit = jax.jit(self._select_additive_prior)
        # CPU can keep one reverse update as the compilation boundary.  This
        # preserves the exact update/RNG order while avoiding an outer scan
        # that inlines a large MJX candidate rollout multiple times.
        self._reverse_step_jit = jax.jit(self._reverse_step)

    @property
    def _prior_active(self):
        """Static-at-trace switch; also supports prior injection in unit tests."""
        return self.prior is not None and self.use_rl_prior

    @property
    def _prior_guided(self):
        return self._prior_active and self.prior_mode == "guided"

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
            batch_size = getattr(
                solver, "candidate_rollout_batch_size", None
            )
            # cfsmbd/mdcoas AL: augment the brax reward with the soft-feasibility
            # penalty from env.constraint_residual. The no-op default residual =>
            # 0 penalty => byte-identical to the plain rollout (== DIAL).
            if self.use_soft_feasibility:
                self._augmented = True                       # rollout takes aug params at call time
                return build_brax_rollout_augmented(
                    env, candidate_batch_size=batch_size
                )
            return build_brax_rollout(
                env, candidate_batch_size=batch_size
            )
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

    def _project_to_prior_union(self, nodes, centers):
        """Project each plan to the nearest expert trust ball."""
        if not self._prior_active or self.prior_trust_radius <= 0.0:
            return nodes
        single = nodes.ndim == 2
        plans = nodes[None] if single else nodes
        # (B, C, Hn, nu): choose the expert center with minimum plan RMS.
        delta = plans[:, None] - centers[None]
        rms = jnp.sqrt(jnp.mean(delta * delta, axis=(-2, -1)) + 1e-8)
        nearest = jnp.argmin(rms, axis=1)
        chosen = centers[nearest]
        selected_delta = plans - chosen
        selected_rms = jnp.sqrt(
            jnp.mean(selected_delta * selected_delta, axis=(-2, -1), keepdims=True)
        )
        scale = jnp.minimum(
            1.0, self.prior_trust_radius / (selected_rms + 1e-8)
        )
        projected = chosen + scale * selected_delta
        return projected[0] if single else projected

    def _reverse_step(
        self, state, rng, Ybar_curr, noise_scale, k, t0, U_rl,
        structured_nodes=None, prior_centers=None,
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
        n_structured = (
            0 if structured_nodes is None else structured_nodes.shape[0]
        )
        n_noise = self.Nsample - n_structured
        eps = self._draw_noise(
            y_rng, (n_noise, Hn1, nu), state=Ybar_curr
        ).astype(Ybar_curr.dtype)
        Y0s = eps * noise_scale[None, :, None] + Ybar_curr[None]  # base DIAL sampling schedule
        Y0s = Y0s.at[:, 0].set(Ybar_curr[0])                      # pin node-0
        if self._prior_guided:
            if prior_centers is not None:
                Y0s = self._project_to_prior_union(Y0s, prior_centers)
            else:
                Y0s = self._project_to_prior(Y0s, U_rl)
            if structured_nodes is not None:
                structured_nodes = structured_nodes.astype(Ybar_curr.dtype)
                structured_nodes = (
                    self._project_to_prior_union(
                        structured_nodes, prior_centers
                    )
                    if prior_centers is not None
                    else self._project_to_prior(structured_nodes, U_rl)
                )
                Y0s = jnp.concatenate([Y0s, structured_nodes], axis=0)
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
        update_gate_action = None
        if self.geometry_gate_fn is not None:
            gate_diag = self.geometry_gate_fn(state, Ybar_curr, t0)
            gate_action = jnp.clip(
                jnp.asarray(gate_diag["action"], Ybar_curr.dtype), 0.0, 1.0
            )[None, :]
            if "update_action" in gate_diag:
                update_gate_action = jnp.clip(
                    jnp.asarray(
                        gate_diag["update_action"], Ybar_curr.dtype
                    ),
                    0.0,
                    1.0,
                )[None, :]
        clean_gate_action = gate_action
        if gate_diag is not None and self._gate_controllability_only:
            # A controllability-aware task may keep its clean tangential
            # geometry fully active while selectively attenuating unreliable
            # normal/stiffness/force projection and retraction blocks.  Tasks
            # without this optional contract retain the previous fully active
            # clean geometry; non-controllability methods keep ``action``.
            clean_action = gate_diag.get("clean_action")
            clean_gate_action = (
                None
                if clean_action is None
                else jnp.clip(
                    jnp.asarray(clean_action, Ybar_curr.dtype), 0.0, 1.0
                )[None, :]
            )

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
            if clean_gate_action is not None:
                shaped_dir = (
                    u_dir + clean_gate_action * (projected_dir - u_dir)
                )
            else:
                shaped_dir = projected_dir
            Ybar_weighted = Ybar_curr + shaped_dir
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
        if self._prior_guided:
            Ybar_next = (
                self._project_to_prior_union(Ybar_next, prior_centers)
                if prior_centers is not None
                else self._project_to_prior(Ybar_next, U_rl)
            )
        # Retraction is a feasibility map on the FINAL proposal.  Applying it
        # before transport lets DDPM/DDIM (and a prior-union projection) move
        # the plan straight back off the clean manifold.
        if self.retraction is not None:
            Ybar_retracted = self.retraction.retract(
                state, Ybar_next,
                {
                    "sched_state": {"k": idx_init, "K": self.Ndiffuse_init},
                    "sched_params": {"t0": t0},
                },
            ).trajectory
            if clean_gate_action is not None:
                Ybar_next = Ybar_next + clean_gate_action * (
                    Ybar_retracted - Ybar_next
                )
            else:
                Ybar_next = Ybar_retracted
        if self._candidate_projection_jit is not None:
            Ybar_next = self._candidate_projection_jit(state, Ybar_next)
        if update_gate_action is not None:
            # Optional task-owned epistemic gate.  ``action`` above retains
            # the original MGA meaning (blend raw and manifold directions),
            # preserving every task that omits this key.  A task with a
            # trusted time-indexed incumbent can additionally freeze action
            # blocks whose local realization is not yet observable, rather
            # than replacing that incumbent with pure Gaussian motion.
            Ybar_next = Ybar_curr + update_gate_action * (
                Ybar_next - Ybar_curr
            )
        info = {"rews": rews, "mean_reward": rews.mean(), "weights_max": weights.max()}
        if structured_nodes is not None:
            n_noise = self.Nsample - structured_nodes.shape[0]
            n_rl = self.prior_stochastic_samples
            n_atacom = self.prior_atacom_samples
            info["proposal_gaussian_best_reward"] = jnp.max(rews[:n_noise])
            info["proposal_gaussian_weight"] = jnp.sum(weights[:n_noise])
            if n_rl > 0:
                rl_slice = slice(n_noise, n_noise + n_rl)
                info["proposal_rl_best_reward"] = jnp.max(rews[rl_slice])
                info["proposal_rl_weight"] = jnp.sum(weights[rl_slice])
            if n_atacom > 0:
                start = n_noise + n_rl
                atacom_slice = slice(start, start + n_atacom)
                info["proposal_atacom_best_reward"] = jnp.max(
                    rews[atacom_slice]
                )
                info["proposal_atacom_weight"] = jnp.sum(weights[atacom_slice])
        if gate_diag is not None:
            info.update({
                f"gate_{name}": gate_diag[name]
                for name in ("scalar", "path", "normal", "stiffness", "force")
            })
        return rng, Ybar_next, info


    def _configure_execution_context(self):
        """Protocol 1 seals task-owned entry_prepared independently of entry
        geometry validity. The task also seals the actual entry stiffness
        matrix AND raw coordinates decoded from the last committed action
        (cold start: nominal zero raw); nominal decoding cannot replace them.
        Repeated UNLOAD preparation reuses the complete seal even
        before commitment; an absent seal in an active context is unknown.
        NORMAL preparation cancels only uncommitted entries. Actual NORMAL
        execution ends the old session. These are task-state semantics; the
        generic backend neither owns anchor fields nor serializes callables.
        """
        model = getattr(self._env, "mga_execution_context", None)
        execution = getattr(self._task_contract_env, "mga_execution_context", None)
        can_select_emergency = self.prior_acceptance and (
            self.prior_mode == "additive"
            or self.prior_fallback_mode == "receding_incumbent"
        )
        if not can_select_emergency or (model is None and execution is None):
            return
        expected = {"schema_version": 1, "normal_mode": 0, "emergency_mode": 1}
        for label, context in (("model", model), ("execution", execution)):
            if not isinstance(context, Mapping):
                raise ValueError(f"MGA {label} execution context is missing")
            # Callables are runtime capabilities, never a serializable contract.
            # Compare only the small scalar protocol, not DR parameter values.
            if any(type(context.get(k)) is not int or context[k] != v
                   for k, v in expected.items()):
                raise ValueError(f"MGA {label} execution context schema mismatch")
            if any(not callable(context.get(k))
                   for k in ("prepare_state", "mode_from_state", "step")):
                raise TypeError(f"MGA {label} execution context hooks must be callable")
        if not all(callable(hook) for hook in (
            self.score_risk_fn, self.emergency_score_risk_fn,
            self.emergency_plan_fn, self.risk_safe_fn,
        )):
            raise ValueError("MGA execution context requires complete task safety hooks")
        self._model_execution_context = model
        self._execution_context = execution
        # When model and execution are the same task instance, this hook shares
        # the already-compiled real-step executable.  A distinct nominal model
        # keeps its own hook, preserving model/execution separation.
        score_context = execution if self._env is self._task_contract_env else model
        host_emergency = score_context.get("score_emergency_host")
        if host_emergency is not None and not callable(host_emergency):
            raise TypeError("MGA score_emergency_host hook must be callable")
        self._host_emergency_score_risk_fn = host_emergency
        recovery_ready = execution.get("normal_recovery_ready")
        if recovery_ready is not None and not callable(recovery_ready):
            raise TypeError("MGA normal_recovery_ready hook must be callable")
        self._normal_recovery_ready_fn = recovery_ready
        self.execution_step = self._execute_with_context
        self._shift_with_mode_jit = jax.jit(self._shift_with_mode)

    @staticmethod
    def _checked_execution_mode(mode):
        mode = jnp.asarray(mode)
        if mode.shape != () or not jnp.issubdtype(mode.dtype, jnp.integer):
            raise ValueError("execution mode must be a scalar integer")
        if not isinstance(mode, jax.core.Tracer) and int(np.asarray(mode)) not in (0, 1):
            raise ValueError("unknown execution mode")
        return mode.astype(jnp.int32)

    def _incumbent_execution_mode(self, state, override=None):
        mode = (
            self._execution_context["mode_from_state"](state)
            if override is None else override
        )
        return self._checked_execution_mode(mode)

    def _normal_rollout_state(self, state):
        # Set pending NORMAL explicitly. Preserve a COMMITTED unload's entry
        # history, but cancel any uncommitted prepared entry when committed
        # mode is NORMAL. prepare_state(NORMAL) is geometry-independent.
        return self._model_execution_context["prepare_state"](
            state, jnp.int32(0)
        )

    def _execute_with_context(self, state, action, plan_info):
        if any(isinstance(leaf, jax.core.Tracer)
               for leaf in jax.tree_util.tree_leaves((state, action, plan_info))):
            raise ValueError("context execution callback must run on the host")
        if not isinstance(plan_info, Mapping) or "execution_mode" not in plan_info:
            raise ValueError("MGA acceptance lost its explicit execution mode")
        if bool(np.asarray(plan_info.get("emergency_unrecoverable", 0.0)) > 0.5):
            raise ExecutionRejected(
                "no_revalidated_safe_candidate",
                {
                    "normal_refined_safe": bool(np.asarray(
                        plan_info.get("refined_revalidated_safe", 0.0)
                    )),
                    "incumbent_safe": bool(np.asarray(
                        plan_info.get("incumbent_revalidated_safe", 0.0)
                    )),
                    "emergency_safe": bool(np.asarray(
                        plan_info.get("emergency_revalidated_safe", 0.0)
                    )),
                },
            )
        mode = self._checked_execution_mode(plan_info["execution_mode"])
        # Record candidate provenance before the real step, then commit it in
        # after_step only if execution actually succeeds.  A shifted recovery
        # remains a recovery only while the exact incumbent is reselected;
        # adopting any refined/expert/emergency candidate ends the exemption.
        selected_new_recovery = bool(np.asarray(
            plan_info.get("normal_recovery_selected", 0.0)
        ) > 0.5) or bool(np.asarray(
            plan_info.get("normal_rescue_selected", 0.0)
        ) > 0.5)
        selected_recovery_incumbent = (
            self._committed_shift_task_recovery
            and bool(np.asarray(
                plan_info.get("incumbent_candidate_selected", 0.0)
            ) > 0.5)
        )
        self._pending_shift_task_recovery = (
            selected_new_recovery or selected_recovery_incumbent
        )
        # Reproduce the scoring preparation order from the current measured
        # state, not a stale pending request or an anchor copied from a plan.
        # Committed unload history survives this NORMAL preparation.
        state = self._execution_context["prepare_state"](state, jnp.int32(0))
        return self._execution_context["step"](state, action, mode)

    def _shift_with_mode(self, plan_var, mode):
        # mode is an explicit dynamic JIT argument, not a captured host cache.
        terminal = (
            self.spline.shift_nodes_certified_terminal_hold(plan_var)
            if self.receding_shift_mode == "certified_terminal_hold"
            else self.spline.shift_nodes_terminal_hold(plan_var)
        )
        ordinary = (
            terminal if (
                self.receding_shift_mode in {
                    "terminal_hold", "certified_terminal_hold",
                }
                or (self.receding_shift_mode == "legacy" and self._prior_active
                    and self.prior_fallback_mode == "receding_incumbent")
            ) else self.spline.shift_nodes(plan_var)
        )
        return jnp.where(mode == 1, terminal, ordinary)

    # --- WarmStartPlanner protocol (consumed by the receding-horizon bridge) --
    def init_plan_var(self) -> jnp.ndarray:
        if self._execution_context is not None:
            self._committed_shift_mode = 0
            self._committed_shift_task_recovery = False
            self._pending_shift_task_recovery = False
        if self.atacom_prior is not None:
            reset = getattr(self.atacom_prior, "reset", None)
            if callable(reset):
                reset()
        return jnp.zeros((self.Hnode + 1, self.nu), dtype=jnp.float32)

    def after_step(self, state, action, next_state) -> None:
        """Commit state carried by a structured expert after real execution."""
        del state, action
        if self._execution_context is not None:
            # This method is called only after REAL execution by the host
            # controller. Planning always reads the dynamic mode from state.
            self._committed_shift_mode = int(np.asarray(
                self._incumbent_execution_mode(next_state)
            ))
            self._committed_shift_task_recovery = (
                self._pending_shift_task_recovery
            )
            self._pending_shift_task_recovery = False
        if self.atacom_prior is not None:
            commit = getattr(self.atacom_prior, "commit", None)
            if callable(commit):
                commit()

    def make_schedule(self, n_diffuse: int) -> jnp.ndarray:
        n_diffuse = int(n_diffuse)
        if not 0 < n_diffuse <= self.Ndiffuse_init:
            raise ValueError(
                "n_diffuse must be in [1, Ndiffuse_init], got "
                f"{n_diffuse} with Ndiffuse_init={self.Ndiffuse_init}"
            )
        active = make_traj_diffuse_factors(
            self.sigma_control, self.traj_diffuse_factor, n_diffuse
        )
        # Keep one static scan shape for cold and steady replans.  Without this,
        # CPU XLA caches a second full MJX/MGA executable for Ndiffuse=2 after
        # compiling Ndiffuse_init=10, which can exceed the host memory budget.
        # NaN rows are masked to exact carry no-ops in _replan_scan, so the
        # number and order of actual reverse updates remain unchanged.
        pad = self.Ndiffuse_init - n_diffuse
        return jnp.pad(
            active,
            ((0, pad), (0, 0)),
            constant_values=jnp.nan,
        )

    def _replan_scan(self, state, warm_start, schedule, rng, t0, U_rl) -> jnp.ndarray:
        n = schedule.shape[0]

        def body(carry, k):
            rng_c, Y = carry
            noise_scale = schedule[k]

            def active_step(active_carry):
                active_rng, active_y = active_carry
                # k = reverse-step index; _reverse_step derives idx_init + the
                # coupled-annealing multipliers from it.
                active_rng, active_y, _ = self._reverse_step(
                    state,
                    active_rng,
                    active_y,
                    noise_scale,
                    k,
                    t0,
                    U_rl,
                )
                return active_rng, active_y

            rng_c, Y = jax.lax.cond(
                jnp.all(jnp.isfinite(noise_scale)),
                active_step,
                lambda inactive_carry: inactive_carry,
                (rng_c, Y),
            )
            return (rng_c, Y), None

        (rng_out, Y), _ = jax.lax.scan(body, (rng, warm_start), jnp.arange(n))
        return Y

    def _replan_stepwise(self, state, warm_start, schedule, rng, t0, U_rl):
        """Host-loop reverse steps around one stable compiled update."""
        Y = warm_start
        for k in range(schedule.shape[0]):
            noise_scale = schedule[k]
            if not bool(np.all(np.isfinite(np.asarray(noise_scale)))):
                continue
            rng, Y, _ = self._reverse_step_jit(
                state, rng, Y, noise_scale, jnp.int32(k), t0, U_rl
            )
            rng, Y = jax.block_until_ready((rng, Y))
        return Y

    def _replan_scan_structured(
        self, state, warm_start, schedule, rng, t0, U_rl, structured_nodes,
        prior_centers,
    ):
        """Reverse scan with a fixed-budget batch of structured proposals."""
        n = schedule.shape[0]

        def body(carry, k):
            rng_c, Y = carry
            noise_scale = schedule[k]

            def active_step(active_carry):
                active_rng, active_y = active_carry
                active_rng, active_y, step_info = self._reverse_step(
                    state, active_rng, active_y, noise_scale, k, t0, U_rl,
                    structured_nodes=structured_nodes,
                    prior_centers=prior_centers,
                )
                diag = jnp.asarray([
                    step_info["proposal_gaussian_best_reward"],
                    step_info.get("proposal_rl_best_reward", jnp.nan),
                    step_info.get("proposal_atacom_best_reward", jnp.nan),
                    step_info["proposal_gaussian_weight"],
                    step_info.get("proposal_rl_weight", jnp.nan),
                    step_info.get("proposal_atacom_weight", jnp.nan),
                ], dtype=active_y.dtype)
                return (active_rng, active_y), diag

            (rng_c, Y), diag = jax.lax.cond(
                jnp.all(jnp.isfinite(noise_scale)),
                active_step,
                lambda inactive_carry: (
                    inactive_carry,
                    jnp.full((6,), jnp.nan, dtype=Y.dtype),
                ),
                (rng_c, Y),
            )
            return (rng_c, Y), diag

        (_, Y), diagnostics = jax.lax.scan(
            body, (rng, warm_start), jnp.arange(n)
        )
        valid = jnp.isfinite(diagnostics)
        totals = jnp.nansum(diagnostics, axis=0)
        counts = jnp.maximum(jnp.sum(valid, axis=0), 1)
        return Y, totals / counts

    def _rollout_node_candidates(self, state, nodes, t0):
        dense = self._node2u_batch(nodes)
        if self._augmented:
            rewards = self._rollout_fn(
                state, dense, t0, self.aug_lambda, self.aug_rho
            )
        else:
            rewards = self._rollout_fn(state, dense, t0)
        return dense, rewards

    def _node2u_batch(self, nodes):
        """Expand nodes while preserving P4's certified endpoint bit-exactly."""
        dense = self.spline.node2u_batch(nodes)
        if self.receding_shift_mode == "certified_terminal_hold":
            dense = dense.at[:, 0].set(nodes[:, 0])
        return dense

    def _node2u(self, nodes):
        dense = self.spline.node2u(nodes)
        if self.receding_shift_mode == "certified_terminal_hold":
            dense = dense.at[0].set(nodes[0])
        return dense

    def _map_candidate_scores_unmaterialized(self, score_fn, candidates):
        """Map candidate certificates without introducing a host barrier."""
        batch_size = self.candidate_rollout_batch_size
        if batch_size is None:
            return jax.vmap(score_fn)(candidates)
        if batch_size == 1:
            return jax.lax.map(score_fn, candidates)
        return jax.lax.map(
                score_fn, candidates, batch_size=batch_size
            )

    def _materialize_candidate_scores(self, mapped):
        if self.stepwise_acceptance:
            mapped = jax.tree.map(
                lambda value: value.block_until_ready(), mapped
            )
        return mapped

    def _map_candidate_scores(self, score_fn, candidates):
        """Map expensive candidate certificates with the rollout memory policy."""
        return self._materialize_candidate_scores(
            self._map_candidate_scores_unmaterialized(score_fn, candidates)
        )

    def _normal_candidate_scores(self, state, candidates):
        aug_lambda = self.aug_lambda if self._augmented else 0.0
        aug_rho = self.aug_rho if self._augmented else 0.0
        return self._map_candidate_scores_unmaterialized(
            lambda us: self.score_risk_fn(
                state, us, aug_lambda, aug_rho
            ),
            candidates,
        )

    def _emergency_candidate_scores(self, state, candidates):
        aug_lambda = self.aug_lambda if self._augmented else 0.0
        aug_rho = self.aug_rho if self._augmented else 0.0
        return self._map_candidate_scores_unmaterialized(
            lambda us: self.emergency_score_risk_fn(
                state, us, aug_lambda, aug_rho
            ),
            candidates,
        )

    def _accept_refinement(
        self, state, fallback, refined, t0, atacom_incumbent=None,
        emergency=None, fallback_mode=None, learned_abstain=False,
    ):
        normal_transition_ready = jnp.asarray(True)
        emergency_scored = jnp.asarray(emergency is not None)
        emergency_validation_candidate_count = jnp.float32(0.0)
        emergency_candidates = None
        if emergency is not None:
            emergency_candidates = jnp.asarray(emergency)
            if emergency_candidates.ndim == 2:
                emergency_candidates = emergency_candidates[None]
            if emergency_candidates.ndim != 3 or emergency_candidates.shape[0] < 1:
                raise ValueError(
                    "emergency candidate bank must have shape (K, Hnode, action_dim)"
                )
        if self._execution_context is not None:
            fallback_mode = self._incumbent_execution_mode(state, fallback_mode)
            original_fallback_mode = fallback_mode
            # A task-owned dwell is a hybrid mode-transition constraint, not
            # merely a filter on one particular recovery proposal.  While an
            # actually committed UNLOAD has not completed that dwell, no
            # Gaussian, learned, or ATACOM NORMAL candidate may bypass it.
            # The shifted UNLOAD incumbent and explicit emergency bank remain
            # eligible and are revalidated in their actual execution mode.
            if self._normal_recovery_ready_fn is not None:
                normal_transition_ready = jnp.where(
                    fallback_mode == 1,
                    self._normal_recovery_ready_fn(state),
                    jnp.asarray(True),
                )
            state = self._normal_rollout_state(state)
            if emergency is None:
                raise ValueError("execution context requires an explicit emergency candidate")
        fallback_source = (
            "prior_risk_incumbent"
            if self.prior_fallback_mode == "receding_incumbent"
            else "prior_risk_rl"
        )
        if not self.prior_acceptance:
            # This ablation always deploys the refined proposal.  Evaluating
            # score/risk cannot affect the selected action and would compile two
            # additional MJX acceptance executables on CPU.  Mark unavailable
            # counterfactual diagnostics explicitly instead of paying that cost.
            nan = jnp.asarray(jnp.nan, dtype=refined.dtype)
            risks = jnp.full((2, 4), nan, dtype=refined.dtype)
            info = {
                "prior_accepted": jnp.asarray(1.0, dtype=refined.dtype),
                "prior_predicted_improvement": nan,
                "prior_score_fallback": nan,
                "prior_score_refined": nan,
                "prior_risk_refined": risks[1],
                "prior_risk_ok": nan,
                "prior_force_veto": jnp.asarray(0.0, dtype=refined.dtype),
            }
            info[fallback_source] = risks[0]
            return refined, info

        # Keep fallback and refinement as separate candidates even at the cold
        # start.  A cold fallback is not trusted implicitly, but a task-owned
        # model-based certificate may validate it just like any other current-
        # state candidate (including a nonzero task initializer).
        has_atacom_incumbent = atacom_incumbent is not None
        base_candidates = (
            [fallback, atacom_incumbent, refined]
            if has_atacom_incumbent
            else [fallback, refined]
        )
        base_candidate_array = jnp.stack(base_candidates, axis=0)
        candidates = (
            jnp.concatenate([base_candidate_array, emergency_candidates], axis=0)
            if emergency_candidates is not None else base_candidate_array
        )
        refined_idx = 2 if has_atacom_incumbent else 1
        emergency_start = len(base_candidates)
        dense = self._node2u_batch(candidates)
        if self.score_risk_fn is not None:
            aug_lambda = self.aug_lambda if self._augmented else 0.0
            aug_rho = self.aug_rho if self._augmented else 0.0
            score_one = lambda us: self.score_risk_fn(
                state, us, aug_lambda, aug_rho
            )
            if self._execution_context is not None:
                # Revalidate the shifted incumbent in its actual committed
                # mode. Every newly refined/ATACOM proposal is NORMAL.
                emergency_state = self._execution_context["prepare_state"](
                    state, jnp.int32(1)
                )
                # prepare_state(UNLOAD) seals entry_prepared independently
                # of entry_geometry_valid; an evaluated-invalid entry is also
                # immutable. Repeated UNLOAD preparation by the nominal
                # scorer must preserve the execution-frame entry even while
                # committed mode is still NORMAL. Missing sealed context is
                # unknown, never permission to recapture in model geometry.
                emergency_score_one = lambda us: self.emergency_score_risk_fn(
                    emergency_state, us, aug_lambda, aug_rho
                )
                base_count = len(base_candidates)
                if self.stepwise_acceptance:
                    base_scores, base_risks = self._materialize_candidate_scores(
                        self._normal_candidate_scores_jit(
                            state, dense[:base_count]
                        )
                    )
                    if (
                        self.lazy_emergency_scoring
                        and self._host_emergency_score_risk_fn is not None
                        and not self._host_emergency_prewarmed
                    ):
                        # Compile the emergency transition/score signatures
                        # while the process still holds only its first normal
                        # planning executable. Deferring this first compilation
                        # until a late viability loss can exceed a memory-limited
                        # CPU container after many receding calls. One candidate
                        # has the same dynamic shapes as the complete bank; its
                        # value is discarded and cannot affect selection.
                        warmup = self._host_emergency_score_risk_fn(
                            emergency_state, dense[emergency_start],
                            self.aug_lambda if self._augmented else 0.0,
                            self.aug_rho if self._augmented else 0.0,
                        )
                        self._materialize_candidate_scores(warmup)
                        self._host_emergency_prewarmed = True
                else:
                    base_scores, base_risks = self._map_candidate_scores(
                        score_one, dense[:base_count]
                    )
                # Reuse one full-H normal graph and one short task-owned
                # emergency graph. A mode-specific fallback must not compile a second
                # copy of the ordinary full-H MJX rollout.
                if self.stepwise_acceptance:
                    score_emergency = True
                    if (
                        self.lazy_emergency_scoring
                        and self.reliability_model is None
                    ):
                        # Emergency safety is relevant only if an emergency can
                        # actually be selected.  With a NORMAL incumbent and at
                        # least one absolutely safe performance candidate, the
                        # lexicographic selector cannot choose UNLOAD unless the
                        # task explicitly overrides.  Avoid compiling a third
                        # large MJX executable on memory-limited CPU hosts.
                        base_safe = jax.vmap(self.risk_safe_fn)(base_risks)
                        any_base_safe = bool(
                            np.asarray(jax.device_get(jnp.any(base_safe)))
                        )
                        fallback_is_unload = int(
                            np.asarray(jax.device_get(fallback_mode))
                        ) == 1
                        task_override = (
                            bool(np.asarray(jax.device_get(
                                self.emergency_override_fn(state)
                            )))
                            if self.emergency_override_fn is not None
                            else False
                        )
                        score_emergency = (
                            fallback_is_unload
                            or task_override
                            or not any_base_safe
                        )
                    if score_emergency:
                        unload_candidates = jnp.concatenate([
                            dense[0:1], dense[emergency_start:]
                        ], axis=0)
                        if self._host_emergency_score_risk_fn is not None:
                            # The task advances functional state through its
                            # existing execution-step cache, then scores the
                            # realized transition without nesting another full
                            # MJX step graph. Keep this host loop sequential:
                            # emergency candidate banks are small and evaluated
                            # only when the safety-first selector can use them.
                            progressive = (
                                self.lazy_emergency_scoring
                                and self.reliability_model is None
                            )
                            if progressive:
                                # Emergency candidates are ordered by the task,
                                # with its primary recovery first.  Safety is
                                # lexicographic here: once a recovery has a
                                # current-state certificate, evaluating the
                                # remaining bank merely optimizes emergency
                                # reward and can retain/compile many complete
                                # MJX states on a memory-limited CPU host.
                                #
                                # Revalidate a committed shifted UNLOAD
                                # incumbent first.  A NORMAL fallback is never
                                # silently reinterpreted as UNLOAD; start with
                                # the first explicitly task-owned candidate.
                                unload_scores = jnp.full(
                                    (unload_candidates.shape[0],), -jnp.inf,
                                    dtype=base_scores.dtype,
                                ).at[0].set(base_scores[0])
                                unload_risks = jnp.full(
                                    (
                                        unload_candidates.shape[0],
                                        base_risks.shape[-1],
                                    ),
                                    jnp.inf,
                                    dtype=base_risks.dtype,
                                ).at[0].set(base_risks[0])
                                evaluation_indices = []
                                if fallback_is_unload:
                                    evaluation_indices.append(0)
                                evaluation_indices.extend(
                                    range(1, int(unload_candidates.shape[0]))
                                )
                                evaluated = 0
                                for candidate_index in evaluation_indices:
                                    output = self._host_emergency_score_risk_fn(
                                        emergency_state,
                                        unload_candidates[candidate_index],
                                        self.aug_lambda if self._augmented else 0.0,
                                        self.aug_rho if self._augmented else 0.0,
                                    )
                                    score, risk = self._materialize_candidate_scores(
                                        output
                                    )
                                    if (
                                        self.normal_recovery_plan_fn is not None
                                        or self.normal_recovery_plans_fn is not None
                                    ):
                                        unload_candidate = unload_candidates[
                                            candidate_index
                                        ]
                                        successor = emergency_state
                                        recovery_ready = False
                                        recovery_entry_index = 0
                                        # Certify the complete task-owned
                                        # UNLOAD dwell before evaluating its
                                        # NORMAL successor. The host loop is
                                        # bounded by the already fixed action
                                        # horizon and uses the same contextual
                                        # model step as actual deployment.
                                        dwell_limit = (
                                            1
                                            if self._normal_recovery_ready_fn is None
                                            else int(unload_candidate.shape[0])
                                        )
                                        for dwell_index in range(dwell_limit):
                                            successor = (
                                                self._model_execution_context[
                                                    "step"
                                                ](
                                                    successor,
                                                    unload_candidate[dwell_index],
                                                    jnp.int32(1),
                                                )
                                            )
                                            recovery_entry_index = dwell_index
                                            recovery_ready = (
                                                True
                                                if self._normal_recovery_ready_fn is None
                                                else bool(np.asarray(jax.device_get(
                                                    self._normal_recovery_ready_fn(
                                                        successor
                                                    )
                                                )))
                                            )
                                            if recovery_ready:
                                                break
                                        # The task recovery uses row zero as
                                        # its exact entry-side posture. Align
                                        # that row with the last simulated
                                        # UNLOAD interval while retaining the
                                        # static horizon shape.
                                        recovery_reference_dense = (
                                            unload_candidate.at[0].set(
                                                unload_candidate[
                                                    recovery_entry_index
                                                ]
                                            )
                                        )
                                        # Task recovery hooks operate in the
                                        # same node space as the receding plan.
                                        # ``unload_candidate`` is already a
                                        # dense sequence here; passing it
                                        # through directly makes a node-space
                                        # pulse collapse to one 20 ms action
                                        # during pre-certification, while the
                                        # committed exit later expands that
                                        # pulse through the spline. Refit once
                                        # and pin the exact entry action so
                                        # both paths certify identical recovery
                                        # semantics.
                                        recovery_reference = self.spline.u2node(
                                            recovery_reference_dense
                                        ).at[0].set(recovery_reference_dense[0])
                                        recoveries = (
                                            self._task_normal_recovery_bank(
                                                successor,
                                                recovery_reference,
                                                0.0,
                                                unload_candidates.dtype,
                                            )
                                            if recovery_ready else None
                                        )
                                        # Reuse the already compiled normal
                                        # scorer; do not add a second MJX graph.
                                        if recoveries is not None:
                                            recovery_state = (
                                                self._model_execution_context[
                                                    "prepare_state"
                                                ](successor, jnp.int32(0))
                                            )
                                            # The task orders least-invasive
                                            # recovery first. Evaluate one at
                                            # a time and stop at the first
                                            # candidate whose NORMAL horizon,
                                            # combined with this UNLOAD dwell,
                                            # is certified. This is the same
                                            # bank used after UNLOAD commits.
                                            first_combined_risk = None
                                            for recovery in recoveries:
                                                recovery_dense = self._node2u(
                                                    recovery
                                                )
                                                recovery_batch = jnp.broadcast_to(
                                                    recovery_dense,
                                                    (
                                                        base_count,
                                                        *recovery_dense.shape,
                                                    ),
                                                )
                                                _, recovery_risks = (
                                                    self._materialize_candidate_scores(
                                                        self._normal_candidate_scores_jit(
                                                            recovery_state,
                                                            recovery_batch,
                                                        )
                                                    )
                                                )
                                                combined_risk = jnp.maximum(
                                                    risk, recovery_risks[0]
                                                )
                                                if first_combined_risk is None:
                                                    first_combined_risk = combined_risk
                                                if bool(np.asarray(jax.device_get(
                                                    self.risk_safe_fn(combined_risk)
                                                ))):
                                                    risk = combined_risk
                                                    break
                                            else:
                                                # Preserve finite diagnostics
                                                # from the least-intervening
                                                # failed recovery while still
                                                # marking the emergency path
                                                # unsafe.
                                                risk = first_combined_risk
                                        elif self._normal_recovery_ready_fn is not None:
                                            # A finite emergency prefix that
                                            # cannot reach the task-owned exit
                                            # within the fixed horizon is not
                                            # recursively viable.
                                            risk = jnp.full_like(risk, jnp.inf)
                                    unload_scores = unload_scores.at[
                                        candidate_index
                                    ].set(score)
                                    unload_risks = unload_risks.at[
                                        candidate_index
                                    ].set(risk)
                                    evaluated += 1
                                    safe = bool(np.asarray(jax.device_get(
                                        self.risk_safe_fn(risk)
                                    )))
                                    # A task override requests an explicit
                                    # task-owned plan, so a safe shifted
                                    # incumbent at index zero does not satisfy
                                    # it.  Any safe task-owned recovery does.
                                    if safe and (
                                        candidate_index > 0 or not task_override
                                    ):
                                        break
                                emergency_validation_candidate_count = (
                                    jnp.float32(evaluated)
                                )
                                unload_scores, unload_risks = (
                                    self._materialize_candidate_scores(
                                        (unload_scores, unload_risks)
                                    )
                                )
                            else:
                                unload_outputs = []
                                for candidate in unload_candidates:
                                    output = self._host_emergency_score_risk_fn(
                                        emergency_state, candidate,
                                        self.aug_lambda if self._augmented else 0.0,
                                        self.aug_rho if self._augmented else 0.0,
                                    )
                                    # JAX dispatch is asynchronous.  A Python
                                    # list comprehension alone enqueues the
                                    # complete emergency bank and retains every
                                    # intermediate MJX state until the final
                                    # stack is blocked, defeating the intended
                                    # low-memory host loop. Materialize each
                                    # candidate before launching the next one.
                                    unload_outputs.append(
                                        self._materialize_candidate_scores(output)
                                    )
                                unload_scores = jnp.stack([
                                    output[0] for output in unload_outputs
                                ])
                                unload_risks = jnp.stack([
                                    output[1] for output in unload_outputs
                                ])
                                unload_scores, unload_risks = (
                                    self._materialize_candidate_scores(
                                        (unload_scores, unload_risks)
                                    )
                                )
                                emergency_validation_candidate_count = jnp.float32(
                                    unload_candidates.shape[0]
                                )
                        else:
                            unload_scores, unload_risks = self._materialize_candidate_scores(
                                self._emergency_candidate_scores_jit(
                                    emergency_state, unload_candidates,
                                )
                            )
                            emergency_validation_candidate_count = jnp.float32(
                                unload_candidates.shape[0]
                            )
                    else:
                        unavailable_count = 1 + emergency_candidates.shape[0]
                        unload_scores = jnp.concatenate([
                            base_scores[0:1],
                            jnp.full(
                                (unavailable_count - 1,), -jnp.inf,
                                dtype=base_scores.dtype,
                            ),
                        ])
                        unload_risks = jnp.concatenate([
                            base_risks[0:1],
                            jnp.full(
                                (unavailable_count - 1, base_risks.shape[-1]),
                                jnp.inf, dtype=base_risks.dtype,
                            ),
                        ], axis=0)
                        emergency_scored = jnp.asarray(False)
                else:
                    unload_candidates = jnp.concatenate([
                        dense[0:1], dense[emergency_start:]
                    ], axis=0)
                    unload_scores, unload_risks = self._map_candidate_scores(
                        emergency_score_one,
                        unload_candidates,
                    )
                    emergency_validation_candidate_count = jnp.float32(
                        unload_candidates.shape[0]
                    )
                scores = base_scores.at[0].set(jnp.where(
                    fallback_mode == 1, unload_scores[0], base_scores[0]
                ))
                risks = base_risks.at[0].set(jnp.where(
                    fallback_mode == 1, unload_risks[0], base_risks[0]
                ))
                scores = jnp.concatenate([scores, unload_scores[1:]])
                risks = jnp.concatenate([risks, unload_risks[1:]])
                known_mode = (fallback_mode == 0) | (fallback_mode == 1)
                # A corrupted dynamic mode cannot accidentally become an
                # ordinary certified fallback inside an outer jit.
                risks = jnp.where(known_mode, risks, jnp.inf)
                scores = jnp.where(known_mode, scores, jnp.nan)
            elif emergency_candidates is not None and self.emergency_score_risk_fn is not None:
                # Performance candidates are certified against the task's
                # robust inner set.  The explicitly task-owned emergency has a
                # separate physical-set contract, so do not accidentally mark
                # a safe unload as unrecoverable merely because the measured
                # state is already inside the robustness buffer band.
                if self.stepwise_acceptance:
                    base_scores, base_risks = self._materialize_candidate_scores(
                        self._normal_candidate_scores_jit(
                            state, dense[:emergency_start]
                        )
                    )
                else:
                    base_scores, base_risks = self._map_candidate_scores(
                        score_one, dense[:emergency_start]
                    )
                emergency_scores, emergency_risks = self._map_candidate_scores(
                    lambda us: self.emergency_score_risk_fn(
                        state, us, aug_lambda, aug_rho
                    ),
                    dense[emergency_start:],
                )
                scores = jnp.concatenate(
                    [base_scores, emergency_scores], axis=0
                )
                risks = jnp.concatenate(
                    [base_risks, emergency_risks], axis=0
                )
            else:
                if self.stepwise_acceptance:
                    scores, risks = self._materialize_candidate_scores(
                        self._normal_candidate_scores_jit(state, dense)
                    )
                else:
                    scores, risks = self._map_candidate_scores(
                        score_one, dense
                    )
        else:
            _, rewards = self._rollout_node_candidates(state, candidates, t0)
            scores = jnp.mean(rewards, axis=-1)
            risks = None
        learned_risks = None
        support_scores = None
        if self.reliability_model is not None:
            if self.reliability_sequence_feature_fn is not None:
                reliability_features = jax.vmap(
                    lambda us: self.reliability_sequence_feature_fn(state, us)
                )(dense)
            else:
                reliability_features = jax.vmap(
                    lambda us: self.reliability_feature_fn(state, us[0])
                )(dense)
            learned_risks = jax.vmap(
                self.reliability_model.predict_upper
            )(reliability_features)
            support_fn = (
                self.reliability_model.support_score_state
                if self.reliability_support_mode == "state"
                else self.reliability_model.support_score
            )
            support_scores = jax.vmap(support_fn)(reliability_features)
            if self._execution_context is not None:
                # This model is calibrated for NORMAL candidates only. A
                # numeric action with a different execution mode is not the
                # same candidate; do not present its prediction as applicable.
                learned_risks = learned_risks.at[0].set(jnp.where(
                    fallback_mode == 1, 0.0, learned_risks[0]
                ))
                support_scores = support_scores.at[0].set(jnp.where(
                    fallback_mode == 1, 0.0, support_scores[0]
                ))
                if emergency_candidates is not None:
                    learned_risks = learned_risks.at[emergency_start:].set(0.0)
                    support_scores = support_scores.at[emergency_start:].set(0.0)

        atacom_selected = jnp.asarray(False)
        if has_atacom_incumbent:
            # At t=0 the zero-initialized receding plan has never been deployed
            # and is not a meaningful safety incumbent.  ATACOM supplies the
            # first model-constrained executable plan; later steps compare it
            # against the shifted plan that was actually deployed.
            cold_atacom = t0 <= 0.0
            atacom_better = scores[1] > (
                scores[0] + self.prior_improvement_epsilon
            )
            if risks is not None or self.risk_fn is not None:
                if risks is None:
                    risks = jax.vmap(
                        lambda us: self.risk_fn(state, us)
                    )(dense)
                atacom_better = atacom_better & jnp.all(
                    risks[1] <= risks[0] + self.prior_risk_tolerance
                ) & (risks[1, 0] <= 1e-8)
            if learned_risks is not None:
                atacom_better = (
                    atacom_better
                    & (support_scores[1] <= 1.0)
                    & jnp.all(
                        learned_risks[1]
                        <= learned_risks[0]
                        + self.reliability_risk_tolerance
                    )
                    & (
                        learned_risks[1, 0]
                        <= self.reliability_force_limit
                    )
                    & (
                        learned_risks[1, 2]
                        <= self.reliability_deformation_limit
                    )
                )
            if self._execution_context is not None:
                # A short emergency certificate is not a performance
                # baseline for the normal H-step ATACOM candidate.
                atacom_normal_safe = (
                    self.risk_safe_fn(risks[1]) & jnp.isfinite(scores[1])
                    & jnp.all(jnp.isfinite(risks[1]))
                )
                if learned_risks is not None:
                    if self.reliability_hard_limits is not None:
                        own_bound = jnp.all(
                            learned_risks[1] <= self.reliability_hard_limits
                        )
                    else:
                        own_bound = (
                            (learned_risks[1, 0] <= self.reliability_force_limit)
                            & (learned_risks[1, 2] <= self.reliability_deformation_limit)
                        )
                    supported = support_scores[1] <= 1.0
                    atacom_normal_safe &= self._reliability_gate_mask(
                        supported, own_bound
                    )
                atacom_better = jnp.where(
                    fallback_mode == 1, atacom_normal_safe, atacom_better
                )
            atacom_selected = (
                jnp.asarray(True)
                if self.prior_atacom_default
                else (cold_atacom | atacom_better)
            )
            if self._execution_context is not None:
                atacom_selected &= normal_transition_ready
            fallback = jnp.where(atacom_selected, candidates[1], candidates[0])
            if self._execution_context is not None:
                fallback_mode = jnp.where(atacom_selected, jnp.int32(0), fallback_mode)
            fallback_score = jnp.where(
                atacom_selected, scores[1], scores[0]
            )
            if risks is not None:
                fallback_risk = jnp.where(
                    atacom_selected, risks[1], risks[0]
                )
            if learned_risks is not None:
                fallback_learned_risk = jnp.where(
                    atacom_selected, learned_risks[1], learned_risks[0]
                )
                fallback_support = jnp.where(
                    atacom_selected, support_scores[1], support_scores[0]
                )
        else:
            fallback_score = scores[0]
            if risks is not None:
                fallback_risk = risks[0]
            if learned_risks is not None:
                fallback_learned_risk = learned_risks[0]
                fallback_support = support_scores[0]

        improvement = scores[refined_idx] - fallback_score
        if self._execution_context is not None:
            # Finite storage placeholder only. Applicability-aware reporting
            # must exclude this row; it is not a measured zero improvement.
            improvement = jnp.where(fallback_mode == 0, improvement, 0.0)
        predicted_ok = improvement > self.prior_improvement_epsilon

        if risks is not None or self.risk_fn is not None:
            if risks is None:
                risks = jax.vmap(lambda us: self.risk_fn(state, us))(dense)
            tolerance = (
                jnp.zeros_like(self.prior_risk_tolerance)
                if (
                    has_atacom_incumbent
                    and self.prior_atacom_strict_risk
                )
                else self.prior_risk_tolerance
            )
            if not has_atacom_incumbent:
                fallback_risk = risks[0]
            risk_ok = (
                self.risk_compare_fn(
                    risks[refined_idx], fallback_risk, tolerance
                )
                if self.risk_compare_fn is not None
                else jnp.all(
                    risks[refined_idx] <= fallback_risk + tolerance
                )
            )
            # The task, rather than the generic backend, owns the semantics of
            # its risk vector.  Legacy tasks without that hook retain the
            # original force-component veto.
            if self.risk_safe_fn is not None:
                refined_safe = self.risk_safe_fn(risks[refined_idx])
                fallback_safe = self.risk_safe_fn(fallback_risk)
                emergency_safe_all = (
                    jax.vmap(self.risk_safe_fn)(risks[emergency_start:])
                    if emergency_candidates is not None
                    else jnp.zeros((0,), dtype=jnp.bool_)
                )
            else:
                refined_safe = risks[refined_idx, 0] <= 1e-8
                fallback_safe = fallback_risk[0] <= 1e-8
                emergency_safe_all = (
                    risks[emergency_start:, 0] <= 1e-8
                    if emergency_candidates is not None
                    else jnp.zeros((0,), dtype=jnp.bool_)
                )
            hard_force_ok = refined_safe
        else:
            risks = jnp.zeros(
                (candidates.shape[0], 0), dtype=refined.dtype
            )
            fallback_risk = risks[0]
            risk_ok = jnp.asarray(True)
            hard_force_ok = jnp.asarray(True)
            refined_safe = jnp.asarray(True)
            fallback_safe = jnp.asarray(True)
            emergency_safe_all = (
                jnp.ones((emergency_candidates.shape[0],), dtype=jnp.bool_)
                if emergency_candidates is not None
                else jnp.zeros((0,), dtype=jnp.bool_)
            )

        if learned_risks is not None:
            # A NORMAL proposal considered while the measured incumbent is a
            # committed task-owned UNLOAD is a hybrid recovery transition.
            # The learned model is fitted only on ordinary NORMAL candidate
            # windows; it must abstain here and leave the complete task-owned
            # model rollout as the authority.  Without this distinction a
            # supported-but-miscalibrated learned balance head can veto every
            # recovery forever, even though UNLOAD itself is only a temporary
            # safe mode.
            recovery_transition = (
                ((fallback_mode == 1) | jnp.asarray(learned_abstain, jnp.bool_))
                if self._execution_context is not None
                else jnp.asarray(False)
            )
            learned_support_ok = support_scores[refined_idx] <= 1.0
            learned_tolerance = (
                jnp.zeros_like(self.reliability_risk_tolerance)
                if (
                    has_atacom_incumbent
                    and self.prior_atacom_strict_risk
                )
                else self.reliability_risk_tolerance
            )
            learned_risk_ok = jnp.all(
                learned_risks[refined_idx]
                <= fallback_learned_risk + learned_tolerance
            )
            if self._execution_context is not None:
                learned_risk_ok = recovery_transition | learned_risk_ok
            if self.reliability_hard_limits is not None:
                learned_hard_ok_all = jnp.all(
                    learned_risks <= self.reliability_hard_limits,
                    axis=-1,
                )
                fallback_learned_hard_ok = jnp.all(
                    fallback_learned_risk <= self.reliability_hard_limits
                )
            else:
                # Exact legacy surface-scanning interpretation.
                learned_hard_ok_all = (
                    (
                        learned_risks[:, 0]
                        <= self.reliability_force_limit
                    )
                    & (
                        learned_risks[:, 2]
                        <= self.reliability_deformation_limit
                    )
                )
                fallback_learned_hard_ok = (
                    (fallback_learned_risk[0] <= self.reliability_force_limit)
                    & (
                        fallback_learned_risk[2]
                        <= self.reliability_deformation_limit
                    )
                )
            learned_hard_ok = learned_hard_ok_all[refined_idx]
            # A calibrated learned bound is authoritative only inside its
            # support.  Some tasks have a complete task-owned model-based
            # sequence certificate for deployment-family OOD states.  In that
            # explicitly selected mode the learned model abstains outside its
            # calibration domain and the existing model-based certificate owns
            # acceptance; it does not turn "unknown" into "unsafe forever".
            # The default remains the conservative legacy veto used by tasks
            # without that certificate.
            learned_acceptance_ok = self._reliability_acceptance_mask(
                learned_support_ok,
                learned_risk_ok,
                learned_hard_ok,
            )
            learned_acceptance_ok = recovery_transition | learned_acceptance_ok
            risk_ok = risk_ok & learned_acceptance_ok
            # A learned realization bound is part of final revalidation, not
            # merely a test of whether refinement improves over the shifted
            # incumbent.  Otherwise an unreliable incumbent remains the
            # fallback (and the first replan bypasses the learned hard limit),
            # even when both performance candidates exceed the calibrated
            # bound.  The task-owned emergency remains the last-resort action.
            refined_safe = (
                refined_safe & learned_acceptance_ok
            )
            fallback_support_ok = fallback_support <= 1.0
            fallback_learned_ok = self._reliability_gate_mask(
                fallback_support_ok, fallback_learned_hard_ok
            )
            if self._execution_context is not None:
                fallback_learned_ok = (fallback_mode == 1) | fallback_learned_ok
            fallback_safe = (
                fallback_safe & fallback_learned_ok
            )
            hard_force_ok = hard_force_ok & (
                recovery_transition
                | self._reliability_gate_mask(
                    learned_support_ok, learned_hard_ok
                )
            )

        # Safety is lexicographic over performance.  A shifted incumbent is a
        # valid fallback only after it has been re-evaluated from the *current*
        # measured state.  If refinement alone is safe, take it even when it is
        # not a score improvement.  If neither performance candidate is safe,
        # deploy the task-owned emergency horizon (its own predicted safety is
        # reported because an existing violation may need time to dissipate).
        incumbent_valid_safe = fallback_safe
        emergency_incumbent_active = (
            fallback_mode == 1
            if self._execution_context is not None
            else (
                self.emergency_active_fn(fallback)
                if self.emergency_active_fn is not None
                else jnp.asarray(False)
            )
        )
        performance_incumbent_valid = (
            incumbent_valid_safe & (~emergency_incumbent_active)
        )
        accepted = refined_safe & (
            (~performance_incumbent_valid)
            | (predicted_ok & risk_ok & hard_force_ok)
        )
        if self._execution_context is not None:
            accepted &= normal_transition_ready
        if emergency_candidates is not None:
            emergency_index = jnp.argmax(jnp.where(
                emergency_safe_all,
                scores[emergency_start:],
                -jnp.inf,
            ))
            selected_emergency = emergency_candidates[emergency_index]
            selected_emergency_risk = risks[emergency_start + emergency_index]
            emergency_safe = jnp.any(emergency_safe_all)
        else:
            selected_emergency = fallback
            selected_emergency_risk = jnp.full_like(fallback_risk, jnp.inf)
            emergency_safe = jnp.asarray(False)
        task_emergency_override = (
            self.emergency_override_fn(state)
            if (
                emergency_candidates is not None
                and self.emergency_override_fn is not None
            )
            else jnp.asarray(False)
        )
        accepted = accepted & (~task_emergency_override)
        emergency_selected = task_emergency_override | (
            (~refined_safe) & (~incumbent_valid_safe)
            if emergency_candidates is not None
            else jnp.asarray(False)
        )
        selected = jnp.where(accepted, refined, fallback)
        if emergency_candidates is not None:
            selected = jnp.where(
                emergency_selected, selected_emergency, selected
            )
        selected_revalidated_safe = jnp.where(
            emergency_selected,
            emergency_safe,
            jnp.where(accepted, refined_safe, incumbent_valid_safe),
        )
        emergency_unrecoverable = emergency_selected & (~emergency_safe)
        incumbent_candidate_selected = (
            (~accepted) & (~emergency_selected) & (~atacom_selected)
        )
        info = {
            "prior_accepted": accepted.astype(jnp.float32),
            "prior_predicted_improvement": improvement,
            "prior_score_fallback": fallback_score,
            "prior_score_refined": scores[refined_idx],
            "prior_risk_refined": risks[refined_idx],
            "prior_risk_ok": risk_ok.astype(jnp.float32),
            "prior_force_veto": (~hard_force_ok).astype(jnp.float32),
            "atacom_incumbent_selected": atacom_selected.astype(jnp.float32),
            "incumbent_candidate_selected": (
                incumbent_candidate_selected.astype(jnp.float32)
            ),
            "incumbent_revalidated_safe": incumbent_valid_safe.astype(jnp.float32),
            "refined_revalidated_safe": refined_safe.astype(jnp.float32),
            "emergency_selected": emergency_selected.astype(jnp.float32),
            "emergency_task_override": task_emergency_override.astype(
                jnp.float32
            ),
            "emergency_revalidated_safe": emergency_safe.astype(jnp.float32),
            "emergency_incumbent_active": emergency_incumbent_active.astype(
                jnp.float32
            ),
            "selected_revalidated_safe": selected_revalidated_safe.astype(
                jnp.float32
            ),
            "emergency_unrecoverable": emergency_unrecoverable.astype(
                jnp.float32
            ),
        }
        if self._execution_context is not None:
            info["execution_mode"] = jnp.where(
                emergency_selected, jnp.int32(1),
                jnp.where(accepted, jnp.int32(0), fallback_mode),
            )
            info["emergency_validation_candidate_count"] = jnp.where(
                emergency_scored,
                emergency_validation_candidate_count,
                jnp.float32(0.0),
            )
            info["emergency_validation_applicable"] = emergency_scored.astype(
                jnp.float32
            )
            info["prior_comparison_applicable"] = (fallback_mode == 0).astype(jnp.float32)
            info["emergency_fallback_recovery"] = (
                (original_fallback_mode == 1) & (info["execution_mode"] == 0)
            ).astype(jnp.float32)
            info["reliability_incumbent_applicable"] = (
                (fallback_mode == 0) & (self.reliability_model is not None)
            ).astype(jnp.float32)
            info["normal_transition_dwell_ready"] = (
                normal_transition_ready.astype(jnp.float32)
            )
        info[fallback_source] = fallback_risk
        if has_atacom_incumbent:
            info["prior_score_atacom"] = scores[1]
            info["prior_risk_atacom"] = risks[1]
        if emergency_candidates is not None:
            info["prior_risk_emergency"] = selected_emergency_risk
        if learned_risks is not None:
            info["reliability_risk_incumbent"] = fallback_learned_risk
            info["reliability_risk_refined"] = learned_risks[refined_idx]
            info["reliability_support_incumbent"] = fallback_support
            info["reliability_support_refined"] = support_scores[refined_idx]
            info["reliability_abstained"] = (
                recovery_transition
                | (~jnp.asarray(self.reliability_gate_authoritative))
                | (
                    (support_scores[refined_idx] > 1.0)
                    & (self.reliability_ood_policy == "model_based")
                )
            ).astype(jnp.float32)
            info["reliability_gate_authoritative"] = jnp.asarray(
                self.reliability_gate_authoritative, jnp.float32
            )
        return selected, info

    def _select_additive_prior(self, state, candidates):
        """Rank expert horizons by the same task certificate as deployment."""
        if self._candidate_projection_jit is not None:
            candidates = jax.vmap(
                lambda nodes: self._candidate_projection_jit(state, nodes)
            )(candidates)
        dense = self._node2u_batch(candidates)
        scores, risks = jax.vmap(lambda us: self.score_risk_fn(
            state, us, self.aug_lambda if self._augmented else 0.0,
            self.aug_rho if self._augmented else 0.0,
        ))(dense)
        safe = jax.vmap(self.risk_safe_fn)(risks)
        if self.reliability_model is not None:
            features = jax.vmap(lambda us: (
                self.reliability_sequence_feature_fn(state, us)
                if self.reliability_sequence_feature_fn is not None
                else self.reliability_feature_fn(state, us[0])
            ))(dense)
            learned = self.reliability_model.predict_upper(features)
            support_fn = (
                self.reliability_model.support_score_state
                if self.reliability_support_mode == "state"
                else self.reliability_model.support_score
            )
            supported = support_fn(features) <= 1.0
            bounded = (
                jnp.all(learned <= self.reliability_hard_limits, axis=-1)
                if self.reliability_hard_limits is not None
                else ((learned[:, 0] <= self.reliability_force_limit)
                      & (learned[:, 2] <= self.reliability_deformation_limit))
            )
            safe &= self._reliability_gate_mask(supported, bounded)
        safe &= jnp.isfinite(scores) & jnp.all(jnp.isfinite(risks), axis=-1)
        index = jnp.argmax(jnp.where(safe, scores, -jnp.inf))
        return candidates[index], jnp.any(safe), scores[index]

    def _reliability_gate_mask(self, supported, bounded):
        """Return the acceptance mask for learned confidence constraints.

        ``model_based`` is an abstaining policy outside calibration and while a
        checkpoint is still a development artifact.  In that state the
        learned model remains observable through diagnostics but cannot veto a
        candidate that passed the task-owned physical certificate.  The
        explicit ``veto`` policy remains conservative and authoritative.
        """
        if self.reliability_ood_policy == "veto":
            return supported & bounded
        if not self.reliability_gate_authoritative:
            return jnp.ones_like(supported, dtype=bool)
        return (~supported) | bounded

    def _reliability_acceptance_mask(self, supported, risk_ok, hard_ok):
        if self.reliability_ood_policy == "veto":
            return supported & risk_ok & hard_ok
        if not self.reliability_gate_authoritative:
            return jnp.ones_like(supported, dtype=bool)
        return (~supported) | (risk_ok & hard_ok)

    def _task_normal_recovery_bank(
        self, state, recovery_incumbent, t0, dtype,
    ):
        """Normalize the task's legacy plan or ordered plan bank."""
        recovery = (
            self.normal_recovery_plans_fn(state, recovery_incumbent, t0)
            if self.normal_recovery_plans_fn is not None
            else self.normal_recovery_plan_fn(state, recovery_incumbent, t0)
        )
        if recovery is None:
            return None
        recoveries = jnp.asarray(recovery, dtype)
        if recoveries.ndim == recovery_incumbent.ndim:
            recoveries = recoveries[None]
        if (
            recoveries.ndim != recovery_incumbent.ndim + 1
            or recoveries.shape[1:] != recovery_incumbent.shape
        ):
            raise ValueError(
                "normal recovery plan bank shape "
                f"{recoveries.shape} is incompatible with incumbent shape "
                f"{recovery_incumbent.shape}"
            )
        return recoveries

    def _attempt_task_normal_recovery(
        self, state, recovery_incumbent, selected, info, t0, emergency,
        incumbent_mode=None,
    ):
        """Revalidate a task-owned NORMAL exit from a committed emergency.

        A shifted UNLOAD incumbent can remain physically safe while its fixed
        recovery posture drifts away from the normal controller's viable set.
        Tasks that own such a hybrid transition may expose one deterministic
        NORMAL proposal, or an ordered bank of bounded proposals, plus an
        optional task-state dwell guard.  Once that guard opens, the same
        full-horizon model certificate used for ordinary candidates must mark
        a proposal safe before the mode can change.  Ordered banks are checked
        one at a time and stop at the first certified member, so the task can
        put its least-intervening recovery first without materializing another
        large MJX candidate batch.
        """
        diagnostics = {
            "normal_recovery_applicable": jnp.float32(0.0),
            "normal_recovery_revalidated_safe": jnp.float32(0.0),
            "normal_recovery_selected": jnp.float32(0.0),
            "normal_recovery_candidate_count": jnp.float32(0.0),
            "normal_recovery_selected_index": jnp.int32(-1),
            "normal_recovery_last_risk": jnp.zeros((4,), jnp.float32),
        }
        recovery_mode = (
            info.get("execution_mode", jnp.int32(0))
            if incumbent_mode is None else jnp.asarray(incumbent_mode, jnp.int32)
        )
        if (
            self._execution_context is None
            or (
                self.normal_recovery_plan_fn is None
                and self.normal_recovery_plans_fn is None
            )
            or int(np.asarray(jax.device_get(recovery_mode))) != 1
        ):
            return selected, {**info, **diagnostics}
        if (
            self._normal_recovery_ready_fn is not None
            and not bool(np.asarray(jax.device_get(
                self._normal_recovery_ready_fn(state)
            )))
        ):
            return selected, {**info, **diagnostics}

        recoveries = self._task_normal_recovery_bank(
            state, recovery_incumbent, t0, selected.dtype
        )
        if recoveries is None:
            return selected, {**info, **diagnostics}
        # This proposal is a task-owned hybrid-mode exit, not a sampled
        # Gaussian/RL candidate.  The generic candidate projection can encode
        # a deliberately tiny policy trust tube (P4 uses 0.05), which would
        # silently erase the task's bounded capture action before validation.
        # Keep only the solver-wide action bound here.  The unchanged complete
        # NORMAL model horizon below remains the authority that can accept it.
        chosen_candidate = selected
        chosen_info = None
        selected_index = -1
        any_safe = False
        last_recovery_risk = diagnostics["normal_recovery_last_risk"]
        for index in range(int(recoveries.shape[0])):
            bounded = jnp.clip(
                recoveries[index], -self.action_limit, self.action_limit
            )
            candidate, recovery_info = self._accept_refinement_jit(
                state, recovery_incumbent, bounded, jnp.maximum(t0, 1.0), None,
                emergency, fallback_mode=jnp.int32(1),
            )
            recovery_safe = recovery_info["refined_revalidated_safe"] > 0.5
            last_recovery_risk = recovery_info["prior_risk_refined"]
            any_safe |= bool(np.asarray(jax.device_get(recovery_safe)))
            recovered = (
                recovery_safe
                & (recovery_info["selected_revalidated_safe"] > 0.5)
                & (recovery_info["execution_mode"] == 0)
            )
            if bool(np.asarray(jax.device_get(recovered))):
                chosen_candidate = candidate
                chosen_info = recovery_info
                selected_index = index
                break

        recovered_host = chosen_info is not None
        merged = {**info, **chosen_info} if recovered_host else info
        merged.update({
            "normal_recovery_applicable": jnp.float32(1.0),
            "normal_recovery_revalidated_safe": jnp.float32(any_safe),
            "normal_recovery_selected": jnp.float32(recovered_host),
            "normal_recovery_candidate_count": jnp.float32(recoveries.shape[0]),
            "normal_recovery_selected_index": jnp.int32(selected_index),
            "normal_recovery_last_risk": last_recovery_risk,
        })
        return chosen_candidate, merged

    def _attempt_task_normal_rescue(
        self, state, rescue_incumbent, selected, info, t0, emergency,
        incumbent_mode=None,
    ):
        """Try a task-owned NORMAL rescue after every primary path is unsafe.

        This hook is deliberately narrower than NORMAL recovery from a
        committed emergency.  It is considered only while the measured task
        mode is already NORMAL and no ordinary NORMAL candidate survived.
        It runs before committing an otherwise-safe UNLOAD transition, since
        a certified lower-load NORMAL successor preserves task progress and
        avoids an unnecessary hybrid-mode switch.  The task receives the
        solver horizon length so a time-indexed reference can construct
        node-aligned candidates without hard-coding a particular spline grid.
        Every rescue is checked by the unchanged complete NORMAL certificate;
        the first safe member wins, otherwise the original emergency or
        unrecoverable decision is preserved.
        """
        diagnostics = {
            "normal_rescue_applicable": jnp.float32(0.0),
            "normal_rescue_revalidated_safe": jnp.float32(0.0),
            "normal_rescue_selected": jnp.float32(0.0),
            "normal_rescue_candidate_count": jnp.float32(0.0),
            "normal_rescue_selected_index": jnp.int32(-1),
            # Keep an inapplicable rescue diagnostic finite so trajectory
            # metadata remains JSON-serializable.  Applicability and the
            # candidate count carry the semantic distinction; infinity here
            # would only create a non-portable artifact.
            "normal_rescue_last_risk": jnp.zeros((4,)),
            "normal_rescue_candidate_risks": jnp.zeros((0, 4)),
            "normal_rescue_forced_selection": jnp.float32(0.0),
        }
        mode = (
            info.get("execution_mode", jnp.int32(0))
            if incumbent_mode is None else jnp.asarray(incumbent_mode, jnp.int32)
        )
        selected_safe_normal = (
            (info["selected_revalidated_safe"] > 0.5)
            & (info.get("execution_mode", jnp.int32(0)) == 0)
        )
        if (
            self._execution_context is None
            or self.normal_rescue_plans_fn is None
            or int(np.asarray(jax.device_get(mode))) != 0
            or bool(np.asarray(jax.device_get(selected_safe_normal)))
        ):
            return selected, {**info, **diagnostics}

        rescue = self.normal_rescue_plans_fn(
            state, rescue_incumbent, int(self.Hsample), t0
        )
        if rescue is None:
            return selected, {**info, **diagnostics}
        rescues = jnp.asarray(rescue, selected.dtype)
        if rescues.ndim == rescue_incumbent.ndim:
            rescues = rescues[None]
        if (
            rescues.ndim != rescue_incumbent.ndim + 1
            or rescues.shape[1:] != rescue_incumbent.shape
        ):
            raise ValueError(
                "normal rescue plan bank shape "
                f"{rescues.shape} is incompatible with incumbent shape "
                f"{rescue_incumbent.shape}"
            )

        chosen_candidate = selected
        chosen_info = None
        selected_index = -1
        any_safe = False
        last_rescue_risk = diagnostics["normal_rescue_last_risk"]
        candidate_risks = []
        for index in range(int(rescues.shape[0])):
            bounded = jnp.clip(
                rescues[index], -self.action_limit, self.action_limit
            )
            candidate, rescue_info = self._accept_refinement_jit(
                state, rescue_incumbent, bounded, jnp.maximum(t0, 1.0),
                None, emergency, fallback_mode=jnp.int32(0),
                learned_abstain=jnp.asarray(True),
            )
            # A NORMAL rescue is deliberately outside performance ordering:
            # it exists to replace a safe-but-unproductive UNLOAD before that
            # mode is committed.  The candidate itself must still pass the
            # unchanged complete NORMAL certificate (and learned reliability
            # has already abstained above), but it need not beat the failed
            # incumbent's task score.  Reconcile the diagnostics with the
            # explicit task-owned selection when this condition holds.
            rescue_safe = rescue_info["refined_revalidated_safe"] > 0.5
            last_rescue_risk = rescue_info["prior_risk_refined"]
            candidate_risks.append(last_rescue_risk)
            any_safe |= bool(np.asarray(jax.device_get(rescue_safe)))
            if bool(np.asarray(jax.device_get(rescue_safe))):
                chosen_candidate = candidate
                chosen_info = {
                    **rescue_info,
                    "selected_revalidated_safe": jnp.float32(1.0),
                    "emergency_selected": jnp.float32(0.0),
                    "emergency_unrecoverable": jnp.float32(0.0),
                    "execution_mode": jnp.int32(0),
                    "normal_rescue_forced_selection": jnp.float32(1.0),
                }
                selected_index = index
                break

        rescued_host = chosen_info is not None
        merged = {**info, **chosen_info} if rescued_host else info
        merged.update({
            "normal_rescue_applicable": jnp.float32(1.0),
            "normal_rescue_revalidated_safe": jnp.float32(any_safe),
            "normal_rescue_selected": jnp.float32(rescued_host),
            "normal_rescue_candidate_count": jnp.float32(rescues.shape[0]),
            "normal_rescue_selected_index": jnp.int32(selected_index),
            "normal_rescue_last_risk": last_rescue_risk,
            "normal_rescue_candidate_risks": jnp.stack(candidate_risks),
            "normal_rescue_forced_selection": jnp.float32(rescued_host),
        })
        return chosen_candidate, merged

    def _replan_additive(self, state, incumbent, schedule, rng, t0, incumbent_mode=None):
        """Keep the no-prior search intact; experts can only add candidates.

        Gaussian RNG, starting point and realization context are identical
        with/without the prior. Nsample is the Gaussian refinement budget;
        expert horizons are evaluated once, with their count reported.
        """
        if self.score_risk_fn is None or self.risk_safe_fn is None:
            raise ValueError("additive prior requires task score and safety hooks")
        if self._execution_context is not None:
            incumbent_mode = self._incumbent_execution_mode(state, incumbent_mode)
            state = self._normal_rollout_state(state)
        if self._prepare_state_jit is not None:
            state = self._prepare_state_jit(state, incumbent, t0)
        gaussian = self._replan_scan_jit(
            state, incumbent, schedule, rng, t0, incumbent
        )
        if self._candidate_projection_jit is not None:
            if self._committed_shift_task_recovery:
                # Preserve only the explicitly tracked task-owned recovery.
                # Its complete model horizon, not the proposal tube, is its
                # deployment certificate.
                incumbent = jnp.clip(
                    incumbent, -self.action_limit, self.action_limit
                )
            elif self._execution_context is not None:
                # A committed UNLOAD incumbent is also a task-owned candidate
                # with an explicit execution identity.  Projecting it through
                # the NORMAL proposal tube before revalidation changes the
                # very fallback whose safety is being checked (P4 used to
                # erase its knee-clearance residual here).  Preserve it only
                # while the measured state says the committed mode is UNLOAD;
                # ordinary NORMAL incumbents retain the historical projection.
                incumbent = jnp.where(
                    incumbent_mode == 1,
                    jnp.clip(
                        incumbent, -self.action_limit, self.action_limit
                    ),
                    self._candidate_projection_jit(state, incumbent),
                )
            else:
                incumbent = self._candidate_projection_jit(state, incumbent)
            gaussian = self._candidate_projection_jit(state, gaussian)
        emergency_factory = self.emergency_plans_fn or self.emergency_plan_fn
        emergency = (
            emergency_factory(state, incumbent, t0)
            if emergency_factory is not None else None
        )
        if emergency is not None:
            # Emergency identity is explicit in the candidate source and its
            # eventual execution_mode.  A task-owned recovery must retain its
            # independently certified action (for P4, the knee bank) instead
            # of passing through a NORMAL policy/Gaussian proposal tube.
            emergency = jnp.clip(
                emergency, -self.action_limit, self.action_limit
            )
        acceptance_kw = (
            {"fallback_mode": incumbent_mode}
            if self._execution_context is not None else {}
        )
        selected, info = self._accept_refinement_jit(
            state, incumbent, gaussian, t0, None, emergency, **acceptance_kw
        )
        selected, info = self._attempt_task_normal_recovery(
            state, incumbent, selected, info, t0, emergency,
            incumbent_mode=incumbent_mode,
        )
        selected, info = self._attempt_task_normal_rescue(
            state, incumbent, selected, info, t0, emergency,
            incumbent_mode=incumbent_mode,
        )
        info = {**info, "additive_prior_selected": jnp.float32(0.0),
                "additive_prior_candidate_count": jnp.float32(0.0)}
        if not self._prior_active:
            return selected, info
        proposals = [jnp.asarray(self.prior.warm_start(state))[None]]
        if self.prior_stochastic_samples:
            batch = self.prior.sample_horizons(
                state, key=jax.random.fold_in(rng, 1701),
                n_samples=self.prior_stochastic_samples,
            )
            proposals.append(batch.trajectories)
        candidates = jnp.clip(jnp.concatenate(proposals),
                              -self.action_limit, self.action_limit)
        expert, any_safe, expert_score = self._select_additive_prior_jit(
            state, candidates
        )
        support_applicable = (
            self._prior_applicable_jit(state)
            if self._prior_applicable_jit is not None
            else jnp.asarray(True)
        )
        any_safe = any_safe & support_applicable
        # The Gaussian decision has already been revalidated at this state,
        # including cold start. Do not invalidate it a second time at t=0.
        second_acceptance_kw = (
            {"fallback_mode": info["execution_mode"]}
            if self._execution_context is not None else {}
        )
        challenger, expert_info = self._accept_refinement_jit(
            state, selected, expert, jnp.maximum(t0, 1.0), None, emergency,
            **second_acceptance_kw,
        )
        # Never let a rejected expert replace the Gaussian fallback with a
        # second emergency. Adoption requires a strict score improvement.
        improves_normal = (
            expert_info["prior_predicted_improvement"] > self.prior_improvement_epsilon
        )
        if self._execution_context is not None:
            # Returning from a certified one-step unload to a safe normal
            # horizon is lexicographic recovery, not a cross-horizon score win.
            improves_normal |= info["execution_mode"] == 1
        # A task-owned NORMAL recovery is the certified successor paired with
        # the previously committed UNLOAD transition.  Once that successor
        # has been selected, a policy proposal must not overwrite it merely
        # because its task score is higher.  Besides breaking the recursive
        # hybrid certificate, doing so made the diagnostics claim that both
        # the recovery and the additive prior were deployed on the same step.
        # The learned prior remains eligible on ordinary NORMAL replans and
        # when no task-owned recovery can be certified.
        recovery_transition = (
            (info.get("normal_recovery_selected", jnp.float32(0.0)) > 0.5)
            | (info.get("normal_rescue_selected", jnp.float32(0.0)) > 0.5)
        )
        adopted = (
            any_safe
            & (expert_info["prior_accepted"] > 0.5)
            & improves_normal
            & ~recovery_transition
        )
        deployed = jnp.where(adopted, challenger, selected)
        combined = {
            k: jnp.where(adopted, expert_info[k], v) if k in expert_info else v
            for k, v in info.items()
        }
        if self._execution_context is not None:
            combined["emergency_fallback_recovery"] = jnp.maximum(
                info["emergency_fallback_recovery"],
                adopted.astype(jnp.float32) * expert_info["emergency_fallback_recovery"],
            )
            combined["additive_prior_comparison_applicable"] = expert_info[
                "prior_comparison_applicable"
            ]
        combined.update({
            "additive_prior_selected": adopted.astype(jnp.float32),
            "additive_prior_candidate_count": jnp.float32(candidates.shape[0]),
            "additive_prior_support_applicable": support_applicable.astype(
                jnp.float32
            ),
            "additive_prior_any_safe": any_safe.astype(jnp.float32),
            "additive_prior_improvement": expert_info["prior_predicted_improvement"],
            "additive_prior_score": expert_score,
            "additive_gaussian_score": expert_info["prior_score_fallback"],
        })
        return deployed, combined

    def replan_with_info(
        self, state, warm_start, schedule, rng, t0=0.0, incumbent_mode=None,
    ):
        t0 = jnp.asarray(t0, jnp.float32)
        if self.prior_mode == "additive":
            if self._execution_context is not None:
                return self._replan_additive(
                    state, warm_start, schedule, rng, t0, incumbent_mode
                )
            return self._replan_additive(state, warm_start, schedule, rng, t0)
        if self._execution_context is not None:
            incumbent_mode = self._incumbent_execution_mode(state, incumbent_mode)
            state = self._normal_rollout_state(state)
        receding_incumbent = warm_start
        if self._prior_active:
            lam = self.prior_lambda_shift
            U_rl = jnp.asarray(self.prior.warm_start(state), warm_start.dtype)
            warm_start = lam * warm_start + (1.0 - lam) * U_rl
            warm_start = self._project_to_prior(warm_start, U_rl)
        else:
            # Dummy of the right static shape; all prior branches are Python
            # static false, preserving the legacy candidate set and arithmetic.
            U_rl = warm_start
        U_atacom = None
        if (
            self._prior_active
            and self.atacom_prior is not None
            and (self.prior_union_trust or self.prior_atacom_incumbent)
        ):
            # Generate the ATACOM anchor from the actual current state.  When it
            # is the declared lower bound, refinement starts locally around it
            # instead of around an unrelated RL/receding mixture.
            U_atacom = jnp.asarray(
                self.atacom_prior.warm_start(state), warm_start.dtype
            )
            if self.prior_atacom_default:
                warm_start = U_atacom
        if self._prepare_state_jit is not None:
            # The measured response is frozen around the actual incumbent, not
            # around an arbitrary shifted/RL mixture.
            if self.prior_atacom_default and U_atacom is not None:
                response_center = U_atacom
            elif self._prior_active and self.prior_fallback_mode == "receding_incumbent":
                response_center = jnp.where(
                    t0 <= 0.0, U_rl, receding_incumbent
                )
            else:
                response_center = U_rl if self._prior_active else warm_start
            state = self._prepare_state_jit(state, response_center, t0)
        proposal_info = {}
        n_structured = (
            self.prior_stochastic_samples + self.prior_atacom_samples
        )
        if self._prior_active and n_structured > 0:
            proposal_nodes = []
            proposal_logps = []
            if self.prior_stochastic_samples > 0:
                sample_horizons = getattr(self.prior, "sample_horizons", None)
                if sample_horizons is None:
                    raise ValueError(
                        "prior_stochastic_samples requires prior.sample_horizons"
                    )
                rng, proposal_key = jax.random.split(rng)
                proposals = sample_horizons(
                    state, key=proposal_key,
                    n_samples=self.prior_stochastic_samples,
                )
                proposal_nodes.append(proposals.trajectories)
                proposal_logps.append(proposals.log_prob)
            if self.prior_atacom_samples > 0:
                rng, atacom_key = jax.random.split(rng)
                atacom_proposals = self.atacom_prior.sample_horizons(
                    state, key=atacom_key,
                    n_samples=self.prior_atacom_samples,
                )
                proposal_nodes.append(atacom_proposals.trajectories)
                proposal_logps.append(atacom_proposals.log_prob)
            structured_nodes = jnp.asarray(
                jnp.concatenate(proposal_nodes, axis=0), warm_start.dtype
            )
            expected = (
                n_structured,
                self.Hnode + 1,
                self.nu,
            )
            if structured_nodes.shape != expected:
                raise ValueError(
                    f"structured proposal shape {structured_nodes.shape} != {expected}"
                )
            if self.prior_union_trust and U_atacom is not None:
                prior_centers = jnp.stack([U_rl, U_atacom], axis=0)
            else:
                prior_centers = U_rl[None]
            refined, mixture_diag = self._replan_scan_structured_jit(
                state, warm_start, schedule, rng, t0, U_rl,
                structured_nodes, prior_centers,
            )
            proposal_info = {
                "proposal_stochastic_count": jnp.asarray(
                    self.prior_stochastic_samples, jnp.float32
                ),
                "proposal_atacom_count": jnp.asarray(
                    self.prior_atacom_samples, jnp.float32
                ),
                "proposal_logp_mean": jnp.mean(
                    jnp.concatenate(proposal_logps, axis=0)
                ),
                "proposal_gaussian_best_reward": mixture_diag[0],
                "proposal_rl_best_reward": mixture_diag[1],
                "proposal_atacom_best_reward": mixture_diag[2],
                "proposal_gaussian_weight": mixture_diag[3],
                "proposal_rl_weight": mixture_diag[4],
                "proposal_atacom_weight": mixture_diag[5],
            }
        else:
            # Exact legacy call: no RNG split, no candidate-shape change.
            refined = (
                self._replan_stepwise(
                    state, warm_start, schedule, rng, t0, U_rl
                )
                if self.stepwise_diffusion
                else self._replan_scan_jit(
                    state, warm_start, schedule, rng, t0, U_rl
                )
            )
        # Receding-incumbent acceptance is a final model-based safety check, not
        # an RL-only feature.  It must remain active during the CPU pre-prior
        # phase so the post-retraction proposal is evaluated before deployment.
        # Other fallback modes preserve their legacy prior-gated behaviour.
        incumbent_acceptance = (
            self.prior_acceptance
            and self.prior_fallback_mode == "receding_incumbent"
            and self.score_risk_fn is not None
        )
        if self._prior_active or incumbent_acceptance:
            fallback = (
                receding_incumbent
                if self.prior_fallback_mode == "receding_incumbent"
                else U_rl
            )
            if self._candidate_projection_jit is not None:
                preserve_task_recovery = (
                    self.prior_fallback_mode == "receding_incumbent"
                    and self._committed_shift_task_recovery
                )
                if preserve_task_recovery:
                    fallback = jnp.clip(
                        fallback, -self.action_limit, self.action_limit
                    )
                elif (
                    self._execution_context is not None
                    and self.prior_fallback_mode == "receding_incumbent"
                ):
                    fallback = jnp.where(
                        incumbent_mode == 1,
                        jnp.clip(
                            fallback, -self.action_limit, self.action_limit
                        ),
                        self._candidate_projection_jit(state, fallback),
                    )
                else:
                    fallback = self._candidate_projection_jit(state, fallback)
                refined = self._candidate_projection_jit(state, refined)
            emergency_factory = self.emergency_plans_fn or self.emergency_plan_fn
            emergency = (
                emergency_factory(state, fallback, t0)
                if (
                    self.prior_fallback_mode == "receding_incumbent"
                    and emergency_factory is not None
                )
                else None
            )
            if emergency is not None:
                emergency = jnp.clip(
                    emergency, -self.action_limit, self.action_limit
                )
            acceptance_kw = (
                {"fallback_mode": incumbent_mode}
                if self._execution_context is not None else {}
            )
            if self.prior_atacom_incumbent:
                selected, acceptance_info = self._accept_refinement_jit(
                    state, fallback, refined, t0, U_atacom, emergency,
                    **acceptance_kw,
                )
            else:
                selected, acceptance_info = self._accept_refinement_jit(
                    state, fallback, refined, t0, None, emergency,
                    **acceptance_kw,
                )
            return selected, {**acceptance_info, **proposal_info}
        return refined, proposal_info

    def replan(self, state, warm_start, schedule, rng, t0=0.0) -> jnp.ndarray:
        if self._execution_context is not None:
            raise ValueError("mode-aware MGA requires replan_with_info and contextual execution")
        return self.replan_with_info(
            state, warm_start, schedule, rng, t0=t0
        )[0]

    def first_action(self, plan_var) -> jnp.ndarray:
        return self._node2u(plan_var)[0]

    def shift(self, plan_var) -> jnp.ndarray:
        if self._execution_context is not None:
            if isinstance(plan_var, jax.core.Tracer):
                raise ValueError("mode-aware host shift cannot be enclosed in jit")
            return self._shift_with_mode_jit(
                plan_var, jnp.int32(self._committed_shift_mode)
            )
        if self.receding_shift_mode == "certified_terminal_hold":
            return self.spline.shift_nodes_certified_terminal_hold(plan_var)
        if self.receding_shift_mode == "terminal_hold":
            return self.spline.shift_nodes_terminal_hold(plan_var)
        if self.receding_shift_mode == "zero":
            # An additive expert must not change the ordinary Gaussian
            # incumbent's tail semantics merely by being loaded.  Emergencies
            # retain their unloaded tail: raw zero can encode nonzero force.
            shifted = self.spline.shift_nodes(plan_var)
            if (
                self.prior_fallback_mode == "receding_incumbent"
                and self.emergency_active_fn is not None
            ):
                shifted = jnp.where(
                    self.emergency_active_fn(plan_var),
                    self.spline.shift_nodes_terminal_hold(plan_var),
                    shifted,
                )
            return shifted
        if (
            self._prior_active
            and self.prior_fallback_mode == "receding_incumbent"
        ):
            return self.spline.shift_nodes_terminal_hold(plan_var)
        if (
            self.prior_fallback_mode == "receding_incumbent"
            and self.emergency_active_fn is not None
        ):
            # Preserve canonical DIAL zero-tail semantics for ordinary CPU
            # incumbents.  Only the task-owned emergency needs terminal hold:
            # zero-padding its normalized force/stiffness channels would turn
            # a 0-N compliant plan back into a loaded nominal-stiffness plan.
            emergency_active = self.emergency_active_fn(plan_var)
            return jnp.where(
                emergency_active,
                self.spline.shift_nodes_terminal_hold(plan_var),
                self.spline.shift_nodes(plan_var),
            )
        return self.spline.shift_nodes(plan_var)

    # --- unified backend interface (single-shot, parallel) -------------------
    def plan(self, x0: Any, rng_key: Optional[Any] = None) -> Dict[str, Any]:
        """Single-shot full-horizon plan (Ndiffuse_init reverse steps from cold),
        the whole reverse-diffuse jit-ed; candidate rollout vmap-ed over Nsample."""
        if self._execution_context is not None:
            raise ValueError(
                "mode-aware MGA requires run_receding; action-only single-shot "
                "rollout cannot discard the selected execution context"
            )
        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.seed)
        replan_jit = jax.jit(self.replan_with_info)
        Y, prior_info = replan_jit(
            x0,
            self.init_plan_var(),
            self.make_schedule(self.Ndiffuse_init),
            rng_key,
        )
        us = self._node2u(Y)
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


__all__ = ["MgaBackendJax"]
