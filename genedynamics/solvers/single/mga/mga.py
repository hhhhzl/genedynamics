"""MGA solver shell (Manifold Generative Annealing).

A self-contained multi-backend solver under `genedynamics/solvers/single/mga/`,
DIAL-derived (brax `PipelineEnv` substrate + node-spline), written in the SAME
shape as `dial.py` / `mdoc.py`: a thin `BaseModelBasedDiffusionSolver` subclass
that selects `MgaBackendJax` at runtime and drives the backend-agnostic
receding-horizon bridge (`solvers/common/receding_horizon.py`):

    MGA == RecedingHorizonController(<MGA backend>)

MGA adds nothing hand-rolled: its upgrades are composed from EXISTING upstream
machinery through the backend's optional seams:
  * `transport`         — `transport/backends` DDPM/DDIM/FM/Adaptive (like mdoc/2go)
  * soft-feasibility / iALM — the cfsmbd / mdcoas AL pattern (`aug_lambda/aug_rho`
                          + `[g]_+` penalty in the rollout reward via the env's
                          `constraint_residual` hook). NB: iALM is cfsmbd's, NOT mdoc's.
  * `constraint_filter` — `core/constraints` action-space ConstraintFilter (mdoc's
                          mechanism, distinct from iALM)
  * geometry            — `genemetry` SdfManifold/CfsRetraction + ScheduleOverlay (like 2go)
  * `noise_sampler`     — `core/prob` seam
The one genuinely-new primitive (position-stiffness SPD) lives upstream in
`genedynamics/core/control/stiffness.py`; the solver only consumes it via
`action_size`. With every seam off MGA reduces byte-identically to DIAL.

`method` selects an ablation via `core/method_registry` (one flag per component);
fairness (same `(M,H,K)`) is enforced there.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from genedynamics.solvers.common.model_based_diffusion.base_solver import (
    BaseModelBasedDiffusionSolver,
)
from genedynamics.solvers.common.receding_horizon import RecedingHorizonController
from genedynamics.solvers.single.mga.backend_impl import to_unified_backend
from genedynamics.solvers.single.mga.core.method_registry import (
    canonical_method_name,
    resolve_method,
)
from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.core.dynamics import DynamicsToEnvAdapter
from genedynamics.core.energy import LegacyEnergyFunctional, EnergyToLegacyAdapter
from genedynamics.core.types import Trajectory

try:
    from genedynamics.core.registry.solvers import register_solver
except Exception:  # pragma: no cover
    register_solver = None


def _get_mga_backend(backend_name: str):
    if backend_name == "jax":
        from genedynamics.solvers.single.mga.backends.mga_jax import MgaBackendJax

        return MgaBackendJax
    return None


class MGASolver(BaseModelBasedDiffusionSolver):
    """Receding-horizon MGA over the parallel node-spline reverse-diffusion backend."""

    def __init__(
        self,
        dynamics: Any,
        energy: Any,
        backend: Any,
        *,
        nu: Optional[int] = None,
        rollout_fn: Optional[Any] = None,
        step_fn: Optional[Any] = None,
        # --- DIAL-inherited sampler knobs ---
        horizon: int = 16,
        dt: float = 0.02,
        ctrl_dt: float = 0.02,
        Hsample: int = 16,
        Hnode: int = 4,
        Nsample: int = 2048,
        Ndiffuse: int = 2,
        Ndiffuse_init: int = 10,
        temp_sample: float = 0.06,
        horizon_diffuse_factor: float = 0.9,
        traj_diffuse_factor: float = 0.5,
        sigma_scale: float = 1.0,
        action_limit: float = 1.0,
        beta0: float = 1e-4,
        betaT: float = 1e-2,
        # --- soft-feasibility / iALM (cfsmbd/mdcoas pattern) ---
        aug_lambda: float = 0.0,
        aug_rho: float = 1.0,
        n_steps: Optional[int] = None,
        seed: int = 0,
        # --- MGA ablation selector (one flag per component) ---
        method: str = "mga_base",
        # --- optional upstream seams (None / NoOp => verbatim DIAL) ---
        transport: Any = None,
        constraint_filter: Any = None,
        noise_sampler: Any = None,
        obstacles: Any = None,
        # --- geometry seam (genemetry SdfManifold/CfsRetraction, like 2go) ---
        geometry_fn: Any = None,        # (state, Ybar_nodes, t0) -> a_geom (Hnode+1, nu)
        retraction: Any = None,         # optional genemetry CfsRetraction
        geometry_gate_fn: Any = None,   # -> {action/scalar/path/normal/stiffness/force}
        prepare_state_fn: Any = None,   # frozen task context before reverse diffusion
        candidate_projection_fn: Any = None,  # task-owned executable-set map
        mga_topk_active: int = 8,
        mga_eps_stab: float = 1e-6,
        mga_geom_gain: float = 1.0,
        # --- prior seam (genedynamics/learning/priors; None => verbatim DIAL) ---
        prior: Any = None,              # Prior: RL/diffusion warm-start (eq:rl_warm_start)
        atacom_prior: Any = None,       # optional structured 7D-tangent expert
        prior_lambda_shift: float = 0.5,  # U_init = lam*U_shift + (1-lam)*U_rl
        risk_fn: Any = None,             # task-owned sequence risk vector
        prior_include_incumbent: bool = True,
        prior_trust_radius: float = 0.5,
        prior_stochastic_samples: int = 0,
        prior_atacom_samples: int = 0,
        prior_union_trust: bool = False,
        prior_atacom_incumbent: bool = False,
        prior_atacom_default: bool = False,
        prior_atacom_strict_risk: bool = False,
        prior_improvement_epsilon: float = 0.0,
        prior_risk_tolerance: Any = (0.0, 0.0, 0.0, 0.0),
        prior_acceptance: bool = True,
        prior_fallback_mode: str = "rl",
        reliability_model: Any = None,
        reliability_risk_tolerance: Any = (0.0, 0.0, 0.0, 0.0),
        reliability_force_limit: float = 0.25,
        reliability_deformation_limit: float = float("inf"),
        reliability_hard_limits: Any = None,
        reliability_support_mode: str = "joint",
        reliability_ood_policy: str = "veto",
        **kwargs: Any,
    ) -> None:
        super().__init__(dynamics, energy, backend, **kwargs)

        self.horizon = int(Hsample)
        self.dt = float(dt)
        self.seed = int(seed)
        self.n_steps = n_steps
        self._is_brax_env = dynamics is not None and hasattr(dynamics, "pipeline_step")
        if nu is not None:
            self.nu = int(nu)
        else:
            self.nu = int(getattr(dynamics, "act_dim", 0) or getattr(dynamics, "action_size", 0))

        self._rollout_fn = rollout_fn
        self._step_fn = step_fn
        self._env_adapter = (
            DynamicsToEnvAdapter(dynamics, dt)
            if dynamics is not None and not self._is_brax_env
            else None
        )
        if isinstance(energy, LegacyEnergyFunctional) or energy is None:
            self._legacy_energy = energy
        else:
            self._legacy_energy = EnergyToLegacyAdapter(energy, dynamics).legacy_energy

        # --- seams ---
        # ``mga`` is the public paper algorithm and resolves to the frozen
        # controllability-plus-gating contract. ``mga_base`` remains available
        # for low-level regression probes of the original controller.
        self.method = canonical_method_name(method)
        self.flags = resolve_method(self.method)
        self.transport = transport
        from genedynamics.core.constraints.action_filters import NoOpConstraintFilter
        self.constraint_filter = constraint_filter or NoOpConstraintFilter()
        self.noise_sampler = noise_sampler
        self.obstacles = obstacles
        self.geometry_fn = geometry_fn      # read by MgaBackendJax (None => geometry off)
        self.retraction = retraction
        self.geometry_gate_fn = geometry_gate_fn
        self.prepare_state_fn = prepare_state_fn
        self.candidate_projection_fn = candidate_projection_fn
        self.prior = prior                  # read by MgaBackendJax (None => no warm-start mix)
        self.atacom_prior = atacom_prior
        self.prior_lambda_shift = float(prior_lambda_shift)
        self.risk_fn = risk_fn
        self.prior_include_incumbent = bool(prior_include_incumbent)
        self.prior_trust_radius = float(prior_trust_radius)
        self.prior_stochastic_samples = int(prior_stochastic_samples)
        self.prior_atacom_samples = int(prior_atacom_samples)
        self.prior_union_trust = bool(prior_union_trust)
        self.prior_atacom_incumbent = bool(prior_atacom_incumbent)
        self.prior_atacom_default = bool(prior_atacom_default)
        self.prior_atacom_strict_risk = bool(prior_atacom_strict_risk)
        if self.prior_atacom_incumbent and self.atacom_prior is None:
            raise ValueError(
                "prior_atacom_incumbent requires an atacom_prior"
            )
        if (
            self.prior_stochastic_samples < 0
            or self.prior_atacom_samples < 0
            or self.prior_stochastic_samples + self.prior_atacom_samples
            >= int(Nsample)
        ):
            raise ValueError(
                "structured prior samples must be nonnegative and sum to less "
                f"than Nsample; got rl={self.prior_stochastic_samples}, "
                f"atacom={self.prior_atacom_samples}, Nsample={Nsample}"
            )
        if self.prior_atacom_samples and self.atacom_prior is None:
            raise ValueError(
                "prior_atacom_samples requires an atacom_prior"
            )
        self.prior_improvement_epsilon = float(prior_improvement_epsilon)
        self.prior_risk_tolerance = tuple(float(x) for x in prior_risk_tolerance)
        self.prior_acceptance = bool(prior_acceptance)
        self.prior_fallback_mode = str(prior_fallback_mode)
        self.reliability_model = reliability_model
        self.reliability_risk_tolerance = tuple(
            float(x) for x in reliability_risk_tolerance
        )
        self.reliability_force_limit = float(reliability_force_limit)
        self.reliability_deformation_limit = float(reliability_deformation_limit)
        self.reliability_hard_limits = (
            None
            if reliability_hard_limits is None
            else tuple(float(x) for x in reliability_hard_limits)
        )
        self.reliability_support_mode = str(reliability_support_mode)
        self.reliability_ood_policy = str(reliability_ood_policy)

        self.config.update(
            dict(
                Hsample=int(Hsample), Hnode=int(Hnode), Nsample=int(Nsample),
                Ndiffuse=int(Ndiffuse), Ndiffuse_init=int(Ndiffuse_init),
                temp_sample=float(temp_sample),
                horizon_diffuse_factor=float(horizon_diffuse_factor),
                traj_diffuse_factor=float(traj_diffuse_factor),
                sigma_scale=float(sigma_scale), action_limit=float(action_limit),
                ctrl_dt=float(ctrl_dt), beta0=float(beta0), betaT=float(betaT),
                aug_lambda=float(aug_lambda), aug_rho=float(aug_rho),
                mga_topk_active=int(mga_topk_active), mga_eps_stab=float(mga_eps_stab),
                mga_geom_gain=float(mga_geom_gain),
                prior_include_incumbent=bool(prior_include_incumbent),
                prior_trust_radius=float(prior_trust_radius),
                prior_stochastic_samples=self.prior_stochastic_samples,
                prior_atacom_samples=self.prior_atacom_samples,
                prior_union_trust=self.prior_union_trust,
                prior_atacom_incumbent=self.prior_atacom_incumbent,
                prior_atacom_default=self.prior_atacom_default,
                prior_atacom_strict_risk=self.prior_atacom_strict_risk,
                prior_improvement_epsilon=float(prior_improvement_epsilon),
                prior_risk_tolerance=self.prior_risk_tolerance,
                prior_acceptance=bool(prior_acceptance),
                prior_fallback_mode=self.prior_fallback_mode,
                reliability_risk_tolerance=self.reliability_risk_tolerance,
                reliability_force_limit=self.reliability_force_limit,
                reliability_deformation_limit=self.reliability_deformation_limit,
                reliability_ood_policy=self.reliability_ood_policy,
                method=str(method),
            )
        )
        self._backend_impl = None

    def _get_backend_impl(self):
        if self._backend_impl is None:
            backend = RuntimeBackendManager.get_backend()
            backend_cls = _get_mga_backend(backend.name)
            if backend_cls is None:
                raise ValueError(f"MGA backend '{backend.name}' not found")
            self._backend_impl = to_unified_backend(backend_cls(solver=self))
        return self._backend_impl

    # --- receding-horizon execution (reuses the DIAL bridge) ---
    def make_controller(self, n_steps: int, **kw: Any) -> RecedingHorizonController:
        backend = self._get_backend_impl()
        step_fn = self._step_fn or getattr(backend, "_step_fn", None)
        if step_fn is None:
            raise ValueError("MGA needs a step_fn (inject one or provide dynamics).")
        return RecedingHorizonController(
            backend,
            step_fn=step_fn,
            n_steps=int(n_steps),
            n_diffuse_init=int(self.config["Ndiffuse_init"]),
            n_diffuse=int(self.config["Ndiffuse"]),
            **kw,
        )

    def run_receding(self, x0: Any, n_steps: int, rng: Any, **kw: Any):
        return self.make_controller(n_steps, **kw).run(x0, rng)

    def solve(self, x0: Any, horizon: int, **kwargs: Any) -> Trajectory:
        import jax

        n = int(kwargs.pop("n_steps", None) or self.n_steps or horizon)
        rng = kwargs.pop("rng_key", None)
        if rng is None:
            rng = jax.random.PRNGKey(self.seed)
        res = self.make_controller(n).run(x0, rng)
        states = [self._flatten_state(s) for s in res.states]
        actions = [np.asarray(a, dtype=np.float32) for a in res.actions]
        return Trajectory(states=states, actions=actions, info={"source": "mga", "method": self.method})

    @staticmethod
    def _flatten_state(s: Any) -> np.ndarray:
        ps = getattr(s, "pipeline_state", None)
        if ps is not None and hasattr(ps, "qpos"):
            return np.concatenate(
                [np.asarray(ps.qpos, np.float32), np.asarray(ps.qvel, np.float32)]
            )
        return np.asarray(s, dtype=np.float32)


if register_solver is not None:
    try:
        register_solver("mga", MGASolver)
    except Exception:  # pragma: no cover
        pass
