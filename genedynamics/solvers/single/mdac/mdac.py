"""MDAC solver shell (Manifold Diffusion Annealing Control).

A self-contained multi-backend solver under `genedynamics/solvers/single/mdac/`,
DIAL-derived (brax `PipelineEnv` substrate + node-spline), written in the SAME
shape as `dial.py` / `mdoc.py`: a thin `BaseModelBasedDiffusionSolver` subclass
that selects `MdacBackendJax` at runtime and drives the backend-agnostic
receding-horizon bridge (`solvers/common/receding_horizon.py`):

    MDAC == RecedingHorizonController(<MDAC backend>)

MDAC adds nothing hand-rolled: its upgrades are composed from EXISTING upstream
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
`action_size`. With every seam off MDAC reduces byte-identically to DIAL.

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
from genedynamics.solvers.single.mdac.backend_impl import to_unified_backend
from genedynamics.solvers.single.mdac.core.method_registry import resolve_method
from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.core.dynamics import DynamicsToEnvAdapter
from genedynamics.core.energy import LegacyEnergyFunctional, EnergyToLegacyAdapter
from genedynamics.core.types import Trajectory

try:
    from genedynamics.core.registry.solvers import register_solver
except Exception:  # pragma: no cover
    register_solver = None


def _get_mdac_backend(backend_name: str):
    if backend_name == "jax":
        from genedynamics.solvers.single.mdac.backends.mdac_jax import MdacBackendJax

        return MdacBackendJax
    return None


class MDACSolver(BaseModelBasedDiffusionSolver):
    """Receding-horizon MDAC over the parallel node-spline reverse-diffusion backend."""

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
        # --- MDAC ablation selector (one flag per component) ---
        method: str = "mdac",
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
        mdac_topk_active: int = 8,
        mdac_eps_stab: float = 1e-6,
        mdac_geom_gain: float = 1.0,
        # --- prior seam (genedynamics/learning/priors; None => verbatim DIAL) ---
        prior: Any = None,              # Prior: RL/diffusion warm-start (eq:rl_warm_start)
        prior_lambda_shift: float = 0.5,  # U_init = lam*U_shift + (1-lam)*U_rl
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
        self.method = str(method)
        self.flags = resolve_method(self.method)
        self.transport = transport
        from genedynamics.core.constraints.action_filters import NoOpConstraintFilter
        self.constraint_filter = constraint_filter or NoOpConstraintFilter()
        self.noise_sampler = noise_sampler
        self.obstacles = obstacles
        self.geometry_fn = geometry_fn      # read by MdacBackendJax (None => geometry off)
        self.retraction = retraction
        self.geometry_gate_fn = geometry_gate_fn
        self.prepare_state_fn = prepare_state_fn
        self.prior = prior                  # read by MdacBackendJax (None => no warm-start mix)
        self.prior_lambda_shift = float(prior_lambda_shift)

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
                mdac_topk_active=int(mdac_topk_active), mdac_eps_stab=float(mdac_eps_stab),
                mdac_geom_gain=float(mdac_geom_gain),
                method=str(method),
            )
        )
        self._backend_impl = None

    def _get_backend_impl(self):
        if self._backend_impl is None:
            backend = RuntimeBackendManager.get_backend()
            backend_cls = _get_mdac_backend(backend.name)
            if backend_cls is None:
                raise ValueError(f"MDAC backend '{backend.name}' not found")
            self._backend_impl = to_unified_backend(backend_cls(solver=self))
        return self._backend_impl

    # --- receding-horizon execution (reuses the DIAL bridge) ---
    def make_controller(self, n_steps: int, **kw: Any) -> RecedingHorizonController:
        backend = self._get_backend_impl()
        step_fn = self._step_fn or getattr(backend, "_step_fn", None)
        if step_fn is None:
            raise ValueError("MDAC needs a step_fn (inject one or provide dynamics).")
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
        return Trajectory(states=states, actions=actions, info={"source": "mdac", "method": self.method})

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
        register_solver("mdac", MDACSolver)
    except Exception:  # pragma: no cover
        pass
