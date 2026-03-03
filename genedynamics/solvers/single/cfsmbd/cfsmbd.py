"""
CFS-MBD (genedynamics version).

Definition:
  CFS-MBD = MBD-style diffusion driver + Augmented Lagrangian objective + CFS-based per-step QP projection
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import time
import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.solvers.common.model_based_diffusion import BaseModelBasedDiffusionSolver
from genedynamics.core.dynamics import DynamicsModel, DynamicsToEnvAdapter
from genedynamics.core.energy import EnergyFunctional, LegacyEnergyFunctional
from genedynamics.core.backends import Backend
from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.core.types import State, Trajectory
from genedynamics.core.energy import EnergyToLegacyAdapter
from genedynamics.core.task_spec import legacy_extract_position
from genedynamics.core.constraints.action_filters import ConstraintFilter, NoOpConstraintFilter
from genedynamics.solvers.single.cfsmbd.backend_impl import to_unified_backend

try:
    from genedynamics.core.registry.solvers import register_solver
except Exception:
    register_solver = None


def _get_cfsmbd_backend(backend_name: str):
    if backend_name == "jax":
        from genedynamics.solvers.single.cfsmbd.backends.cfsmbd_jax import CFSMBDBackendJax
        return CFSMBDBackendJax
    if backend_name == "numpy":
        from genedynamics.solvers.single.cfsmbd.backends.cfsmbd_numpy import CFSMBDBackendNumpy
        return CFSMBDBackendNumpy
    return None


class CFSMBDSolver(BaseModelBasedDiffusionSolver):
    """
    Solver wrapper: delegates to CFS-MBD backend.
    
    CFS-MBD combines:
    - MBD-style reverse diffusion
    - Augmented Lagrangian objective (J + λ^T [g]_+ + (ρ/2) ||[g]_+||^2)
    - CFS-based per-step QP projection for constraint satisfaction
    """

    def __init__(
        self,
        dynamics: DynamicsModel,
        energy: EnergyFunctional,
        backend: Backend,
        *,
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
        constraint_filter: Optional[ConstraintFilter] = None,
        obstacles: Any = None,
        show_tqdm: bool = False,
        aug_lambda: float = 0.0,
        aug_rho: float = 1.0,
        action_extra_sigma: float = 0.0,
        num_modes: int = 1,
        mode_strategy: str = "multirun",
        diversity_eta: float = 1.0,
        diversity_topK_cand: int = None,
        diversity_use_state: bool = True,
        position_extractor=None,
        position_dim: int = 2,
        **kwargs: Any,
    ):
        super().__init__(dynamics, energy, backend, **kwargs)

        if backend.name not in {"jax", "numpy"}:
            raise ValueError(
                f"CFS-MBD solver currently supports backends {{'jax','numpy'}}, got {backend.name}."
            )

        self.horizon = int(horizon)
        self.dt = float(dt)
        self.seed = int(seed)
        self.config.update(
            dict(
                Nsample=int(Nsample),
                Ndiffuse=int(Ndiffuse),
                temp_sample=float(temp_sample),
                beta0=float(beta0),
                betaT=float(betaT),
                action_limit=float(action_limit),
                scheduler=scheduler,
                show_tqdm=bool(show_tqdm),
                aug_lambda=float(aug_lambda),
                aug_rho=float(aug_rho),
                action_extra_sigma=float(action_extra_sigma),
                num_modes=int(num_modes),
                mode_strategy=str(mode_strategy),
                diversity_eta=float(diversity_eta),
                diversity_topK_cand=int(diversity_topK_cand) if diversity_topK_cand is not None else None,
                diversity_use_state=bool(diversity_use_state),
                use_target_line=bool(kwargs.get("use_target_line", False)),
                num_targets=int(kwargs.get("num_targets", 4)),
            )
        )

        self._env_adapter = DynamicsToEnvAdapter(dynamics, dt)
        if isinstance(energy, LegacyEnergyFunctional):
            self._legacy_energy = energy
        else:
            self._legacy_energy = EnergyToLegacyAdapter(energy, dynamics).legacy_energy

        self._backend_impl = None
        self.constraint_filter = constraint_filter or NoOpConstraintFilter()
        self.obstacles = obstacles
        self.position_extractor = position_extractor or legacy_extract_position
        self.position_dim = int(position_dim)

    def _get_backend_impl(self):
        if self._backend_impl is None:
            backend = RuntimeBackendManager.get_backend()
            backend_cls = _get_cfsmbd_backend(backend.name)
            if backend_cls is None:
                raise ValueError(f"CFS-MBD backend '{backend.name}' not found")
            # JAX backend accepts solver parameter, NumPy backend does not
            if backend.name == "jax":
                self._backend_impl = backend_cls(
                    solver=self,  # Pass solver to access num_modes
                    env_adapter=self._env_adapter,
                    legacy_energy=self._legacy_energy,
                    horizon=self.horizon,
                    dt=self.dt,
                    Nsample=self.config["Nsample"],
                    Ndiffuse=self.config["Ndiffuse"],
                    temp_sample=self.config["temp_sample"],
                    beta0=self.config["beta0"],
                    betaT=self.config["betaT"],
                    action_limit=self.config["action_limit"],
                    seed=self.seed,
                    scheduler=self.config.get("scheduler"),
                    show_tqdm=self.config.get("show_tqdm", False),
                    constraint_filter=self.constraint_filter,
                    obstacles=self.obstacles,
                    aug_lambda=self.config.get("aug_lambda", 0.0),
                    aug_rho=self.config.get("aug_rho", 1.0),
                    action_extra_sigma=self.config.get("action_extra_sigma", 0.0),
                )
            else:
                self._backend_impl = backend_cls(
                    env_adapter=self._env_adapter,
                    legacy_energy=self._legacy_energy,
                    horizon=self.horizon,
                    dt=self.dt,
                    Nsample=self.config["Nsample"],
                    Ndiffuse=self.config["Ndiffuse"],
                    temp_sample=self.config["temp_sample"],
                    beta0=self.config["beta0"],
                    betaT=self.config["betaT"],
                    action_limit=self.config["action_limit"],
                    seed=self.seed,
                    scheduler=self.config.get("scheduler"),
                    show_tqdm=self.config.get("show_tqdm", False),
                    constraint_filter=self.constraint_filter,
                    obstacles=self.obstacles,
                    aug_lambda=self.config.get("aug_lambda", 0.0),
                    aug_rho=self.config.get("aug_rho", 1.0),
                    action_extra_sigma=self.config.get("action_extra_sigma", 0.0),
                )
            self._backend_impl = to_unified_backend(self._backend_impl)
        return self._backend_impl

    def _postprocess_multirun_result(
        self, best_result: Dict[str, Any], results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        # Stack all C modes' adaptive metrics for (C, K, 11) storage in adaptive/metrics.json
        planner = self._get_backend_impl()
        K_ref = None
        for r in results:
            for key in ("r_hist", "v_rate_hist", "v_mean_hist"):
                h = r.get(key)
                if h is not None and hasattr(h, "__len__"):
                    K_ref = len(np.asarray(h).ravel())
                    break
            if K_ref is not None:
                break
        if K_ref is None:
            K_ref = int(getattr(planner, "Ndiffuse", 100))

        def _stack_hist(key, default_val=np.nan):
            arrs = []
            for r in results:
                h = r.get(key)
                if h is not None and hasattr(h, "__len__"):
                    a = np.asarray(h, dtype=np.float64).ravel()
                    arrs.append(a[:K_ref] if len(a) >= K_ref else np.resize(a, K_ref))
                else:
                    arrs.append(np.full(K_ref, default_val, dtype=np.float64))
            return np.stack(arrs, axis=0)

        for key in (
            "r_hist",
            "v_rate_hist",
            "v_mean_hist",
            "rho_hist",
            "topK_hist",
            "I_QP_hist",
            "eps_hist",
            "lambda_hist",
            "p_hist",
        ):
            best_result["all_" + key] = _stack_hist(key)
        best_result["all_nu_hist"] = _stack_hist("nu_hist", default_val=np.nan)
        best_result["all_compute_cost_hist"] = _stack_hist("compute_cost_hist", default_val=np.nan)
        return best_result


if register_solver is not None:
    try:
        register_solver("cfsmbd", CFSMBDSolver)
    except Exception:
        pass
