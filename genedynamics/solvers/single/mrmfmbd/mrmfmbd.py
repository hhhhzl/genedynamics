"""
MRMFMBD (Soft-robot S1+S3) solver.

Multi-Resolution, Multi-Fidelity Model-Based Diffusion for deformable/soft robots.
Uses fidelity ladder (coarse S1 -> fine S3) and optional mode-marginalization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from genedynamics.solvers.common.model_based_diffusion import BaseModelBasedDiffusionSolver
from genedynamics.core.dynamics import DynamicsModel
from genedynamics.core.energy import EnergyFunctional
from genedynamics.core.backends import Backend
from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.core.types import State, Trajectory
from genedynamics.solvers.single.mrmfmbd.backend_impl import to_unified_backend

try:
    from genedynamics.core.registry.solvers import register_solver
except Exception:
    register_solver = None


def _get_mrmfmbd_backend(backend_name: str):
    if backend_name == "jax":
        from genedynamics.solvers.single.mrmfmbd.backends.mrmfmbd_jax import MRMFMBDBackendJax
        return MRMFMBDBackendJax
    return None


class MRMFMBDSolver(BaseModelBasedDiffusionSolver):
    """
    Solver for soft-robot MR-MF-MBD.

    Multi-fidelity diffusion: early steps use coarse sim (S1), late steps use fine (S3).
    """

    def __init__(
        self,
        dynamics: DynamicsModel,
        energy: EnergyFunctional,
        backend: Backend,
        *,
        fidelity_simulator: Any,
        fidelity_ladder: Any,
        horizon: int = 80,
        Nsample: int = 64,
        Ndiffuse: int = 100,
        temp_sample: float = 0.5,
        beta0: float = 1e-4,
        betaT: float = 1e-2,
        action_limit: float = 1.0,
        action_extra_sigma: float = 0.0,
        seed: int = 0,
        scheduler: Any = None,
        show_tqdm: bool = False,
        mode_marginalizer: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__(dynamics, energy, backend, **kwargs)

        if backend.name != "jax":
            raise ValueError("MRMFMBD currently supports only JAX backend.")

        self.horizon = horizon
        self.seed = seed

        self.config.update(
            dict(
                fidelity_simulator=fidelity_simulator,
                fidelity_ladder=fidelity_ladder,
                Nsample=Nsample,
                Ndiffuse=Ndiffuse,
                temp_sample=temp_sample,
                beta0=beta0,
                betaT=betaT,
                action_limit=action_limit,
                action_extra_sigma=action_extra_sigma,
                scheduler=scheduler,
                show_tqdm=bool(show_tqdm),
                mode_marginalizer=mode_marginalizer,
            )
        )

        self._backend_impl = None

    def _get_backend_impl(self):
        if self._backend_impl is None:
            backend = RuntimeBackendManager.get_backend()
            backend_cls = _get_mrmfmbd_backend(backend.name)
            if backend_cls is None:
                raise ValueError(f"MRMFMBD backend '{backend.name}' not found")

            self._backend_impl = backend_cls(
                fidelity_simulator=self.config["fidelity_simulator"],
                fidelity_ladder=self.config["fidelity_ladder"],
                horizon=self.horizon,
                Nsample=self.config["Nsample"],
                Ndiffuse=self.config["Ndiffuse"],
                temp_sample=self.config["temp_sample"],
                beta0=self.config["beta0"],
                betaT=self.config["betaT"],
                action_limit=self.config["action_limit"],
                action_extra_sigma=self.config.get("action_extra_sigma", 0.0),
                seed=self.seed,
                scheduler=self.config.get("scheduler"),
                show_tqdm=self.config.get("show_tqdm", False),
                mode_marginalizer=self.config.get("mode_marginalizer"),
            )
            self._backend_impl = to_unified_backend(self._backend_impl)
        return self._backend_impl


if register_solver is not None:
    try:
        register_solver("mrmfmbd", MRMFMBDSolver)
    except Exception:
        pass
