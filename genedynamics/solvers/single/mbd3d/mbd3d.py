"""
MBD3D (3DGS robust mapping) solver.

Annealed bridge + MCSA for joint scene and camera trajectory estimation
with observation likelihood. Uses core.inference and core.prob.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import jax

from genedynamics.solvers.common.model_based_diffusion import BaseModelBasedDiffusionSolver
from genedynamics.core.dynamics import DynamicsModel
from genedynamics.core.energy import EnergyFunctional
from genedynamics.core.backends import Backend
from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.core.types import State, Trajectory
from genedynamics.solvers.single.mbd3d.backend_impl import to_unified_backend

try:
    from genedynamics.core.registry.solvers import register_solver
except Exception:
    register_solver = None


def _get_mbd3d_backend(backend_name: str):
    if backend_name == "jax":
        from genedynamics.solvers.single.mbd3d.backends.mbd3d_jax import MBD3DBackendJax
        return MBD3DBackendJax
    return None


class MBD3DSolver(BaseModelBasedDiffusionSolver):
    """
    Solver for 3DGS robust mapping via annealed bridge + MCSA.

    State θ = (scene_params, camera_trajectory).
    Bridge: π_k(θ) ∝ p0(θ) * p(y|θ)^β_k.
    Requires observations (ObservationBundle) passed in solve(..., observations=...).
    """

    def __init__(
        self,
        dynamics: DynamicsModel,
        energy: EnergyFunctional,
        backend: Backend,
        *,
        scene_repr: Any,
        renderer: Any,
        likelihood: Any,
        bridge_schedule: Any,
        camera_prior: Optional[Any] = None,
        horizon: int = 10,
        n_gaussians: int = 32,
        M: int = 16,
        sigma_mcsa: float = 0.05,
        ess_min: float = 1.0,
        seed: int = 0,
        show_tqdm: bool = False,
        fidelity_ladder: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__(dynamics, energy, backend, **kwargs)

        if backend.name != "jax":
            raise ValueError("MBD3D currently supports only JAX backend.")

        self.horizon = horizon
        self.seed = seed
        self._current_observations = None

        self.config.update(
            dict(
                scene_repr=scene_repr,
                renderer=renderer,
                likelihood=likelihood,
                bridge_schedule=bridge_schedule,
                camera_prior=camera_prior,
                n_gaussians=n_gaussians,
                M=M,
                sigma_mcsa=sigma_mcsa,
                ess_min=ess_min,
                show_tqdm=bool(show_tqdm),
                fidelity_ladder=fidelity_ladder,
            )
        )

        self._backend_impl = None

    def _get_backend_impl(self):
        if self._backend_impl is None:
            backend = RuntimeBackendManager.get_backend()
            backend_cls = _get_mbd3d_backend(backend.name)
            if backend_cls is None:
                raise ValueError(f"MBD3D backend '{backend.name}' not found")

            raw = backend_cls(
                scene_repr=self.config["scene_repr"],
                renderer=self.config["renderer"],
                likelihood=self.config["likelihood"],
                bridge_schedule=self.config["bridge_schedule"],
                camera_prior=self.config.get("camera_prior"),
                horizon=self.horizon,
                n_gaussians=self.config["n_gaussians"],
                M=self.config["M"],
                sigma_mcsa=self.config["sigma_mcsa"],
                ess_min=self.config["ess_min"],
                seed=self.seed,
                show_tqdm=self.config.get("show_tqdm", False),
                fidelity_ladder=self.config.get("fidelity_ladder"),
            )
            self._backend_impl = _MBD3DObservationAdapter(raw, self)
            self._backend_impl = to_unified_backend(self._backend_impl)
        return self._backend_impl


    def solve(
        self,
        x0: State,
        horizon: int,
        observations: Optional[Any] = None,
        **kwargs: Any,
    ) -> Trajectory:
        """
        Solve for scene and camera trajectory given observations.

        Args:
            x0: initial state (camera pose hint or scene init)
            horizon: planning horizon (number of views - 1)
            observations: ObservationBundle (required)
            **kwargs: passed to base (e.g. rng_key)

        Returns:
            Trajectory with states=camera poses, actions=pose deltas, info=scene_params, etc.
        """
        if observations is None:
            raise ValueError("MBD3D requires observations. Pass observations=ObservationBundle(...)")
        self._current_observations = observations
        self._ensure_horizon(horizon)

        planner = self._get_backend_impl()
        x0_data = self._prepare_state(x0)
        rng_key = self._resolve_rng_key(kwargs)

        result = planner.plan(x0_data, rng_key=rng_key)

        states_list = [np.asarray(s, dtype=np.float32) for s in result["states"]]
        actions_list = [np.asarray(a, dtype=np.float32) for a in result["actions"]]
        return Trajectory(states=states_list, actions=actions_list, info=result)

    def sample_trajectories(
        self,
        x0: State,
        horizon: int,
        n_samples: int,
        observations: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Trajectory]:
        self._current_observations = observations
        self._ensure_horizon(horizon)
        planner = self._get_backend_impl()
        x0_data = self._prepare_state(x0)
        rng_key = self._resolve_rng_key(kwargs)
        return planner.sample_trajectories(x0_data, n_samples, rng_key=rng_key)


class _MBD3DObservationAdapter:
    """Adapter that injects solver's _current_observations into plan()."""

    def __init__(self, backend: Any, solver: Any):
        self._backend = backend
        self._solver = solver

    def plan(self, x0: Any, rng_key: Any = None) -> Dict[str, Any]:
        obs = getattr(self._solver, "_current_observations", None)
        return self._backend.plan(x0, rng_key=rng_key, observations=obs)

    def sample_trajectories(
        self, x0: Any, n_samples: int, rng_key: Any = None
    ) -> List[Trajectory]:
        obs = getattr(self._solver, "_current_observations", None)
        return self._backend.sample_trajectories(x0, n_samples, rng_key=rng_key, observations=obs)

    def plan_batch(self, x0: Any, keys: Any) -> List[Dict[str, Any]]:
        obs = getattr(self._solver, "_current_observations", None)
        return self._backend.plan_batch(x0, keys, observations=obs)


if register_solver is not None:
    try:
        register_solver("mbd3d", MBD3DSolver)
    except Exception:
        pass
