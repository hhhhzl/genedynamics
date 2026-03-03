"""
Unified wrapper base for model-based diffusion solvers.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.core.solvers import SamplingSolver
from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.core.types import State, Trajectory

from .multirun import split_multirun_keys, run_multirun, aggregate_multirun_results


class BaseModelBasedDiffusionSolver(SamplingSolver):
    """Common solve/sample skeleton for MBD-family wrappers."""

    @abstractmethod
    def _get_backend_impl(self):
        raise NotImplementedError

    def _on_horizon_changed(self) -> None:
        """Invalidate backend cache when horizon changes."""
        if hasattr(self, "_backend_impl"):
            self._backend_impl = None

    def _prepare_state(self, x0: State) -> Any:
        return np.asarray(x0, dtype=np.float32) if not isinstance(x0, jnp.ndarray) else x0

    def _resolve_rng_key(self, kwargs: Dict[str, Any]) -> Any:
        return kwargs.get("rng_key", jax.random.PRNGKey(int(getattr(self, "seed", 0))))

    def _get_backend_name(self) -> str:
        return RuntimeBackendManager.get_backend().name

    def _plan_once(self, planner: Any, x0_data: Any, rng_key: Any) -> Dict[str, Any]:
        return planner.plan(x0_data, rng_key)

    def _plan_batch_or_none(self, planner: Any, x0_data: Any, keys: List[Any]):
        if hasattr(planner, "plan_batch"):
            return planner.plan_batch(x0_data, keys)
        return None

    def _postprocess_multirun_result(
        self, best_result: Dict[str, Any], results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Subclass hook for additional aggregate fields."""
        _ = results
        return best_result

    def _ensure_horizon(self, horizon: int) -> None:
        if horizon != self.horizon:
            self.horizon = int(horizon)
            self._on_horizon_changed()

    def sample_trajectories(
        self,
        x0: State,
        horizon: int,
        n_samples: int,
        **kwargs,
    ) -> List[Trajectory]:
        self._ensure_horizon(horizon)
        planner = self._get_backend_impl()
        x0_data = self._prepare_state(x0)
        rng_key = self._resolve_rng_key(kwargs)
        return planner.sample_trajectories(x0_data, n_samples, rng_key=rng_key)

    def solve(
        self,
        x0: State,
        horizon: int,
        **kwargs,
    ) -> Trajectory:
        self._ensure_horizon(horizon)
        planner = self._get_backend_impl()
        x0_data = self._prepare_state(x0)
        rng_key = self._resolve_rng_key(kwargs)

        C = int(self.config.get("num_modes", 1))
        mode_strategy = str(self.config.get("mode_strategy", "multirun")).lower()
        if C > 1 and mode_strategy == "multirun":
            backend_name = self._get_backend_name()
            keys = split_multirun_keys(rng_key, C, backend_name, int(getattr(self, "seed", 0)))
            batch_fn = self._plan_batch_or_none
            results = run_multirun(
                planner=planner,
                x0_data=x0_data,
                keys=keys,
                plan_once_fn=self._plan_once,
                plan_batch_fn=batch_fn,
            )
            result = aggregate_multirun_results(results, keys)
            result = self._postprocess_multirun_result(result, results)
        else:
            result = self._plan_once(planner, x0_data, rng_key)

        states_list = [np.asarray(s, dtype=np.float32) for s in result["states"]]
        actions_list = [np.asarray(a, dtype=np.float32) for a in result["actions"]]
        return Trajectory(states=states_list, actions=actions_list, info=result)

