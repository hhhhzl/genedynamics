"""PegasusFlow baseline solver (brax-native basis sampling MPC, WBFO-style).

Thin multi-backend wrapper (cf. ``dial.py``): selects the jax backend
(``backends/pegasusflow_jax.py`` — a ``WarmStartPlanner``) and drives the shared
receding-horizon bridge. No flat-state dynamics/energy adapters — it owns the brax env.
"""

from __future__ import annotations

from typing import Any, Optional

from genedynamics.solvers.common.receding_horizon import RecedingHorizonController

try:
    from genedynamics.core.registry.solvers import register_solver
except Exception:                                   # pragma: no cover
    register_solver = None


def _get_pegasusflow_backend(backend_name: str):
    if backend_name in ("jax", "brax"):
        from genedynamics.solvers.single.pegasusflow.backends.pegasusflow_jax import (
            PegasusFlowBackendJax,
        )
        return PegasusFlowBackendJax
    return None


class PegasusFlowSolver:
    """Selects the jax backend (WarmStartPlanner) and runs receding-horizon control."""

    def __init__(self, dynamics: Any, energy: Any = None, backend: Any = None, *,
                 nu: Optional[int] = None, Hsample: int = 16, Hnode: int = 4, Nsample: int = 2048,
                 noise_sigma: float = 0.3, sigma_decay: float = 0.9, temp_sample: float = 0.1,
                 action_limit: float = 1.0, Ndiffuse: int = 2, Ndiffuse_init: int = 10,
                 seed: int = 0, rollout_fn: Any = None, step_fn: Any = None, **kwargs: Any) -> None:
        self.dynamics = dynamics
        self.nu = int(nu if nu is not None
                      else getattr(dynamics, "action_size", 0) or getattr(dynamics, "act_dim", 0))
        self.seed = int(seed)
        self._rollout_fn, self._step_fn = rollout_fn, step_fn
        self.config = dict(
            Hsample=int(Hsample), Hnode=int(Hnode), Nsample=int(Nsample),
            noise_sigma=float(noise_sigma), sigma_decay=float(sigma_decay),
            temp_sample=float(temp_sample), action_limit=float(action_limit),
            Ndiffuse=int(Ndiffuse), Ndiffuse_init=int(Ndiffuse_init),
        )
        self._backend_impl = None

    def _get_backend_impl(self):
        if self._backend_impl is None:
            cls = _get_pegasusflow_backend("jax")
            self._backend_impl = cls(solver=self)
        return self._backend_impl

    def make_controller(self, n_steps: int, **kw: Any) -> RecedingHorizonController:
        b = self._get_backend_impl()
        return RecedingHorizonController(
            b, step_fn=b._step_fn, n_steps=int(n_steps),
            n_diffuse_init=int(self.config["Ndiffuse_init"]),
            n_diffuse=int(self.config["Ndiffuse"]), **kw)

    def run_receding(self, x0: Any, n_steps: int, rng: Any, **kw: Any):
        return self.make_controller(int(n_steps), **kw).run(x0, rng)


if register_solver is not None:
    try:
        register_solver("pegasusflow", PegasusFlowSolver)
    except Exception:                                # pragma: no cover
        pass


__all__ = ["PegasusFlowSolver"]
