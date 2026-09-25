"""ISSA RL baseline solver (multi-backend wrapper).

Implicit Safe Set Algorithm (CoRL'21 / JAIR). A trained brax policy (the ``learning/`` PPO/SAC
engine) driven closed-loop by the shared ``RLPolicyController``, with the jax backend's AdamBA
projection (``backends/issa_jax.py``) pulling each action into the safe set ``{g ≤ 0}`` before
execution. Thin wrapper: select backend + run_receding.
"""

from __future__ import annotations

from typing import Any, Callable

from genedynamics.solvers.common.rl_policy_controller import RLPolicyController

try:
    from genedynamics.core.registry.solvers import register_solver
except Exception:                                   # pragma: no cover
    register_solver = None


def _get_issa_backend(backend_name: str):
    if backend_name in ("jax", "brax"):
        from genedynamics.solvers.single.issa.backends.issa_jax import make_issa_projection
        return make_issa_projection
    return None


class IssaSolver:
    """Trained policy + implicit-safe-set (AdamBA) projection, driven closed-loop."""

    def __init__(self, env: Any, act_fn: Callable[[Any, Any], Any], *, n_dirs: int = 20,
                 n_iters: int = 50, bound: float = 1e-4, threshold: float = 0.0,
                 enforce_absolute: bool = True, action_limit: float = 1.0,
                 seed: int = 0, step_env: Any = None) -> None:
        make_projection = _get_issa_backend("jax")
        self.projection = make_projection(
            env, n_dirs=n_dirs, n_iters=n_iters, bound=bound,
            threshold=threshold, enforce_absolute=enforce_absolute,
            action_limit=action_limit, seed=seed, step_env=step_env,
        )
        self._ctrl = RLPolicyController(
            step_env or env, act_fn,
            action_projection=self.projection, seed=seed
        )

    def run_receding(self, x0: Any, n_steps: int, rng: Any, **kwargs: Any):
        return self._ctrl.run_receding(x0, n_steps, rng, **kwargs)


if register_solver is not None:
    try:
        register_solver("issa", IssaSolver)
    except Exception:                                # pragma: no cover
        pass


__all__ = ["IssaSolver"]
