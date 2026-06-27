"""DIAL-MPC solver (multi-backend wrapper).

Follows the genedynamics MBD-family pattern (cf. mbd.py / twogo.py): a thin
``BaseModelBasedDiffusionSolver`` subclass that selects a backend at runtime via
``RuntimeBackendManager`` + ``_get_dial_backend`` and wraps it with
``to_unified_backend``. The backend (``backends/dial_jax.py``) carries the
reverse-update math AND the ``WarmStartPlanner`` capability.

DIAL-MPC is receding-horizon, so ``solve()`` is overridden to drive the
backend-agnostic bridge (``solvers/common/receding_horizon.py``):

    DIAL-MPC == RecedingHorizonController(<DIAL backend>)

The backend's single-shot ``plan()`` remains available (open-loop full-horizon
plan) for the base Solver/multirun path. Reward is the env's own reward (= the
solver's ``LegacyEnergyFunctional``, negated), kept free of MBD guidance/terminal
shaping so the reproduction is faithful to upstream dial-mpc.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from genedynamics.solvers.common.model_based_diffusion.base_solver import (
    BaseModelBasedDiffusionSolver,
)
from genedynamics.solvers.common.receding_horizon import RecedingHorizonController
from genedynamics.solvers.single.dial.backend_impl import to_unified_backend
from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.core.dynamics import DynamicsToEnvAdapter
from genedynamics.core.energy import LegacyEnergyFunctional, EnergyToLegacyAdapter
from genedynamics.core.types import Trajectory

try:
    from genedynamics.core.registry.solvers import register_solver
except Exception:  # pragma: no cover
    register_solver = None


def _get_dial_backend(backend_name: str):
    if backend_name == "jax":
        from genedynamics.solvers.single.dial.backends.dial_jax import DialBackendJax

        return DialBackendJax
    return None


class DIALMPCSolver(BaseModelBasedDiffusionSolver):
    """Receding-horizon DIAL-MPC over the node-spline weighted-mean backend."""

    def __init__(
        self,
        dynamics: Any,
        energy: Any,
        backend: Any,
        *,
        nu: Optional[int] = None,
        rollout_fn: Optional[Any] = None,
        step_fn: Optional[Any] = None,
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
        n_steps: Optional[int] = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(dynamics, energy, backend, **kwargs)

        self.horizon = int(Hsample)
        self.dt = float(dt)
        self.seed = int(seed)
        self.n_steps = n_steps
        # brax PipelineEnv exposes action_size (not act_dim); flat envs use act_dim
        self._is_brax_env = dynamics is not None and hasattr(dynamics, "pipeline_step")
        if nu is not None:
            self.nu = int(nu)
        else:
            self.nu = int(getattr(dynamics, "act_dim", 0) or getattr(dynamics, "action_size", 0))

        # injected rollout/step (CPU tests) — backend reads these off the solver
        self._rollout_fn = rollout_fn
        self._step_fn = step_fn

        # env transition + reward adapters (flat-state mjx path, mirrors mbd.py).
        # brax envs roll out via env.step directly, so they need no flat adapter.
        self._env_adapter = (
            DynamicsToEnvAdapter(dynamics, dt)
            if dynamics is not None and not self._is_brax_env
            else None
        )
        if isinstance(energy, LegacyEnergyFunctional) or energy is None:
            self._legacy_energy = energy
        else:
            self._legacy_energy = EnergyToLegacyAdapter(energy, dynamics).legacy_energy

        self.config.update(
            dict(
                Hsample=int(Hsample), Hnode=int(Hnode), Nsample=int(Nsample),
                Ndiffuse=int(Ndiffuse), Ndiffuse_init=int(Ndiffuse_init),
                temp_sample=float(temp_sample),
                horizon_diffuse_factor=float(horizon_diffuse_factor),
                traj_diffuse_factor=float(traj_diffuse_factor),
                sigma_scale=float(sigma_scale), action_limit=float(action_limit),
                ctrl_dt=float(ctrl_dt),
            )
        )
        self._backend_impl = None

    def _get_backend_impl(self):
        if self._backend_impl is None:
            backend = RuntimeBackendManager.get_backend()
            backend_cls = _get_dial_backend(backend.name)
            if backend_cls is None:
                raise ValueError(f"DIAL backend '{backend.name}' not found")
            self._backend_impl = to_unified_backend(backend_cls(solver=self))
        return self._backend_impl

    # --- receding-horizon execution (DIAL's defining behaviour) ---

    def make_controller(self, n_steps: int, **kw: Any) -> RecedingHorizonController:
        backend = self._get_backend_impl()
        step_fn = self._step_fn or getattr(backend, "_step_fn", None)
        if step_fn is None:
            raise ValueError("DIAL needs a step_fn (inject one or provide dynamics).")
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
        """Run DIAL receding-horizon control. ``horizon`` = episode length
        (executed steps); the planning horizon is ``Hsample``."""
        import jax

        n = int(kwargs.pop("n_steps", None) or self.n_steps or horizon)
        rng = kwargs.pop("rng_key", None)
        if rng is None:
            rng = jax.random.PRNGKey(self.seed)
        res = self.make_controller(n).run(x0, rng)
        states = [self._flatten_state(s) for s in res.states]
        actions = [np.asarray(a, dtype=np.float32) for a in res.actions]
        return Trajectory(states=states, actions=actions, info={"source": "dial_mpc"})

    @staticmethod
    def _flatten_state(s: Any) -> np.ndarray:
        """Flat [qpos; qvel] for the Trajectory. brax State -> pipeline_state
        qpos/qvel; flat arrays pass through."""
        ps = getattr(s, "pipeline_state", None)
        if ps is not None and hasattr(ps, "qpos"):
            return np.concatenate(
                [np.asarray(ps.qpos, np.float32), np.asarray(ps.qvel, np.float32)]
            )
        return np.asarray(s, dtype=np.float32)


if register_solver is not None:
    try:
        register_solver("dial", DIALMPCSolver)
    except Exception:  # pragma: no cover
        pass
