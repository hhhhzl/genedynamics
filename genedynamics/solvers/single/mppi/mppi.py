"""
MPPI (Model Predictive Path Integral) solver with EDOC-style architecture.

This refactor mirrors EDOC's planner + solver separation while
implementing only the JAX backend for now.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.core.solvers import SamplingSolver
from genedynamics.core.dynamics import DynamicsModel, DynamicsToEnvAdapter
from genedynamics.core.energy import EnergyFunctional, LegacyEnergyFunctional
from genedynamics.core.backends import Backend
from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.core.types import State, Trajectory
from genedynamics.core.energy import EnergyToLegacyAdapter

try:
    from genedynamics.core.registry.solvers import register_solver
except Exception:
    register_solver = None


# ============================================================================
# Backend registry helpers
# ============================================================================
try:
    from genedynamics.solvers.single.mppi.backends import mppi_jax, mppi_numpy  
except ImportError:
    pass


def _get_mppi_backend(backend_name: str):
    if backend_name == "jax":
        from genedynamics.solvers.single.mppi.backends.mppi_jax import MPPIBackendJax

        return MPPIBackendJax
    if backend_name == "numpy":
        from genedynamics.solvers.single.mppi.backends.mppi_numpy import MPPIBackendNumpy

        return MPPIBackendNumpy
    return None


# ============================================================================
# MPPI Solver (wrapper)
# ============================================================================

class MPPISolver(SamplingSolver):
    """Solver wrapper that follows EDOC-like structure but runs MPPI."""

    def __init__(
        self,
        dynamics: DynamicsModel,
        energy: EnergyFunctional,
        backend: Backend,
        *,
        horizon: int = 80,
        dt: float = 0.1,
        num_samples: int = 512,
        num_iterations: int = 6,
        noise_sigma: float = 0.3,
        lambda_: float = 1.0,
        action_limit: float = 1.0,
        terminal_energy_weight: float = 0.0,
        guide_weight: float = 0.0,
        position_extractor: Optional[Any] = None,
        position_dim: int = 2,
        seed: int = 0,
        # brax sampling-MPC budget (used only on the brax-env path; flat path ignores them)
        Hsample: int = 16,
        Hnode: int = 4,
        Nsample: int = 2048,
        Ndiffuse: int = 2,
        Ndiffuse_init: int = 10,
        rollout_fn: Optional[Any] = None,
        step_fn: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(dynamics, energy, backend, **kwargs)

        if backend.name not in {"jax", "numpy"}:
            raise ValueError(
                f"MPPI solver supports backends {{'jax','numpy'}}, got {backend.name}."
            )

        self.horizon = horizon
        self.dt = dt
        self.seed = seed
        self.config.update(
            dict(
                num_samples=num_samples,
                num_iterations=num_iterations,
                noise_sigma=noise_sigma,
                lambda_=lambda_,
                action_limit=action_limit,
                terminal_energy_weight=float(terminal_energy_weight),
                guide_weight=float(guide_weight),
                Hsample=int(Hsample), Hnode=int(Hnode), Nsample=int(Nsample),
                Ndiffuse=int(Ndiffuse), Ndiffuse_init=int(Ndiffuse_init),
            )
        )
        self.position_extractor = position_extractor
        self.position_dim = int(position_dim)

        # brax PipelineEnv: roll out env.step directly (no flat-state adapter, like DIAL);
        # flat-state dynamics keep the DynamicsToEnvAdapter path unchanged.
        self._is_brax_env = dynamics is not None and hasattr(dynamics, "pipeline_step")
        self.nu = int(getattr(dynamics, "action_size", 0) or getattr(dynamics, "act_dim", 0))
        self._rollout_fn, self._step_fn = rollout_fn, step_fn
        self._env_adapter = None if self._is_brax_env else DynamicsToEnvAdapter(dynamics, dt)
        if energy is None or isinstance(energy, LegacyEnergyFunctional):
            self._legacy_energy = energy
        else:
            self._legacy_energy = EnergyToLegacyAdapter(energy, dynamics).legacy_energy

        self._backend_impl = None
        self._brax_backend_impl = None

    def _get_backend_impl(self):
        if self._backend_impl is None:
            backend = RuntimeBackendManager.get_backend()
            backend_cls = _get_mppi_backend(backend.name)
            if backend_cls is None:
                raise ValueError(f"MPPI backend '{backend.name}' not found")
            self._backend_impl = backend_cls(
                env_adapter=self._env_adapter,
                legacy_energy=self._legacy_energy,
                horizon=self.horizon,
                dt=self.dt,
                num_samples=self.config["num_samples"],
                num_iterations=self.config["num_iterations"],
                noise_sigma=self.config["noise_sigma"],
                lambda_=self.config["lambda_"],
                action_limit=self.config["action_limit"],
                terminal_energy_weight=self.config["terminal_energy_weight"],
                guide_weight=self.config["guide_weight"],
                position_extractor=self.position_extractor,
                position_dim=self.position_dim,
                seed=self.seed,
            )
        return self._backend_impl

    def sample_trajectories(
        self,
        x0: State,
        horizon: int,
        n_samples: int,
        **kwargs,
    ) -> List[Trajectory]:
        if horizon != self.horizon:
            self.horizon = horizon
            self._planner = None
        planner = self._get_backend_impl()

        x0_data = np.asarray(x0, dtype=np.float32) if not isinstance(x0, jnp.ndarray) else x0
        rng_key = kwargs.get("rng_key", jax.random.PRNGKey(self.seed))
        return planner.sample_trajectories(x0_data, n_samples, rng_key=rng_key)

    def solve(
        self,
        x0: State,
        horizon: int,
        **kwargs,
    ) -> Trajectory:
        if horizon != self.horizon:
            self.horizon = horizon
            self._planner = None
        planner = self._get_backend_impl()

        x0_data = np.asarray(x0, dtype=np.float32) if not isinstance(x0, jnp.ndarray) else x0
        rng_key = kwargs.get("rng_key", jax.random.PRNGKey(self.seed))
        result = planner.plan(x0_data, rng_key)

        actions_list = [np.asarray(a, dtype=np.float32) for a in result["actions"]]
        states_list = [np.asarray(s, dtype=np.float32) for s in result["states"]]

        return Trajectory(states=states_list, actions=actions_list, info=result)

    # --- brax-native receding-horizon path (arm / box-push comparison) ---
    # On a brax env, MPPI runs through the SAME shared bridge as DIAL/MGA using the
    # path-integral backend (backends/mppi_brax_jax.py), warm-started across steps.
    def _get_brax_backend_impl(self):
        if self._brax_backend_impl is None:
            from genedynamics.solvers.single.mppi.backends.mppi_brax_jax import MPPIBraxBackendJax
            self._brax_backend_impl = MPPIBraxBackendJax(solver=self)
        return self._brax_backend_impl

    def make_controller(self, n_steps: int, **kw: Any):
        from genedynamics.solvers.common.receding_horizon import RecedingHorizonController
        b = self._get_brax_backend_impl()
        return RecedingHorizonController(
            b, step_fn=b._step_fn, n_steps=int(n_steps),
            n_diffuse_init=int(self.config.get("Ndiffuse_init", 10)),
            n_diffuse=int(self.config.get("Ndiffuse", 2)), **kw)

    def run_receding(self, x0: Any, n_steps: int, rng: Any, **kw: Any):
        return self.make_controller(int(n_steps), **kw).run(x0, rng)


if register_solver is not None:
    try:
        register_solver("mppi", MPPISolver)
    except Exception:
        pass


# ============================================================================
# Main Entry Point (backward compatibility)
# ============================================================================

def run_mppi(args, initial_state: Optional[np.ndarray] = None):
    """Run MPPI planner (backward-compatible entry point)."""
    from genedynamics.envs.factories import make_env, make_energy
    from genedynamics.core import get_backend

    env = make_env(args.env_name)
    energy = make_energy(args.env_name)

    if hasattr(env, "dt"):
        env.dt = args.dt
    if hasattr(env, "horizon"):
        env.horizon = args.horizon
    if hasattr(env, "control_limit"):
        env.control_limit = args.action_limit

    dynamics = DynamicsToEnvAdapter(env, dt=args.dt)
    backend = get_backend("jax")

    solver = MPPISolver(
        dynamics=dynamics,
        energy=energy,
        backend=backend,
        horizon=args.horizon,
        dt=args.dt,
        num_samples=args.num_samples,
        num_iterations=args.num_iterations,
        noise_sigma=args.noise_sigma,
        lambda_=args.lambda_,
        action_limit=args.action_limit,
        seed=args.seed,
    )

    if initial_state is None:
        rng = jax.random.PRNGKey(args.seed)
        try:
            x0, _ = env.reset(rng)
        except TypeError:
            x0, _ = env.reset()
    else:
        x0 = np.asarray(initial_state, dtype=np.float32)

    traj = solver.solve(x0, horizon=args.horizon, rng_key=jax.random.PRNGKey(args.seed))
    return traj.info if hasattr(traj, "info") else {
        "actions": np.stack(traj.actions, axis=0),
        "states": np.stack(traj.states, axis=0),
    }
