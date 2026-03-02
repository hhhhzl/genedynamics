"""
MDOC (genedynamics version).

Definition:
  MDOC = MBD-style diffusion driver + ConstraintFilter (action-space filtering per diffusion step)
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
from genedynamics.solvers.single.edoc import EnergyToLegacyAdapter
from genedynamics.core.constraints.action_filters import ConstraintFilter, NoOpConstraintFilter

try:
    from genedynamics.core.registry.solvers import register_solver
except Exception:
    register_solver = None


def _get_mdoc_backend(backend_name: str):
    if backend_name == "jax":
        from genedynamics.solvers.single.mdoc.backends.mdoc_jax import MDOCBackendJax

        return MDOCBackendJax
    if backend_name == "numpy":
        from genedynamics.solvers.single.mdoc.backends.mdoc_numpy import MDOCBackendNumpy

        return MDOCBackendNumpy
    return None


class MDOCSolver(SamplingSolver):
    """
    Solver wrapper: delegates to MDOC backend.
    """

    def __init__(
        self,
        dynamics: DynamicsModel,
        energy: EnergyFunctional,
        backend: Backend,
        *,
        horizon: int = 64,
        dt: float = 0.05,
        Nsample: int = 256,
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
        num_modes: int = 1,
        mode_strategy: str = "multirun",
        diversity_eta: float = 1.0,
        diversity_topK_cand: int = None,
        diversity_use_state: bool = True,
        use_target_line: bool = False,
        num_targets: int = 4,
        **kwargs: Any,
    ):
        super().__init__(dynamics, energy, backend, **kwargs)

        if backend.name not in {"jax", "numpy"}:
            raise ValueError(
                f"MDOC solver currently supports backends {{'jax','numpy'}}, got {backend.name}."
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
                num_modes=int(num_modes),
                mode_strategy=str(mode_strategy),
                diversity_eta=float(diversity_eta),
                diversity_topK_cand=int(diversity_topK_cand) if diversity_topK_cand is not None else None,
                diversity_use_state=bool(diversity_use_state),
                use_target_line=bool(use_target_line),
                num_targets=int(num_targets),
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
        # Store additional CBF params
        self.cbf_params = {
            "cbf_tau": kwargs.get("cbf_tau", 0.005),
            "cbf_eta": kwargs.get("cbf_eta", 1.5),
            "cbf_margin": kwargs.get("cbf_margin", 0.1),
            "base_beta": kwargs.get("base_beta", 0.05),
            "terminal_energy_weight": kwargs.get("terminal_energy_weight", 50.0),
            "guide_weight": kwargs.get("guide_weight", 10.0),
        }

    def _get_backend_impl(self):
        if self._backend_impl is None:
            backend = RuntimeBackendManager.get_backend()
            backend_cls = _get_mdoc_backend(backend.name)
            if backend_cls is None:
                raise ValueError(f"MDOC backend '{backend.name}' not found")
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
                    **self.cbf_params,
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
                    **self.cbf_params,
                )
        return self._backend_impl

    def sample_trajectories(self, x0: State, horizon: int, n_samples: int, **kwargs) -> List[Trajectory]:
        if horizon != self.horizon:
            self.horizon = int(horizon)
            self._backend_impl = None
        planner = self._get_backend_impl()
        x0_data = np.asarray(x0, dtype=np.float32) if not isinstance(x0, jnp.ndarray) else x0
        rng_key = kwargs.get("rng_key", jax.random.PRNGKey(self.seed))
        return planner.sample_trajectories(x0_data, n_samples, rng_key=rng_key)

    def solve(self, x0: State, horizon: int, **kwargs) -> Trajectory:
        if horizon != self.horizon:
            self.horizon = int(horizon)
            self._backend_impl = None
        planner = self._get_backend_impl()
        x0_data = np.asarray(x0, dtype=np.float32) if not isinstance(x0, jnp.ndarray) else x0
        rng_key = kwargs.get("rng_key", jax.random.PRNGKey(self.seed))
        C = int(self.config.get("num_modes", 1))
        mode_strategy = str(self.config.get("mode_strategy", "multirun")).lower()
        if C > 1 and mode_strategy == "multirun":
            backend = RuntimeBackendManager.get_backend()
            if backend.name == "jax":
                # Ensure rng_key is a JAX PRNG key (convert if needed)
                if not isinstance(rng_key, jnp.ndarray) or rng_key.shape != (2,):
                    # If not a JAX key, create one from seed or convert
                    if isinstance(rng_key, (int, np.integer)):
                        rng_key = jax.random.PRNGKey(int(rng_key))
                    else:
                        # Try to extract seed or use default
                        rng_key = jax.random.PRNGKey(self.seed)
                keys = jax.random.split(rng_key, C)
            else:
                # NumPy backend: convert to integer seeds
                if isinstance(rng_key, jnp.ndarray) and rng_key.shape == (2,):
                    # Convert JAX key to integer seed (use hash of key values)
                    base_seed = int(rng_key[0]) ^ int(rng_key[1])
                elif isinstance(rng_key, (int, np.integer)):
                    base_seed = int(rng_key)
                else:
                    base_seed = self.seed
                # Generate C different seeds
                keys = [base_seed + i for i in range(C)]
            planner_num_modes_orig = int(getattr(planner, "num_modes", 1))
            planner.num_modes = 1
            try:
                # Use batch version for parallel execution if available
                if hasattr(planner, "plan_batch"):
                    results = planner.plan_batch(x0_data, keys)
                else:
                    # Fallback to sequential if batch not available
                    results = [planner.plan(x0_data, k) for k in keys]
            finally:
                planner.num_modes = planner_num_modes_orig

            candidate_states_list = [np.asarray(r["states"], dtype=np.float32) for r in results]
            candidate_actions_list = [np.asarray(r["actions"], dtype=np.float32) for r in results]
            candidate_costs = np.asarray(
                [float(np.asarray(r.get("candidate_costs", [np.nan]))[int(r.get("best_idx", 0))]) for r in results],
                dtype=np.float32,
            )
            best_idx = int(np.nanargmin(candidate_costs))
            best_result = dict(results[best_idx])
            best_result["candidate_states"] = candidate_states_list
            best_result["candidate_actions"] = candidate_actions_list
            best_result["candidate_costs"] = candidate_costs
            best_result["best_idx"] = best_idx
            best_result["mode_strategy"] = "multirun"
            best_result["multirun_keys"] = keys
            best_result["multirun_diffusion_data"] = [
                {"diffusion_actions_traj": r.get("diffusion_actions_traj"), "diffusion_sampled_actions": r.get("diffusion_sampled_actions")}
                for r in results
            ]
            result = best_result
        else:
            result = planner.plan(x0_data, rng_key)
        states_list = [np.asarray(s, dtype=np.float32) for s in result["states"]]
        actions_list = [np.asarray(a, dtype=np.float32) for a in result["actions"]]
        return Trajectory(states=states_list, actions=actions_list, info=result)


if register_solver is not None:
    try:
        register_solver("mdoc", MDOCSolver)
    except Exception:
        pass


def run_mdoc(args):
    """Backward-compatible entry point (similar to run_mbd)."""
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
    backend = get_backend(getattr(args, "backend", "jax"))

    solver = MDOCSolver(
        dynamics=dynamics,
        energy=energy,
        backend=backend,
        horizon=args.horizon,
        dt=args.dt,
        Nsample=args.Nsample,
        Ndiffuse=args.Ndiffuse,
        temp_sample=args.temp_sample,
        beta0=args.beta0,
        betaT=args.betaT,
        action_limit=args.action_limit,
        seed=args.seed,
        scheduler=getattr(args, "scheduler", None),
        obstacles=getattr(args, "obstacles", None),
    )

    traj = solver.solve(x0=env.reset()[0], horizon=args.horizon, rng_key=jax.random.PRNGKey(args.seed))
    return traj.info if hasattr(traj, "info") else {"actions": np.stack(traj.actions), "states": np.stack(traj.states)}


