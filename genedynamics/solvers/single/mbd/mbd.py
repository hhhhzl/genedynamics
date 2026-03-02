"""
MBD (Multi-scale Barrier Diffusion) solver with EDOC-style architecture.

This refactor separates planner/backend logic (JAX-only for now) from the
solver wrapper to match the EDOC packaging pattern.
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
from genedynamics.core.task_spec import legacy_extract_position

try:
    from genedynamics.core.registry.solvers import register_solver
except Exception:
    register_solver = None


# ============================================================================
# Backend registry helpers
# ============================================================================
try:
    from genedynamics.solvers.single.mbd.backends import mbd_jax  
except ImportError:
    pass


def _get_mbd_backend(backend_name: str):
    if backend_name == "jax":
        from genedynamics.solvers.single.mbd.backends.mbd_jax import MBDBackendJax

        return MBDBackendJax
    if backend_name == "numpy":
        from genedynamics.solvers.single.mbd.backends.mbd_numpy import MBDBackendNumpy

        return MBDBackendNumpy
    return None


# ============================================================================
# MBD Solver (wrapper)
# ============================================================================

class MBDSolver(SamplingSolver):
    """Solver wrapper that mirrors EDOC structure but runs MBD diffusion."""

    def __init__(
        self,
        dynamics: DynamicsModel,
        energy: EnergyFunctional,
        backend: Backend,
        *,
        horizon: int = 80,
        dt: float = 0.1,
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
        num_modes: int = 1,
        mode_strategy: str = "multirun",
        diversity_eta: float = 1.0,
        diversity_topK_cand: int = None,
        diversity_use_state: bool = True,
        terminal_energy_weight: float = 100.0,
        use_target_line: bool = False,
        num_targets: int = 4,
        position_extractor=None,
        position_dim: int = 2,
        **kwargs,
    ):
        super().__init__(dynamics, energy, backend, **kwargs)

        if backend.name not in {"jax", "numpy"}:
            raise ValueError(
                f"MBD solver supports backends {{'jax','numpy'}}, got {backend.name}."
            )

        self.horizon = horizon
        self.dt = dt
        self.seed = seed
        self.action_extra_sigma = action_extra_sigma
        self.config.update(
            dict(
                Nsample=Nsample,
                Ndiffuse=Ndiffuse,
                temp_sample=temp_sample,
                beta0=beta0,
                betaT=betaT,
                action_limit=action_limit,
                action_extra_sigma=action_extra_sigma,
                scheduler=scheduler,
                show_tqdm=bool(show_tqdm),
                num_modes=int(num_modes),
                mode_strategy=str(mode_strategy),
                diversity_eta=float(diversity_eta),
                diversity_topK_cand=int(diversity_topK_cand) if diversity_topK_cand is not None else None,
                diversity_use_state=bool(diversity_use_state),
                terminal_energy_weight=float(terminal_energy_weight),
                use_target_line=bool(use_target_line),
                num_targets=int(num_targets),
            )
        )

        self.position_extractor = position_extractor or legacy_extract_position
        self.position_dim = int(position_dim)
        self._env_adapter = DynamicsToEnvAdapter(dynamics, dt)
        if isinstance(energy, LegacyEnergyFunctional):
            self._legacy_energy = energy
        else:
            self._legacy_energy = EnergyToLegacyAdapter(energy, dynamics).legacy_energy

        self._backend_impl = None

    def _get_backend_impl(self):
        if self._backend_impl is None:
            backend = RuntimeBackendManager.get_backend()
            backend_cls = _get_mbd_backend(backend.name)
            if backend_cls is None:
                raise ValueError(f"MBD backend '{backend.name}' not found")
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
                    terminal_energy_weight=self.config.get("terminal_energy_weight", 100.0),
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
                    terminal_energy_weight=self.config.get("terminal_energy_weight", 100.0),
                    position_extractor=self.position_extractor,
                    position_dim=self.position_dim,
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
        # Multi-mode strategy: run C independent solves and take best-of-each-run
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
                # Use batch version for parallel execution
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
        register_solver("mbd", MBDSolver)
    except Exception:
        pass


# ============================================================================
# Main Entry Point (backward compatibility)
# ============================================================================

def run_mbd(args):
    """Run MBD diffusion planner (backward-compatible entry point)."""
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

    solver = MBDSolver(
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
    )

    rng = jax.random.PRNGKey(args.seed)
    try:
        x0, _ = env.reset(rng)
    except TypeError:
        x0, _ = env.reset()

    traj = solver.solve(x0, horizon=args.horizon, rng_key=jax.random.PRNGKey(args.seed))
    return traj.info if hasattr(traj, "info") else {
        "actions": np.stack(traj.actions, axis=0),
        "states": np.stack(traj.states, axis=0),
    }
