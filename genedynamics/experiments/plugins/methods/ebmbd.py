"""
EB-MBD method plugin.
"""

from typing import Dict, Any
import numpy as np

from genedynamics.solvers.single.ebmbd import EBMBDSolver
from genedynamics.core.dynamics.adapters import EnvDynamicsAdapter
from genedynamics.core.backends.runtime import RuntimeBackendManager
from ...framework.base import MethodPlugin


class EBMBDMethodPlugin(MethodPlugin):
    """
    Plugin for EB-MBD solver method.
    """

    @property
    def name(self) -> str:
        return "ebmbd"

    def create_planner(self, env: Any, energy: Any, config: Dict[str, Any]) -> EBMBDSolver:
        """
        Create EBMBDSolver. Wrap env as dynamics via EnvDynamicsAdapter.
        """
        backend = RuntimeBackendManager.get_backend()
        dynamics = EnvDynamicsAdapter(env)
        horizon = config.get("horizon", getattr(env, "horizon", 80))
        dt = config.get("dt", getattr(env, "dt", 0.1))
        obstacles = config.get("obstacles", None)
        obstacle_config = config.get("obstacle_config", None)

        solver = EBMBDSolver(
            dynamics=dynamics,
            energy=energy,
            backend=backend,
            horizon=horizon,
            dt=dt,
            Nsample=config.get("Nsample", 2048),
            Ndiffuse=config.get("Ndiffuse", 100),
            temp_sample=config.get("temp_sample", 0.1),
            beta0=config.get("beta0", 1e-4),
            betaT=config.get("betaT", 1e-2),
            action_limit=config.get("action_limit", getattr(env, "control_limit", 1.0)),
            action_extra_sigma=config.get("action_extra_sigma", 0.0),
            mu=config.get("mu", 10.0),
            alpha=config.get("alpha", 1.0),
            bound=config.get("bound", 0.8),
            use_min_over_time=config.get("use_min_over_time", True),
            terminal_energy_weight=config.get("terminal_energy_weight", 0.0),
            obstacles=obstacles,
            obstacle_config=obstacle_config,
            seed=config.get("np_random_seed", None) or 0,
            scheduler=config.get("scheduler"),
            show_tqdm=config.get("show_tqdm", False),
            num_modes=config.get("num_modes", 1),  # Number of candidate trajectories to return
            mode_strategy=config.get("mode_strategy", "multirun"),
            diversity_eta=config.get("diversity_eta", 1.0),  # Diversity weight
            diversity_topK_cand=config.get("diversity_topK_cand", None),  # Pre-filter candidates
            diversity_use_state=config.get("diversity_use_state", True),  # Use state or action features
        )
        return solver

    def plan(self, planner: EBMBDSolver, initial_state: np.ndarray, rng: Any) -> Dict[str, Any]:
        """
        Execute EB-MBD planning and return result dict.
        """
        traj = planner.solve(initial_state, horizon=planner.horizon, rng_key=rng)
        result = traj.info if hasattr(traj, "info") else {}
        # Ensure required fields present
        result.setdefault("states", traj.states)
        result.setdefault("actions", traj.actions)
        result.setdefault("initial_state", traj.states[0] if traj.states else initial_state)
        return result

