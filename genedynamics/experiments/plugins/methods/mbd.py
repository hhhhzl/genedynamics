"""
MBD method plugin.
"""

from typing import Dict, Any
import numpy as np

from genedynamics.solvers.single.mbd import MBDSolver
from genedynamics.core.dynamics.adapters import EnvDynamicsAdapter
from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.core.task_spec import get_default_task_spec
from ...framework.base import MethodPlugin
from ._result_utils import normalize_result_from_trajectory


class MBDMethodPlugin(MethodPlugin):
    """
    Plugin for MBD solver method.
    """

    @property
    def name(self) -> str:
        return "mbd"

    def create_planner(self, env: Any, energy: Any, config: Dict[str, Any]) -> MBDSolver:
        """
        Create MBDSolver. Wrap env as dynamics via EnvDynamicsAdapter.
        """
        backend = RuntimeBackendManager.get_backend()
        dynamics = EnvDynamicsAdapter(env)
        horizon = config.get("horizon", getattr(env, "horizon", 80))
        dt = config.get("dt", getattr(env, "dt", 0.1))

        task_spec = get_default_task_spec(config.get("env_plugin"), config.get("env_name"))
        solver = MBDSolver(
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
            action_extra_sigma=config.get("action_extra_sigma", 0.0),  # Extra noise for diversity
            seed=config.get("np_random_seed", None) or 0,
            scheduler=config.get("scheduler"),  # pass composite scheduler (diffusion_schedulers)
            show_tqdm=config.get("show_tqdm", False),
            num_modes=config.get("num_modes", 1),  # Number of candidate trajectories to return
            mode_strategy=config.get("mode_strategy", "multirun"),
            diversity_eta=config.get("diversity_eta", 1.0),  # Diversity weight
            diversity_topK_cand=config.get("diversity_topK_cand", None),  # Pre-filter candidates
            diversity_use_state=config.get("diversity_use_state", True),  # Use state or action features
            terminal_energy_weight=float(config.get("terminal_energy_weight", 100.0)),
            position_extractor=task_spec.extract_position,
            position_dim=task_spec.position_dim,
        )
        return solver

    def plan(self, planner: MBDSolver, initial_state: np.ndarray, rng: Any) -> Dict[str, Any]:
        """
        Execute MBD planning and return result dict.
        """
        traj = planner.solve(initial_state, horizon=planner.horizon, rng_key=rng)
        result = normalize_result_from_trajectory(traj, initial_state=initial_state, preserve_info=True)
        env = getattr(getattr(planner, "dynamics", None), "env", None)
        if env is None:
            return result

        def _rollout(actions_arr: np.ndarray) -> np.ndarray:
            s = np.asarray(initial_state, dtype=np.float32).copy()
            states = [s.copy()]
            for a in np.asarray(actions_arr, dtype=np.float32):
                s = np.asarray(env.transition(s, a), dtype=np.float32)
                states.append(s.copy())
            return np.asarray(states, dtype=np.float32)

        actions = result.get("actions")
        if actions is not None:
            try:
                result["states"] = _rollout(np.asarray(actions, dtype=np.float32))
            except Exception:
                pass

        candidate_actions = result.get("candidate_actions")
        if candidate_actions is not None:
            try:
                ca = np.asarray(candidate_actions, dtype=np.float32)
                if ca.ndim == 2:
                    ca = ca[None, ...]
                candidate_states = [_rollout(ca[i]) for i in range(ca.shape[0])]
                result["candidate_states"] = np.asarray(candidate_states, dtype=np.float32)
            except Exception:
                pass
        return result

