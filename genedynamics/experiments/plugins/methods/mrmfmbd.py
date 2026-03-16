"""
MRMFMBD method plugin for Soft-robot S1+S3.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from genedynamics.solvers.single.mrmfmbd import MRMFMBDSolver
from genedynamics.solvers.single.mrmfmbd.implementations import EnvFidelitySimulator
from genedynamics.core.dynamics.adapters import EnvDynamicsAdapter
from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.core.inference.fidelity import FidelityConfig, FidelityLadder
from genedynamics.core.task_spec import get_default_task_spec
from ...framework.base import MethodPlugin
from ._result_utils import normalize_result_from_trajectory


class MRMFMBDMethodPlugin(MethodPlugin):
    """
    Plugin for MRMFMBD (Soft-robot S1+S3) solver.
    """

    @property
    def name(self) -> str:
        return "mrmfmbd"

    def create_planner(self, env: Any, energy: Any, config: Dict[str, Any]) -> MRMFMBDSolver:
        backend = RuntimeBackendManager.get_backend()
        dynamics = EnvDynamicsAdapter(env)
        horizon = config.get("horizon", getattr(env, "horizon", 80))
        dt = config.get("dt", getattr(env, "dt", 0.01))

        task_spec = get_default_task_spec(config.get("env_plugin"), config.get("env_name"))

        fidelity_config = FidelityConfig(
            num_levels=config.get("fidelity_num_levels", 3),
            step_ratio=config.get("fidelity_step_ratio", 1.5),
            levels=config.get("fidelity_levels"),
            ladder=config.get("fidelity_ladder"),
        )
        fidelity_ladder = FidelityLadder(fidelity_config, K=config.get("Ndiffuse", 100))

        fidelity_sim = EnvFidelitySimulator(
            env=env,
            energy=energy,
            position_extractor=task_spec.extract_position,
            position_dim=task_spec.position_dim,
            terminal_weight=float(config.get("terminal_energy_weight", 100.0)),
            num_levels=fidelity_config.num_levels,
        )

        solver = MRMFMBDSolver(
            dynamics=dynamics,
            energy=energy,
            backend=backend,
            fidelity_simulator=fidelity_sim,
            fidelity_ladder=fidelity_ladder,
            horizon=horizon,
            Nsample=config.get("Nsample", 64),
            Ndiffuse=config.get("Ndiffuse", 100),
            temp_sample=config.get("temp_sample", 0.5),
            beta0=float(config.get("beta0", 1e-4)),
            betaT=float(config.get("betaT", 1e-2)),
            action_limit=config.get("action_limit", getattr(env, "control_limit", 1.0)),
            action_extra_sigma=config.get("action_extra_sigma", 0.0),
            seed=config.get("np_random_seed", 0),
            scheduler=config.get("scheduler"),
            show_tqdm=config.get("show_tqdm", False),
        )
        return solver

    def plan(self, planner: MRMFMBDSolver, initial_state: np.ndarray, rng: Any) -> Dict[str, Any]:
        traj = planner.solve(initial_state, horizon=planner.horizon, rng_key=rng)
        return normalize_result_from_trajectory(traj, initial_state=initial_state, preserve_info=True)
