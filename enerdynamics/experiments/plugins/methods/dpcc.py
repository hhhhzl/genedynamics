"""
DPCC method plugin implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import sys

import numpy as np

from enerdynamics.core.dynamics import DynamicsToEnvAdapter
from enerdynamics.core.backends.runtime import RuntimeBackendManager
from enerdynamics.solvers.single.dpcc import DPCCSolver
from ...framework.base import MethodPlugin


def _ensure_dpcc_on_path(dpcc_root: str | Path | None) -> Path:
    if dpcc_root is None:
        project_root = Path(__file__).resolve().parents[4]
        dpcc_root = project_root / "dpcc"
    dpcc_root = Path(dpcc_root).resolve()
    if str(dpcc_root) not in sys.path:
        sys.path.insert(0, str(dpcc_root))
    return dpcc_root


class DPCCMethodPlugin(MethodPlugin):
    @property
    def name(self) -> str:
        return "dpcc"

    def create_planner(self, env: Any, energy: Any, config: Dict[str, Any]) -> DPCCSolver:
        dpcc_root = _ensure_dpcc_on_path(config.get("dpcc_root"))

        try:
            import yaml
        except Exception as exc:
            raise ImportError("PyYAML is required for DPCC config loading.") from exc

        import diffuser.utils as dpcc_utils

        dpcc_config_path = Path(config.get("dpcc_config_path", dpcc_root / "config" / "projection_eval.yaml"))
        with open(dpcc_config_path, "r", encoding="utf-8") as f:
            dpcc_config = yaml.safe_load(f) or {}

        exp = config.get("exp", "avoiding-d3il")
        robot_name = exp.split("-")[0]
        seed = int(config.get("seed", config.get("np_random_seed", 0)))
        device = config.get("device", "cuda")

        loadbase = config.get("loadbase", "logs")
        dataset = config.get("dataset", exp)
        diffusion_loadpath = config.get("diffusion_loadpath", "diffusion")
        diffusion_epoch = config.get("diffusion_epoch", None)

        diffusion_experiment = dpcc_utils.load_diffusion(
            loadbase,
            dataset,
            diffusion_loadpath,
            str(seed),
            epoch=diffusion_epoch,
            device=device,
        )
        diffusion = diffusion_experiment.diffusion
        normalizer = diffusion_experiment.dataset.normalizer

        robot_name = exp.split("-")[0]
        obs_indices = dpcc_config.get("observation_indices", {}).get(robot_name, {})
        act_indices = dpcc_config.get("action_indices", {}).get(robot_name, {})
        indices = {"observations": obs_indices, "actions": act_indices}

        plan_config = dict(config.get("plan_config", {}))
        plan_config.setdefault("exp", exp)
        plan_config.setdefault("horizon", config.get("horizon", getattr(env, "horizon", 64)))
        # DPCC convention: dt is keyed by robot_name in projection_eval.yaml
        dt_default = None
        dt_cfg = dpcc_config.get("dt")
        if isinstance(dt_cfg, dict):
            dt_default = dt_cfg.get(robot_name)
        plan_config.setdefault(
            "dt", config.get("dt", dt_default if dt_default is not None else getattr(env, "dt", 0.1))
        )
        plan_config.setdefault("batch_size", config.get("batch_size", 8))
        plan_config.setdefault("max_episode_length", config.get("max_episode_length", 200))
        plan_config.setdefault("variant", config.get("variant", "dpcc"))
        plan_config.setdefault("solver", config.get("solver", "scipy"))
        plan_config.setdefault("halfspace_variant", config.get("halfspace_variant"))

        dynamics = DynamicsToEnvAdapter(env, dt=plan_config["dt"])
        backend = RuntimeBackendManager.get_backend()

        solver = DPCCSolver(
            dynamics=dynamics,
            energy=energy,
            backend=backend,
            env=env,
            diffusion=diffusion,
            normalizer=normalizer,
            plan_config=plan_config,
            constraint_config=dpcc_config,
            indices=indices,
            device=device,
            seed=seed,
        )
        return solver

    def plan(self, planner: DPCCSolver, initial_state: np.ndarray, rng: Any) -> Dict[str, Any]:
        horizon = planner.plan_config.get("horizon", getattr(planner.env, "horizon", 64))
        traj = planner.solve(initial_state, horizon=horizon, rng_key=rng)
        return {
            "states": traj.states,
            "actions": traj.actions,
            "initial_state": initial_state,
            "info": traj.info,
        }

