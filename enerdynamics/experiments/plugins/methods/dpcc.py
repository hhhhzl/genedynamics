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
    """
    Resolve a usable DPCC root and add it to sys.path.

    Priority:
    1) Explicit `dpcc_root` from config.
    2) Local vendored DPCC module: enerdynamics/solvers/single/dpcc
    3) Repo-root sibling: ../dpcc
    4) Repo-root child: ./dpcc
    When a candidate is found, also add its `diffuser/` subdir if present so pickled
    objects referencing `diffuser.*` can be imported.
    """
    candidates = []
    if dpcc_root is not None:
        candidates.append(Path(dpcc_root))

    project_root = Path(__file__).resolve().parents[4]
    # NOTE: we iterate in increasing priority but insert with sys.path.insert(0),
    # so later entries end up ahead. Keep highest priority last in this list.
    candidates.append(project_root / "dpcc")         # child in repo (lowest priority)
    candidates.append(project_root.parent / "dpcc")  # sibling repo
    # Prefer the vendored copy inside this repo before any external checkouts.
    local_dpcc = Path(__file__).resolve().parents[3] / "solvers" / "single" / "dpcc"
    candidates.append(local_dpcc)

    config_dir = None
    for cand in candidates:
        cand = cand.resolve()
        if not cand.exists():
            continue
        if str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
        diffuser_subdir = cand / "diffuser"
        if diffuser_subdir.exists() and str(diffuser_subdir) not in sys.path:
            sys.path.insert(0, str(diffuser_subdir))
        # Add d3il packages if present (for dpcc datasets)
        d3il_pkg = cand / "d3il"
        if d3il_pkg.exists() and str(d3il_pkg) not in sys.path:
            sys.path.insert(0, str(d3il_pkg))
        d3il_pkg_inner = d3il_pkg / "d3il"
        if d3il_pkg_inner.exists() and str(d3il_pkg_inner) not in sys.path:
            sys.path.insert(0, str(d3il_pkg_inner))
        src_d3il_pkg = cand / "src" / "d3il"
        if src_d3il_pkg.exists() and str(src_d3il_pkg) not in sys.path:
            sys.path.insert(0, str(src_d3il_pkg))

        # Pick the first candidate with a config, but keep scanning to collect
        # additional import roots (so external `diffuser/` can still be found).
        if config_dir is None:
            if (cand / "config" / "projection_eval.yaml").exists():
                config_dir = cand / "config"
            elif (cand / "projection_eval.yaml").exists():
                config_dir = cand.parent if cand.is_file() else cand
            else:
                config_dir = cand

    if config_dir is None:
        raise FileNotFoundError("No DPCC root found; checked config path, local vendored dpcc, ../dpcc, ./dpcc")
    return config_dir


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

        from enerdynamics.solvers.single.dpcc import diffuser_utils as dpcc_utils

        project_root = Path(__file__).resolve().parents[4]
        local_default_cfg = project_root / "enerdynamics" / "solvers" / "single" / "dpcc" / "config" / "projection_eval.yaml"

        # Resolve config path with fallbacks: explicit -> relative to repo root -> vendored default.
        dpcc_config_raw = config.get("dpcc_config_path")
        candidate_paths = []
        if dpcc_config_raw:
            candidate_paths.append(Path(dpcc_config_raw))
            candidate_paths.append(project_root / dpcc_config_raw)
        # if dpcc_root points to a config dir, respect it
        if (dpcc_root / "projection_eval.yaml").exists():
            candidate_paths.append(dpcc_root / "projection_eval.yaml")
        candidate_paths.append(local_default_cfg)

        dpcc_config_path = None
        for cand in candidate_paths:
            if cand.exists():
                dpcc_config_path = cand
                break
        if dpcc_config_path is None:
            raise FileNotFoundError(f"DPCC config not found. Tried: {[str(c) for c in candidate_paths]}")

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
        if diffusion_epoch is None:
            diffusion_epoch = "latest"

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
