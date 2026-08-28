"""Unified-runner adapters for the three MDAC contact tasks.

The adapters contain no physics.  They resolve the same environment kwargs as
the existing MDAC controller factory and delegate construction to
``genedynamics.envs.factories.make_env``.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from genedynamics.envs.factories import make_env
from genedynamics.experiments.framework.base import EnvironmentPlugin
from genedynamics.solvers.single.mdac.core.method_registry import (
    METHOD_TABLE,
    resolve_method,
)

ARM_TASK = "manipulator_surface_scan"
HUMANOID_TASK = "humanoid_box_push"
INSERT_TASK = "manipulator_peg_insert"


def stiffness_mode_for(method: str, flags: Any) -> str:
    """Map an MDAC method contract to the task's stiffness chart."""
    if not flags.use_stiffness:
        return "none"
    if flags.log_spd_stiffness:
        return "log_spd"
    return "fixed" if "fixed" in method else "euclid"


_PRIVATE_KEYS = {
    "_controller_method",
    "_execution_env_params",
    "_experiment_seed",
}


def _env_kwargs(task: str, config: Dict[str, Any]) -> Dict[str, Any]:
    method = str(config.get("_controller_method", "mdac"))
    seed = int(config.get("_experiment_seed", 0))
    kwargs = {k: v for k, v in config.items() if k not in _PRIVATE_KEYS}
    if method in METHOD_TABLE:
        flags = resolve_method(method)
        kwargs["stiffness_mode"] = stiffness_mode_for(method, flags)
        if task == ARM_TASK:
            kwargs["clean_manifold_force"] = flags.use_force_manifold
    else:
        kwargs["stiffness_mode"] = "log_spd"
        kwargs.pop("clean_manifold_force", None)
    if task == ARM_TASK:
        kwargs.setdefault("surface_seed", seed)
    elif task == HUMANOID_TASK:
        kwargs.setdefault("dr_seed", seed)
    elif task == INSERT_TASK:
        kwargs.setdefault("domain_seed", seed)
    return kwargs


class ContactTaskEnvironmentPlugin(EnvironmentPlugin):
    """Thin environment adapter shared by scan, insertion, and push."""

    task: str

    def __init__(self, task: str) -> None:
        self.task = str(task)

    @property
    def name(self) -> str:
        return self.task

    def create_env(self, config: Dict[str, Any]) -> Any:
        model_kwargs = _env_kwargs(self.task, config)
        env = make_env(self.task, **model_kwargs)
        execution_overrides = dict(config.get("_execution_env_params") or {})
        execution_env = None
        if execution_overrides:
            execution_config = dict(config)
            execution_config.update(execution_overrides)
            execution_env = make_env(
                self.task, **_env_kwargs(self.task, execution_config)
            )
            if int(execution_env.action_size) != int(env.action_size):
                raise ValueError("model/execution environment action mismatch")
            if int(execution_env.observation_size) != int(env.observation_size):
                raise ValueError("model/execution environment observation mismatch")
        env._experiment_execution_env = execution_env
        env._experiment_task = self.task
        env._experiment_controller_method = str(
            config.get("_controller_method", "mdac")
        )
        return env

    def create_energy(self, env: Any = None) -> Any:
        del env
        return None

    def get_state_dim(self) -> int:
        # Structured Brax states are intentionally not flattened by this plugin.
        return 0

    def extract_position(self, state: Any) -> np.ndarray:
        info = getattr(state, "info", None)
        if isinstance(info, dict):
            for key in ("ee_position", "peg_position", "box_position"):
                if key in info:
                    return np.asarray(info[key], dtype=np.float32).reshape(-1)[:3]
        return np.zeros(3, dtype=np.float32)

    def reset_state(self, env: Any, seed: int) -> Any:
        import jax

        return self.execution_env(env).reset(jax.random.PRNGKey(int(seed)))


__all__ = [
    "ARM_TASK",
    "HUMANOID_TASK",
    "INSERT_TASK",
    "ContactTaskEnvironmentPlugin",
    "stiffness_mode_for",
]
