"""Unified-runner environment adapters for the four transfer workcells."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.experiments.framework.base import EnvironmentPlugin
from genedynamics.tasks.transfer.specs import mechanism_spec


class TransferTaskEnvironmentPlugin(EnvironmentPlugin):
    """Construct one public task adapter and keep physics inside ``envs``."""

    def __init__(self, task_name: str) -> None:
        self.task_name = task_name

    @property
    def name(self) -> str:
        return self.task_name

    def create_env(self, config: Dict[str, Any]) -> Any:
        values = dict(config)
        seed = int(values.pop("_experiment_seed", 0))
        values.pop("_controller_method", None)
        values.pop("_execution_env_params", None)
        robot = values.pop("robot", None)
        history_len = int(values.pop("history_len", 4))
        initial = values.pop("initial", None)
        mode = int(values.pop("mode", 0))
        shared_execution = values.pop("shared_execution", None)
        mechanism_values = dict(values.pop("mechanism", {}))
        body_support = dict(values.pop("body_support", {}))
        mechanism = mechanism_spec(
            self.task_name,
            mechanism_values,
            body_support_parameters=body_support,
        )
        task = instantiate_task(
            self.task_name,
            robot=robot,
            mechanism_spec=mechanism,
            seed=seed,
            history_len=history_len,
            config_overrides=values or None,
            shared_execution=shared_execution,
        )
        task.target = np.asarray(
            task.config.goal_position
            if task.spec.family == "obstacle"
            else task.config.goal,
            dtype=np.float32,
        )
        task._experiment_initial = initial
        task._experiment_mode = mode
        return task

    def create_energy(self, env: Any = None) -> None:
        del env
        return None

    def get_state_dim(self) -> int:
        return 0

    def extract_position(self, state: Any) -> np.ndarray:
        if isinstance(state, dict) and "tcp_position" in state:
            return np.asarray(state["tcp_position"]).reshape(-1, 3)[0]
        if not hasattr(self, "_public_tcp_slice"):
            raise ValueError("reset the transfer environment before decoding flat states")
        return np.asarray(state).reshape(-1)[self._public_tcp_slice]

    def reset_state(self, env: Any, seed: int) -> Any:
        try:
            public = env.reset(
                (int(seed),), seed=int(seed), initial=env._experiment_initial,
                modes=[env._experiment_mode],
            )
            field = next(row for row in env.public_state_fields() if row["name"] == "tcp_position")
            self._public_tcp_slice = slice(field["start"], field["stop"])
            return public
        except Exception as error:
            from genedynamics.tasks.transfer.episode import episode_evidence
            error.recorded_evidence = {
                "transfer_episodes": episode_evidence(env.records, env.spec, env.evaluate()),
            }
            error.execution_accounting = dict(env.accounting)
            raise


def transfer_environment_plugins() -> list[TransferTaskEnvironmentPlugin]:
    return [
        TransferTaskEnvironmentPlugin(name)
        for name in (
            "obstacle_transfer",
            "guided_contact_alignment",
            "keyed_insertion_recovery",
            "mobile_keyed_insertion_recovery",
            "compliant_keyed_insertion_recovery",
            "aperture_brake",
            "corner_guided_seating",
            "compliant_fixture_insertion",
        )
    ]


__all__ = ["TransferTaskEnvironmentPlugin", "transfer_environment_plugins"]
