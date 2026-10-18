"""Unified-runner adapters for transfer-task policies.

The common task reference is executable now. Method adapters that require
frozen checkpoints are registered under their paper names and reject execution
until the config supplies the new package-owned binding. This prevents old
script workflows from silently producing evidence during the refactor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from genedynamics.core.types import Trajectory
from genedynamics.experiments.framework.base import MethodPlugin
from genedynamics.tasks.transfer.teachers import PublicTeacher


@dataclass
class _TransferPlanner:
    task: Any
    actor: Any
    n_steps: int
    method: str


class TransferMethodPlugin(MethodPlugin):
    """Run an actor exclusively through the public transfer-task contract."""

    def __init__(self, name: str, *, executable_reference: bool = False) -> None:
        self._name = name
        self._executable_reference = executable_reference

    @property
    def name(self) -> str:
        return self._name

    def create_planner(self, env: Any, energy: Any, config: Dict[str, Any]) -> Any:
        del energy
        if not self._executable_reference:
            raise RuntimeError(
                f"transfer method {self.name!r} has no frozen package-owned "
                "actor binding after the architecture refactor; keep the "
                "protocol pending and migrate its checkpoint adapter first"
            )
        actor = PublicTeacher(
            env,
            mode=int(config.get("mode", getattr(env, "_experiment_mode", 0))),
            recipe=str(config.get("recipe", "default")),
        )
        return _TransferPlanner(
            task=env,
            actor=actor,
            n_steps=int(config.get("n_steps", 1)),
            method=self.name,
        )

    def plan(self, planner: _TransferPlanner, initial_state: Any, rng: Any) -> Dict[str, Any]:
        del rng
        public = initial_state
        states = [planner.task.flat_observation(public)[0].copy()]
        actions = []
        infos = []
        for _ in range(planner.n_steps):
            if bool(np.asarray(planner.task.state.status)[0]):
                break
            action = np.asarray(planner.actor.act(public), dtype=np.float32)
            step = planner.task.step(action)
            public = step.observation
            actions.append(action.copy())
            states.append(planner.task.flat_observation(public)[0].copy())
            infos.append({
                "status": int(step.status[0]),
                "success": bool(step.status[0] == 1),
                "collision": bool(step.status[0] == 2),
            })
        evaluation = planner.task.evaluate()
        trajectory = Trajectory(
            states=states,
            actions=actions,
            info={
                "infos": infos,
                "success": bool(evaluation.get("successes", 0)),
                "collision": bool(evaluation.get("forbidden_collisions", 0)),
                "task_evaluation": evaluation,
            },
        )
        return {
            "trajectory": trajectory,
            "states": states,
            "actions": actions,
            "infos": infos,
            "initial_state": initial_state,
            "task_evaluation": evaluation,
            "component_contract": {
                "plugin": self.name,
                "public_task_interface": True,
                "implementation": "task_reference",
            },
        }


def transfer_method_plugins() -> list[TransferMethodPlugin]:
    methods = [
        TransferMethodPlugin("task_reference", executable_reference=True),
    ]
    methods.extend(
        TransferMethodPlugin(name)
        for name in (
            "cross_ebdm",
            "fada",
            "acp",
            "human2sim2robot",
            "play2perfect",
            "dial_transfer",
            "response_mpc",
        )
    )
    return methods


__all__ = ["TransferMethodPlugin", "transfer_method_plugins"]
