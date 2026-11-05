"""Thin unified-runner plugin for task-owned transfer actor bindings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from genedynamics.core.types import Trajectory
from genedynamics.experiments.framework.base import MethodPlugin
from genedynamics.tasks.transfer.controllers.bindings import create_transfer_actor
from genedynamics.tasks.transfer.episode import controller_diagnostics, episode_evidence
from genedynamics.solvers.single.cross_ebdm.learning.substep_risk import (
    RiskSetEmptyError,
)


@dataclass
class _TransferPlanner:
    task: Any
    actor: Any
    n_steps: int
    method: str
    binding_version: str


class TransferMethodPlugin(MethodPlugin):
    """Run an actor exclusively through the public transfer-task contract."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def create_planner(self, env: Any, energy: Any, config: Dict[str, Any]) -> Any:
        del energy
        actor = create_transfer_actor(env, self.name, config)
        return _TransferPlanner(
            task=env,
            actor=actor,
            n_steps=int(config.get("n_steps", 1)),
            method=self.name,
            binding_version=str(
                (config.get("actor_binding") or {}).get("version", "task-reference")
            ),
        )

    def plan(self, planner: _TransferPlanner, initial_state: Any, rng: Any) -> Dict[str, Any]:
        del rng
        public = initial_state
        states = [planner.task.flat_observation(public)[0].copy()]
        actions = []
        infos = []
        execution_status = None
        initial_controls = int(planner.task.accounting["control_steps"])
        for _ in range(planner.n_steps):
            if bool(np.asarray(planner.task.state.status)[0]):
                break
            try:
                action = np.asarray(planner.actor.act(public), dtype=np.float32)
                diagnostic = controller_diagnostics(planner.actor)
                body_intent = planner.task.public_body_intent(public)
                step = (
                    planner.task.step(action)
                    if body_intent is None
                    else planner.task.step_contact(action, body_twist=body_intent)
                )
                public = step.observation
                actions.append(action.copy())
                states.append(planner.task.flat_observation(public)[0].copy())
                infos.append({
                    "status": int(step.status[0]),
                    "success": bool(step.status[0] == 1),
                    "collision": bool(step.status[0] == 2),
                })
                planner.task.record_control_diagnostics(diagnostic)
            except RiskSetEmptyError as error:
                planner.task.records[-1].state = "rejected"
                planner.task.records[-1].rejection_reason = "risk_set_empty"
                execution_status = {
                    "state": "aborted_unrecoverable", "reason": str(error),
                    "error_type": type(error).__name__,
                    "executed_control_steps": int(planner.task.accounting["control_steps"]) - initial_controls,
                }
                break
            except Exception as error:
                # Preserve the physical prefix in the framework's existing
                # retained-failure path, rather than losing it to an exception.
                execution_status = {
                    "state": "aborted_unrecoverable", "reason": str(error),
                    "error_type": type(error).__name__,
                    "executed_control_steps": int(planner.task.accounting["control_steps"]) - initial_controls,
                }
                break
        evaluation = planner.task.evaluate()
        runtime = {
            "model_calls": int(getattr(planner.actor, "irreversible_model_calls", 0)),
            "policy_network_calls": int(
                getattr(planner.actor, "irreversible_network_calls", 0)
            ),
            "candidate_rollouts": int(
                getattr(planner.actor, "irreversible_candidate_rollouts", 0)
            ),
            "action_updates": int(
                getattr(planner.actor, "irreversible_action_updates", 0)
            ),
        }
        trajectory = Trajectory(
            states=states,
            actions=actions,
            info={
                "infos": infos,
                "success": bool(evaluation.get("successes", 0)),
                "collision": bool(evaluation.get("constraint_violations", 0)),
                "task_evaluation": evaluation,
                "runtime": runtime,
            },
        )
        return {
            "trajectory": trajectory,
            "state_layout": getattr(planner.task, "state_layout", "transfer_public_flat_v2"),
            "state_fields": planner.task.public_state_fields(),
            "states": states,
            "actions": actions,
            "infos": infos,
            "initial_state": initial_state,
            "task_evaluation": evaluation,
            "component_contract": {
                "plugin": self.name,
                "public_task_interface": True,
                "binding_version": planner.binding_version,
                "implementation": (
                    f"{type(planner.actor).__module__}."
                    f"{type(planner.actor).__qualname__}"
                ),
                "shared_execution": dict(planner.task.shared_execution),
            },
            "runtime": runtime,
            "recorded_evidence": {
                "transfer_episodes": episode_evidence(
                    planner.task.records, planner.task.spec, evaluation,
                ),
            },
            **({"execution_status": execution_status} if execution_status else {}),
        }


def transfer_method_plugins() -> list[TransferMethodPlugin]:
    return [
        TransferMethodPlugin(name)
        for name in (
            "task_reference",
            "cross_ebdm",
            "task_only",
            "fada",
            "adaptsim",
            "asid",
            "acp",
            "human2sim2robot",
            "dial_transfer",
            "response_mpc",
            "public_geometry",
            "public_source_prior",
        )
    ]


__all__ = ["TransferMethodPlugin", "transfer_method_plugins"]
