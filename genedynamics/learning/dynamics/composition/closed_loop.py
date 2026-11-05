"""Functional composition supports Torch graphs and JAX traced arrays alike."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClosedLoopState:
    physical: Any
    controller: Any
    task: Any


@dataclass(frozen=True)
class RolloutPrediction:
    states: tuple
    actions: tuple
    references: tuple
    terminal: ClosedLoopState


class ClosedLoopRollout:
    def __init__(
        self,
        *,
        dynamics: Callable,
        controller: Callable,
        reference: Callable,
        observe: Callable,
        update_controller: Callable,
        advance_task: Callable,
    ):
        self.dynamics = dynamics
        self.controller = controller
        self.reference = reference
        self.observe = observe
        self.update_controller = update_controller
        self.advance_task = advance_task

    def rollout(self, initial: ClosedLoopState, controls, context=None):
        current = initial
        states, actions, references = [current.physical], [], []
        for control in controls:
            reference = self.reference(current.task, control, context)
            action = self.controller(
                current.physical, current.controller, reference, context
            )
            physical = self.dynamics(current.physical, action, context)
            task = self.advance_task(current.task, control, context)
            next_reference = self.reference(task, control, context)
            observation = self.observe(physical, action, next_reference, context)
            memory = self.update_controller(
                current.controller, observation, action, context
            )
            current = ClosedLoopState(physical, memory, task)
            states.append(physical)
            actions.append(action)
            references.append(reference)
        return RolloutPrediction(
            tuple(states), tuple(actions), tuple(references), current
        )
