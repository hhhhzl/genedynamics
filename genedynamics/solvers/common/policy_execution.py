"""Host execution bridge for native policies and task-owned public interfaces.

This bridge owns no policy, controller, observation or success math. In
particular it never hands a simulator state to a policy by default. It supports
one episode; vectorized evaluation and partial resets belong to task adapters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from genedynamics.solvers.common.receding_horizon import RecedingHorizonResult


class NativePolicy(Protocol):
    def reset(self) -> None: ...

    def act(self, observation: Any) -> Any: ...


@dataclass(frozen=True)
class ExecutedStep:
    """Actual transition and applied command returned by the task controller.

    ``executed_action`` may differ from the policy request (e.g. ACP virtual
    targets become physical controls). The task adapter must record the applied
    command, and may retain the requested action in ``info`` for diagnostics.
    ``done`` is a single episode flag, never a vector silently reduced here.
    """

    state: Any
    executed_action: Any
    done: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)


class PolicyExecutionController:
    def __init__(
        self,
        policy: NativePolicy,
        *,
        observe: Callable[[Any], Any],
        execute: Callable[[Any, Any], ExecutedStep],
    ) -> None:
        self.policy = policy
        self.observe = observe
        self.execute = execute

    def run(self, initial_state: Any, *, n_steps: int) -> RecedingHorizonResult:
        """Reset policy history and run until termination or the step budget.

        Randomness is owned explicitly by each native policy, rather than
        accepting and then ignoring a JAX random key. Observation conversion and
        Torch/CPU/device synchronization are explicit task adapter operations.
        """
        if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 0:
            raise ValueError("n_steps must be a nonnegative integer")
        from genedynamics.core.types import ExecutionRejected

        self.policy.reset()
        state = initial_state
        states, actions, infos = [state], [], []
        for step in range(n_steps):
            requested = self.policy.act(self.observe(state))
            try:
                transition = self.execute(state, requested)
            except ExecutionRejected as exc:
                # The established exception contract rejects BEFORE physical
                # execution. Preserve the real prefix; never pad or retry it.
                exc.partial_result = RecedingHorizonResult(
                    states=list(states), actions=list(actions), infos=list(infos),
                )
                exc.partial_states_complete = True
                exc.requested_steps = n_steps
                exc.rejected_step = step
                exc.rejected_action = requested
                raise
            if not isinstance(transition, ExecutedStep):
                raise TypeError("execute must return ExecutedStep")
            if type(transition.done) is not bool:
                raise TypeError("task adapter must supply a scalar Python bool done")
            state = transition.state
            states.append(state)
            actions.append(transition.executed_action)
            infos.append(dict(transition.info))
            if transition.done:
                break
        return RecedingHorizonResult(states=states, actions=actions, infos=infos)
