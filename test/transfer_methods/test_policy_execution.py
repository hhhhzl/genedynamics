"""Exercise the actual host bridge with structured controller commands."""
import pytest

from genedynamics.solvers.common.policy_execution import ExecutedStep, PolicyExecutionController


class Policy:
    def __init__(self):
        self.resets = 0
        self.observations = []

    def reset(self):
        self.resets += 1
        self.observations = []

    def act(self, observation):
        assert set(observation) == {"public_position"}
        self.observations.append(observation)
        return {"virtual_target": observation["public_position"] + 2, "stiffness": 3}


def test_closed_loop_records_executed_commands_and_terminal_state():
    policy = Policy()

    def execute(state, request):
        applied = request["stiffness"] * (request["virtual_target"] - state["position"])
        next_state = {**state, "position": state["position"] + 1}
        return ExecutedStep(next_state, applied, next_state["position"] == 2,
                            {"requested": request})

    controller = PolicyExecutionController(
        policy, observe=lambda s: {"public_position": s["position"]}, execute=execute,
    )
    initial = {"position": 0, "sealed_mechanism": "must never reach actor"}
    result = controller.run(initial, n_steps=10)
    assert len(result.states) == len(result.actions) + 1 == 3
    assert result.actions == [6, 6]
    assert result.infos[1]["requested"]["virtual_target"] == 3
    assert result.states[-1]["position"] == 2
    controller.run(initial, n_steps=0)
    assert policy.resets == 2
    assert policy.observations == []


def test_bridge_never_retries_a_failed_execution():
    calls = []

    def failed(state, command):
        calls.append(command)
        raise RuntimeError("controller failed after target call")

    controller = PolicyExecutionController(Policy(), observe=lambda _: {"public_position": 0}, execute=failed)
    with pytest.raises(RuntimeError, match="controller failed"):
        controller.run(None, n_steps=3)
    assert len(calls) == 1


def test_vectorized_terminal_flag_requires_a_task_adapter():
    import numpy as np
    controller = PolicyExecutionController(
        Policy(), observe=lambda _: {"public_position": 0},
        execute=lambda s, a: ExecutedStep(s, a, np.array([True, False])),
    )
    with pytest.raises(TypeError, match="scalar Python bool"):
        controller.run(None, n_steps=1)


def test_execution_rejection_retains_real_prefix_without_fabricating_action():
    from genedynamics.core.types import ExecutionRejected

    def execute(state, request):
        if state == 1:
            raise ExecutionRejected("adapter rejected command", {"model_calls": 2})
        return ExecutedStep(state + 1, 7)

    controller = PolicyExecutionController(
        Policy(), observe=lambda s: {"public_position": s}, execute=execute,
    )
    with pytest.raises(ExecutionRejected) as failure:
        controller.run(0, n_steps=10)
    exc = failure.value
    assert exc.partial_result.states == [0, 1]
    assert exc.partial_result.actions == [7]
    assert exc.partial_states_complete
    assert exc.requested_steps == 10 and exc.rejected_step == 1
    assert exc.rejected_action == {"virtual_target": 3, "stiffness": 3}
    assert exc.details == {"model_calls": 2}
