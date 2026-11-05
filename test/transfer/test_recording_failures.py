"""Already executed fake transitions remain charged when later recording fails."""
from collections import namedtuple
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from genedynamics.envs.adapters.transfer_task import CpuTaskAdapter
from genedynamics.tasks.transfer.episode import EpisodeRecord
from genedynamics.tasks.transfer.specs import task_spec


def _adapter_and_transition(batch=1):
    """Only plain state fixtures: no adapter constructor or simulator startup."""
    spec = task_spec("T3")
    records = [EpisodeRecord(index, 937100 + index, 0, "synthetic_fixture") for index in range(batch)]
    action = np.zeros((batch, 13), np.float32)
    action[:, 0] = .01
    position = np.broadcast_to([.44, -.2, .61], (batch, 1, 3)).copy()
    raw_type = namedtuple("PhysicalTrace", ("valid", "command", "tcp_position", "status"))
    raw = raw_type(np.ones((batch, 1), bool), action[:, None].copy(), position,
                   np.zeros((batch, 1), np.int32))
    fake = SimpleNamespace(
        spec=spec, config=spec.config, num_envs=batch,
        _episodes=records, _current=list(range(batch)),
        _state=SimpleNamespace(last_command=action.copy(), status=np.zeros(batch, np.int32)),
        _previous_request=np.zeros_like(action), _previous_action=np.zeros_like(action),
        _public_reference_position=position[:, 0].copy(),
        _public_reference_rotation=np.broadcast_to(np.diag([1., -1., -1.]), (batch, 3, 3)).copy(),
        accounting={"control_steps": 0, "physics_steps": 0},
        lifetime_accounting={"control_steps": 10, "physics_steps": 100},
        _push_history=Mock(), observe=Mock(return_value={"fixture": np.zeros((batch, 1))}),
    )
    fake._charge = MethodType(CpuTaskAdapter._charge, fake)
    fake._row = Mock(side_effect=lambda trace, index, requested: {
        "tcp_position": trace.tcp_position[index, 0].copy(),
        "backend_status": 0, "valid": True,
    })
    return fake, action, raw, np.ones(batch, bool)


@pytest.mark.parametrize("failure_stage", ("row", "history", "observe"))
def test_recording_failure_cannot_erase_incurred_control_or_physics_charge(failure_stage):
    fake, action, raw, active = _adapter_and_transition()
    failing = {"row": fake._row, "history": fake._push_history, "observe": fake.observe}[failure_stage]
    failing.side_effect = RuntimeError("recording fixture failed")
    with pytest.raises(RuntimeError, match="recording fixture failed"):
        CpuTaskAdapter._finish_step(fake, action, raw, active)
    assert fake.accounting == {"control_steps": 1, "physics_steps": fake.config.substeps}
    assert fake.lifetime_accounting == {"control_steps": 11, "physics_steps": 100 + fake.config.substeps}
    record = fake._episodes[0]
    if failure_stage == "row":
        assert record.state == "aborted"
        assert record.requested_actions == record.traces == record.control_diagnostics == []
        retained = record.incomplete_transitions[0]
        np.testing.assert_array_equal(retained["requested_action"], action[0])
        np.testing.assert_array_equal(retained["physical_trace"]["command"], raw.command[0, 0])
        assert bool(retained["physical_trace"]["valid"])
        assert retained["error_type"] == "RuntimeError"
    else:
        assert len(record.requested_actions) == len(record.traces) == len(record.control_diagnostics) == 1
        assert record.control_diagnostics[0] == {"recording": "unavailable"}
        assert record.incomplete_transitions == []


def test_successful_step_has_one_replaceable_diagnostic_placeholder_without_recharging():
    fake, action, raw, active = _adapter_and_transition()
    step = CpuTaskAdapter._finish_step(fake, action, raw, active)
    np.testing.assert_array_equal(step.executed_action, action)
    record = fake._episodes[0]
    assert record.control_diagnostics == [{"recording": "unavailable"}]
    snapshot = {"stage": 2, "compensation": np.asarray([.01, 0., 0.])}
    CpuTaskAdapter.record_control_diagnostics(fake, snapshot)
    snapshot["compensation"][0] = 99.
    assert record.control_diagnostics[0]["compensation"][0] == .01
    assert fake.accounting["control_steps"] == 1
    with pytest.raises(ValueError, match="one diagnostic snapshot"):
        CpuTaskAdapter.record_control_diagnostics(fake, {"stage": 3})
    assert fake.accounting["control_steps"] == 1


def test_batched_row_failure_preserves_every_already_executed_world():
    fake, action, raw, active = _adapter_and_transition(batch=2)
    normal_row = fake._row.side_effect

    def first_world_fails(trace, index, requested):
        if index == 0:
            raise RuntimeError("first world recording failed")
        return normal_row(trace, index, requested)

    fake._row.side_effect = first_world_fails
    with pytest.raises(RuntimeError, match="first world recording failed"):
        CpuTaskAdapter._finish_step(fake, action, raw, active)
    assert fake.accounting["control_steps"] == 2
    assert fake.accounting["physics_steps"] == 2 * fake.config.substeps
    for record in fake._episodes:
        # A row-construction failure may retain raw evidence for the entire
        # batch or only the failed world; either preserves every paid step.
        assert len(record.traces) + len(record.incomplete_transitions) == 1
        if record.traces:
            assert len(record.requested_actions) == len(record.control_diagnostics) == 1
