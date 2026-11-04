"""Frozen controller weights must retain derivatives with respect to references."""

import torch
from genedynamics.learning.dynamics import ClosedLoopRollout, ClosedLoopState


def test_reference_gradient_through_frozen_controller():
    controller = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        controller.weight.fill_(2.0)
    controller.requires_grad_(False)
    model = ClosedLoopRollout(
        dynamics=lambda state, action, context: state + action,
        controller=lambda state, history, reference, context: controller(reference),
        reference=lambda task, control, context: control,
        observe=lambda state, action, reference, context: state,
        update_controller=lambda history, observation, action, context: observation,
        advance_task=lambda task, control, context: task + 1,
    )
    controls = torch.tensor([[0.1], [0.2]], requires_grad=True)
    result = model.rollout(ClosedLoopState(torch.zeros(1), torch.zeros(1), 0), controls)
    result.terminal.physical.sum().backward()
    torch.testing.assert_close(controls.grad, torch.full_like(controls, 2.0))
    assert controller.weight.grad is None
    assert result.terminal.task == 2
