from dataclasses import dataclass

import torch
from genedynamics.core.constraints.action_filters import (
    TorchTrajectoryFeasibilityFilter,
)


@dataclass(frozen=True)
class Evaluation:
    maximum_violation: torch.Tensor
    predicted_feasible: torch.Tensor


class Binding:
    def __init__(self):
        self.model_call_count = 0

    def evaluate_action_sequences(self, actions):
        batch, horizon, _ = actions.shape
        self.model_call_count += batch * horizon
        violation = actions.abs().amax(dim=(1, 2)) - 0.2
        return Evaluation(violation, violation <= 0)


def test_trajectory_filter_contracts_only_unresolved_candidates_and_counts_calls():
    binding = Binding()
    actions = torch.tensor([[[1.0], [1.0]], [[0.1], [0.1]]])
    filter_ = TorchTrajectoryFeasibilityFilter((0.5, 0.1))

    result = filter_.apply_actions(None, actions, env=binding)

    torch.testing.assert_close(
        result,
        torch.tensor([[[0.1], [0.1]], [[0.1], [0.1]]]),
    )
    assert binding.model_call_count == 8


def test_trajectory_filter_preserves_best_available_infeasible_contraction():
    binding = Binding()
    actions = torch.tensor([[[-1.0], [1.0]]])
    filter_ = TorchTrajectoryFeasibilityFilter((0.8, 0.4))

    result = filter_.apply_actions(None, actions, env=binding)

    torch.testing.assert_close(result, actions * 0.4)
    assert binding.model_call_count == 6
