"""Play2Perfect solver lifecycle without Isaac Sim or upstream assets."""

import torch

from genedynamics.solvers.single.play2perfect import Play2PerfectSolver


class _Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def get_default_rnn_state(self):
        return [torch.zeros((1, 1, 2))]

    def forward(self, batch):
        state = batch["rnn_states"][0]
        return {
            "mus": batch["obs"][:, :2] + state[0],
            "rnn_states": [state + 1],
        }


def test_actor_keeps_episode_state_and_requires_explicit_reset_on_batch_change():
    solver = Play2PerfectSolver(_Actor(), observation_dim=2)
    first = solver.act([[1.0, 2.0]], coefficient=0.0)
    second = solver.act([[1.0, 2.0]], coefficient=0.0)
    torch.testing.assert_close(first, torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(second, torch.tensor([[2.0, 3.0]]))
    solver.reset(2)
    result = solver.act([[1.0, 2.0], [3.0, 4.0]], coefficient=[0.0, 50.0])
    assert result.shape == (2, 2)
