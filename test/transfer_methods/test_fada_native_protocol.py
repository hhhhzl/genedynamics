"""Regression tests for public episode boundaries in native robot execution."""

import pytest

torch = pytest.importorskip("torch")

from genedynamics.solvers.single.fada.learning.native_diagnostics import reset_public_histories


@pytest.mark.parametrize("done", [torch.tensor([0, 1, 0], dtype=torch.long),
                                 torch.tensor([False, True, False])])
def test_partial_reset_preserves_live_environment_histories(done):
    obs = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3) + 1
    act = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2) + 1
    valid = torch.ones(3, 2, dtype=torch.bool)
    previous = torch.arange(6, dtype=torch.float32).reshape(3, 2) + 1
    original = [value.clone() for value in (obs, act, valid, previous)]
    mask = reset_public_histories(done, obs, act, valid, previous)
    assert mask.tolist() == [False, True, False]
    for value, before in zip((obs, act, valid, previous), original):
        assert torch.equal(value[[0, 2]], before[[0, 2]])
        assert not bool(value[1].any())


def test_single_native_integer_done_resets_env_zero_without_indexing_env_one():
    obs, act = torch.ones(1, 30, 64), torch.ones(1, 30, 29)
    valid, previous = torch.ones(1, 30, dtype=torch.bool), torch.ones(1, 29)
    reset_public_histories(torch.ones(1, dtype=torch.long), obs, act, valid, previous)
    assert all(not bool(value.any()) for value in (obs, act, valid, previous))
