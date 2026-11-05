import numpy as np
import pytest

from genedynamics.tasks.transfer.visualization import replay_frame_indices


def test_replay_frame_indices_preserve_endpoints_and_order():
    indices = replay_frame_indices(501, 73)
    assert len(indices) == 73
    assert indices[0] == 0
    assert indices[-1] == 500
    assert np.all(np.diff(indices) > 0)


def test_replay_frame_indices_validate_empty_inputs():
    with pytest.raises(ValueError, match="no recorded states"):
        replay_frame_indices(0, 10)
    with pytest.raises(ValueError, match="must be positive"):
        replay_frame_indices(10, 0)
