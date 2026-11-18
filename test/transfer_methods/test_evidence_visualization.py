import numpy as np
import pytest

from genedynamics.tasks.transfer.visualization import replay_frame_indices, recorded_event_at_time


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


def test_replay_event_labels_never_announce_future_or_missing_events():
    evaluation = {"event_physics_steps": {
        "first_aperture_cleared": 100,
        "reorientation_completed": 300,
        "terminal_hold_completed": None,
    }}
    assert recorded_event_at_time(evaluation, .1, .002) == "approach / carry"
    assert recorded_event_at_time(evaluation, .2, .002) == "first aperture cleared"
    assert recorded_event_at_time(evaluation, .5, .002) == "first aperture cleared"
    assert recorded_event_at_time(evaluation, .6, .002) == "reorientation completed"
