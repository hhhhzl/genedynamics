import hashlib

import numpy as np
import pytest

pytest.importorskip("torch")

from genedynamics.solvers.single.fada.learning.closed_loop_diagnostics import (
    _checked_path,
    task_trajectory_from_rollout,
)


def _public(count, *, joints=7):
    rotation = np.repeat(np.eye(3, dtype=np.float32)[None], count, axis=0)
    return {
        "joint_position": np.zeros((count, joints), np.float32),
        "joint_velocity": np.zeros((count, joints), np.float32),
        "tcp_position": np.repeat(np.array([[.44, -.2, .61]], np.float32), count, axis=0),
        "tcp_rotation": rotation.copy(),
        "tcp_twist": np.zeros((count, 6), np.float32),
        "measured_wrench": np.zeros((count, 6), np.float32),
        "goal_position": np.repeat(np.array([[.44, -.2, .59]], np.float32), count, axis=0),
        "goal_rotation": rotation.copy(),
        "previous_action": np.zeros((count, 13), np.float32),
        "previous_request": np.zeros((count, 13), np.float32),
        "reference_position": np.repeat(np.array([[.44, -.2, .61]], np.float32), count, axis=0),
        "reference_rotation": rotation.copy(),
        "elapsed_seconds": np.arange(count, dtype=np.float32)[:, None] * .02,
        "body_position": np.repeat(np.array([[1., 2., .3]], np.float32), count, axis=0),
        "body_twist": np.zeros((count, 6), np.float32),
        "controller_feedback": np.zeros((count, 2), np.float32),
    }


@pytest.mark.parametrize("task,robot,joints", [("T3", "xarm7", 7), ("T4", "go2_z1", 19)])
def test_reconstructs_exact_199d_public_command_without_private_state(task, robot, joints):
    public = _public(9, joints=joints)
    trajectory = task_trajectory_from_rollout(
        public,
        np.zeros((8, 13), np.float32),
        task=task,
        robot=robot,
        mode=0,
        payload={"cmd_dim": 199, "genedynamics_task_adapter": {}},
    )
    assert trajectory.obs.shape == (9, 35)
    assert trajectory.current_command.shape == (9, 199)
    assert trajectory.previous_issued.shape == (9, 13)
    assert np.isfinite(trajectory.current_command).all()
    if task == "T4":
        np.testing.assert_array_equal(trajectory.obs[:, 24:27], 0.)


def test_bound_artifact_rejects_byte_change(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"frozen")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert _checked_path(artifact, digest, label="fixture") == artifact
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA changed"):
        _checked_path(artifact, digest, label="fixture")
