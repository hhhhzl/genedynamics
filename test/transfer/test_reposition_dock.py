import numpy as np

from genedynamics.tasks.transfer.reposition_dock import (
    PublicDockingMethodExecution, PublicRepositionDockExecution, TARGET_BOUNDS,
    TARGET_OFFSETS, target_from_reset,
)
from genedynamics.tasks.transfer.specs import task_spec


def test_reposition_semantic_task_uses_existing_mobile_family():
    spec = task_spec("reposition_then_dock")
    assert spec.name == "reposition_then_dock"
    assert spec.alias == "T4"
    assert spec.config.environment == "reposition_then_dock"
    assert spec.config.robot == "go2_z1"
    assert spec.config.mobile
    assert spec.config.execution_profile == "joint_margin_public_reference"


def test_reposition_targets_are_disjoint_bounded_and_reset_relative():
    start = np.asarray([.004, -.002, .01], dtype=np.float32)
    targets = np.stack([target_from_reset(start, mode) for mode in (0, 1)])
    np.testing.assert_allclose(targets - start, TARGET_OFFSETS)
    assert targets[0, 1] > start[1] > targets[1, 1]
    assert np.all(TARGET_OFFSETS >= TARGET_BOUNDS[0])
    assert np.all(TARGET_OFFSETS <= TARGET_BOUNDS[1])


def test_reposition_execution_waits_for_full_stability_dwell():
    class Actor:
        def __init__(self):
            self.elapsed = []

        def act(self, public):
            self.elapsed.append(float(public["elapsed_seconds"][0]))
            return np.ones((1, 13), dtype=np.float32)

    spec = task_spec("reposition_then_dock")
    actor = Actor()
    execution = PublicRepositionDockExecution(spec, actor, mode=0)
    public = {
        "body_planar_pose": np.zeros((1, 3), dtype=np.float32),
        "body_target": np.zeros((1, 3), dtype=np.float32),
        "body_twist": np.zeros((1, 6), dtype=np.float32),
        "tcp_twist": np.zeros((1, 6), dtype=np.float32),
        "controller_feedback": np.zeros((1, 2), dtype=np.float32),
        "elapsed_seconds": np.asarray([5.0], dtype=np.float32),
    }
    for _ in range(11):
        np.testing.assert_array_equal(execution.act(public), np.zeros((1, 13)))
    np.testing.assert_array_equal(execution.act(public), np.ones((1, 13)))
    public["elapsed_seconds"] = np.asarray([5.02], dtype=np.float32)
    np.testing.assert_array_equal(execution.act(public), np.ones((1, 13)))
    np.testing.assert_allclose(actor.elapsed, [0.0, 0.02], atol=1e-6)


def test_docking_method_execution_removes_semantic_only_fields():
    from genedynamics.solvers.single.cross_ebdm.learning.response import PUBLIC_FIELDS

    class Actor:
        def act(self, public):
            assert set(public) == PUBLIC_FIELDS
            return np.zeros((1, 13), dtype=np.float32)

    public = {key: np.zeros((1, 1), dtype=np.float32) for key in PUBLIC_FIELDS}
    public["body_target"] = np.zeros((1, 3), dtype=np.float32)
    public["body_planar_pose"] = np.zeros((1, 3), dtype=np.float32)
    execution = PublicDockingMethodExecution(Actor())
    np.testing.assert_array_equal(
        execution.act(public), np.zeros((1, 13), dtype=np.float32),
    )
