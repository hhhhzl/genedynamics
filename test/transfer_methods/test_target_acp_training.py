"""Target episode split and numerical normalizer remain separate from source identity."""
import numpy as np
import pytest

pytest.importorskip("mujoco")
from test.transfer_methods.test_public_target_data import trajectory
from genedynamics.learning.datasets.transfer.target import PublicTargetTrajectory
from genedynamics.solvers.single.acp.learning.target_task_training import PublicACPTaskDataset
from genedynamics.solvers.single.acp.learning.task_training import pose
from genedynamics.solvers.single.acp.backends import spatial


def test_target_native_dataset_rejects_validation_leakage_and_roundtrips_chart():
    source=trajectory();world=spatial.SE3_to_pose9(pose(source.public["tcp_position"][:-1],source.public["tcp_rotation"][:-1]))
    commands=np.concatenate((world,world,np.full((20,1),200.)),axis=-1)
    row=PublicTargetTrajectory(dict(source.metadata,permission="C_supervised",chart="acp_world19"),source.public,commands)
    dataset=PublicACPTaskDataset([row],purpose="fit")
    normalizer=dataset.normalizer();normalized=normalizer["action"].normalize(dataset.actions)
    np.testing.assert_allclose(normalizer["action"].unnormalize(normalized).numpy(),dataset.actions.numpy(),atol=1e-6)
    assert dataset.receipts[0]["robot"]=="xarm7" and "source_success" not in dataset.receipts[0]
    with pytest.raises(ValueError,match="split"):PublicACPTaskDataset([row],purpose="validation")
    with pytest.raises(ValueError,match="duplicate"):PublicACPTaskDataset([row,row],purpose="fit")
