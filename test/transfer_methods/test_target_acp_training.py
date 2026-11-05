"""Target episode split and numerical normalizer remain separate from source identity."""
import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")
from test.transfer_methods.test_public_target_data import trajectory
from genedynamics.learning.datasets.transfer.target import PublicTargetTrajectory
from genedynamics.solvers.single.acp.learning.target_task_training import PublicACPTaskDataset
from genedynamics.solvers.single.acp.learning.task_training import pose,response_chart,sampling_profile
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


def test_sampling_profile_compares_inverse_chart_prediction_to_native_actions():
    identity=np.asarray([1.,0.,0.,0.,1.,0.],dtype=np.float32)
    native=np.zeros((2,3,19),dtype=np.float32)
    native[...,3:9]=identity;native[...,12:18]=identity
    native[...,9]=.01;native[...,18]=200.

    class Dataset:
        action_chart="response19"
        native_actions=torch.from_numpy(native)
        actions=response_chart(native_actions)
        def __len__(self):return len(self.actions)
        def batch(self,indices):
            return {"obs":{"sparse":{"index":torch.as_tensor(indices,dtype=torch.long)}}}

    class Policy(torch.nn.Module):
        def __init__(self):super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()))
        def predict_action(self,obs):
            return {"sparse":Dataset.native_actions[obs["sparse"]["index"]].to(self.anchor.device)}

    report=sampling_profile(Policy(),Dataset(),indices=(0,1),seed_base=700)
    assert report["native_action_chart_comparison"]
    assert report["dataset_action_chart"]=="response19"
    assert report["aggregate"]["reference_translation_error_m"]["maximum"]==0.
    assert report["aggregate"]["virtual_translation_error_m"]["maximum"]==0.
    assert report["aggregate"]["stiffness_absolute_error_N_per_m"]["maximum"]==0.
