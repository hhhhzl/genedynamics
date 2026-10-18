"""Full task logs stay public, episode-bound and distinct from short probes."""
import json

import numpy as np
import pytest

pytest.importorskip("mujoco")
from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.learning.datasets.transfer.target import PublicTargetTrajectory,VERSION,FOUR_TASK_VERSION
from genedynamics.solvers.single.fada.learning.task_training import public_target_trajectory
from genedynamics.solvers.single.acp.learning.target_task_training import public_target_windows
from genedynamics.solvers.single.acp.learning.task_training import pose
from genedynamics.solvers.single.acp.backends import spatial


def trajectory():
    task=instantiate_task("T3",robot="xarm7");p=task.reset((28900,),seed=28900);observations=[p];actions=[]
    for _ in range(20):
        action=np.zeros((1,13),np.float32);actions.append(action[0]);observations.append(task.step(action).observation)
    public={k:np.concatenate([v[k] for v in observations]) for k in p}
    meta=dict(version=VERSION,permission="B_log",robot="xarm7",mode=0,seed=28900,purpose="fit",chart="shared13",controls=20,recording_end="budget_stop")
    return PublicTargetTrajectory(meta,public,np.array(actions))


def test_target_data_rejects_truth_and_preserves_budget_prefix_and_native_windows(tmp_path):
    row=trajectory();prefix=row.prefix(15);assert len(prefix.requests)==15 and len(prefix.public["tcp_position"])==16
    prefix.save(tmp_path/"public");loaded=PublicTargetTrajectory.load(tmp_path/"public")
    assert loaded.digest==prefix.digest
    native=public_target_trajectory(loaded);assert native.obs.shape==(16,35) and native.current_command.shape==(16,67)
    with pytest.raises(ValueError):PublicTargetTrajectory(dict(row.metadata,servo_gain=.65),row.public,row.requests)
    with pytest.raises(ValueError):PublicTargetTrajectory(row.metadata,dict(row.public,axial_support=np.ones(21)),row.requests)
    with pytest.raises(ValueError):row.prefix(21)
    p=tmp_path/"public/public.json";m=json.loads(p.read_text());m["mode"]=1;p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match="SHA"):PublicTargetTrajectory.load(tmp_path/"public")


def test_native_acp_labels_keep_world_commands_and_relative_action_time():
    from genedynamics.solvers.single.acp.learning.task_training import pose
    from genedynamics.solvers.single.acp.backends import spatial
    row=trajectory();world=spatial.SE3_to_pose9(pose(row.public["tcp_position"][:-1],row.public["tcp_rotation"][:-1]))
    commands=np.concatenate((world,world,np.full((20,1),200)),axis=-1)
    acp=PublicTargetTrajectory(dict(row.metadata,permission="C_supervised",chart="acp_world19"),row.public,commands)
    observations,actions=public_target_windows(acp,horizon=16)
    assert len(actions)==5 and actions.shape==(5,16,19)
    np.testing.assert_allclose(actions[0,0,:3],0.,atol=1e-7)
    np.testing.assert_array_equal(actions[...,18],200.)
    with pytest.raises(ValueError):public_target_trajectory(acp)


@pytest.mark.parametrize("task,robot", [("T1","xarm7"),("T2","xarm7"),("T3","xarm7"),("T4","go2_z1")])
def test_v2_native_acp_data_binds_task_specific_public_context(task,robot):
    source=trajectory();world=spatial.SE3_to_pose9(pose(source.public["tcp_position"][:-1],source.public["tcp_rotation"][:-1]))
    commands=np.concatenate((world,world,np.full((20,1),800.)),axis=-1)
    metadata=dict(source.metadata,version=FOUR_TASK_VERSION,permission="C_supervised",chart="acp_world19",robot=robot,task=task)
    public=dict(source.public)
    if task=="T4":
        public["joint_position"]=np.pad(public["joint_position"],((0,0),(0,18)))
        public["joint_velocity"]=np.pad(public["joint_velocity"],((0,0),(0,17)))
    row=PublicTargetTrajectory(metadata,public,commands)
    observations,actions=public_target_windows(row,horizon=16)
    assert row.task==task and actions.shape==(5,16,19) and observations["controller_context"].shape==(5,1,53)
