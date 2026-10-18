"""Public deployment matches native adapted env clock and recurrent player."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from genedynamics.envs.adapters.transfer_task import FLAT_FIELDS
from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.solvers.single.human2sim2robot.learning.task_env import ObjectTrajectoryTaskEnv
from genedynamics.solvers.single.human2sim2robot import Human2Sim2RobotSolver
from genedynamics.experiments.plugins.methods.transfer.human2sim2robot import PublicObjectTrajectoryPPOActor,VERSION
from genedynamics.solvers.single.human2sim2robot.backends.human2sim2robot_torch import native_lstm_config
from genedynamics.solvers.single.cross_ebdm.learning.response import PUBLIC_FIELDS

def public(tick,position):
    result={k:np.zeros((1,3),np.float32) for k in PUBLIC_FIELDS}
    for k in ('tcp_rotation','goal_rotation','reference_rotation'):result[k]=np.eye(3,dtype=np.float32)[None]
    for k in ('joint_position','joint_velocity'):result[k]=np.zeros((1,7),np.float32)
    for k in ('tcp_twist','measured_wrench','body_twist'):result[k]=np.zeros((1,6),np.float32)
    for k in ('previous_action','previous_request'):result[k]=np.zeros((1,13),np.float32)
    result['elapsed_seconds']=np.asarray([[tick*.02]],np.float32);result['tcp_position']=np.asarray([position],np.float32)
    return result

class FixtureTask:
    def __init__(self):
        self.spec=task_spec('T1',robot='xarm7');self.config=self.spec.config;self.num_envs=1
        self.runner=SimpleNamespace(skill=SimpleNamespace(linear_speed=.15,angular_speed=.6,log_stiffness_limit=1.))
        self.row=public(0,[0,0,0]);self.observation_size=self.flat_observation().shape[1];self.state=SimpleNamespace(steps=np.zeros(1))
    def flat_observation(self,observation=None):
        row=self.row if observation is None else observation
        return np.concatenate([row[k].reshape(1,-1) for k in FLAT_FIELDS],axis=-1)
    def reset(self,*args,**kwargs):self.row=public(0,[0,0,0]);self.state.steps[:]=0
    def step(self,action):return SimpleNamespace(observation=self.row,done=np.zeros(1,bool),status=np.zeros(1,int))

@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_native_clock_player_and_public_continuation(device):
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(21);task=FixtureTask();demo=np.tile(np.eye(4,dtype=np.float32),(8,1,1));demo[:,0,3]=np.arange(8)*.01
    env=ObjectTrajectoryTaskEnv(task,demo,num_envs=1,demo_dt=.02);obs=env.reset();network=native_lstm_config()
    native=Human2Sim2RobotSolver(network_config=network,observation_shape=env.observation_space.shape,action_low=env.action_space.low,action_high=env.action_space.high,device=device)
    weights=dict(model=deepcopy(native.model.state_dict()));native.restore(weights)
    meta=dict(version=VERSION,task='T1',body='xarm7',mode=0,flat_fields=list(FLAT_FIELDS),demonstration=demo.tolist(),demonstration_sha256=hashlib.sha256(demo.tobytes()).hexdigest(),demo_dt=.02,network_config=asdict(network),observation_shape=list(obs.shape[1:]),action_low=env.action_space.low.tolist(),action_high=env.action_space.high.tolist())
    actor=PublicObjectTrajectoryPPOActor(task.spec,dict(public_task_adapter=meta,native_policy=weights),mode=0,seed=21,device=device)
    positions=[[0,0,0],[.31,0,0],[.01,0,0],[.02,0,0]]
    for tick,position in enumerate(positions):
        if tick:
            task.row=public(tick,position);task.state.steps[:]=tick;obs,_,_,_=env.step(np.zeros((1,13),np.float32),auto_reset=False)
        before=actor.get_state();actual=actor.act(task.row);after=actor.get_state();actor.restore(before);again=actor.act(task.row);actor.restore(after)
        expected=native.act(obs).detach().cpu().numpy().astype(np.float32)
        np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(again,actual);np.testing.assert_array_equal(actor.observation(task.row),obs)
        assert actor.reference_index==env.reference_index[0]
    bad=dict(task.row,private_carry=np.zeros(1))
    with pytest.raises(ValueError,match='exact public'):actor.act(bad)
