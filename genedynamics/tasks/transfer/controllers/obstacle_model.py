"""Owned nominal T1 prediction from public encoders and causal controller memory.

No real executor/checkpoint/plant mechanism can be supplied. Controller memory
is reconstructed under the declared nominal zero-delay model; target delay
mismatch remains a prediction error, never privileged simulator assimilation.
"""
from copy import deepcopy
import numpy as np
import mujoco
from genedynamics.tasks.transfer.runtime import ExecutionSpec,MechanismSpec,SkillSpec,ControllerState,ExecutionState
from genedynamics.core.control.transfer import integrate_error
from genedynamics.envs.domains.manipulation.transfer_workcell.backends.mujoco import MujocoExecutionBackend


class PublicObstacleNominalModel:
    VERSION='t1-public-encoder-owned-nominal-model-v1'
    def __init__(self,spec,*,charge=None):
        if spec.alias!='T1' or spec.config.robot not in {'panda','xarm7'}:raise ValueError('explicit fixed-arm T1 model only')
        self.spec=spec;self.backend=MujocoExecutionBackend(execution=ExecutionSpec(backend='numpy',device='cpu',physics_backend='mujoco'),task=spec.config,skill=SkillSpec(),mechanism=MechanismSpec());self.charge=charge;self.irreversible_model_controls=0;self.irreversible_forward_calls=0;self.reset()
    def reset(self):self.previous=None;self.tick=-1;self.integral=np.zeros(6)
    def get_state(self):return deepcopy(dict(version=self.VERSION,robot=self.spec.config.robot,previous=self.previous,tick=self.tick,integral=self.integral))
    def restore(self,s):
        if s['version']!=self.VERSION or s['robot']!=self.spec.config.robot:raise ValueError('nominal public model chart differs')
        self.previous=deepcopy(s['previous']);self.tick=s['tick'];self.integral=s['integral'].copy()
    def observe(self,public):
        tick=int(np.rint(np.asarray(public['elapsed_seconds']).item()/self.spec.config.control_dt))
        if tick==self.tick:return
        if tick!=self.tick+1:raise ValueError('public model reconstruction requires all observed ticks')
        if self.previous is not None:
            self.integral=integrate_error(np,self.integral,self.previous['tcp_position'][0],self.previous['tcp_rotation'][0],public['reference_position'][0],public['reference_rotation'][0],self.spec.config.control_dt)
        self.previous={k:np.asarray(v).copy() for k,v in public.items()};self.tick=tick
    def predict(self,commands):
        if self.previous is None:raise RuntimeError('observe public packet before prediction')
        commands=np.asarray(commands,np.float32)
        if commands.ndim!=2 or commands.shape[-1]!=13 or not len(commands):raise ValueError('nonempty Hx13 public requests required')
        p=self.previous;data=mujoco.MjData(self.backend.model);data.qpos[:]=p['joint_position'][0];data.qvel[:]=p['joint_velocity'][0];mujoco.mj_forward(self.backend.model,data);self.irreversible_forward_calls+=1
        # Public encoder assimilation owns a fresh MjData. No real warmstart,
        # contact solver cache, hidden delay queue or integration state is used.
        controller=ControllerState(p['reference_position'][0].astype(np.float64),p['reference_rotation'][0].astype(np.float64),np.zeros((self.spec.config.max_delay_steps+1,13)),p['previous_action'][0].astype(np.float64),self.integral.copy())
        state=ExecutionState(data,controller,0,self.tick,0,0);traces=[]
        for command in commands:
            if self.charge:self.charge('command_requests',1)
            state,trace=self.backend._step(state,command)
            if trace.valid:
                self.irreversible_model_controls+=1
                if self.charge:
                    self.charge('control_steps',1);self.charge('physics_steps',self.spec.config.substeps)
            traces.append(dict(position=np.asarray(trace.tcp_position).copy(),rotation=np.asarray(trace.tcp_rotation).copy(),joint_position=state.physics.qpos.copy(),joint_velocity=state.physics.qvel.copy(),joint_speed_peak=float(trace.max_joint_speed),clearance=float(trace.clearance_lower_bound),status=int(trace.status),valid=bool(trace.valid)))
            if trace.terminal:break
        return traces
