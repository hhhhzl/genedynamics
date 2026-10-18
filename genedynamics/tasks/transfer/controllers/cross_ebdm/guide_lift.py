"""Public T2 lift-route state/features for ordinary response MPC.

The ordinary planner uses target history and the shared response model.
Source-query support is measured but does not disable this source-free planner.
Source/body phase semantics remain explicit; transport benefits are unqualified.
"""
import numpy as np
from genedynamics.solvers.single.cross_ebdm.learning.response import public_frame,features
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import BOUNDS
from genedynamics.tasks.transfer.behavior import teacher_reference
from genedynamics.tasks.transfer.controllers.guide_lift_route import (
    CLEAR_Z,
    PublicGuideLiftRoutePolicy,
)
from genedynamics.solvers.single.response_mpc.arm import PublicArmTaskAdamMPCActor
from .arm_dense import PublicArmTaskDenseActor
from .arm_noise import PublicArmTaskNoiseActor

def lift_state(base,public):
    spec=base.spec
    if spec.config.robot=='panda':
        memory=base.base.observer.memory;phase=memory.phase.copy();integral=memory.tracking_integral.copy();ref=teacher_reference(memory,insertion=False,goal_z=spec.config.goal[2]);target=ref[:,:3];force=ref[:,4:5]
    else:
        phase=np.array([base.stage],np.int64);integral=base.compensation.copy();target=base.initial.copy();target[:,2]=CLEAR_Z;axis=0 if base.mode==0 else 1;goal=np.asarray(spec.config.goal)
        if base.stage>=1:target[:,axis]=goal[axis]
        if base.stage>=2:target[:,:2]=goal[:2]
        if base.stage==3:target[:,2]=goal[2]
        force=np.full((1,1),5. if base.stage==3 else 0.)
    rotation=np.broadcast_to(np.diag([1.,-1.,-1.]),(1,3,3));return phase,integral,np.concatenate((target,rotation.reshape(1,9),force),axis=1).astype(np.float32)

class _LiftPrepare:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.spec.alias!='T2':raise ValueError('explicit T2 lift interface')
        self.base=PublicGuideLiftRoutePolicy(self.spec,mode=self.mode)
    def prepare(self,public):
        if len(public['tcp_position'])!=1:raise ValueError('explicit batch1')
        tick=int(np.rint(np.asarray(public['elapsed_seconds']).item()/self.spec.config.control_dt))
        if tick<self.last_tick or tick>self.last_tick+1:raise ValueError('all public ticks required')
        if tick!=self.last_tick:self.frames.append(public_frame(public,self.spec.config.robot,self.policy.response_spec)[0]);self.frames=self.frames[-self.policy.response_spec.history_len:];self.last_tick=tick
        base=self.base.act(public);phase,integral,ref=lift_state(self.base,public);source,distance,index=self.bank.query(public,phase,[self.mode]);target=features(np.stack(self.frames),[len(self.frames)-1],self.spec.config.robot,'T2',[0,0,0],self.policy.response_spec)
        return dict(target_features=target,source_features=source['features'],source_commands=source['commands'],base_proto=np.repeat(base[:,None],self.policy.response_spec.horizon,axis=1)/BOUNDS,phase=phase,own_context=np.concatenate((integral/.2,np.zeros((1,3))),axis=1).astype(np.float32),current=dict(position=public['tcp_position'].copy(),rotation=public['tcp_rotation'].copy(),wrench=public['measured_wrench'].copy()),task_reference=ref,source_query_distance=distance),base

class PublicLiftGuideDenseActor(_LiftPrepare,PublicArmTaskDenseActor):
    VERSION='public-T2-lift-reference-dense-v1'

class PublicLiftGuideMPCActor(_LiftPrepare,PublicArmTaskAdamMPCActor):
    VERSION='public-T2-lift-ordinary-sourcefree-AdamMPC-v1'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.maximum_query_distance=np.inf

class PublicLiftGuideNoiseActor(PublicArmTaskNoiseActor):
    VERSION='public-T2-lift-local-noise-v1'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.spec.alias!='T2':raise ValueError('T2 only')
        self.base=PublicGuideLiftRoutePolicy(self.spec,mode=self.mode)
