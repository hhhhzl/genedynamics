"""Ordinary T2 lift MPC needs no source query or source phase correspondence."""
import numpy as np
from genedynamics.solvers.single.cross_ebdm.learning.response import public_frame,features
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import BOUNDS
from genedynamics.tasks.transfer.controllers.cross_ebdm.guide_lift import (
    PublicLiftGuideMPCActor,
    lift_state,
)
class PublicSourceFreeLiftGuideMPCActor(PublicLiftGuideMPCActor):
    VERSION='public-T2-sourcefree-lift-AdamMPC-v2'
    def prepare(self,public):
        if len(public['tcp_position'])!=1:raise ValueError('explicit batch1')
        tick=int(np.rint(np.asarray(public['elapsed_seconds']).item()/self.spec.config.control_dt))
        if tick<self.last_tick or tick>self.last_tick+1:raise ValueError('all public ticks required')
        if tick!=self.last_tick:self.frames.append(public_frame(public,self.spec.config.robot,self.policy.response_spec)[0]);self.frames=self.frames[-self.policy.response_spec.history_len:];self.last_tick=tick
        base=self.base.act(public);phase,integral,ref=lift_state(self.base,public);target=features(np.stack(self.frames),[len(self.frames)-1],self.spec.config.robot,'T2',[0,0,0],self.policy.response_spec)
        return dict(target_features=target,base_proto=np.repeat(base[:,None],self.policy.response_spec.horizon,axis=1)/BOUNDS,phase=phase,own_context=np.concatenate((integral/.2,np.zeros((1,3))),axis=1).astype(np.float32),current=dict(position=public['tcp_position'].copy(),rotation=public['tcp_rotation'].copy(),wrench=public['measured_wrench'].copy()),task_reference=ref,source_query_distance=np.zeros(1)),base
