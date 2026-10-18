"""T2 corridor task binding with explicit source-compatible phase categories.

Lift stages 1 and 2 are alignment/motion (source phase2); phase1 is absent in
the frozen guide source bank. This encoding asserts no equality of contact
dynamics. Ordinary MPC uses no source query; Full/task-only retain measured
source support distances and their predeclared fallback.
"""
import numpy as np
from genedynamics.solvers.single.cross_ebdm.learning.response import public_frame,features
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import BOUNDS
from genedynamics.solvers.single.response_mpc.arm import PublicArmTaskAdamMPCActor
from genedynamics.tasks.transfer.controllers.guide_corridor_lift_route import PublicGuideCorridorLiftRoutePolicy
from .arm_dense import PublicArmTaskDenseActor
from .arm_noise import PublicArmTaskNoiseActor
from .guide_lift import lift_state
PHASES=np.asarray([0,2,2,3],np.int64)
def corridor_state(base,public):
    phase,integral,reference=lift_state(base,public)
    if base.spec.config.robot=='xarm7':phase=PHASES[np.asarray([base.stage])]
    return phase,integral,reference
class _CorridorPrepare:
    requires_source_query=True
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.spec.alias!='T2':raise ValueError('explicit T2 corridor binding')
        self.base=PublicGuideCorridorLiftRoutePolicy(self.spec,mode=self.mode)
    def prepare(self,public):
        tick=int(np.rint(np.asarray(public['elapsed_seconds']).item()/self.spec.config.control_dt))
        if len(public['tcp_position'])!=1 or tick<self.last_tick or tick>self.last_tick+1:raise ValueError('one world and every public tick required')
        if tick!=self.last_tick:self.frames.append(public_frame(public,self.spec.config.robot,self.policy.response_spec)[0]);self.frames=self.frames[-self.policy.response_spec.history_len:];self.last_tick=tick
        base=self.base.act(public);phase,integral,reference=corridor_state(self.base,public);target=features(np.stack(self.frames),[len(self.frames)-1],self.spec.config.robot,'T2',[0,0,0],self.policy.response_spec)
        batch=dict(target_features=target,base_proto=np.repeat(base[:,None],self.policy.response_spec.horizon,axis=1)/BOUNDS,phase=phase,own_context=np.concatenate((integral/.2,np.zeros((1,3))),axis=1).astype(np.float32),current=dict(position=public['tcp_position'].copy(),rotation=public['tcp_rotation'].copy(),wrench=public['measured_wrench'].copy()),task_reference=reference,source_query_distance=np.zeros(1))
        if self.requires_source_query:
            source,distance,_=self.bank.query(public,phase,[self.mode]);batch.update(source_features=source['features'],source_commands=source['commands'],source_query_distance=distance)
        return batch,base
class PublicCorridorGuideDenseActor(_CorridorPrepare,PublicArmTaskDenseActor):
    VERSION='public-T2-corridor-semantic-phase-dense-v1'
class PublicCorridorGuideMPCActor(_CorridorPrepare,PublicArmTaskAdamMPCActor):
    VERSION='public-T2-corridor-sourcefree-AdamMPC-v1'
    requires_source_query=False
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.maximum_query_distance=np.inf
class PublicCorridorGuideNoiseActor(PublicArmTaskNoiseActor):
    VERSION='public-T2-corridor-local-noise-v1'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.spec.alias!='T2':raise ValueError('explicit T2 corridor excitation')
        self.base=PublicGuideCorridorLiftRoutePolicy(self.spec,mode=self.mode)
