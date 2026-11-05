"""Explicit T2 target corridor memory for native FADA current-command input."""
from copy import deepcopy
import numpy as np
from genedynamics.tasks.transfer.guide_observation import public_memory
from genedynamics.tasks.transfer.controllers.guide_corridor_lift_route import PublicGuideCorridorLiftRoutePolicy
from genedynamics.tasks.transfer.controllers.cross_ebdm.guide_corridor_v2 import PHASES
from .joint import PublicJointFADATaskActor
VERSION='public-T2-corridor-FADA-memory-v2'
class PublicCorridorGuideMemory:
    def __init__(self,spec,modes):
        if spec.alias!='T2' or spec.config.robot!='xarm7' or len(modes)!=1:raise ValueError('explicit target T2 batch1 memory')
        self.spec=spec;self.mode=int(modes[0]);self.base=PublicGuideCorridorLiftRoutePolicy(spec,mode=self.mode);self.tick=-1;self.start=None;self.cached=None
    def get_state(self):return deepcopy(dict(base=self.base.get_state(),tick=self.tick,start=self.start,cached=self.cached))
    def restore(self,s):self.base.restore(s['base']);self.tick=s['tick'];self.start=deepcopy(s['start']);self.cached=deepcopy(s['cached'])
    def update(self,public):
        tick=int(np.rint(np.asarray(public['elapsed_seconds']).item()/self.spec.config.control_dt))
        if tick==self.tick:return self.cached.copy()
        if tick!=self.tick+1:raise ValueError('every public corridor memory tick required')
        self.base.act(public);p=np.asarray(public['tcp_position']);r=np.asarray(public['tcp_rotation']);rp=np.asarray(public['reference_position']);rr=np.asarray(public['reference_rotation'])
        if self.start is None or tick%10==0:self.start=(p.copy(),r.copy(),rp.copy(),rr.copy())
        def relative(position,rotation):
            return np.concatenate((np.einsum('bji,bj->bi',r,position-p)/.05,(np.swapaxes(r,-1,-2)@rotation)[:,:,:2].swapaxes(-1,-2).reshape(1,6)),axis=1)
        high=np.asarray([.08]*3+[.4]*3+[1.]*6+[25.]);phase=np.eye(4)[[PHASES[self.base.stage]]];mode=np.eye(2)[[self.mode]]
        self.cached=np.concatenate((relative(rp,rr),np.asarray(public['previous_action'])/high,phase,mode,self.base.compensation/.08,np.asarray([[(tick%10)/10.]]),relative(self.start[0],self.start[1]),relative(self.start[2],self.start[3])),axis=1).astype(np.float32);self.tick=tick;return self.cached.copy()
def corridor_body_memory(spec,modes):
    if spec.alias!='T2':raise ValueError('explicit T2 memory binding required')
    return public_memory(spec,modes) if spec.config.robot=='panda' else PublicCorridorGuideMemory(spec,modes)
def corridor_context_sequence(public,spec,mode):
    memory=corridor_body_memory(spec,[mode]);return np.stack([memory.update({k:np.asarray(v)[i:i+1] for k,v in public.items()})[0] for i in range(len(public['tcp_position']))])
def corridor_checkpoint_binding(payload):
    result=deepcopy(payload);result['genedynamics_task_adapter'].update(public_memory_version=VERSION,source_memory_unchanged=True,target_phase_semantics='stage0->approach0, stages1/2->guide-motion1, stage3->terminal3; no contact-response equality asserted');return result
class PublicCorridorGuideFADAActor(PublicJointFADATaskActor):
    VERSION='public-T2-corridor-FADA-actor199-v2'
    def __init__(self,spec,payload,*,mode,device='cpu'):
        if spec.alias!='T2' or spec.config.robot!='xarm7' or payload['genedynamics_task_adapter'].get('public_memory_version')!=VERSION:raise ValueError('fit/deploy same declared target corridor memory')
        super().__init__(spec,payload,mode=mode,device=device)
    def reset(self):
        super().reset();self.policy.public_memory=PublicCorridorGuideMemory(self.spec,[self.mode])
