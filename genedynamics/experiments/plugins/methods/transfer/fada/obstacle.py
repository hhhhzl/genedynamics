"""T1 knownbody public-memory correspondence; native FADA core unchanged.

Panda keeps the frozen source observer byte-for-byte. xArm uses the same known
workspace route/integral rule available to the common public feedback actor.
This is an explicit adapter version, not a claim that native FADA failed.
"""
from copy import deepcopy
import numpy as np
from genedynamics.tasks.transfer.observation import PublicControllerMemory
from genedynamics.tasks.transfer.controllers.obstacle_feedback import PublicObstacleFeedbackPolicy
from .base import PublicFADATaskActor

BODY_MEMORY_VERSION='public-T1-knownbody-memory-v2'


class PublicObstacleBodyMemory:
    def __init__(self,spec,modes):
        if spec.alias!='T1' or spec.config.robot!='xarm7':raise ValueError('explicitT1xArm publicbody observer')
        self.spec=spec;self.base=PublicObstacleFeedbackPolicy(spec,modes=modes)
    def get_state(self):return self.base.get_state()
    def restore(self,state):self.base.restore(state)
    def reorder(self,indices):self.base.reorder(indices)
    def update(self,public):
        self.base.act(public);p=np.asarray(public['tcp_position'],np.float64);r=np.asarray(public['tcp_rotation'],np.float64);rp=np.asarray(public['reference_position']);rr=np.asarray(public['reference_rotation']);n=len(p);scale=max(.05,float(np.linalg.norm(np.asarray(self.spec.config.goal_position)-self.spec.config.start_position)))
        def relative(position,rotation):
            translation=np.einsum('bji,bj->bi',r,position-p)/scale;rot=np.swapaxes(r,-1,-2)@rotation;return np.concatenate((translation,rot[:,:,:2].swapaxes(-1,-2).reshape(n,6)),axis=-1)
        high=np.asarray([.12]*3+[.6]*3+[1.]*6+[1.]);return np.concatenate((relative(rp,rr),np.asarray(public['previous_action'])/high,np.eye(4)[self.base.phase],np.eye(2)[self.base.modes],self.base.integral/.2,np.zeros((n,1)),relative(p,r),relative(rp,rr)),axis=-1).astype(np.float32)


def obstacle_body_memory(spec,modes):
    if spec.alias!='T1':raise ValueError('T1 only')
    return PublicControllerMemory(spec,modes) if spec.config.robot=='panda' else PublicObstacleBodyMemory(spec,modes)


def body_context_sequence(public,spec,mode):
    observer=obstacle_body_memory(spec,[mode]);return np.stack([observer.update({k:np.asarray(v)[i:i+1] for k,v in public.items()})[0] for i in range(len(public['tcp_position']))])


def obstacle_body_checkpoint_binding(payload):
    result=deepcopy(payload);result['genedynamics_task_adapter']['public_memory_version']=BODY_MEMORY_VERSION;result['genedynamics_task_adapter']['source_memory_unchanged']=True;return result


class PublicObstacleFADATaskActor(PublicFADATaskActor):
    def __init__(self,spec,payload,*,mode,device='cpu'):
        if spec.alias!='T1' or spec.config.robot not in {'panda','xarm7'}:raise ValueError('explicitT1 knownbodybinding')
        if spec.config.robot=='xarm7' and payload['genedynamics_task_adapter'].get('public_memory_version')!=BODY_MEMORY_VERSION:raise ValueError('checkpoint mustbe fit withsamepublicbody memory')
        super().__init__(spec,payload,mode=mode,device=device);self.policy.memory_factory=obstacle_body_memory
