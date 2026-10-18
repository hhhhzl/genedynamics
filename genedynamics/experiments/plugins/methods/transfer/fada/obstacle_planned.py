"""The same T1 model-selected path in FADA fitting and deployment memory."""
from copy import deepcopy
import numpy as np
from .base import PublicFADATaskActor
from .obstacle import PublicObstacleBodyMemory, obstacle_body_memory
from genedynamics.tasks.transfer.controllers.obstacle_path import PublicObstaclePathPolicy

VERSION='public-T1-common-planned-FADA-memory-v1'


def planned_body_memory(spec,modes,path):
    if spec.alias!='T1':raise ValueError('T1 only')
    if spec.config.robot=='panda':return obstacle_body_memory(spec,modes)
    memory=PublicObstacleBodyMemory(spec,modes);memory.base=PublicObstaclePathPolicy(spec,modes=modes,path=path);return memory


def planned_context_sequence(public,spec,mode,path):
    memory=planned_body_memory(spec,[mode],path)
    return np.stack([memory.update({k:np.asarray(v)[i:i+1] for k,v in public.items()})[0] for i in range(len(public['tcp_position']))])


def planned_checkpoint_binding(payload,path):
    result=deepcopy(payload);result['genedynamics_task_adapter'].update(public_memory_version=VERSION,public_known_body_path=np.asarray(path).tolist(),source_memory_unchanged=True);return result


class PublicPlannedObstacleFADAActor(PublicFADATaskActor):
    def __init__(self,spec,payload,*,mode,device='cpu'):
        metadata=payload['genedynamics_task_adapter']
        if metadata.get('public_memory_version')!=VERSION:raise ValueError('checkpoint must be fit with same planned memory')
        self.path=np.asarray(metadata['public_known_body_path']).copy();super().__init__(spec,payload,mode=mode,device=device);self.policy.memory_factory=lambda s,m:planned_body_memory(s,m,self.path)
