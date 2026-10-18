"""Causal portable FADA task binding owning only public packets and actor memory."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
from genedynamics.envs.adapters.transfer_task import FLAT_FIELDS
from genedynamics.tasks.transfer.guide_observation import public_memory
from genedynamics.solvers.single.fada.learning.source_training import SourceTaskPolicy
from genedynamics.solvers.single.fada.learning.task_training import portable_history
from genedynamics.tasks.transfer.runtime import SkillSpec


class PublicTaskHistory:
    def __init__(self,spec,mode,history_len):
        self.spec=spec;self.config=spec.config;self.mode=mode;self.history_len=history_len;self.num_envs=1
        self.runner=SimpleNamespace(skill=SkillSpec());self.packets=[]
    def observe(self):return deepcopy(self.packets[-1])
    def fada_observation(self):
        n=len(self.packets);public=self.packets[-1]
        flat=np.concatenate([np.concatenate([row[k].reshape(1,-1) for k in FLAT_FIELDS],axis=-1) for row in self.packets])
        obs=np.zeros((1,self.history_len,flat.shape[-1]),np.float32);actions=np.zeros((1,self.history_len,13),np.float32);mask=np.zeros((1,self.history_len),bool)
        obs[0,-n:]=flat;actions[0,-n:]=np.concatenate([row['previous_action'] for row in self.packets]);mask[0,-n:]=True
        command=np.concatenate((public['goal_position'],public['goal_rotation'].reshape(1,9),np.eye(2,dtype=np.float32)[self.mode:self.mode+1]),axis=-1)
        return dict(history_obs=obs,history_act=actions,history_valid_mask=mask,current_command=command)


class PublicFADATaskActor:
    VERSION='public-fada-portable-owned-history-v1'
    def __init__(self,spec,payload,*,mode,device='cpu'):
        self.spec=spec;self.mode=mode;self.view=PublicTaskHistory(spec,mode,payload['cfg']['history_len'])
        self.policy=SourceTaskPolicy.from_checkpoint(payload,task=self.view,device=device,memory_factory=public_memory)
        self.reset();self.irreversible_network_calls=0
    def reset(self):
        self.view.packets=[];self.policy.reset();self.tick=-1;self.cached=None
    def get_state(self):
        return deepcopy(dict(version=self.VERSION,mode=self.mode,task=self.spec.name,robot=self.spec.config.robot,packets=self.view.packets,tick=self.tick,cached=self.cached,memory=self.policy.public_memory,actions=self.policy.solver._actions,index=self.policy.solver._index))
    def restore(self,state):
        if (state['version'],state['mode'],state['task'],state['robot'])!=(self.VERSION,self.mode,self.spec.name,self.spec.config.robot):raise ValueError('public FADA binding differs')
        self.view.packets=deepcopy(state['packets']);self.tick=state['tick'];self.cached=deepcopy(state['cached']);self.policy.public_memory=deepcopy(state['memory']);self.policy.solver._actions=deepcopy(state['actions']);self.policy.solver._index=state['index']
    def act(self,public):
        tick=int(np.rint(np.asarray(public['elapsed_seconds']).item()/self.spec.config.control_dt))
        if tick==self.tick:return self.cached.copy()
        if tick!=self.tick+1:raise ValueError('FADA must observe every public control tick; reset at episode start')
        if np.asarray(public['tcp_position']).shape[0]!=1:raise ValueError('this explicit task binding is batch1')
        self.view.packets.append({k:np.asarray(v).copy() for k,v in public.items()});self.view.packets=self.view.packets[-self.view.history_len:]
        self.cached=self.policy.act(portable_history(self.view)).astype(np.float32);self.tick=tick;self.irreversible_network_calls+=1
        return self.cached.copy()
