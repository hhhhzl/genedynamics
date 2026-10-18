"""Public-history dense transport execution through the shared NumPy decoder."""
from copy import deepcopy
import numpy as np
import torch

from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.bias_feedback import PublicBiasFeedbackPolicy
from genedynamics.solvers.single.cross_ebdm.learning.response import public_frame,features
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import BOUNDS,decoder_state,decoder_context


class ProtoInterceptor:
    def __init__(self):self.override=None;self.raw=None
    def apply(self,value,phase):
        self.raw=value.copy()
        return value if self.override is None else self.override.copy()


def canonical_proto(raw):
    value=np.array(raw,copy=True);value[...,:3]=np.clip(value[...,:3],-.02,.02);value[...,3:6]=np.clip(value[...,3:6],-.12,.12)
    value[...,6:12]=np.clip(value[...,6:12],-1.,1.);value[...,12]=np.clip(value[...,12],0.,25.)
    return value


class PublicDenseTransportActor:
    """One independent world; optional batching shares immutable neural weights."""
    def __init__(self,spec,policy,source_bank,feedback,*,prior_path=None,mode=0,seed=0,device='cpu',maximum_query_distance=5.):
        if spec.alias!='T3':raise ValueError('dense decoder currently physically binds T3; other tasks require separate qualification')
        if source_bank.spec!=policy.response_spec:raise ValueError('source query and learned policy schemas differ')
        self.spec,self.policy,self.source_bank,self.feedback=spec,policy,source_bank,feedback;self.device=torch.device(device);self.mode=int(mode);self.seed=int(seed);self.maximum_query_distance=maximum_query_distance
        self.base=PublicBiasFeedbackPolicy.load(spec,prior_path,feedback=feedback,modes=[mode],seed=seed);self.interceptor=ProtoInterceptor();self.base.pose_transport=self.interceptor
        self.policy.to(self.device).eval();self.frames=[];self.last_tick=-1;self.last_query_distance=None;self.fallback_controls=0;self.irreversible_network_calls=0

    def get_state(self):return deepcopy(dict(base=self.base.get_state(),frames=self.frames,last_tick=self.last_tick,last_query_distance=self.last_query_distance,fallback_controls=self.fallback_controls))

    def restore(self,state):
        self.base.restore(state['base']);self.frames=deepcopy(state['frames']);self.last_tick=state['last_tick'];self.last_query_distance=state['last_query_distance'];self.fallback_controls=state['fallback_controls'];self.interceptor.override=None

    def reset(self):
        self.base.reset(modes=[self.mode],seed=self.seed);self.frames=[];self.last_tick=-1;self.last_query_distance=None;self.fallback_controls=0;self.interceptor.override=None

    def prepare(self,public):
        if len(public['tcp_position'])!=1:raise ValueError('one causal history per actor; use independent-world batching')
        tick=int(round(float(np.asarray(public['elapsed_seconds']).reshape(-1)[0])/self.spec.config.control_dt))
        if tick<self.last_tick or (self.last_tick>=0 and tick>self.last_tick+1):raise ValueError('explicit reset/reorder and every observation tick required')
        if tick!=self.last_tick:
            self.frames.append(public_frame(public,self.spec.config.robot,self.policy.response_spec)[0]);self.frames=self.frames[-self.policy.response_spec.history_len:];self.last_tick=tick
        self.interceptor.override=None;base_action=self.base.act(public);raw=canonical_proto(self.interceptor.raw)
        state=decoder_state(self.base,public);source,distance,index=self.source_bank.query(public,state['phase'],[self.mode]);self.last_query_distance=float(distance[0])
        target=features(np.stack(self.frames),[len(self.frames)-1],self.spec.config.robot,self.spec.alias,[0,0,0],self.policy.response_spec)
        return dict(target_features=target,source_features=source['features'],source_commands=source['commands'],base_proto=np.repeat(raw,self.policy.response_spec.horizon,axis=1)/BOUNDS,
            decoder=state,current=dict(position=np.asarray(public['tcp_position']).copy(),rotation=np.asarray(public['tcp_rotation']).copy(),wrench=np.asarray(public['measured_wrench']).copy()),
            source_current=dict(position=source['position'],rotation=source['rotation'],wrench=source['wrench']),source_query_index=index,source_query_distance=distance),base_action

    def act(self,public):
        batch,base_action=self.prepare(public)
        if self.last_query_distance>self.maximum_query_distance:
            self.fallback_controls+=1;return base_action
        state={k:torch.as_tensor(v,device=self.device) for k,v in batch['decoder'].items()};state={k:v.float() if v.is_floating_point() else v for k,v in state.items()}
        with torch.inference_mode():
            values=[torch.as_tensor(batch[k],device=self.device,dtype=torch.float32) for k in ('target_features','source_features','source_commands','base_proto')]
            normalized=self.policy(*values,state['phase'],decoder_context(state,self.feedback));self.irreversible_network_calls+=1
            self.interceptor.override=(normalized[:,0:1]*normalized.new_tensor(BOUNDS)).cpu().numpy()
        return self.base.act(public)


class IndependentWorldDenseSolver:
    """Explicit per-world memory, stable under reordering and partial reset."""
    def __init__(self,actors):self.actors=list(actors)
    def act(self,public):return np.concatenate([actor.act({k:np.asarray(v)[i:i+1] for k,v in public.items()}) for i,actor in enumerate(self.actors)])
    def get_state(self):return [a.get_state() for a in self.actors]
    def restore(self,states):
        if len(states)!=len(self.actors):raise ValueError('actor state/world count differs')
        for actor,state in zip(self.actors,states):actor.restore(state)
    def reorder(self,indices):self.actors=[self.actors[int(i)] for i in indices]
    def reset(self,mask=None):
        mask=np.ones(len(self.actors),bool) if mask is None else np.asarray(mask,bool)
        if mask.shape!=(len(self.actors),):raise ValueError('one reset mask entry per world')
        for actor,selected in zip(self.actors,mask):
            if selected:actor.reset()
