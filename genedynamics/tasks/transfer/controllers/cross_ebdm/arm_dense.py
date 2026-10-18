"""Causal T1/T2 public dense and ordinary response-MPC bindings."""
from copy import deepcopy
import numpy as np
import torch
from genedynamics.solvers.single.cross_ebdm.learning.response import public_frame,features
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import BOUNDS,task_proxy
from genedynamics.solvers.single.cross_ebdm.learning.task_response import task_amplitude
from genedynamics.tasks.transfer.behavior import teacher_reference
from genedynamics.tasks.transfer.controllers.guide_feedback import PublicGuideFeedbackPolicy
from genedynamics.tasks.transfer.controllers.obstacle_feedback import PublicObstacleFeedbackPolicy


class PublicArmTaskDenseActor:
    VERSION='public-T1-T2-actualrequest-dense-v1'
    def __init__(self,spec,policy,bank,*,mode,device='cpu',maximum_query_distance=5.):
        if spec.alias not in {'T1','T2'} or bank.task!=spec.alias or bank.spec!=policy.response_spec:raise ValueError('matching actualtask/sourcebank required')
        self.spec,self.policy,self.bank=spec,policy,bank;self.mode=mode;self.device=torch.device(device);self.maximum_query_distance=maximum_query_distance;self.policy.to(self.device).eval();self.base=(PublicObstacleFeedbackPolicy(spec,modes=[mode]) if spec.alias=='T1' else PublicGuideFeedbackPolicy(spec,modes=[mode],lead_xy_m=.008,position_admittance=True));self.frames=[];self.last_tick=-1;self.fallback_controls=0;self.irreversible_network_calls=0;self.last_action_tick=-1;self.cached_action=None
    def reset(self):self.base.reset();self.frames=[];self.last_tick=-1;self.fallback_controls=0;self.last_action_tick=-1;self.cached_action=None
    def get_state(self):return deepcopy(dict(version=self.VERSION,base=self.base.get_state(),frames=self.frames,last_tick=self.last_tick,fallback_controls=self.fallback_controls,last_action_tick=self.last_action_tick,cached_action=self.cached_action))
    def restore(self,s):
        if s['version']!=self.VERSION:raise ValueError('armtask densebinding differs')
        self.base.restore(s['base']);self.frames=deepcopy(s['frames']);self.last_tick=s['last_tick'];self.fallback_controls=s['fallback_controls'];self.last_action_tick=s['last_action_tick'];self.cached_action=deepcopy(s['cached_action'])
    def prepare(self,public):
        if len(public['tcp_position'])!=1:raise ValueError('explicit batch1 publichistory')
        tick=int(np.rint(np.asarray(public['elapsed_seconds']).item()/self.spec.config.control_dt))
        if tick<self.last_tick or tick>self.last_tick+1:raise ValueError('observe everytick andreset explicitly')
        if tick!=self.last_tick:self.frames.append(public_frame(public,self.spec.config.robot,self.policy.response_spec)[0]);self.frames=self.frames[-self.policy.response_spec.history_len:];self.last_tick=tick
        base=self.base.act(public)
        if self.spec.alias=='T1':phase=self.base.phase.copy();integral=self.base.integral.copy();ref=np.concatenate((self.base.reference(),public['goal_rotation'].reshape(1,9),np.zeros((1,1))),axis=-1)
        else:
            memory=self.base.observer.memory;phase=memory.phase.copy();integral=memory.tracking_integral.copy();waypoint=teacher_reference(memory,insertion=False,goal_z=self.spec.config.goal[2]);rotation=np.broadcast_to(np.diag([1.,-1.,-1.]),(1,3,3));ref=np.concatenate((waypoint[:,:3],rotation.reshape(1,9),waypoint[:,4:5]),axis=-1)
        source,distance,index=self.bank.query(public,phase,[self.mode]);target=features(np.stack(self.frames),[len(self.frames)-1],self.spec.config.robot,self.spec.alias,[0,0,0],self.policy.response_spec)
        context=np.concatenate((integral/.2,np.zeros((1,3))),axis=1).astype(np.float32)
        return dict(target_features=target,source_features=source['features'],source_commands=source['commands'],base_proto=np.repeat(base[:,None],self.policy.response_spec.horizon,axis=1)/BOUNDS,phase=phase,own_context=context,current=dict(position=public['tcp_position'].copy(),rotation=public['tcp_rotation'].copy(),wrench=public['measured_wrench'].copy()),task_reference=ref.astype(np.float32),source_query_distance=distance),base
    def act(self,public):
        tick=int(np.rint(np.asarray(public['elapsed_seconds']).item()/self.spec.config.control_dt))
        if tick==self.last_action_tick:return self.cached_action.copy()
        b,base=self.prepare(public)
        if b['source_query_distance'][0]>self.maximum_query_distance:self.fallback_controls+=1;self.cached_action=base.copy();self.last_action_tick=tick;return base
        with torch.inference_mode():
            args=[torch.as_tensor(b[k],device=self.device,dtype=torch.float32) for k in ('target_features','source_features','source_commands','base_proto')];phase=torch.as_tensor(b['phase'],device=self.device);context=torch.as_tensor(b['own_context'],device=self.device);value=self.policy(*args,phase,context);self.irreversible_network_calls+=1
        self.cached_action=(value[:,0]*value.new_tensor(BOUNDS)).cpu().numpy().astype(np.float32);self.last_action_tick=tick;return self.cached_action.copy()


class PublicArmTaskResponseMPCActor(PublicArmTaskDenseActor):
    def __init__(self,*args,response,**kwargs):
        super().__init__(*args,**kwargs);self.response=response.to(self.device).eval();self.irreversible_model_calls=0;self.irreversible_action_updates=0
        for p in self.response.parameters():p.requires_grad_(False)
    def act(self,public):
        tick=int(np.rint(np.asarray(public['elapsed_seconds']).item()/self.spec.config.control_dt))
        if tick==self.last_action_tick:return self.cached_action.copy()
        b,base=self.prepare(public)
        if b['source_query_distance'][0]>self.maximum_query_distance:self.fallback_controls+=1;self.cached_action=base.copy();self.last_action_tick=tick;return base
        f=torch.as_tensor(b['target_features'],device=self.device,dtype=torch.float32);proto=torch.as_tensor(b['base_proto'],device=self.device,dtype=torch.float32);amplitude=task_amplitude(f)[:,None];current={k:torch.as_tensor(v,device=self.device,dtype=torch.float32) for k,v in b['current'].items()};reference=torch.as_tensor(b['task_reference'],device=self.device,dtype=torch.float32);value=proto.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            for _ in range(5):
                predicted=self.response(f,value);self.irreversible_model_calls+=1;loss=task_proxy(predicted,current,reference).mean()+.01*((value-proto)**2).mean();gradient,=torch.autograd.grad(loss,value);value=value-.25*gradient;value=torch.minimum(torch.maximum(value,proto-.025*amplitude),proto+.025*amplitude);value=torch.minimum(torch.maximum(value,-amplitude),amplitude);value=torch.cat((value[...,:12],torch.where(proto[...,12:13]*25>.1,value[...,12:13].clamp(min=0),torch.zeros_like(value[...,12:13]))),dim=-1).detach().requires_grad_(True);self.irreversible_action_updates+=1
        self.cached_action=(value[:,0].detach()*value.new_tensor(BOUNDS)).cpu().numpy().astype(np.float32);self.last_action_tick=tick;return self.cached_action.copy()


class PublicArmTaskSourcePriorActor(PublicArmTaskDenseActor):
    """Issue the nearest frozen Panda source request in shared-13 coordinates."""
    VERSION = 'public-T1-T2-frozen-source-skill-prior-v1'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.irreversible_source_prior_calls = 0

    def act(self, public):
        tick = int(np.rint(np.asarray(public['elapsed_seconds']).item() / self.spec.config.control_dt))
        if tick == self.last_action_tick:
            return self.cached_action.copy()
        batch, base = self.prepare(public)
        if batch['source_query_distance'][0] > self.maximum_query_distance:
            self.fallback_controls += 1
            action = base
        else:
            action = np.asarray(batch['source_commands'][:, 0]) * BOUNDS
            self.irreversible_source_prior_calls += 1
        self.cached_action = np.asarray(action, dtype=np.float32)
        self.last_action_tick = tick
        return self.cached_action.copy()
