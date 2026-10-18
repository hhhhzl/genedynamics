"""Ordinary endpoint-response MPC with the same public model/local support."""
import time
import numpy as np
import torch
from genedynamics.tasks.transfer.behavior import teacher_reference
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import BOUNDS,decode_pose_transport,task_proxy
from genedynamics.tasks.transfer.controllers.cross_ebdm.dense import PublicDenseTransportActor


class PublicResponseMPCActor(PublicDenseTransportActor):
    """Fixed action-gradient steps; frozen model; report20ms wall overruns."""
    def __init__(self,*args,response,iterations=5,learning_rate=.25,soft_deadline_seconds=.02,**kwargs):
        super().__init__(*args,**kwargs)
        if iterations!=5 or learning_rate!=.25:raise ValueError('fixed declared ordinary MPC opportunity')
        self.response=response.to(self.device).eval()
        for p in self.response.parameters():p.requires_grad_(False)
        self.iterations,self.learning_rate=iterations,learning_rate;self.soft_deadline_seconds=soft_deadline_seconds
        self.irreversible_model_calls=0;self.irreversible_action_updates=0;self.model_acceptance_fallbacks=0;self.inference_seconds=[]

    def get_state(self):
        value=super().get_state();value['model_acceptance_fallbacks']=self.model_acceptance_fallbacks;return value

    def restore(self,state):
        super().restore(state);self.model_acceptance_fallbacks=state['model_acceptance_fallbacks']

    def reset(self):
        super().reset();self.model_acceptance_fallbacks=0

    def act(self,public):
        begun=time.perf_counter();batch,base_action=self.prepare(public)
        if self.last_query_distance>self.maximum_query_distance:
            self.fallback_controls+=1;self.inference_seconds.append(time.perf_counter()-begun);return base_action
        def tensor(value):
            if isinstance(value,dict):return {k:tensor(v) for k,v in value.items()}
            v=torch.as_tensor(value,device=self.device);return v.float() if v.is_floating_point() else v
        state=tensor(batch['decoder']);feature=tensor(batch['target_features']);base=tensor(batch['base_proto']);current=tensor(batch['current']);bounds=base.new_tensor(BOUNDS)
        ref=teacher_reference(self.base.observer.memory,goal_z=self.spec.config.goal[2]);c,s=np.cos(ref[:,3]),np.sin(ref[:,3]);rotation=np.zeros((1,3,3));rotation[:,0,0]=c;rotation[:,0,1]=s;rotation[:,1,0]=s;rotation[:,1,1]=-c;rotation[:,2,2]=-1.
        reference=tensor(np.concatenate((ref[:,:3],rotation.reshape(1,9),ref[:,4:5]),axis=1))
        with torch.inference_mode(False),torch.enable_grad():
            latent=torch.nn.Parameter(torch.zeros_like(base));optimizer=torch.optim.Adam([latent],lr=self.learning_rate)
            def commands():
                value=base+.025*torch.tanh(latent);value=torch.cat((value[...,:12].clamp(-1,1),value[...,12:13].clamp(0,1)),dim=-1)
                return torch.cat((value[...,:12],torch.where(base[...,12:13]*25>.1,value[...,12:13],torch.zeros_like(value[...,12:13]))),dim=-1)
            def objective(value):
                actual=decode_pose_transport(value*bounds,state,self.feedback)/bounds;pred=self.response(feature,actual);self.irreversible_model_calls+=1
                task=task_proxy(pred,current,reference).mean();peaks=torch.expm1(pred[...,18:20]).clamp(min=0)
                risk=((peaks[...,0]/25.-1).clamp(min=0)**2+(peaks[...,1]/10.-1).clamp(min=0)**2).max()
                return task+risk+.01*((value-base)**2).mean()
            base_cost=None
            for _ in range(self.iterations):
                loss=objective(commands())
                if base_cost is None:base_cost=float(loss.detach())
                if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite ordinary model MPC objective')
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();self.irreversible_action_updates+=1
            with torch.no_grad():
                selected=commands();cost=float(objective(selected))
                if not np.isfinite(cost) or cost>base_cost:selected=base;self.model_acceptance_fallbacks+=1
                self.interceptor.override=(selected[:,0:1]*bounds).cpu().numpy()
        value=self.base.act(public);self.inference_seconds.append(time.perf_counter()-begun);return value
