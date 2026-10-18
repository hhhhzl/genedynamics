"""Stronger ordinary T1/T2 Adam response MPC matching the T3 optimizer recipe."""
import numpy as np
import torch
from genedynamics.solvers.single.cross_ebdm.learning.task_response import task_amplitude
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import BOUNDS,task_proxy
from genedynamics.tasks.transfer.controllers.cross_ebdm.arm_dense import PublicArmTaskDenseActor


class PublicArmTaskAdamMPCActor(PublicArmTaskDenseActor):
    def __init__(self,*args,response,**kwargs):
        super().__init__(*args,**kwargs);self.response=response.to(self.device).eval();self.irreversible_model_calls=0;self.irreversible_action_updates=0;self.model_acceptance_fallbacks=0
        for p in self.response.parameters():p.requires_grad_(False)
    def get_state(self):return dict(super().get_state(),model_acceptance_fallbacks=self.model_acceptance_fallbacks)
    def restore(self,s):super().restore(s);self.model_acceptance_fallbacks=s['model_acceptance_fallbacks']
    def reset(self):super().reset();self.model_acceptance_fallbacks=0
    def act(self,public):
        tick=int(np.rint(np.asarray(public['elapsed_seconds']).item()/self.spec.config.control_dt))
        if tick==self.last_action_tick:return self.cached_action.copy()
        b,base_action=self.prepare(public)
        if b['source_query_distance'][0]>self.maximum_query_distance:self.fallback_controls+=1;self.cached_action=base_action.copy();self.last_action_tick=tick;return self.cached_action.copy()
        f=torch.as_tensor(b['target_features'],device=self.device,dtype=torch.float32);base=torch.as_tensor(b['base_proto'],device=self.device,dtype=torch.float32);amplitude=task_amplitude(f)[:,None];current={k:torch.as_tensor(v,device=self.device,dtype=torch.float32) for k,v in b['current'].items()};reference=torch.as_tensor(b['task_reference'],device=self.device,dtype=torch.float32)
        with torch.inference_mode(False),torch.enable_grad():
            latent=torch.nn.Parameter(torch.zeros_like(base));optimizer=torch.optim.Adam([latent],lr=.25)
            def commands():
                value=base+.025*amplitude*torch.tanh(latent);value=torch.minimum(torch.maximum(value,-amplitude),amplitude);force=torch.where(base[...,12:13]*25>.1,value[...,12:13].clamp(min=0),torch.zeros_like(value[...,12:13]));return torch.cat((value[...,:12],force),dim=-1)
            def objective(value):
                predicted=self.response(f,value);self.irreversible_model_calls+=1;peaks=torch.expm1(predicted[...,18:20]).clamp(min=0);risk=((peaks[...,0]/25.-1).clamp(min=0)**2+(peaks[...,1]/10.-1).clamp(min=0)**2).max();return task_proxy(predicted,current,reference).mean()+risk+.01*((value-base)**2).mean()
            initial=None
            for _ in range(5):
                loss=objective(commands())
                if initial is None:initial=float(loss.detach())
                if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite ordinaryAdamMPC modelobjective')
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();self.irreversible_action_updates+=1
            with torch.no_grad():
                value=commands();cost=float(objective(value))
                if not np.isfinite(cost) or cost>initial:value=base;self.model_acceptance_fallbacks+=1
                self.cached_action=(value[:,0]*value.new_tensor(BOUNDS)).cpu().numpy().astype(np.float32)
        self.last_action_tick=tick;return self.cached_action.copy()
