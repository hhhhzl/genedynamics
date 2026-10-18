"""Dense deterministic source-conditioned transport over public histories.

The differentiable decoder reproduces the shared pose/admittance/governor
operations using public observations and the controller's own memory. Response
training never consumes a simulator carry. Model task proxies remain distinct
from the independently evaluated physical task cost.
"""
from dataclasses import asdict,dataclass
from copy import deepcopy
import numpy as np
import torch
from torch import nn

from genedynamics.core.control.transfer import orientation_error
from .response import PublicResponseSpec,DELTA_SCALE
from genedynamics.learning.checkpoints.torch import capture_rng_state,restore_rng_state,to_cpu

BOUNDS=np.asarray([.08]*3+[.4]*3+[1.]*6+[25.],np.float32)


@dataclass(frozen=True)
class PublicTransportSpec:
    residual_scale:float=1.
    hidden:tuple=(64,64)
    learning_rate:float=3e-4
    response_weight:float=1.
    change_weight:float=.01
    risk_weight:float=1.

    def __post_init__(self):
        if any(not np.isfinite(v) or v<0 for v in (self.residual_scale,self.learning_rate,self.response_weight,self.change_weight,self.risk_weight)) or self.learning_rate==0:raise ValueError('finite nonnegative public transport coefficients required')


def decoder_state(actor,public):
    """Extract only this actor's memory and arithmetic on public observations."""
    phase=actor.observer.memory.phase.copy();fb=actor.pose_feedback;p=np.asarray(public['tcp_position']);r=np.asarray(public['tcp_rotation'])
    approaching=(phase==3)&(p[:,2]>actor.spec.config.goal[2]+fb.guard_seat_gap_m)
    return dict(phase=phase,compensation=actor.compensation.copy(),clear_count=actor.clear_count.copy(),retreating=actor.retreating.copy(),
        waiting=approaching&(actor.clear_count<fb.guard_settle_ticks),position=p.copy(),reference_position=np.asarray(public['reference_position']).copy(),
        rotation_lead=np.stack([orientation_error(np,a,b) for a,b in zip(r,public['reference_rotation'])]),wrench=np.asarray(public['measured_wrench']).copy())


def decode_pose_transport(proto,state,feedback,dt=.02):
    """Torch equivalent of the post-transport T3/T4 pose feedback decoder."""
    if feedback.normal_control!='position_admittance':raise ValueError('explicit pose/admittance decoder required')
    def clip(x,low,high):return torch.minimum(torch.maximum(x,torch.as_tensor(low,device=x.device,dtype=x.dtype)),torch.as_tensor(high,device=x.device,dtype=x.dtype))
    phase=state['phase'][:,None];active=(phase>=2);insert=phase==3;p=state['position'][:,None];rp=state['reference_position'][:,None];rl=state['rotation_lead'][:,None]
    command=proto.clone()
    normal=feedback.normal_admittance_gain*(state['wrench'][:,None,2]-command[...,12])+command[...,2]
    z=torch.where(insert,clip(normal,-feedback.normal_velocity_limit,feedback.normal_velocity_limit),command[...,2]);force=torch.where(insert,0.,command[...,12])
    command=torch.cat((command[...,:2],z[...,None],command[...,3:12],force[...,None]),dim=-1)
    command=torch.cat((clip(command[...,:3],-.02,.02),clip(command[...,3:6],-.12,.12),command[...,6:]),dim=-1)
    lead=command.new_tensor([feedback.lead_xy_m,feedback.lead_xy_m,feedback.lead_z_m]);rotation=clip(command[...,3:6],(-feedback.lead_rotation_rad-rl)/dt,(feedback.lead_rotation_rad-rl)/dt)
    velocity=clip(command[...,:3],(p-lead-rp)/dt,(p+lead-rp)/dt)
    command=torch.cat((torch.where(active[...,None],velocity,command[...,:3]),torch.where(active[...,None],rotation,command[...,3:6]),command[...,6:]),dim=-1)
    z=torch.where(state['waiting'][:,None],torch.maximum(command[...,2],command.new_zeros(())),command[...,2]);z=torch.where(state['retreating'][:,None],feedback.guard_retreat_speed,z)
    z=torch.where(active,clip(z,(p[...,2]-feedback.lead_z_m-rp[...,2])/dt,(p[...,2]+feedback.lead_z_m-rp[...,2])/dt),z)
    z=torch.where(insert,clip(z,-feedback.normal_velocity_limit,feedback.normal_velocity_limit),z)
    command=torch.cat((command[...,:2],z[...,None],command[...,3:]),dim=-1)
    command=torch.cat((command[...,:6]+torch.where(active[...,None],state['compensation'][:,None],0.),command[...,6:]),dim=-1)
    velocity=clip(command[...,:3],(p-lead-rp)/dt,(p+lead-rp)/dt);rotation=clip(command[...,3:6],(-feedback.lead_rotation_rad-rl)/dt,(feedback.lead_rotation_rad-rl)/dt)
    command=torch.cat((torch.where(active[...,None],velocity,command[...,:3]),torch.where(active[...,None],rotation,command[...,3:6]),command[...,6:]),dim=-1)
    command=torch.cat((clip(command[...,:3],-.02,.02),clip(command[...,3:6],-.12,.12),command[...,6:]),dim=-1)
    z=torch.where(insert,clip(command[...,2],-feedback.normal_velocity_limit,feedback.normal_velocity_limit),command[...,2])
    return torch.cat((command[...,:2],z[...,None],command[...,3:]),dim=-1)


def decoder_context(state,feedback):
    return torch.cat((state['compensation']/feedback.compensation_limit,state['waiting'][:,None].to(state['compensation'].dtype),
        state['retreating'][:,None].to(state['compensation'].dtype),(state['clear_count'][:,None]/feedback.guard_settle_ticks).clamp(max=1)),dim=1)


class PublicResidualTransport(nn.Module):
    def __init__(self,response_spec=None,spec=None):
        super().__init__();self.response_spec=response_spec or PublicResponseSpec();self.spec=spec or PublicTransportSpec()
        sizes=(2*self.response_spec.feature_dim+26*self.response_spec.horizon+4+9,*self.spec.hidden,13*self.response_spec.horizon);layers=[]
        for i,(a,b) in enumerate(zip(sizes[:-1],sizes[1:])):
            layers.append(nn.Linear(a,b))
            if i<len(sizes)-2:layers.append(nn.SiLU())
        self.network=nn.Sequential(*layers);nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self,target_features,source_features,source_commands,base_proto,phase,own_decoder_context):
        x=torch.cat((target_features,source_features,source_commands.flatten(start_dim=1),base_proto.flatten(start_dim=1),torch.nn.functional.one_hot(phase.long(),4).to(target_features.dtype),own_decoder_context),dim=1)
        residual=torch.tanh(self.network(x)).reshape(-1,self.response_spec.horizon,13);value=base_proto+self.spec.residual_scale*residual
        return torch.cat((value[...,:12].clamp(-1,1),value[...,12:13].clamp(0,1)),dim=-1)

    def checkpoint(self):return dict(version='public-source-conditioned-transport-v1',response_spec=asdict(self.response_spec),transport_spec=asdict(self.spec),state_dict=to_cpu(self.state_dict()),private_state=False,stochastic_noise=False)

    @classmethod
    def from_checkpoint(cls,payload,device='cpu'):
        if payload.get('version')!='public-source-conditioned-transport-v1' or payload.get('private_state') is not False:raise ValueError('public transport checkpoint required')
        r=dict(payload['response_spec']);r['hidden']=tuple(r['hidden']);s=dict(payload['transport_spec']);s['hidden']=tuple(s['hidden']);model=cls(PublicResponseSpec(**r),PublicTransportSpec(**s)).to(device);model.load_state_dict(payload['state_dict'],strict=True);return model


def task_proxy(predicted,current,reference):
    scale=predicted.new_tensor(DELTA_SCALE);delta=predicted[...,:18]*scale
    position=current['position'][None]+delta[...,:3]
    omega=delta[...,3:6];zero=torch.zeros_like(omega[...,0]);x,y,z=omega.unbind(-1)
    skew=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),dim=-1).reshape(*omega.shape[:-1],3,3)
    rotation=torch.matrix_exp(skew)@current['rotation'][None]
    wrench=current['wrench'][None]+delta[...,12:18]
    return (((position-reference[None,:,:3])/.01)**2).mean(-1)+(((rotation-reference[None,:,3:12].reshape(1,-1,3,3))/.1)**2).mean((-1,-2))+((wrench[...,2]-reference[None,:,12])/5.)**2


class PublicTransportTrainer:
    def __init__(self,policy,response,feedback):
        self.policy,self.response,self.feedback=policy,response,feedback;self.steps=0
        for p in response.parameters():p.requires_grad_(False)
        response.eval();self.optimizer=torch.optim.AdamW(policy.parameters(),lr=policy.spec.learning_rate)

    def objective(self,batch,*,response_weight=None,task_band_reference=None,tolerance=.05):
        normalized=self.policy(batch['target_features'],batch['source_features'],batch['source_commands'],batch['base_proto'],batch['decoder']['phase'],decoder_context(batch['decoder'],self.feedback))
        actual=decode_pose_transport(normalized*normalized.new_tensor(BOUNDS),batch['decoder'],self.feedback)/normalized.new_tensor(BOUNDS)
        predicted=self.response(batch['target_features'],actual)
        with torch.no_grad():source=self.response(batch['source_features'],batch['source_commands'])
        task=task_proxy(predicted,batch['current'],batch['task_reference']).mean(0);matching=((predicted[...,:18]-source[...,:18])**2).mean((0,2))
        peaks=torch.expm1(predicted[...,18:20]).clamp(min=0);risk=((peaks[...,0]/25.-1).clamp(min=0)**2+(peaks[...,1]/10.-1).clamp(min=0)**2).amax(0)
        change=((normalized-batch['base_proto'])**2).mean((1,2));weight=self.policy.spec.response_weight if response_weight is None else response_weight
        if task_band_reference is None:task_term=task
        else:task_term=100.*(task-task_band_reference-tolerance).clamp(min=0)
        loss=(task_term+weight*matching+self.policy.spec.risk_weight*risk+self.policy.spec.change_weight*change).mean()
        return loss,dict(task_proxy=task.mean(),response_matching=matching.mean(),risk_proxy=risk.mean(),command_change=change.mean())

    def step(self,batch,**kwargs):
        self.policy.train();loss,metrics=self.objective(batch,**kwargs);self.optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(self.policy.parameters(),1.);self.optimizer.step();self.steps+=1
        return dict(loss=float(loss.detach().cpu()),**{k:float(v.detach().cpu()) for k,v in metrics.items()})

    def state(self):return dict(checkpoint=self.policy.checkpoint(),optimizer=to_cpu(self.optimizer.state_dict()),steps=self.steps,rng=capture_rng_state())

    def restore(self,state):
        current=self.policy.checkpoint()
        if any(state['checkpoint'][k]!=current[k] for k in ('response_spec','transport_spec','version')):raise ValueError('public transport continuation schema differs')
        self.policy.load_state_dict(state['checkpoint']['state_dict'],strict=True);self.optimizer.load_state_dict(deepcopy(state['optimizer']));self.steps=int(state['steps']);restore_rng_state(state['rng'])
