"""Observed-history response ensemble; no simulator carry or private labels.

Only recorded public arrays and known task/body identities enter the encoder.
The model predicts an H-command public response, including control-rate wrench
peaks. Those peaks do not certify the physical evaluator's substep limits.
"""
from dataclasses import asdict,dataclass
from copy import deepcopy
import numpy as np
import torch
from torch import nn

from genedynamics.tasks.transfer.episode import TaskSpec
from genedynamics.learning.checkpoints.torch import capture_rng_state,restore_rng_state,to_cpu
from genedynamics.learning.datasets.transfer.chart import portable_observation

PUBLIC_FIELDS=frozenset(('joint_position','joint_velocity','tcp_position','tcp_rotation','tcp_twist','measured_wrench','goal_position','goal_rotation','previous_action','previous_request','reference_position','reference_rotation','elapsed_seconds','body_position','body_twist','controller_feedback'))
BODIES=('panda','xarm7','ur5e','go2_z1')
TASKS=('T1','T2','T3','T4')
DELTA_SCALE=np.asarray([.01]*3+[.1]*3+[.1]*6+[5.]*3+[1.]*3,dtype=np.float32)


@dataclass(frozen=True)
class PublicResponseSpec:
    history_len:int=4
    horizon:int=10
    maximum_joints:int=32
    hidden:tuple=(128,128)
    members:int=3
    learning_rate:float=3e-4

    def __post_init__(self):
        if min(self.history_len,self.horizon,self.maximum_joints,self.members)<1 or not np.isfinite(self.learning_rate) or self.learning_rate<=0:raise ValueError('positive finite public response configuration required')

    @property
    def frame_dim(self):return 86+4*self.maximum_joints
    @property
    def feature_dim(self):return self.history_len*(self.frame_dim+1)+len(BODIES)+len(TASKS)+3


def public_frame(public,body,spec):
    if set(public)!=PUBLIC_FIELDS:raise ValueError('public response requires exactly the declared sensor fields; no private/future/physics fields')
    if body not in BODIES:raise ValueError('unqualified body identity')
    obs=portable_observation(public).copy();n=len(obs)
    # Fixed SI scales, independent of target data and held-out records.
    obs[:,:3]/=.5;obs[:,12:15]/=.08;obs[:,15:18]/=.4;obs[:,18:21]/=25.;obs[:,21:24]/=4.
    joints=[]
    for name,denominator,free in (('joint_position',np.pi,7),('joint_velocity',3.,6)):
        a=np.asarray(public[name],dtype=np.float32).reshape(n,-1);size=a.shape[1]
        if size>spec.maximum_joints:raise ValueError('joint chart exceeds declared padding; new schema required')
        scale=np.full(size,denominator,np.float32)
        if body=='go2_z1':scale[:free]=1.
        value=np.zeros((n,spec.maximum_joints),np.float32);mask=np.zeros_like(value);value[:,:size]=a/scale;mask[:,:size]=1.
        joints.extend((value,mask))
    reference=np.concatenate((np.asarray(public['reference_position'])/.5,np.asarray(public['reference_rotation']).reshape(n,9)),axis=1)
    goal=np.concatenate((np.asarray(public['goal_position'])/.5,np.asarray(public['goal_rotation']).reshape(n,9)),axis=1)
    bounds=np.asarray([.08]*3+[.4]*3+[1.]*6+[25.],np.float32)
    actions=np.concatenate((np.asarray(public['previous_action'])/bounds,np.asarray(public['previous_request'])/bounds),axis=1)
    return np.concatenate((obs,reference,goal,np.asarray(public['elapsed_seconds']).reshape(n,1)/20.,actions,*joints),axis=1).astype(np.float32)


def features(frames,indices,body,task,calibration_context,spec):
    indices=np.asarray(indices,dtype=np.int64).reshape(-1)
    if task not in TASKS or body not in BODIES or np.any(indices<0) or np.any(indices>=len(frames)):raise ValueError('public feature indices/body/task invalid')
    context=np.asarray(calibration_context,dtype=np.float32)
    if context.shape!=(3,) or not np.isfinite(context).all():raise ValueError('three finite fitted/public context values required; construction truth is not an input')
    lag=indices[:,None]-np.arange(spec.history_len-1,-1,-1)[None,:];valid=lag>=0
    history=frames[np.maximum(lag,0)]*valid[...,None]
    return np.concatenate((np.concatenate((history,valid[...,None]),axis=-1).reshape(len(indices),-1),np.tile(np.eye(len(BODIES))[BODIES.index(body)],(len(indices),1)),
        np.tile(np.eye(len(TASKS))[TASKS.index(task)],(len(indices),1)),np.tile(context,(len(indices),1))),axis=1).astype(np.float32)


def rotation_log(rotation):
    r=np.asarray(rotation,dtype=np.float64);v=np.stack((r[...,2,1]-r[...,1,2],r[...,0,2]-r[...,2,0],r[...,1,0]-r[...,0,1]),axis=-1)*.5
    sine=np.linalg.norm(v,axis=-1);cosine=np.clip((np.trace(r,axis1=-2,axis2=-1)-1)*.5,-1.,1.);angle=np.arctan2(sine,cosine)
    if np.any((sine<1e-8)&(cosine<0)):raise ValueError('near-pi response requires a separately qualified rotation chart')
    return (v*np.where(sine>1e-8,angle/np.maximum(sine,1e-30),1+angle**2/6)[...,None]).astype(np.float32)


def response_windows(public,requests,*,body,task,calibration_context,spec):
    requests=np.asarray(requests,dtype=np.float32);n=len(requests)
    if requests.shape!=(n,13) or any(len(v)!=n+1 for v in public.values()):raise ValueError('one complete public transition per requested command required')
    f=public_frame(public,body,spec);t=np.arange(max(0,n-spec.horizon+1));future=t+spec.horizon
    rotation=np.asarray(public['tcp_rotation']);delta_rotation=rotation_log(rotation[future]@np.swapaxes(rotation[t],-1,-2))
    delta=np.concatenate((np.asarray(public['tcp_position'])[future]-np.asarray(public['tcp_position'])[t],delta_rotation,
        np.asarray(public['tcp_twist'])[future]-np.asarray(public['tcp_twist'])[t],np.asarray(public['measured_wrench'])[future]-np.asarray(public['measured_wrench'])[t]),axis=1)
    control_wrench=np.asarray(public['measured_wrench'])[t[:,None]+np.arange(1,spec.horizon+1)[None,:],:3]
    peaks=np.stack((np.linalg.norm(control_wrench,axis=-1).max(axis=1),np.linalg.norm(control_wrench[...,:2],axis=-1).max(axis=1)),axis=1) if len(t) else np.empty((0,2))
    bounds=np.asarray([.08]*3+[.4]*3+[1.]*6+[25.],np.float32)
    chunks=requests[t[:,None]+np.arange(spec.horizon)[None,:]]/bounds
    return dict(features=features(f,t,body,task,calibration_context,spec),commands=chunks.astype(np.float32),targets=np.concatenate((delta/DELTA_SCALE,np.log1p(peaks)),axis=1).astype(np.float32),ticks=t)


class PublicResponseEnsemble(nn.Module):
    def __init__(self,spec=None):
        super().__init__();self.spec=spec or PublicResponseSpec();self.members=nn.ModuleList()
        for _ in range(self.spec.members):
            sizes=(self.spec.feature_dim+13*self.spec.horizon,*self.spec.hidden,20);layers=[]
            for i,(a,b) in enumerate(zip(sizes[:-1],sizes[1:])):
                layers.append(nn.Linear(a,b))
                if i<len(sizes)-2:layers.append(nn.SiLU())
            self.members.append(nn.Sequential(*layers))

    def forward(self,features,commands):
        if features.shape[-1]!=self.spec.feature_dim or commands.shape[-2:]!=(self.spec.horizon,13):raise ValueError('public response input schema mismatch')
        x=torch.cat((features,commands.flatten(start_dim=-2)),dim=-1)
        return torch.stack([m(x) for m in self.members])

    def checkpoint(self):return dict(version='public-observed-response-v1',spec=asdict(self.spec),state_dict=to_cpu(self.state_dict()),private_state=False,output_semantics='normalized18D public pose/twist/wrench increments and log1p control-rate force peaks; not substep safety')

    @classmethod
    def from_checkpoint(cls,payload,device='cpu'):
        if payload.get('version')!='public-observed-response-v1' or payload.get('private_state') is not False:raise ValueError('public response checkpoint required')
        s=dict(payload['spec']);s['hidden']=tuple(s['hidden']);model=cls(PublicResponseSpec(**s)).to(device);model.load_state_dict(payload['state_dict'],strict=True);return model


class PublicResponseTrainer:
    def __init__(self,model):
        self.model=model;self.optimizer=torch.optim.AdamW(model.parameters(),lr=model.spec.learning_rate);self.steps=0

    def step(self,batch):
        self.model.train();predicted=self.model(batch['features'],batch['commands']);truth=batch['targets'];losses=[]
        for member in predicted:
            ids=torch.randint(len(truth),(len(truth),),device=truth.device);losses.append(torch.nn.functional.mse_loss(member[ids],truth[ids]))
        loss=torch.stack(losses).mean();self.optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.);self.optimizer.step();self.steps+=1
        return float(loss.detach().cpu())

    def state(self):return dict(checkpoint=self.model.checkpoint(),optimizer=to_cpu(self.optimizer.state_dict()),steps=self.steps,rng=capture_rng_state())

    def restore(self,state):
        if state['checkpoint']['spec']!=asdict(self.model.spec):raise ValueError('public response continuation schema differs')
        self.model.load_state_dict(state['checkpoint']['state_dict'],strict=True);self.optimizer.load_state_dict(deepcopy(state['optimizer']));self.steps=int(state['steps']);restore_rng_state(state['rng'])
