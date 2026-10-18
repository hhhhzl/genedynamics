"""Full public current-joint context for the native task-space FADA adapter.

The common predicted future stays physical35D: no source-to-target joint
correspondence or future joint targets are asserted. Current encoders, padding
masks and known body identity enter configurable native command conditioning.
"""
from copy import deepcopy
import numpy as np
import torch
from genedynamics.tasks.transfer.guide_observation import public_memory
from genedynamics.solvers.single.cross_ebdm.learning.response import PUBLIC_FIELDS,BODIES
from genedynamics.solvers.single.fada.learning.source_training import bind_source_action
from genedynamics.solvers.single.fada import FADASolver
from genedynamics.tasks.transfer.controllers.obstacle_path import PublicObstaclePathPolicy
from .base import PublicFADATaskActor, PublicTaskHistory
from .mobile import anchor_public_body
from .obstacle import PublicObstacleBodyMemory

VERSION='public-current-joint-context199-v1'
JOINT_CONTEXT_DIM=132


def joint_context(public,body):
    if set(public)!=PUBLIC_FIELDS or body not in BODIES:raise ValueError('exact public sensor schema andknownbody required')
    n=len(public['tcp_position']);parts=[]
    for name,scale,free in [('joint_position',np.pi,7),('joint_velocity',3.,6)]:
        a=np.asarray(public[name],np.float32).reshape(n,-1)
        if a.shape[1]>32 or not np.isfinite(a).all():raise ValueError('finitejointchart atmost32 coordinates')
        denominator=np.full(a.shape[1],scale,np.float32)
        if body=='go2_z1':denominator[:free]=1.
        values=np.zeros((n,32),np.float32);mask=np.zeros_like(values);values[:,:a.shape[1]]=a/denominator;mask[:,:a.shape[1]]=1.;parts.extend([values,mask])
    return np.concatenate((*parts,np.tile(np.eye(4,dtype=np.float32)[BODIES.index(body)],(n,1))),axis=-1)


def lift_joint_checkpoint(payload):
    if payload['cmd_dim']!=67 or payload['obs_dim']!=35:raise ValueError('qualified35D/67D task source required')
    result=deepcopy(payload);result['cmd_dim']=result['cfg']['cmd_dim']=67+JOINT_CONTEXT_DIM;weights=result['model_state_dict']
    # Native source disables IDM command conditioning. This explicit task
    # adaptation enables it; old command columns start zero to retain its
    # previous function, while current joint columns have native initialization.
    if not result['cfg'].get('idm_use_current_command_for_history',False):
        result['cfg']['idm_use_current_command_for_history']=True
        width=result['cfg']['idm_d_model']
        weights['idm.history_cmd_embed.weight']=torch.zeros((width,67))
        weights['idm.history_cmd_embed.bias']=torch.zeros(width)
    for key,value in list(weights.items()):
        if not key.endswith('cmd_embed.weight'):continue
        if key.startswith('planner.'):extra=torch.zeros((value.shape[0],JOINT_CONTEXT_DIM),dtype=value.dtype)
        else:extra=torch.nn.Linear(result['cmd_dim'],value.shape[0]).weight.detach()[:,67:].clone().to(value.dtype)
        weights[key]=torch.cat((value,extra),dim=-1)
    result['planner_state_dict']={k[8:]:v for k,v in weights.items() if k.startswith('planner.')};result['idm_state_dict']={k[4:]:v for k,v in weights.items() if k.startswith('idm.')}
    for key,fill in [('command_mean',0.),('command_std',1.)]:result['command_norm_stats'][key]=torch.cat((result['command_norm_stats'][key],torch.full((JOINT_CONTEXT_DIM,),fill,dtype=result['command_norm_stats'][key].dtype)))
    result['genedynamics_task_adapter'].update(public_joint_context_version=VERSION,source_planner_joint_columns_initially_zero=True,idm_current_command_conditioning_enabled=True,requires_new_source_qualification=True,joint_context='currentq/qdot withfixedSIscales,paddingmasks,knownbodyID; nofuturejointlabels/mapping, no privatecarry')
    return result


def joint_memory(spec,modes,metadata):
    if spec.alias=='T1' and spec.config.robot=='xarm7':
        observer=PublicObstacleBodyMemory(spec,modes)
        if metadata.get('public_known_body_path') is not None:observer.base=PublicObstaclePathPolicy(spec,modes=modes,path=metadata['public_known_body_path'])
        return observer
    return public_memory(spec,modes)


class JointTaskPolicy:
    def __init__(self,view,payload,device):
        self.view=view;self.metadata=deepcopy(payload['genedynamics_task_adapter']);self.solver=FADASolver.from_checkpoint(payload,device=device,task_action_binding=True);self.decoder=deepcopy(self.metadata['action_decoder']);self.public_memory=None
    def reset(self):self.solver.reset();self.public_memory=None
    def act(self,history):
        public=self.view.observe();modes=np.argmax(history['current_command'][:,12:14],axis=-1)
        if self.public_memory is None:self.public_memory=joint_memory(self.view.spec,modes,self.metadata)
        history=dict(history);history['current_command']=np.concatenate((history['current_command'],self.public_memory.update(public),joint_context(public,self.view.spec.config.robot)),axis=-1)
        return bind_source_action(self.view,self.solver.act(history),self.decoder)


class PublicJointFADATaskActor(PublicFADATaskActor):
    VERSION=VERSION
    def __init__(self,spec,payload,*,mode,device='cpu'):
        if payload['cmd_dim']!=199 or payload['genedynamics_task_adapter'].get('public_joint_context_version')!=VERSION or payload['genedynamics_task_adapter']['task_name']!=spec.name:raise ValueError('explicit full-public jointtask checkpoint required')
        self.spec=spec;self.mode=mode;self.view=PublicTaskHistory(spec,mode,payload['cfg']['history_len']);self.policy=JointTaskPolicy(self.view,payload,device);self.irreversible_network_calls=0;self.reset()
    def reset(self):super().reset();self.body_anchor=None
    def get_state(self):return dict(super().get_state(),body_anchor=deepcopy(self.body_anchor))
    def restore(self,state):super().restore(state);self.body_anchor=deepcopy(state['body_anchor'])
    def act(self,public):
        if self.spec.alias=='T4':
            if self.body_anchor is None:self.body_anchor=np.asarray(public['body_position']).copy()
            public=anchor_public_body(public,self.body_anchor)
        return super().act(public)
