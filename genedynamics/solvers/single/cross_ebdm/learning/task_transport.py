"""Explicit T1/T2 local residual in actual shared 13D request coordinates.

T2 transports tangent and orientation twist only.  Its normal contact command,
impedance chart and force target are owned by the shared finite-state task cell.
"""
import torch
from .residual_transport import PublicResidualTransport,PublicTransportTrainer,task_proxy,BOUNDS
from .task_response import task_amplitude


T2_TRANSPORT_COORDINATES=(0,1,3,4,5)
T2_TASK_OWNED_COORDINATES=(2,6,7,8,9,10,11,12)


def project_task_owned_requests(target_features,candidate,base_proto,*,t2_execution_scale=1.):
    """Project T2 candidates onto its task-owned contact-unit affine set."""
    if not 0.<=float(t2_execution_scale)<=1.:
        raise ValueError('T2 execution scale must be in [0,1]')
    obstacle=(target_features[...,-7:-6]>.5)[:,None]
    mask=base_proto.new_zeros(13);mask[list(T2_TRANSPORT_COORDINATES)]=1.
    projected=torch.where(obstacle, candidate,
                          base_proto+float(t2_execution_scale)*(candidate-base_proto)*mask)
    legacy_force=torch.where(base_proto[...,12:13]*25>.1,
                             projected[...,12:13].clamp(min=0),
                             torch.zeros_like(projected[...,12:13]))
    force=torch.where(obstacle,legacy_force,base_proto[...,12:13])
    return torch.cat((projected[...,:12],force),dim=-1)


class PublicTaskResidualTransport(PublicResidualTransport):
    VERSION='public-T1-T2-task-owned-request-transport-v3'
    V2_VERSION='public-T1-T2-task-owned-request-transport-v2'
    LEGACY_VERSION='public-T1-T2-actual-request-local-transport-v1'
    T2_TRANSPORT_COORDINATES=T2_TRANSPORT_COORDINATES
    T2_TASK_OWNED_COORDINATES=T2_TASK_OWNED_COORDINATES

    def __init__(self,*args,legacy_full_request=False,t2_execution_scale=.25,**kwargs):
        super().__init__(*args,**kwargs)
        self.legacy_full_request=bool(legacy_full_request)
        self.t2_execution_scale=float(t2_execution_scale)
        if not 0.<self.t2_execution_scale<=1.:
            raise ValueError('positive T2 execution scale at most one required')

    def forward(self,target_features,source_features,source_commands,base_proto,phase,own_decoder_context):
        if self.spec.residual_scale!=.025:raise ValueError('fixed inherited localradius .025')
        x=torch.cat((target_features,source_features,source_commands.flatten(start_dim=1),base_proto.flatten(start_dim=1),torch.nn.functional.one_hot(phase.long(),4).to(target_features.dtype),own_decoder_context),dim=1)
        amplitude=task_amplitude(target_features)[:,None]
        residual=torch.tanh(self.network(x)).reshape(-1,self.response_spec.horizon,13)
        t2_mask=base_proto.new_zeros(13);t2_mask[list(self.T2_TRANSPORT_COORDINATES)]=1.
        obstacle=(target_features[...,-7:-6]>.5)[:,None]
        mask=torch.where(obstacle,torch.ones_like(t2_mask),t2_mask)
        if self.legacy_full_request:mask=torch.ones_like(mask)
        value=base_proto+self.spec.residual_scale*amplitude*residual*mask
        value=torch.minimum(torch.maximum(value,-amplitude),amplitude)
        if self.legacy_full_request:
            force=torch.where(base_proto[...,12:13]*25>.1,value[...,12:13].clamp(min=0),torch.zeros_like(value[...,12:13]))
            return torch.cat((value[...,:12],force),dim=-1)
        return project_task_owned_requests(target_features,value,base_proto,
                                           t2_execution_scale=self.t2_execution_scale)

    def checkpoint(self):
        payload=super().checkpoint()
        if self.legacy_full_request:
            return dict(payload,version=self.LEGACY_VERSION,source_force_off_threshold_N=.1)
        version=self.V2_VERSION if self.t2_execution_scale==1. else self.VERSION
        return dict(payload,version=version,source_force_off_threshold_N=.1,
                    t2_transport_coordinates=list(self.T2_TRANSPORT_COORDINATES),
                    t2_task_owned_coordinates=list(self.T2_TASK_OWNED_COORDINATES),
                    **({} if version==self.V2_VERSION else {
                        't2_execution_scale':self.t2_execution_scale,
                        't2_normalized_residual_radius':self.spec.residual_scale*self.t2_execution_scale,
                    }))

    @classmethod
    def from_checkpoint(cls,payload,device='cpu'):
        version=payload.get('version')
        if payload.get('source_force_off_threshold_N')!=.1:
            raise ValueError('task explicit actual-request transport checkpoint required')
        if version in {cls.VERSION,cls.V2_VERSION}:
            if (payload.get('t2_transport_coordinates')!=list(cls.T2_TRANSPORT_COORDINATES)
                    or payload.get('t2_task_owned_coordinates')!=list(cls.T2_TASK_OWNED_COORDINATES)):
                raise ValueError('T2 task-owned request projection differs')
            legacy=False
            scale=1. if version==cls.V2_VERSION else payload.get('t2_execution_scale')
            if version==cls.VERSION and (scale!=.25 or payload.get('t2_normalized_residual_radius')!=.00625):
                raise ValueError('T2 execution trust radius differs')
        elif version==cls.LEGACY_VERSION:
            legacy=True;scale=1.
        else:
            raise ValueError('task explicit actual-request transport checkpoint required')
        parent=dict(payload,version='public-source-conditioned-transport-v1')
        model=super().from_checkpoint(parent,device)
        model.legacy_full_request=legacy
        model.t2_execution_scale=float(scale)
        return model


class PublicTaskTransportTrainer(PublicTransportTrainer):
    def objective(self,batch,*,response_weight=None,task_band_reference=None,tolerance=.05):
        normalized=self.policy(batch['target_features'],batch['source_features'],batch['source_commands'],batch['base_proto'],batch['phase'],batch['own_context']);predicted=self.response(batch['target_features'],normalized)
        with torch.no_grad():source=self.response(batch['source_features'],batch['source_commands'])
        task=task_proxy(predicted,batch['current'],batch['task_reference']).mean(0);matching=((predicted[...,:18]-source[...,:18])**2).mean((0,2));peaks=torch.expm1(predicted[...,18:20]).clamp(min=0);risk=((peaks[...,0]/25.-1).clamp(min=0)**2+(peaks[...,1]/10.-1).clamp(min=0)**2).amax(0);change=((normalized-batch['base_proto'])**2).mean((1,2));weight=self.policy.spec.response_weight if response_weight is None else response_weight;task_term=task if task_band_reference is None else 100.*(task-task_band_reference-tolerance).clamp(min=0);loss=(task_term+weight*matching+self.policy.spec.risk_weight*risk+self.policy.spec.change_weight*change).mean()
        return loss,dict(task_proxy=task.mean(),response_matching=matching.mean(),risk_proxy=risk.mean(),command_change=change.mean())


class PublicTaskGatedTransportTrainer(PublicTransportTrainer):
    """Task-explicit T1/T2 frontend for the separately versioned gated loss."""
    def objective(self,batch,*,response_weight=None,task_only_reference=None,
                  gate_spec=None):
        from .gated_transport import PublicAnchoringGateSpec,_gated_objective
        if not isinstance(gate_spec,PublicAnchoringGateSpec):
            raise ValueError('explicit public anchoring-gate spec required')
        normalized=self.policy(
            batch['target_features'],batch['source_features'],batch['source_commands'],
            batch['base_proto'],batch['phase'],batch['own_context'],
        )
        return _gated_objective(
            policy=self.policy,response=self.response,batch=batch,
            normalized=normalized,actual=normalized,response_weight=response_weight,
            task_only_reference=task_only_reference,gate_spec=gate_spec,
        )


class PublicTaskAdditiveBandTransportTrainer(PublicTransportTrainer):
    """Task-explicit ungated C+H+wR control aligned with gated-v2."""
    def objective(self,batch,*,response_weight=None,task_only_reference=None,
                  gate_spec=None):
        from .gated_transport import PublicAnchoringGateSpec,_gated_objective
        if not isinstance(gate_spec,PublicAnchoringGateSpec):
            raise ValueError('explicit public anchoring-gate spec required')
        normalized=self.policy(
            batch['target_features'],batch['source_features'],batch['source_commands'],
            batch['base_proto'],batch['phase'],batch['own_context'],
        )
        return _gated_objective(
            policy=self.policy,response=self.response,batch=batch,
            normalized=normalized,actual=normalized,response_weight=response_weight,
            task_only_reference=task_only_reference,gate_spec=gate_spec,
            apply_anchoring_gate=False,
        )
