"""Explicit T1/T2 local residual in actual shared13D request coordinates.

Task-specific amplitude/issuance is known; this version does not reuse the
T3/T4 pose/admittance decoder or presume its 0.02m/s control chart.
"""
import torch
from .residual_transport import PublicResidualTransport,PublicTransportTrainer,task_proxy,BOUNDS
from .task_response import task_amplitude


class PublicTaskResidualTransport(PublicResidualTransport):
    def forward(self,target_features,source_features,source_commands,base_proto,phase,own_decoder_context):
        if self.spec.residual_scale!=.025:raise ValueError('fixed inherited localradius .025')
        x=torch.cat((target_features,source_features,source_commands.flatten(start_dim=1),base_proto.flatten(start_dim=1),torch.nn.functional.one_hot(phase.long(),4).to(target_features.dtype),own_decoder_context),dim=1)
        amplitude=task_amplitude(target_features)[:,None];residual=torch.tanh(self.network(x)).reshape(-1,self.response_spec.horizon,13);value=base_proto+self.spec.residual_scale*amplitude*residual;value=torch.minimum(torch.maximum(value,-amplitude),amplitude);force=torch.where(base_proto[...,12:13]*25>.1,value[...,12:13].clamp(min=0),torch.zeros_like(value[...,12:13]));return torch.cat((value[...,:12],force),dim=-1)
    def checkpoint(self):return dict(super().checkpoint(),version='public-T1-T2-actual-request-local-transport-v1',source_force_off_threshold_N=.1)
    @classmethod
    def from_checkpoint(cls,payload,device='cpu'):
        if payload.get('version')!='public-T1-T2-actual-request-local-transport-v1' or payload.get('source_force_off_threshold_N')!=.1:raise ValueError('task explicit actualrequesttransport checkpoint required')
        return super().from_checkpoint(dict(payload,version='public-source-conditioned-transport-v1'),device)


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
