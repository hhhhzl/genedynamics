"""Contact response with the known public command limiter modeled explicitly.

The previous issued command is observable before the unknown actuator delay.
Projecting requests through this known limiter does not reveal that delay or
copy plant/controller carry. Private delayed queues remain outside the model.
"""
from dataclasses import asdict
import torch

from .response import PublicResponseEnsemble,PublicResponseSpec
from genedynamics.learning.checkpoints.torch import to_cpu


def project_contact_requests(commands,previous,*,force_rate=10.,control_dt=.02):
    """Normalized13D request -> known issued chart, for the contact Box."""
    rates=commands.new_tensor([.4/.08]*3+[2./.4]*3+[4.]*6+[force_rate/25.])*control_dt
    value=previous;rows=[]
    for i in range(commands.shape[-2]):
        bounded=torch.cat((commands[...,i,:12].clamp(-1,1),commands[...,i,12:13].clamp(0,1)),dim=-1)
        value=value+torch.maximum(torch.minimum(bounded-value,rates),-rates)
        value=torch.cat((value[...,:12],value[...,12:13].clamp(min=0)),dim=-1);rows.append(value)
    return torch.stack(rows,dim=-2)


class PublicIssuedResponseEnsemble(PublicResponseEnsemble):
    def forward(self,features,commands):
        offset=(self.spec.history_len-1)*(self.spec.frame_dim+1)+60
        previous=features[...,offset:offset+13]
        issued=project_contact_requests(commands,previous)
        # The physical termination/loading law also reads raw force-enable
        # (>0.1N). Preserve that public bit: issued alone is not a sufficient
        # action chart near force-off. Disjoint ranges pack the pair without
        # changing the model's parameter count or losing issued-force units.
        enabled=(commands[...,12:13]>(.1/25.)).to(commands.dtype)
        chart=torch.cat((issued[...,:12],issued[...,12:13]+2*enabled),dim=-1)
        return super().forward(features,chart)

    def checkpoint(self):
        return dict(version='public-issued-contact-response-v1',spec=asdict(self.spec),state_dict=to_cpu(self.state_dict()),private_state=False,
            command_chart='known issued13D from public previous_action; force feature=issued_force/25+2*raw_force_enable(>0.1N); unknown actuator delay not read',
            output_semantics='same20D public response as raw-request model; contact20ms and declared force_rate10; no substep safety guarantee')

    @classmethod
    def from_checkpoint(cls,payload,device='cpu'):
        if payload.get('version')!='public-issued-contact-response-v1' or payload.get('private_state') is not False:raise ValueError('issued-contact public response checkpoint required')
        s=dict(payload['spec']);s['hidden']=tuple(s['hidden']);model=cls(PublicResponseSpec(**s)).to(device);model.load_state_dict(payload['state_dict'],strict=True);return model
