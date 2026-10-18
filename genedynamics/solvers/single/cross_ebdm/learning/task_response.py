"""Task-explicit known issued-command response for T1 and T2 public features."""
from dataclasses import asdict
import torch
from .response import PublicResponseEnsemble,PublicResponseSpec
from .issued_response import project_contact_requests
from genedynamics.learning.checkpoints.torch import to_cpu


def task_amplitude(features):
    obstacle=features[...,-7:-6]>.5
    common=features.new_tensor([1.]*12+[1.]);arm=features.new_tensor([1.5]*6+[1.]*6+[0.])
    return torch.where(obstacle,arm,common)


def project_task_requests(features,commands,spec):
    task=features[...,-7:-3]
    if not bool(torch.all((task[...,:2].sum(-1)==1)&(task[...,2:].sum(-1)==0))):raise ValueError('explicit T1/T2 public chart required')
    offset=(spec.history_len-1)*(spec.frame_dim+1)+60;previous=features[...,offset:offset+13];contact=project_contact_requests(commands,previous);amplitude=task_amplitude(features);rates=commands.new_tensor([.6/.08]*3+[3./.4]*3+[4.]*6+[0.])*.02;value=previous;rows=[]
    for i in range(commands.shape[-2]):
        bounded=torch.minimum(torch.maximum(commands[...,i,:],-amplitude),amplitude);value=value+torch.minimum(torch.maximum(bounded-value,-rates),rates);value=torch.cat((value[...,:12],torch.zeros_like(value[...,12:13])),dim=-1);rows.append(value)
    obstacle=torch.stack(rows,dim=-2);issued=torch.where((task[...,:1]>.5)[...,None],obstacle,contact);enabled=(commands[...,12:13]>(.1/25.)).to(commands.dtype)*(task[...,1:2]>.5)[...,None]
    return torch.cat((issued[...,:12],issued[...,12:13]+2*enabled),dim=-1)


class PublicTaskIssuedResponseEnsemble(PublicResponseEnsemble):
    def forward(self,features,commands):return super().forward(features,project_task_requests(features,commands,self.spec))
    def checkpoint(self):return dict(version='public-issued-T1-T2-response-v1',spec=asdict(self.spec),state_dict=to_cpu(self.state_dict()),private_state=False,command_chart='features remaincommonSI/.08/.4; T1Box.12/.6,acceleration.6/3,force0;T2originalcontactBox/limiter/enablebit. Explicittaskonehot, nohidden delayqueue',output_semantics='20Dcontrol-rate publicobservable response, notsubstep safety')
    @classmethod
    def from_checkpoint(cls,payload,device='cpu'):
        if payload.get('version')!='public-issued-T1-T2-response-v1' or payload.get('private_state') is not False:raise ValueError('explicit T1/T2 publicresponse required')
        s=dict(payload['spec']);s['hidden']=tuple(s['hidden']);model=cls(PublicResponseSpec(**s)).to(device);model.load_state_dict(payload['state_dict'],strict=True);return model
