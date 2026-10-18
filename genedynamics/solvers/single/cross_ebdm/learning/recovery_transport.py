"""Train and validate on the recovery commands that deployment executes.

The recovery controller owns the command while ``recovery_enabled`` is false.
Every learned-policy objective and model-only validation proxy must therefore
replace the candidate by ``common_action`` for those samples.  Keeping the
command construction here gives training and checkpoint selection one semantic
entry point and prevents another P5-style factorial mismatch.
"""
import torch
from .residual_transport import PublicTransportTrainer,decoder_context,decode_pose_transport,task_proxy,BOUNDS


def recovery_mask(batch, *, dtype=None):
    mask = batch['recovery_enabled'].reshape(-1).bool()
    return mask if dtype is None else mask.to(dtype=dtype)


def effective_commands(candidate,batch):
    return torch.where(recovery_mask(batch)[:,None,None],candidate,batch['common_action'][:,None,:].expand_as(candidate))


def recovery_policy_commands(policy,feedback,batch):
    """Return normalized candidates and the commands visible to the model."""
    normalized=policy(batch['target_features'],batch['source_features'],batch['source_commands'],batch['base_proto'],batch['decoder']['phase'],decoder_context(batch['decoder'],feedback))
    decoded=decode_pose_transport(normalized*normalized.new_tensor(BOUNDS),batch['decoder'],feedback)/normalized.new_tensor(BOUNDS)
    return normalized,effective_commands(decoded,batch)


def recovery_task_proxy(policy,response,feedback,batch):
    """Per-sample model task proxy under the deployed recovery command rule."""
    _,commands=recovery_policy_commands(policy,feedback,batch)
    return task_proxy(response(batch['target_features'],commands),batch['current'],batch['task_reference']).mean(0)

class PublicRecoveryTransportTrainer(PublicTransportTrainer):
    def objective(self,batch,*,response_weight=None,task_band_reference=None,tolerance=.05):
        normalized,actual=recovery_policy_commands(self.policy,self.feedback,batch)
        predicted=self.response(batch['target_features'],actual)
        with torch.no_grad():source=self.response(batch['source_features'],batch['source_commands'])
        task=task_proxy(predicted,batch['current'],batch['task_reference']).mean(0);matching=((predicted[...,:18]-source[...,:18])**2).mean((0,2))
        peaks=torch.expm1(predicted[...,18:20]).clamp(min=0);risk=((peaks[...,0]/25.-1).clamp(min=0)**2+(peaks[...,1]/10.-1).clamp(min=0)**2).amax(0)
        change=((normalized-batch['base_proto'])**2).mean((1,2))*recovery_mask(batch,dtype=normalized.dtype);weight=self.policy.spec.response_weight if response_weight is None else response_weight
        if task_band_reference is None:task_term=task
        else:task_term=100.*(task-task_band_reference-tolerance).clamp(min=0)
        loss=(task_term+weight*matching+self.policy.spec.risk_weight*risk+self.policy.spec.change_weight*change).mean()
        return loss,dict(task_proxy=task.mean(),response_matching=matching.mean(),risk_proxy=risk.mean(),command_change=change.mean())

