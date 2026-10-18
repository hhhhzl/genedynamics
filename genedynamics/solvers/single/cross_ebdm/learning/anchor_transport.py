"""Absolute task-observable anchors alongside public response increments.

New objective version only. Shared world/task coordinates are required for
these pose anchors; they are not joint/body correspondences or a safety bound.
The zero-anchor branch retains the original task-only objective exactly.
"""
import torch
from .response import DELTA_SCALE
from .residual_transport import PublicTransportTrainer,BOUNDS,decode_pose_transport,decoder_context,task_proxy


def source_current_from_features(features,spec):
    if features.shape[-1]!=spec.feature_dim:raise ValueError('public source feature chart differs')
    offset=(spec.history_len-1)*(spec.frame_dim+1)
    if not bool(torch.all(features[...,offset+spec.frame_dim]==1)):raise ValueError('source current frame must be valid')
    return dict(position=features[...,offset:offset+3]*.5,
        rotation=features[...,offset+3:offset+12].reshape(-1,3,3),
        wrench=features[...,offset+18:offset+24]*features.new_tensor([25.]*3+[4.]*3))


def future_observables(predicted,current):
    delta=predicted[...,:18]*predicted.new_tensor(DELTA_SCALE);omega=delta[...,3:6];zero=torch.zeros_like(omega[...,0]);x,y,z=omega.unbind(-1)
    skew=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),dim=-1).reshape(*omega.shape[:-1],3,3)
    return dict(position=current['position'][None]+delta[...,:3],rotation=torch.matrix_exp(skew)@current['rotation'][None],wrench=current['wrench'][None]+delta[...,12:18])


def absolute_anchor_error(predicted,source,current,source_current):
    target=future_observables(predicted,current);reference=future_observables(source,source_current)
    position=((target['position']-reference['position'])/.01)**2
    rotation=((target['rotation']-reference['rotation'])/.1)**2
    wrench=((target['wrench']-reference['wrench'])/predicted.new_tensor([5.]*3+[1.]*3))**2
    return (position.sum(-1)+rotation.sum((-1,-2))+wrench.sum(-1)).mean(0)/18.


class PublicAnchorTransportTrainer(PublicTransportTrainer):
    """Same networks/two model forwards/update; one declared extra energy."""
    def objective(self,batch,*,response_weight=None,anchor_weight=1.,task_band_reference=None,tolerance=.05):
        if anchor_weight not in (0.,1.):raise ValueError('fixed anchor weights0/1 only')
        normalized=self.policy(batch['target_features'],batch['source_features'],batch['source_commands'],batch['base_proto'],batch['decoder']['phase'],decoder_context(batch['decoder'],self.feedback))
        actual=decode_pose_transport(normalized*normalized.new_tensor(BOUNDS),batch['decoder'],self.feedback)/normalized.new_tensor(BOUNDS)
        predicted=self.response(batch['target_features'],actual)
        with torch.no_grad():source=self.response(batch['source_features'],batch['source_commands'])
        task=task_proxy(predicted,batch['current'],batch['task_reference']).mean(0);matching=((predicted[...,:18]-source[...,:18])**2).mean((0,2))
        anchor=absolute_anchor_error(predicted,source,batch['current'],source_current_from_features(batch['source_features'],self.response.spec))
        peaks=torch.expm1(predicted[...,18:20]).clamp(min=0);risk=((peaks[...,0]/25.-1).clamp(min=0)**2+(peaks[...,1]/10.-1).clamp(min=0)**2).amax(0)
        change=((normalized-batch['base_proto'])**2).mean((1,2));weight=self.policy.spec.response_weight if response_weight is None else response_weight
        task_term=task if task_band_reference is None else 100.*(task-task_band_reference-tolerance).clamp(min=0)
        loss=(task_term+weight*(matching+anchor_weight*anchor)+self.policy.spec.risk_weight*risk+self.policy.spec.change_weight*change).mean()
        return loss,dict(task_proxy=task.mean(),response_matching=matching.mean(),absolute_anchor_matching=anchor.mean(),risk_proxy=risk.mean(),command_change=change.mean())
