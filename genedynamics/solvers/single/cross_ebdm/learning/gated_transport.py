"""Task-first, support-gated response anchoring for a new method version.

The existing Full objective is intentionally left unchanged.  This module
defines a separately versioned training rule with three properties:

* the task objective remains active inside the model task band;
* response matching is applied only to samples that are task compatible and
  lie inside declared public-condition/disagreement thresholds; and
* final checkpoint selection is relative to the matched task-only policy.

The gates use public task-space histories and response-ensemble predictions.
They are model-side filters, not physical safety, reachability, or event-margin
certificates.  Thresholds must be frozen from source/known-nominal development
data before an unknown target cohort is accessed.
"""
from dataclasses import dataclass
import math

import torch

from .recovery_transport import recovery_mask,recovery_policy_commands
from .residual_transport import (
    BOUNDS,PublicTransportTrainer,decode_pose_transport,decoder_context,task_proxy,
)


@dataclass(frozen=True)
class PublicAnchoringGateSpec:
    task_tolerance: float
    maximum_condition_distance: float
    maximum_response_disagreement: float
    band_penalty: float = 100.0

    def __post_init__(self):
        values=(self.task_tolerance,self.maximum_condition_distance,
                self.maximum_response_disagreement,self.band_penalty)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("finite nonnegative anchoring-gate values required")
        if self.band_penalty == 0:
            raise ValueError("positive task-band penalty required")


def _source_current(source_features,response_spec):
    if source_features.shape[-1] != response_spec.feature_dim:
        raise ValueError("public source feature chart differs")
    offset=(response_spec.history_len-1)*(response_spec.frame_dim+1)
    if not bool(torch.all(source_features[...,offset+response_spec.frame_dim] == 1)):
        raise ValueError("source current history frame must be valid")
    return dict(
        position=source_features[...,offset:offset+3]*.5,
        rotation=source_features[...,offset+3:offset+12].reshape(-1,3,3),
        wrench=source_features[...,offset+18:offset+24]
            *source_features.new_tensor([25.]*3+[4.]*3),
    )


def public_condition_distance(batch,response_spec):
    """Normalized current task-space distance to the selected source history."""
    source=_source_current(batch["source_features"],response_spec)
    current=batch["current"]
    position=((current["position"]-source["position"])/.02).square().mean(-1)
    rotation=((current["rotation"]-source["rotation"])/.1).square().mean((-1,-2))
    wrench=((current["wrench"]-source["wrench"])
            /current["wrench"].new_tensor([5.]*3+[1.]*3)).square().mean(-1)
    return ((position+rotation+wrench)/3).sqrt()


def response_disagreement(predicted,source):
    """RMS ensemble disagreement in the normalized 18D public response chart."""
    if predicted.ndim != 3 or source.shape != predicted.shape or predicted.shape[-1] < 18:
        raise ValueError("matching response ensembles must have shape members x batch x outputs")
    target=predicted[...,:18].std(0,unbiased=False).square().mean(-1).sqrt()
    reference=source[...,:18].std(0,unbiased=False).square().mean(-1).sqrt()
    return torch.maximum(target,reference)


def conditional_anchoring_mask(*,task,task_only_reference,condition_distance,
                               disagreement,spec,extra_mask=None):
    """Detached per-sample gate; no gradient crosses a discrete gate decision."""
    shape=task.shape
    if any(value.shape != shape for value in
           (task_only_reference,condition_distance,disagreement)):
        raise ValueError("one task/reference/support/disagreement value per sample required")
    compatible=task.detach() <= task_only_reference.detach()+spec.task_tolerance
    supported=condition_distance.detach() <= spec.maximum_condition_distance
    certain=disagreement.detach() <= spec.maximum_response_disagreement
    mask=compatible & supported & certain
    if extra_mask is not None:
        if extra_mask.reshape(-1).shape != shape:
            raise ValueError("one extra gate value per sample required")
        mask=mask & extra_mask.reshape(-1).bool()
    return mask,dict(task_compatible=compatible,support_accepted=supported,
                     disagreement_accepted=certain)


def select_task_compatible_checkpoint(*,candidate_proxy,task_only_proxy,tolerance,
                                      candidate,task_only):
    """Select the candidate only inside the task-only-relative validation band."""
    values=(candidate_proxy,task_only_proxy,tolerance)
    if any(not math.isfinite(float(value)) for value in values) or tolerance < 0:
        raise ValueError("finite proxies and nonnegative tolerance required")
    accepted=float(candidate_proxy) <= float(task_only_proxy)+float(tolerance)
    return dict(accepted=accepted,selected=candidate if accepted else task_only,
                reference="matched_task_only",
                task_only_relative_gap=float(candidate_proxy)-float(task_only_proxy))


def _gated_objective(*,policy,response,batch,normalized,actual,response_weight,
                     task_only_reference,gate_spec,change_mask=None,
                     apply_anchoring_gate=True):
    if task_only_reference is None:
        raise ValueError("gated anchoring requires a matched task-only reference")
    predicted=response(batch["target_features"],actual)
    with torch.no_grad():
        source=response(batch["source_features"],batch["source_commands"])
    task=task_proxy(predicted,batch["current"],batch["task_reference"]).mean(0)
    matching=((predicted[...,:18]-source[...,:18])**2).mean((0,2))
    disagreement=response_disagreement(predicted,source)
    condition=public_condition_distance(batch,response.spec)
    gated_mask,parts=conditional_anchoring_mask(
        task=task,task_only_reference=task_only_reference,
        condition_distance=condition,disagreement=disagreement,spec=gate_spec,
        extra_mask=change_mask,
    )
    if apply_anchoring_gate:
        mask=gated_mask
    elif change_mask is None:
        mask=torch.ones_like(gated_mask)
    else:
        mask=change_mask.reshape(-1).bool()
    peaks=torch.expm1(predicted[...,18:20]).clamp(min=0)
    risk=((peaks[...,0]/25.-1).clamp(min=0)**2
          +(peaks[...,1]/10.-1).clamp(min=0)**2).amax(0)
    change=((normalized-batch["base_proto"])**2).mean((1,2))
    if change_mask is not None:
        change=change*change_mask.reshape(-1).to(change.dtype)
    weight=policy.spec.response_weight if response_weight is None else response_weight
    violation=(task-task_only_reference-gate_spec.task_tolerance).clamp(min=0)
    # Keep the task objective active throughout the band.  The detached gate
    # only controls the extra response term.
    loss=(task+gate_spec.band_penalty*violation
          +weight*mask.to(matching.dtype)*matching
          +policy.spec.risk_weight*risk+policy.spec.change_weight*change).mean()
    metrics=dict(
        task_proxy=task.mean(),response_matching=matching.mean(),
        gated_response_matching=(mask.to(matching.dtype)*matching).mean(),
        anchoring_fraction=mask.to(task.dtype).mean(),
        task_compatible_fraction=parts["task_compatible"].to(task.dtype).mean(),
        support_accepted_fraction=parts["support_accepted"].to(task.dtype).mean(),
        disagreement_accepted_fraction=parts["disagreement_accepted"].to(task.dtype).mean(),
        condition_distance=condition.mean(),response_disagreement=disagreement.mean(),
        task_band_violation=violation.mean(),risk_proxy=risk.mean(),
        command_change=change.mean(),
    )
    return loss,metrics


class PublicAdditiveBandTransportTrainer(PublicTransportTrainer):
    """Ungated C+H+wR control aligned with the gated-v2 task objective."""
    def objective(self,batch,*,response_weight=None,task_only_reference=None,
                  gate_spec=None):
        if not isinstance(gate_spec,PublicAnchoringGateSpec):
            raise ValueError("explicit public anchoring-gate spec required")
        normalized=self.policy(
            batch["target_features"],batch["source_features"],
            batch["source_commands"],batch["base_proto"],
            batch["decoder"]["phase"],decoder_context(batch["decoder"],self.feedback),
        )
        actual=decode_pose_transport(
            normalized*normalized.new_tensor(BOUNDS),batch["decoder"],self.feedback,
        )/normalized.new_tensor(BOUNDS)
        return _gated_objective(
            policy=self.policy,response=self.response,batch=batch,normalized=normalized,
            actual=actual,response_weight=response_weight,
            task_only_reference=task_only_reference,gate_spec=gate_spec,
            apply_anchoring_gate=False,
        )


class PublicRecoveryAdditiveBandTransportTrainer(PublicTransportTrainer):
    """Recovery-effective ungated C+H+wR control for the aligned ablation."""
    def objective(self,batch,*,response_weight=None,task_only_reference=None,
                  gate_spec=None):
        if not isinstance(gate_spec,PublicAnchoringGateSpec):
            raise ValueError("explicit public anchoring-gate spec required")
        normalized,actual=recovery_policy_commands(self.policy,self.feedback,batch)
        return _gated_objective(
            policy=self.policy,response=self.response,batch=batch,normalized=normalized,
            actual=actual,response_weight=response_weight,
            task_only_reference=task_only_reference,gate_spec=gate_spec,
            change_mask=recovery_mask(batch),apply_anchoring_gate=False,
        )


class PublicGatedTransportTrainer(PublicTransportTrainer):
    def objective(self,batch,*,response_weight=None,task_only_reference=None,
                  gate_spec=None):
        if not isinstance(gate_spec,PublicAnchoringGateSpec):
            raise ValueError("explicit public anchoring-gate spec required")
        normalized=self.policy(
            batch["target_features"],batch["source_features"],
            batch["source_commands"],batch["base_proto"],
            batch["decoder"]["phase"],decoder_context(batch["decoder"],self.feedback),
        )
        actual=decode_pose_transport(
            normalized*normalized.new_tensor(BOUNDS),batch["decoder"],self.feedback,
        )/normalized.new_tensor(BOUNDS)
        return _gated_objective(
            policy=self.policy,response=self.response,batch=batch,normalized=normalized,
            actual=actual,response_weight=response_weight,
            task_only_reference=task_only_reference,gate_spec=gate_spec,
        )


class PublicRecoveryGatedTransportTrainer(PublicTransportTrainer):
    def objective(self,batch,*,response_weight=None,task_only_reference=None,
                  gate_spec=None):
        if not isinstance(gate_spec,PublicAnchoringGateSpec):
            raise ValueError("explicit public anchoring-gate spec required")
        normalized,actual=recovery_policy_commands(self.policy,self.feedback,batch)
        return _gated_objective(
            policy=self.policy,response=self.response,batch=batch,normalized=normalized,
            actual=actual,response_weight=response_weight,
            task_only_reference=task_only_reference,gate_spec=gate_spec,
            change_mask=recovery_mask(batch),
        )

