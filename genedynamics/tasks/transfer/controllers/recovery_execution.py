"""Shared, public contact/withdrawal prefix for arm-skill comparisons.

The fixed task classifies recovery mode before first12mm withdrawal. This gate
uses only control-rate wrench/position and a conservative public upper bound
on first contact height. It supplies no evaluator contact bit or plant state.
"""
from copy import deepcopy
import numpy as np
from .mating_feedback import PublicLivenessMatingFeedbackPolicy, PublicMatingFeedbackPolicy
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.bias_feedback import PublicBiasFeedbackPolicy
class PublicRecoveryGate:
    VERSION='public-contact-withdrawal-prefix-v1'
    def __init__(self):self.reset()
    def reset(self):
        self.tick=-1;self.contact_seen=False;self.maximum_precontact_z=-np.inf;self.enabled=False
    def get_state(self):return deepcopy(vars(self))
    def restore(self,state):self.__dict__.update(deepcopy(state))
    def observe(self,public):
        tick=int(np.rint(np.asarray(public['elapsed_seconds']).item()/.02))
        if tick==self.tick:return self.enabled
        if tick!=self.tick+1:raise ValueError('every public recovery tick required')
        z=float(np.asarray(public['tcp_position'])[0,2]);fz=float(np.asarray(public['measured_wrench'])[0,2])
        if not self.contact_seen:
            self.maximum_precontact_z=max(self.maximum_precontact_z,z)
            if tick>=1 and fz>.3001:self.contact_seen=True
        if self.contact_seen and z>self.maximum_precontact_z+.012+.0002:self.enabled=True
        self.tick=tick;return self.enabled
class PublicRecoveryExecution:
    VERSION='public-common-recovery-execution-v1'
    COMMON_T4_POLICY=PublicMatingFeedbackPolicy
    def __init__(self,candidate,spec,feedback,*,mode,seed):
        if spec.alias not in {'T3','T4'}:raise ValueError('explicit insertion/recovery binding')
        self.candidate,self.spec,self.mode,self.seed=candidate,spec,int(mode),int(seed)
        cls=self.COMMON_T4_POLICY if spec.alias=='T4' else PublicBiasFeedbackPolicy
        self.common=cls.load(spec,None,feedback=feedback,modes=[mode],seed=seed);self.gate=PublicRecoveryGate();self.common_controls=0;self.learned_controls=0
    def __getattr__(self,name):return getattr(self.candidate,name)
    def get_state(self):return deepcopy(dict(candidate=self.candidate.get_state(),common=self.common.get_state(),gate=self.gate.get_state(),common_controls=self.common_controls,learned_controls=self.learned_controls))
    def restore(self,state):
        self.candidate.restore(state['candidate']);self.common.restore(state['common']);self.gate.restore(state['gate']);self.common_controls=state['common_controls'];self.learned_controls=state['learned_controls']
    def reset(self):
        self.candidate.reset();self.common.reset(modes=[self.mode],seed=self.seed);self.gate.reset();self.common_controls=self.learned_controls=0
    def act(self,public):
        enabled=self.gate.observe(public);common=self.common.act(public);candidate=self.candidate.act(public)
        if enabled:self.learned_controls+=1;return candidate
        self.common_controls+=1;return common


class PublicLivenessRecoveryExecution(PublicRecoveryExecution):
    """Versioned common prefix using the same P5 release as every candidate."""

    VERSION='public-common-recovery-execution-safe-settled-v1'
    COMMON_T4_POLICY=PublicLivenessMatingFeedbackPolicy
