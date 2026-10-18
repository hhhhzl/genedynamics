"""Ordinary known-jammed T4 recovery without optional initial force buildup.

The task already declares a jammed reset. Begin its public withdrawal phase
at tick0 instead of waiting to establish5N first. Original lift/side/align/
insert control and every physical criterion remain unchanged. Panda source
continues exactly the qualified original controller; this is body-specific
common-reference geometry, not learned transport.
"""
import numpy as np
from .mating_feedback import PublicMatingFeedbackPolicy


class PublicMobileImmediateRecoveryPolicy(PublicMatingFeedbackPolicy):
    VERSION='public-T4-known-jam-immediate-withdrawal-v1'
    def act(self,public):
        if self.physical_task_spec.alias=='T4' and np.all(self.observer.last_tick<0) and np.all(np.asarray(public['elapsed_seconds'])==0):
            memory=self.observer.memory;self.observer.memory=memory._replace(phase=np.ones_like(memory.phase),contact_count=np.zeros_like(memory.contact_count))
        return super().act(public)
