"""Public T3/T4 mating feedback binding; physical task identity stays explicit."""
from dataclasses import replace
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.bias_feedback import PublicBiasFeedbackPolicy
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.liveness import SafeSettledRetreatReleaseMixin


class PublicMatingFeedbackPolicy(PublicBiasFeedbackPolicy):
    """Reuse the qualified public recovery geometry on the actual T4 body.

    Only the observer template uses T3's recovery clock. This class neither
    creates a simulator nor accesses WBC/actuator carry or axial-support truth.
    T4 physical reachability and body stability require separate qualification.
    """
    def __init__(self,spec,**kwargs):
        if spec.alias not in {'T3','T4'} or not spec.config.insertion:
            raise ValueError('matched mating T3/T4 public feedback binding required')
        self.physical_task_spec=spec
        super().__init__(replace(spec,alias='T3'),**kwargs)

    def reset(self,**kwargs):
        self.spec=replace(self.physical_task_spec,alias='T3')
        try:super().reset(**kwargs)
        finally:self.spec=self.physical_task_spec


class PublicLivenessMatingFeedbackPolicy(SafeSettledRetreatReleaseMixin,PublicMatingFeedbackPolicy):
    """P5 T3/T4 binding with the safe-settled retreat release."""

    VERSION='public-mating-feedback-safe-settled-v1'
