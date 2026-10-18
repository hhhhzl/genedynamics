"""T2 free-air order-event route under the unchanged frozen task cost.

The task fixes which axis crosses first; it does not require the first axis
to finish before the second starts. Panda remains exactly the qualified
contact teacher. Only xArm lift-route progression changes, with fixed geometry,
gains, force holding and all physical/mode/deadline predicates unchanged.
"""
from copy import deepcopy
import numpy as np
from .guide_lift_route import PublicGuideLiftRoutePolicy

class PublicGuideEventLiftRoutePolicy(PublicGuideLiftRoutePolicy):
    VERSION='public-T2-knowngeometry-order-event-lift-route-v1'
    def reset(self):
        super().reset();self.mode_anchor=None
    def get_state(self):return dict(super().get_state(),mode_anchor=deepcopy(self.mode_anchor))
    def restore(self,s):
        parent=dict(s);anchor=parent.pop('mode_anchor');super().restore(parent);self.mode_anchor=deepcopy(anchor)
    def act(self,public):
        if self.spec.config.robot=='panda':return super().act(public)
        tick=int(np.rint(np.asarray(public['elapsed_seconds']).item()/self.spec.config.control_dt))
        if tick==self.tick:return super().act(public)
        # The classifier's origin is the first physical trace, not reset pose.
        if self.mode_anchor is None and tick>=1:self.mode_anchor=np.asarray(public['tcp_position'])[:,:2].copy()
        if self.stage==1 and self.mode_anchor is not None:
            axis=0 if self.mode==0 else 1
            # Actual frozen thresholds .015/.008 plus a fixed2mm buffer.
            threshold=.017 if self.mode==0 else .010
            if np.asarray(public['tcp_position'])[0,axis]>self.mode_anchor[0,axis]+threshold:self.stage=2
        return super().act(public)
