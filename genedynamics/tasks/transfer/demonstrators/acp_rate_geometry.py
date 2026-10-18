"""Source-preserving rate-limited native ACP teacher with common xArm geometry."""
from copy import deepcopy
import numpy as np
from .acp import RateLimitedACPTeacher
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.bias_feedback import PublicBiasFeedbackSpec
from genedynamics.tasks.transfer.controllers.mating_feedback import PublicMatingFeedbackPolicy

class _CommonGeometry:
    def __init__(self,spec,mode,feedback):
        self.spec=spec;self.base=PublicMatingFeedbackPolicy.load(spec,None,feedback=PublicBiasFeedbackSpec(**feedback),modes=[mode],seed=0)
    def act(self,public):
        action=self.base.act(public)[0].copy()
        # Native VT force intent remains explicit after the shared13
        # admittance reference sets its force-mode bit to zero.
        if self.base.observer.memory.phase[0]==3:action[12]=self.spec.config.desired_force
        return action

class PublicRateGeometryACPTeacher:
    VERSION='public-T3-source-preserving-rate-native19-xarm-geometry-v1'
    def __init__(self,spec,*,mode,feedback,force_recipe):
        if spec.alias!='T3' or spec.config.robot not in {'panda','xarm7'}:raise ValueError('explicit T3 fixedarm teacher')
        self.spec=spec;self.mode=int(mode);self.feedback=deepcopy(feedback);self.recipe=deepcopy(force_recipe);self.reset()
    def reset(self):
        self.native=RateLimitedACPTeacher(self.spec,self.mode,self.recipe)
        if self.spec.config.robot=='xarm7':self.native.teacher=_CommonGeometry(self.spec,self.mode,self.feedback)
        self.tick=-1;self.cached=None
    @property
    def requests(self):return self.native.requests
    @property
    def limiter_receipts(self):return self.native.limiter_receipts
    def get_state(self):return deepcopy(dict(version=self.VERSION,robot=self.spec.config.robot,mode=self.mode,native=vars(self.native),tick=self.tick,cached=self.cached))
    def restore(self,s):
        if (s['version'],s['robot'],s['mode'])!=(self.VERSION,self.spec.config.robot,self.mode):raise ValueError('native teacherbinding changed')
        self.native.__dict__=deepcopy(s['native']);self.tick=s['tick'];self.cached=deepcopy(s['cached'])
    def act(self,public):
        tick=int(np.rint(np.asarray(public['elapsed_seconds']).item()/self.spec.config.control_dt))
        if tick==self.tick:return deepcopy(self.cached)
        if tick!=self.tick+1:raise ValueError('all public ticks required')
        self.cached=self.native.act(public);self.tick=tick;return deepcopy(self.cached)
