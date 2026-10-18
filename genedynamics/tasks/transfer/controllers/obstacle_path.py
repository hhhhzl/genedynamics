"""Ordinary knownbody waypoint search binding, with unchanged T1 control law."""
from copy import deepcopy
import numpy as np
from .obstacle_feedback import PublicObstacleFeedbackPolicy


class PublicObstaclePathPolicy(PublicObstacleFeedbackPolicy):
    VERSION='public-obstacle-knownbody-path-v1'
    def __init__(self,spec,*,modes,path):
        self.path=np.asarray(path,np.float64).copy()
        if self.path.shape!=(4,) or not np.isfinite(self.path).all():raise ValueError('x2/z2/x3/z3 public path parameters')
        super().__init__(spec,modes=modes)
        if spec.config.robot=='xarm7':self.routes[0,1,[0,2]]=self.path[:2];self.routes[0,2,[0,2]]=self.path[2:]
    def get_state(self):return dict(super().get_state(),path=self.path.copy())
    def restore(self,state):
        if not np.array_equal(state['path'],self.path):raise ValueError('planned route differs')
        super().restore({k:v for k,v in state.items() if k!='path'})
