"""Ordinary T1 finite-horizon nominal-model command governor.

Explicit additional planning reference, not original FADA or transport. Every
candidate is simulated on an owned nominal model from public encoders; finite
predictions provide no safety certificate on unknown dynamics.
"""
from copy import deepcopy
import numpy as np
from .obstacle_feedback import PublicObstacleFeedbackPolicy
from .obstacle_model import PublicObstacleNominalModel


class PublicObstacleNominalGovernor:
    VERSION='public-t1-five-step-nominal-governor-v1'
    def __init__(self,spec,*,mode,charge=None):
        self.spec=spec;self.base=PublicObstacleFeedbackPolicy(spec,modes=[mode]);self.model=PublicObstacleNominalModel(spec,charge=charge);self.reset()
    def reset(self):self.base.reset();self.model.reset();self.tick=-1;self.cached=None;self.last_decision=None;self.irreversible_candidates=0
    def get_state(self):return deepcopy(dict(version=self.VERSION,base=self.base.get_state(),model=self.model.get_state(),tick=self.tick,cached=self.cached,last_decision=self.last_decision))
    def restore(self,s):
        if s['version']!=self.VERSION:raise ValueError('governor version differs')
        self.base.restore(s['base']);self.model.restore(s['model']);self.tick=s['tick'];self.cached=deepcopy(s['cached']);self.last_decision=deepcopy(s['last_decision'])
    def act(self,public):
        tick=int(np.rint(np.asarray(public['elapsed_seconds']).item()/self.spec.config.control_dt))
        if tick==self.tick:return self.cached.copy()
        if tick!=self.tick+1:raise ValueError('observe every publictick andreset explicitly')
        raw=self.base.act(public);self.model.observe(public);options=[];selected=None
        for scale in (1.,.75,.5,.25,0.):
            command=raw.copy();command[:,:6]*=scale;traces=self.model.predict(np.repeat(command,5,axis=0));self.irreversible_candidates+=1
            speed=max(t['joint_speed_peak'] for t in traces);clearance=min(t['clearance'] for t in traces);status=traces[-1]['status'];safe=speed<=2.5 and clearance>=-self.spec.config.collision_tolerance and status in (0,1,5)
            candidate=dict(scale=scale,speed_peak=speed,clearance=clearance,status=status,safe=safe,model_controls=len(traces));options.append((candidate,command))
            if safe:selected=(candidate,command);break
        if selected is None:selected=min(options,key=lambda x:(x[0]['status'] in (2,3,6),x[0]['speed_peak'],-x[0]['scale']))
        self.last_decision=dict(chosen=selected[0],candidates=[v[0] for v in options]);self.cached=selected[1].astype(np.float32);self.tick=tick;return self.cached.copy()
