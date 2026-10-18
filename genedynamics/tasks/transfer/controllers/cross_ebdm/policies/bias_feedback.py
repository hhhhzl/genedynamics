"""Continuous public pose-error compensation across alignment→insertion.

Source behavior memory keeps its original phase resets. This separate shared
decoder integrator preserves tangential/angular load compensation; the normal
channel stays under measured-force admittance.
"""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from genedynamics.core.control.transfer import orientation_error
from .governor_feedback import PublicGovernorFeedbackPolicy, PublicGovernorFeedbackSpec


@dataclass(frozen=True)
class PublicBiasFeedbackSpec(PublicGovernorFeedbackSpec):
    position_gain: float = 6.
    integral_scale: float = 0.
    lead_xy_m: float = .008
    compensation_gain: float = .8
    compensation_limit: float = .04

    def __post_init__(self):
        super().__post_init__()
        if self.integral_scale!=0 or not np.isfinite([self.compensation_gain,self.compensation_limit]).all() or not (0<self.compensation_gain<=2 and 0<self.compensation_limit<=.08):
            raise ValueError("explicit separate bounded pose compensation required")


class PublicBiasFeedbackPolicy(PublicGovernorFeedbackPolicy):
    def __init__(self,spec,*,feedback=None,**kwargs):
        feedback=feedback or PublicBiasFeedbackSpec(insertion_force_scale=1.,lift_before_lateral=True)
        if not isinstance(feedback,PublicBiasFeedbackSpec):raise TypeError("bias-preserving decoder specification required")
        super().__init__(spec,feedback=feedback,**kwargs)

    def reset(self,**kwargs):
        super().reset(**kwargs);n=len(self._reset_modes)
        self.compensation=np.zeros((n,6));self.compensation_tick=np.full(n,-1,np.int64)

    def get_state(self):
        value=super().get_state();value["pose_compensation"]=deepcopy(dict(value=self.compensation,tick=self.compensation_tick))
        return value

    def restore(self,state):
        value=deepcopy(state);compensation=value.pop("pose_compensation");super().restore(value)
        self.compensation=compensation["value"];self.compensation_tick=compensation["tick"]

    def reorder(self,indices):
        super().reorder(indices);self.compensation=self.compensation[indices].copy();self.compensation_tick=self.compensation_tick[indices].copy()

    def act(self,public):
        command=super().act(public).copy();fb=self.pose_feedback;phase=self.observer.memory.phase;dt=self.spec.config.control_dt
        p=np.asarray(public["tcp_position"]);r=np.asarray(public["tcp_rotation"]);reference=self.reference
        c,s=np.cos(reference[:,3]),np.sin(reference[:,3]);desired=np.zeros_like(r)
        desired[:,0,0]=c;desired[:,0,1]=s;desired[:,1,0]=s;desired[:,1,1]=-c;desired[:,2,2]=-1
        angular=np.stack([orientation_error(np,a,b) for a,b in zip(r,desired)])
        error=np.concatenate((reference[:,:3]-p,angular),axis=-1);fresh=self.compensation_tick!=self.observer.last_tick
        self.compensation=np.where((fresh&(phase>=2))[:,None],np.clip(self.compensation+fb.compensation_gain*dt*error,-fb.compensation_limit,fb.compensation_limit),self.compensation)
        self.compensation=np.where((phase<2)[:,None],0.,self.compensation)
        self.compensation[:,2]=np.where(phase==3,0.,self.compensation[:,2])
        command[:,:6]+=np.where((phase>=2)[:,None],self.compensation,0.)
        lead=np.asarray([fb.lead_xy_m,fb.lead_xy_m,fb.lead_z_m]);rp=np.asarray(public["reference_position"])
        command[:,:3]=np.where((phase>=2)[:,None],np.clip(command[:,:3],(p-lead-rp)/dt,(p+lead-rp)/dt),command[:,:3])
        rotation_lead=np.stack([orientation_error(np,a,b) for a,b in zip(r,public["reference_rotation"])])
        command[:,3:6]=np.where((phase>=2)[:,None],np.clip(command[:,3:6],(-fb.lead_rotation_rad-rotation_lead)/dt,(fb.lead_rotation_rad-rotation_lead)/dt),command[:,3:6])
        command[:,:6]=np.clip(command[:,:6],[-.02]*3+[-.12]*3,[.02]*3+[.12]*3)
        command[:,2]=np.where(phase==3,np.clip(command[:,2],-fb.normal_velocity_limit,fb.normal_velocity_limit),command[:,2])
        self.compensation_tick=self.observer.last_tick.copy()
        return command.astype(np.float32)
