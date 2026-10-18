"""Public reference lead bounds and continued insertion-envelope monitoring.

This is a new decoder version. It never rewinds an actuator reference or reads
the plant delay: corrections are issued through the normal bounded interface.
"""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from genedynamics.core.control.transfer import integrate_rotation, orientation_error
from genedynamics.tasks.transfer.behavior import keyway_alignment
from .pose_feedback import PublicPoseFeedbackPolicy, PublicPoseFeedbackSpec


@dataclass(frozen=True)
class PublicGovernorFeedbackSpec(PublicPoseFeedbackSpec):
    position_gain: float = 4.
    rotation_gain: float = 4.
    issued_reference_gain: float = 2.
    integral_scale: float = 1.
    twist_damping: float = 1.
    normal_velocity_limit: float = .015
    lead_xy_m: float = .004
    lead_z_m: float = .010
    lead_rotation_rad: float = .04
    guard_prediction_seconds: float = .08
    guard_settle_ticks: int = 4
    guard_clear_z_m: float = .610
    guard_retreat_speed: float = .006
    guard_seat_gap_m: float = .006

    def __post_init__(self):
        super().__post_init__()
        values=(self.lead_xy_m,self.lead_z_m,self.lead_rotation_rad,self.guard_prediction_seconds,
            self.guard_clear_z_m,self.guard_retreat_speed,self.guard_seat_gap_m)
        if not np.isfinite(values).all() or not (.001<=self.lead_xy_m<=.010 and .006<=self.lead_z_m<=.020
                and .01<=self.lead_rotation_rad<=.1 and 0<=self.guard_prediction_seconds<=.2
                and .609<=self.guard_clear_z_m<=.614 and 0<self.guard_retreat_speed<=.02 and .002<=self.guard_seat_gap_m<=.008):
            raise ValueError("reference governor outside declared public support")
        if isinstance(self.guard_settle_ticks,bool) or not isinstance(self.guard_settle_ticks,int) or not 1<=self.guard_settle_ticks<=10:
            raise ValueError("finite public insertion settling clock required")


class PublicGovernorFeedbackPolicy(PublicPoseFeedbackPolicy):
    def __init__(self,spec,*,feedback=None,**kwargs):
        feedback=feedback or PublicGovernorFeedbackSpec(insertion_force_scale=1.,lift_before_lateral=True)
        if not isinstance(feedback,PublicGovernorFeedbackSpec):raise TypeError("explicit governor specification required")
        super().__init__(spec,feedback=feedback,**kwargs)

    def reset(self,**kwargs):
        super().reset(**kwargs);n=len(self._reset_modes)
        self.retreating=np.zeros(n,bool);self.clear_count=np.zeros(n,np.int32);self.guard_tick=np.full(n,-1,np.int64)

    def get_state(self):
        value=super().get_state();value["governor"]=deepcopy(dict(retreating=self.retreating,clear_count=self.clear_count,guard_tick=self.guard_tick))
        return value

    def restore(self,state):
        value=deepcopy(state);governor=value.pop("governor");super().restore(value)
        self.retreating=governor["retreating"];self.clear_count=governor["clear_count"];self.guard_tick=governor["guard_tick"]

    def reorder(self,indices):
        super().reorder(indices)
        for name in ("retreating","clear_count","guard_tick"):setattr(self,name,getattr(self,name)[indices].copy())

    def _update_retreating(self,phase,p,safe,approaching):
        """Update the frozen v1 height-latched retreat state.

        Kept as a separate hook so later, explicitly versioned controllers can
        change the public release certificate without changing v1 behaviour.
        """
        fb=self.pose_feedback
        self.retreating|=approaching&~safe&(p[:,2]<fb.guard_clear_z_m-.001)
        self.retreating&=p[:,2]<fb.guard_clear_z_m

    def act(self,public):
        command=super().act(public).copy();fb=self.pose_feedback;dt=self.spec.config.control_dt
        p=np.asarray(public["tcp_position"]);r=np.asarray(public["tcp_rotation"]);twist=np.asarray(public["tcp_twist"])
        phase=self.observer.memory.phase;active=phase>=2
        # Clamp the NEXT self-issued pose relative to actual public pose.
        # A lead already outside the box is unwound at the same velocity cap;
        # there is no direct reset of hidden/public actuator carry.
        lead=np.asarray([fb.lead_xy_m,fb.lead_xy_m,fb.lead_z_m])
        rp=np.asarray(public["reference_position"])
        velocity=np.clip(command[:,:3],(p-lead-rp)/dt,(p+lead-rp)/dt)
        rotation_lead=np.stack([orientation_error(np,a,b) for a,b in zip(r,public["reference_rotation"])])
        angular=np.clip(command[:,3:6],(-fb.lead_rotation_rad-rotation_lead)/dt,(fb.lead_rotation_rad-rotation_lead)/dt)
        command[:,:3]=np.where(active[:,None],velocity,command[:,:3]);command[:,3:6]=np.where(active[:,None],angular,command[:,3:6])
        predicted_p=p+fb.guard_prediction_seconds*twist[:,:3]
        predicted_r=np.stack([integrate_rotation(np,a,w,fb.guard_prediction_seconds) for a,w in zip(r,twist[:,3:])])
        safe=keyway_alignment(p,r,self.spec.config.mating_clearance)&keyway_alignment(predicted_p,predicted_r,self.spec.config.mating_clearance)
        fresh=self.guard_tick!=self.observer.last_tick
        self.clear_count=np.where(fresh,np.where(safe,self.clear_count+1,0),self.clear_count)
        # Below the known final6mm guide region, keep the continuous force
        # holding law. Retreating from established seating can unload support.
        approaching=(phase==3)&(p[:,2]>self.spec.config.goal[2]+fb.guard_seat_gap_m)
        self._update_retreating(phase,p,safe,approaching)
        waiting=approaching&(self.clear_count<fb.guard_settle_ticks)
        command[:,2]=np.where(waiting,np.maximum(command[:,2],0.),command[:,2])
        command[:,2]=np.where(self.retreating,fb.guard_retreat_speed,command[:,2])
        bounded_z=np.clip(command[:,2],(p[:,2]-fb.lead_z_m-rp[:,2])/dt,(p[:,2]+fb.lead_z_m-rp[:,2])/dt)
        command[:,2]=np.where(active,bounded_z,command[:,2])
        command[:,2]=np.where(phase==3,np.clip(command[:,2],-fb.normal_velocity_limit,fb.normal_velocity_limit),command[:,2])
        self.guard_tick=self.observer.last_tick.copy()
        command[:,:6]=np.clip(command[:,:6],[-.02]*3+[-.12]*3,[.02]*3+[.12]*3)
        return command.astype(np.float32)
