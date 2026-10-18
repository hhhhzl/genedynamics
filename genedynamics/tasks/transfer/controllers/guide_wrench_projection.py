"""Exact desired-wrench projection on ball intersect known issued-velocity Box.

Unlike sequential radial projection/coordinate clipping, the KKT projection
respects both constraints whenever their intersection is nonempty. This is a
known-controller arithmetic property, not a true-contact force certificate.
"""
import numpy as np
from genedynamics.tasks.transfer.contact import bounded_contact_command
from .guide_pose_feedback import PublicGuidePoseFeedbackPolicy


def project_wrench_box_ball(desired,low,high,radius=9.):
    desired=np.asarray(desired,np.float64);low=np.asarray(low,np.float64);high=np.asarray(high,np.float64)
    if desired.shape!=(2,) or low.shape!=(2,) or high.shape!=(2,) or np.any(low>high) or not np.isfinite(np.r_[desired,low,high,radius]).all() or radius<=0:raise ValueError('finite ordered2D wrenchBox/positive radius required')
    closest=np.clip(np.zeros(2),low,high)
    if np.linalg.norm(closest)>radius:return closest,False
    value=np.clip(desired,low,high)
    if np.linalg.norm(value)<=radius:return value,True
    left=0.;right=1.
    while np.linalg.norm(np.clip(desired/(1+right),low,high))>radius:right*=2.
    for _ in range(48):
        middle=(left+right)/2;candidate=np.clip(desired/(1+middle),low,high)
        if np.linalg.norm(candidate)>radius:left=middle
        else:right=middle
    return np.clip(desired/(1+right),low,high),True


class PublicGuideWrenchProjectionPolicy(PublicGuidePoseFeedbackPolicy):
    VERSION='public-guide-taskforce-exactwrenchprojection-v2'
    def act(self,public):
        action=super().act(public);p=np.asarray(public['tcp_position']);phase=self.observer.memory.phase;distance=np.linalg.norm(p[:,:2]-np.asarray(self.spec.config.goal[:2]),axis=-1);normal=np.where((phase==0)|(distance<=.75*self.spec.config.position_tolerance),5.,.5);action[:,2]=np.clip(.004*(np.asarray(public['measured_wrench'])[:,2]-normal),-.01,.01)
        for i in range(len(action)):
            previous=np.asarray(public['previous_action'][i]);issued=bounded_contact_command(np,action[i],previous,self.spec.config);k=800.*np.exp(issued[6:8]);d=2*np.sqrt(k);lead=np.asarray(public['reference_position'][i,:2])-p[i,:2];twist=np.asarray(public['tcp_twist'][i,:2]);coefficient=self.spec.config.control_dt*k+d;offset=k*lead-d*twist;rate=.4*self.spec.config.control_dt
            velocity_low=previous[:2]+np.clip(-.02-previous[:2],-rate,rate);velocity_high=previous[:2]+np.clip(.02-previous[:2],-rate,rate);target,feasible=project_wrench_box_ball(offset+coefficient*issued[:2],offset+coefficient*velocity_low,offset+coefficient*velocity_high);velocity=(target-offset)/coefficient;action[i,:2]=np.clip(velocity,-.02,.02)
        action[:,12]=0.;return action.astype(np.float32)
