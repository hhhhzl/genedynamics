"""One ordinary task-aware guide reference with public desired-wrench budgeting.

Moving at the teacher's constant 5 N is not a task requirement. Establish
contact, move with 0.5 N, then restore 5 N near the terminal goal. A 9 N desired
Cartesian tangent budget uses the known controller equation/issued limiter;
it does not bound true contact peaks or certify unknown plant dynamics.
"""
import numpy as np
from genedynamics.tasks.transfer.contact import bounded_contact_command
from .guide_pose_feedback import PublicGuidePoseFeedbackPolicy


class PublicGuideWrenchSchedulePolicy(PublicGuidePoseFeedbackPolicy):
    VERSION='public-guide-taskforce-wrenchbudget-v1'
    def act(self,public):
        action=super().act(public);p=np.asarray(public['tcp_position']);phase=self.observer.memory.phase;distance=np.linalg.norm(p[:,:2]-np.asarray(self.spec.config.goal[:2]),axis=-1);normal=np.where((phase==0)|(distance<=.75*self.spec.config.position_tolerance),5.,.5);action[:,2]=np.clip(.004*(np.asarray(public['measured_wrench'])[:,2]-normal),-.01,.01)
        for i in range(len(action)):
            previous=np.asarray(public['previous_action'][i]);issued=bounded_contact_command(np,action[i],previous,self.spec.config);k=800.*np.exp(issued[6:8]);d=2*np.sqrt(k);lead=np.asarray(public['reference_position'][i,:2])-p[i,:2];twist=np.asarray(public['tcp_twist'][i,:2]);wrench=k*(lead+self.spec.config.control_dt*issued[:2])+d*(issued[:2]-twist);magnitude=np.linalg.norm(wrench);target=wrench*min(1.,9./max(magnitude,1e-12));velocity=(target-k*lead+d*twist)/(self.spec.config.control_dt*k+d);low=np.maximum(-.02,previous[:2]-.4*self.spec.config.control_dt);high=np.minimum(.02,previous[:2]+.4*self.spec.config.control_dt);action[i,:2]=np.clip(velocity,low,high)
        action[:,12]=0.;return action.astype(np.float32)
