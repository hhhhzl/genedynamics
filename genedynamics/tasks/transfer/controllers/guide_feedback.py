"""Public T2 guide clock with separately declared lead/normal primitives."""
from copy import deepcopy
import numpy as np
from genedynamics.tasks.transfer.guide_observation import GuidePublicMemory
from genedynamics.tasks.transfer.behavior import reference_commands,teacher_reference


class PublicGuideFeedbackPolicy:
    def __init__(self,spec,*,modes=(0,),lead_xy_m=None,position_admittance=False):
        if spec.alias!='T2' or lead_xy_m not in (None,.008):raise ValueError('fixed T2 guide primitive binding required')
        self.spec,self.lead_xy_m,self.position_admittance=spec,lead_xy_m,bool(position_admittance);self.reset(modes=modes)

    def reset(self,*,modes=None):
        if modes is not None:self.modes=tuple(modes)
        self.observer=GuidePublicMemory(self.spec,self.modes);self.commands=[]

    def binding(self):return dict(task=self.spec.name,robot=self.spec.config.robot,modes=list(self.modes),lead_xy_m=self.lead_xy_m,position_admittance=self.position_admittance)
    def get_state(self):return deepcopy(dict(version='public-guide-feedback-v1',binding=self.binding(),observer=self.observer.get_state(),commands=self.commands))
    def restore(self,state):
        if state['version']!='public-guide-feedback-v1' or state['binding']!=self.binding():raise ValueError('guide continuation binding differs')
        self.observer.restore(state['observer']);self.commands=deepcopy(state['commands'])
    def reorder(self,indices):
        indices=np.asarray(indices,dtype=int);self.observer.reorder(indices);self.modes=tuple(np.asarray(self.modes)[indices]);self.commands=[v[indices].copy() for v in self.commands]

    def act(self,public):
        self.observer.update(public)
        if not self.commands:
            reference=teacher_reference(self.observer.memory,insertion=False,goal_z=self.spec.config.goal[2]);chunk=reference_commands(reference,public['reference_position'],public['reference_rotation'],public['tcp_position'],public['tcp_rotation'],integral_bias=self.observer.memory.tracking_integral,dt=self.spec.config.control_dt)
            self.commands=[v.copy() for v in chunk.swapaxes(0,1)]
        command=self.commands.pop(0).copy();dt=self.spec.config.control_dt
        if self.lead_xy_m is not None:
            p=np.asarray(public['tcp_position']);rp=np.asarray(public['reference_position'])
            command[:,:2]=np.clip(command[:,:2],(p[:,:2]-self.lead_xy_m-rp[:,:2])/dt,(p[:,:2]+self.lead_xy_m-rp[:,:2])/dt)
            # An already infeasible lead cannot be corrected instantaneously
            # under the original teacher's velocity bound; no safety theorem.
            command[:,:2]=np.clip(command[:,:2],-.02,.02)
        if self.position_admittance:
            command[:,2]=np.clip(.004*(np.asarray(public['measured_wrench'])[:,2]-command[:,12]),-.01,.01);command[:,12]=0.
        return command.astype(np.float32)
