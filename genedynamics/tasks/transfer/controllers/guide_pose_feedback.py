"""One declared T2 ordinary pose reference, using public feedback only.

The fixed gains and stiffness are inherited from the T3 common controller.
This is a new common-reference opportunity, not a transported learned policy
or a change to plant mechanics. All 13D commands use the existing executor.
"""
from copy import deepcopy
import numpy as np
from genedynamics.tasks.transfer.guide_observation import GuidePublicMemory
from genedynamics.tasks.transfer.behavior import teacher_reference
from genedynamics.core.control.transfer import orientation_error


class PublicGuidePoseFeedbackPolicy:
    VERSION='public-guide-pose-feedback-v1'
    def __init__(self,spec,*,modes=(0,)):
        if spec.alias!='T2': raise ValueError('T2 guide pose binding only')
        self.spec=spec; self.reset(modes=modes)

    def reset(self,*,modes=None):
        if modes is not None: self.modes=tuple(modes)
        self.observer=GuidePublicMemory(self.spec,self.modes)
        self.compensation=np.zeros((len(self.modes),6)); self.compensation_tick=np.full(len(self.modes),-1,np.int64)

    def get_state(self):
        return deepcopy(dict(version=self.VERSION,robot=self.spec.config.robot,modes=self.modes,
            observer=self.observer.get_state(),compensation=self.compensation,compensation_tick=self.compensation_tick))

    def restore(self,state):
        if state['version']!=self.VERSION or state['robot']!=self.spec.config.robot or state['modes']!=self.modes:
            raise ValueError('guide pose binding differs')
        self.observer.restore(state['observer']); self.compensation=deepcopy(state['compensation']); self.compensation_tick=deepcopy(state['compensation_tick'])

    def reorder(self,indices):
        indices=np.asarray(indices,dtype=int); self.observer.reorder(indices); self.modes=tuple(np.asarray(self.modes)[indices])
        self.compensation=self.compensation[indices].copy(); self.compensation_tick=self.compensation_tick[indices].copy()

    def act(self,public):
        self.observer.update(public); dt=self.spec.config.control_dt
        reference=teacher_reference(self.observer.memory,insertion=False,goal_z=self.spec.config.goal[2])
        p=np.asarray(public['tcp_position']); r=np.asarray(public['tcp_rotation']); rp=np.asarray(public['reference_position'])
        desired=np.broadcast_to(np.diag([1.,-1.,-1.]),r.shape)
        angular=np.stack([orientation_error(np,a,b) for a,b in zip(r,desired)])
        issued_angular=np.stack([orientation_error(np,a,b) for a,b in zip(public['reference_rotation'],desired)])
        error=np.concatenate((reference[:,:3]-p,angular),axis=-1)
        issued=np.concatenate((reference[:,:3]-rp,issued_angular),axis=-1)
        fresh=self.compensation_tick!=self.observer.last_tick
        self.compensation=np.where(fresh[:,None],np.clip(self.compensation+2.*dt*error,-.04,.04),self.compensation)
        self.compensation[:,2]=0.
        velocity=np.asarray([6.]*3+[4.]*3)*error+2.*issued-np.asarray(public['tcp_twist'])+self.compensation
        # Fixed log stiffness1 in tangent/orientation,0 in normal, matching
        # the established T3 spatial mask. No new torque/friction override.
        action=np.concatenate((velocity,np.broadcast_to([1.,1.,0.,1.,1.,1.],(len(p),6)),np.zeros((len(p),1))),axis=-1)
        action[:,:2]=np.clip(action[:,:2],(p[:,:2]-.008-rp[:,:2])/dt,(p[:,:2]+.008-rp[:,:2])/dt)
        rotation_lead=np.stack([orientation_error(np,a,b) for a,b in zip(r,public['reference_rotation'])])
        action[:,3:6]=np.clip(action[:,3:6],(-.04-rotation_lead)/dt,(.04-rotation_lead)/dt)
        action[:,:6]=np.clip(action[:,:6],[-.02]*3+[-.12]*3,[.02]*3+[.12]*3)
        action[:,2]=np.clip(.004*(np.asarray(public['measured_wrench'])[:,2]-5.),-.01,.01)
        self.compensation_tick=self.observer.last_tick.copy()
        return action.astype(np.float32)
