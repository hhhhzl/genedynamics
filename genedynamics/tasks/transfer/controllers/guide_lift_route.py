"""Ordinary T2 lift/traverse/descend reference under the frozen task cost.

Current T2 cost requires terminal support and x/y order, not continuous contact.
The route clears the known guide walls (top .524) with a .106 tool envelope and
4mm clearance, then restores terminal5N. It is a declared free-space reference,
not contact-preserving transport or a change to the environment.
"""
from copy import deepcopy
import numpy as np
from genedynamics.core.control.transfer import orientation_error
from .guide_feedback import PublicGuideFeedbackPolicy

CLEAR_Z=.524+.106+.004


class PublicGuideLiftRoutePolicy:
    VERSION='public-T2-knowngeometry-lift-traverse-descend-v1'
    def __init__(self,spec,*,mode):
        if spec.alias!='T2' or spec.config.robot not in {'panda','xarm7'}:raise ValueError('explicit fixedarm T2')
        self.spec,self.mode=spec,int(mode);self.base=PublicGuideFeedbackPolicy(spec,modes=[mode],lead_xy_m=.008,position_admittance=True) if spec.config.robot=='panda' else None;self.reset()
    def reset(self):
        if self.base is not None:self.base.reset()
        self.stage=0;self.initial=None;self.compensation=np.zeros((1,6));self.tick=-1;self.cached=None
    def get_state(self):return deepcopy(dict(version=self.VERSION,robot=self.spec.config.robot,mode=self.mode,stage=self.stage,initial=self.initial,compensation=self.compensation,tick=self.tick,cached=self.cached,base=self.base.get_state() if self.base is not None else None))
    def restore(self,s):
        if (s['version'],s['robot'],s['mode'])!=(self.VERSION,self.spec.config.robot,self.mode):raise ValueError('lift route binding differs')
        self.stage=s['stage'];self.initial=deepcopy(s['initial']);self.compensation=s['compensation'].copy();self.tick=s['tick'];self.cached=deepcopy(s['cached'])
        if self.base is not None:self.base.restore(s['base'])
    def act(self,public):
        tick=int(np.rint(np.asarray(public['elapsed_seconds']).item()/self.spec.config.control_dt))
        if tick==self.tick:return self.cached.copy()
        if tick!=self.tick+1:raise ValueError('every public control tick required')
        if self.base is not None:self.cached=self.base.act(public).copy();self.tick=tick;return self.cached.copy()
        if len(public['tcp_position'])!=1:raise ValueError('explicit batch1')
        p=np.asarray(public['tcp_position']);r=np.asarray(public['tcp_rotation']);rp=np.asarray(public['reference_position']);dt=self.spec.config.control_dt;goal=np.asarray(self.spec.config.goal)
        if self.initial is None:self.initial=p.copy()
        tol=.75*self.spec.config.position_tolerance
        if self.stage==0 and p[0,2]>=CLEAR_Z-.002:self.stage=1
        axis=0 if self.mode==0 else 1
        if self.stage==1 and abs(p[0,axis]-goal[axis])<=tol:self.stage=2
        if self.stage==2 and np.linalg.norm(p[0,:2]-goal[:2])<=tol:self.stage=3
        target=self.initial.copy();target[:,2]=CLEAR_Z
        if self.stage>=1:target[:,axis]=goal[axis]
        if self.stage>=2:target[:,:2]=goal[:2]
        if self.stage==3:target[:,2]=goal[2]
        desired=np.broadcast_to(np.diag([1.,-1.,-1.]),r.shape);angular=np.stack([orientation_error(np,a,b) for a,b in zip(r,desired)]);issued_angular=np.stack([orientation_error(np,a,b) for a,b in zip(public['reference_rotation'],desired)]);error=np.concatenate((target-p,angular),-1);issued=np.concatenate((target-rp,issued_angular),-1);self.compensation=np.clip(self.compensation+2.*dt*error,-.04,.04)
        if self.stage==3:self.compensation[:,2]=0.
        velocity=np.asarray([6.]*3+[4.]*3)*error+2.*issued-np.asarray(public['tcp_twist'])+self.compensation;velocity[:,:3]=np.clip(velocity[:,:3],-.02,.02);velocity[:,3:]=np.clip(velocity[:,3:],-.12,.12);stiffness=np.ones((1,6));force=np.zeros((1,1))
        if self.stage==3:
            velocity[:,2]=np.clip(.004*(np.asarray(public['measured_wrench'])[:,2]-5.),-.01,.01);stiffness[:,2]=0.;velocity[:,:2]=np.clip(velocity[:,:2],(p[:,:2]-.008-rp[:,:2])/dt,(p[:,:2]+.008-rp[:,:2])/dt);velocity[:,:2]=np.clip(velocity[:,:2],-.02,.02)
        self.cached=np.concatenate((velocity,stiffness,force),-1).astype(np.float32);self.tick=tick;return self.cached.copy()
