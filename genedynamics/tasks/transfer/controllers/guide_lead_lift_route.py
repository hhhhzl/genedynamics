"""Shared known issued-reference lead governor for the fixed T2 event route.

All xArm target geometry, gains, mode events, deadlines and force criteria are
unchanged. Panda remains exactly the original qualified guide reference.
"""
import numpy as np
from genedynamics.core.control.transfer import orientation_error
from .guide_antiwindup_lift_route import PublicGuideAntiWindupLiftRoutePolicy
from .guide_lift_route import CLEAR_Z
class PublicGuideLeadLiftRoutePolicy(PublicGuideAntiWindupLiftRoutePolicy):
    VERSION='public-T2-order-event-shared-reference-lead-v1'
    def act(self,public):
        tick=int(np.rint(np.asarray(public['elapsed_seconds']).item()/self.spec.config.control_dt))
        if tick==self.tick:return self.cached.copy()
        if tick!=self.tick+1:raise ValueError('every public control tick required')
        if self.base is not None:self.cached=self.base.act(public).copy();self.tick=tick;return self.cached.copy()
        if len(public['tcp_position'])!=1:raise ValueError('explicit batch1')
        p=np.asarray(public['tcp_position']);r=np.asarray(public['tcp_rotation']);rp=np.asarray(public['reference_position']);dt=self.spec.config.control_dt;goal=np.asarray(self.spec.config.goal)
        if self.initial is None:self.initial=p.copy()
        tol=.75*self.spec.config.position_tolerance
        if self.mode_anchor is None and tick>=1:self.mode_anchor=p[:,:2].copy()
        if self.stage==1 and self.mode_anchor is not None:
            first=0 if self.mode==0 else 1;threshold=.017 if self.mode==0 else .010
            if p[0,first]>self.mode_anchor[0,first]+threshold:self.stage=2
        if self.stage==0 and p[0,2]>=CLEAR_Z-.002:self.stage=1
        axis=0 if self.mode==0 else 1
        if self.stage==1 and abs(p[0,axis]-goal[axis])<=tol:self.stage=2
        if self.stage==2 and np.linalg.norm(p[0,:2]-goal[:2])<=tol:self.stage=3
        target=self.initial.copy();target[:,2]=CLEAR_Z
        if self.stage>=1:target[:,axis]=goal[axis]
        if self.stage>=2:target[:,:2]=goal[:2]
        if self.stage==3:target[:,2]=goal[2]
        desired=np.broadcast_to(np.diag([1.,-1.,-1.]),r.shape);angular=np.stack([orientation_error(np,a,b) for a,b in zip(r,desired)]);issued_angular=np.stack([orientation_error(np,a,b) for a,b in zip(public['reference_rotation'],desired)]);error=np.concatenate((target-p,angular),-1);issued=np.concatenate((target-rp,issued_angular),-1);limits=np.asarray([.02]*3+[.12]*3);raw=np.asarray([6.]*3+[4.]*3)*error+2.*issued-np.asarray(public['tcp_twist'])+self.compensation;increment=2.*dt*error
        # Standard conditional integration: a saturated command cannot wind
        # farther in the saturation direction. Gains and limits are unchanged.
        allowed=(np.abs(raw)<limits)|(raw*increment<=0.)
        self.compensation=np.clip(self.compensation+np.where(allowed,increment,0.),-.04,.04)
        if self.stage==3:self.compensation[:,2]=0.
        velocity=np.asarray([6.]*3+[4.]*3)*error+2.*issued-np.asarray(public['tcp_twist'])+self.compensation;velocity[:,:3]=np.clip(velocity[:,:3],-.02,.02);velocity[:,3:]=np.clip(velocity[:,3:],-.12,.12);stiffness=np.ones((1,6));force=np.zeros((1,1))
        if self.stage==3:
            velocity[:,2]=np.clip(.004*(np.asarray(public['measured_wrench'])[:,2]-5.),-.01,.01);stiffness[:,2]=0.;velocity[:,:2]=np.clip(velocity[:,:2],(p[:,:2]-.008-rp[:,:2])/dt,(p[:,:2]+.008-rp[:,:2])/dt);velocity[:,:2]=np.clip(velocity[:,:2],-.02,.02)
        # Same public lead limits already used by the shared T3/T4 decoder.
        # Apply throughout free-space travel as well as the loaded approach,
        # so the issued reference cannot accumulate an unbounded intent lead.
        lead=np.asarray([.008,.008,.01]);velocity[:,:3]=np.clip(velocity[:,:3],(p-lead-rp)/dt,(p+lead-rp)/dt)
        rotation_lead=np.stack([orientation_error(np,a,b) for a,b in zip(r,public['reference_rotation'])]);velocity[:,3:]=np.clip(velocity[:,3:],(-.04-rotation_lead)/dt,(.04-rotation_lead)/dt)
        velocity[:,:3]=np.clip(velocity[:,:3],-.02,.02);velocity[:,3:]=np.clip(velocity[:,3:],-.12,.12)
        self.cached=np.concatenate((velocity,stiffness,force),-1).astype(np.float32);self.tick=tick;return self.cached.copy()
