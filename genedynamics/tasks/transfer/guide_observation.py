"""Public memory for all four tasks; preserves the frozen T1/T3 implementation.

T4 intentionally reuses the T3 recovery clock/geometry with the mobile task's
physical configuration. This adds no private support or WBC state. T2 has its
own public guide-outlet events, matching the qualified guide teacher.
"""
from dataclasses import replace
import numpy as np

from .observation import PublicControllerMemory
from genedynamics.tasks.transfer.behavior import advance_behavior

FEATURE_VERSION='public-controller-memory-four-task-v1'
CONTEXT_DIM=53


class GuidePublicMemory(PublicControllerMemory):
    def __init__(self,spec,modes):
        if spec.alias!='T2':raise ValueError('guide memory requires T2')
        super().__init__(replace(spec,alias='T3'),modes)
        self.spec=spec

    def update(self,public):
        p=np.asarray(public['tcp_position'],dtype=np.float64);r=np.asarray(public['tcp_rotation'],dtype=np.float64)
        rp=np.asarray(public['reference_position']);rr=np.asarray(public['reference_rotation'])
        tick=np.rint(np.asarray(public['elapsed_seconds']).reshape(-1)/self.spec.config.control_dt).astype(np.int64)
        if len(p)!=len(self.modes) or np.any(tick<self.last_tick) or np.any((self.last_tick>=0)&(tick>self.last_tick+1)):
            raise ValueError('guide memory requires explicit reset/reorder and every observed tick')
        fresh=tick!=self.last_tick;boundary=fresh&(tick%10==0)
        if self.start_position is None:
            self.start_position,self.start_rotation=p.copy(),r.copy();self.start_reference_position,self.start_reference_rotation=rp.copy(),rr.copy()
        outlet=np.where(self.modes==0,p[:,0]>.455,p[:,1]>-.181)
        updated=advance_behavior(self.memory,p,r,np.asarray(public['measured_wrench'])[:,2],insertion=False,
            dt=10*self.spec.config.control_dt,goal_z=self.spec.config.goal[2],alignment_mask=outlet)
        self.memory=type(updated)(*(np.where(boundary.reshape((-1,)+(1,)*(np.ndim(a)-1)),a,b) for a,b in zip(updated,self.memory)))
        # The shared source guide teacher clears tangential integral on capture.
        capture=(self.memory.phase==3)&(np.max(np.abs(p[:,:2]-np.asarray(self.spec.config.goal)[:2]),axis=1)<.003)&boundary
        integral=self.memory.tracking_integral.copy();integral[capture,:2]=0.;self.memory=self.memory._replace(tracking_integral=integral)
        for key,value in (('start_position',p),('start_rotation',r),('start_reference_position',rp),('start_reference_rotation',rr)):
            setattr(self,key,np.where(boundary.reshape((-1,)+(1,)*(value.ndim-1)),value,getattr(self,key)))
        self.last_tick=tick.copy()
        def relative(position,rotation):
            translation=np.einsum('bji,bj->bi',r,position-p)/.05
            rotation=np.swapaxes(r,-1,-2)@rotation
            return np.concatenate((translation,rotation[...,:,:2].swapaxes(-1,-2).reshape(len(p),6)),axis=1)
        high=np.asarray([.08]*3+[.4]*3+[1.]*6+[self.spec.config.force_limit])
        return np.concatenate((relative(rp,rr),np.asarray(public['previous_action'])/high,np.eye(4)[self.memory.phase],np.eye(2)[self.modes],
            self.memory.tracking_integral/.08,((tick%10)/10.)[:,None],relative(self.start_position,self.start_rotation),relative(self.start_reference_position,self.start_reference_rotation)),axis=1).astype(np.float32)


def public_memory(spec,modes):
    if spec.alias in {'T1','T3'}:return PublicControllerMemory(spec,modes)
    if spec.alias=='T2':return GuidePublicMemory(spec,modes)
    if spec.alias=='T4':return PublicControllerMemory(replace(spec,alias='T3'),modes)
    raise ValueError('unknown four-task memory')


def context_sequence(public,spec,mode):
    memory=public_memory(spec,[mode])
    return np.stack([memory.update({k:np.asarray(v)[i:i+1] for k,v in public.items()})[0] for i in range(len(public['tcp_position']))])
