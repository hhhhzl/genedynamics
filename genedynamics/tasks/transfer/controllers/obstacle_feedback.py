"""Public T1 body-specific geometric reference; original skill Box/plant intact.

Known Panda/xArm waypoint and servo recipes were qualified in P3. They are a
common ordinary reference, not learned transport or inferred target dynamics.
Every mutable value below belongs to the actor; no executor is stored.
"""
from copy import deepcopy
import numpy as np
from genedynamics.core.control.transfer import orientation_error

OBSTACLE_BOUNDS=np.asarray([.12]*3+[.6]*3+[1.]*6+[1.],np.float32)


def routes(spec):
    if spec.alias!='T1' or spec.config.robot not in {'panda','xarm7','ur5e'}:raise ValueError('qualified T1 fixed-arm geometric recipes only')
    goal=spec.config.goal_position
    if spec.config.robot=='ur5e':
        start=spec.config.start_position
        return np.asarray([
            [start,[.38,-.12,.67],[.54,-.12,.65],[.57,.08,.62],goal],
            [start,[.38,-.16,.55],[.66,-.16,.55],[.66,.08,.55],goal],
        ],np.float64)
    upper=([ [.38,-.12,.67],[.61,-.12,.67],[.64,.08,.58],goal] if spec.config.robot=='panda' else
           [ [.38,-.12,.67],[.54,-.12,.65],[.57,.08,.62],goal])
    return np.asarray([upper,[[.38,-.16,.55],[.66,-.16,.55],[.66,.08,.55],goal]],np.float64)


class PublicObstacleFeedbackPolicy:
    VERSION='public-obstacle-body-feedback-v1'
    def __init__(self,spec,*,modes=(0,)):
        self.spec=spec;self.routes=routes(spec);self.pose_transport=None;self._refresh_routes();self.reset(modes=modes)

    def _refresh_routes(self):
        self.route_arcs=(np.concatenate((np.zeros((2,1)),np.cumsum(np.linalg.norm(np.diff(self.routes,axis=1),axis=-1),axis=1)),axis=1)
                         if self.spec.config.robot=='ur5e' else None)

    def reset(self,*,modes=None):
        if modes is not None:self.modes=np.asarray(modes,np.int64).reshape(-1)
        if np.any((self.modes<0)|(self.modes>1)):raise ValueError('two T1 request modes')
        self.phase=np.zeros(len(self.modes),np.int64);self.path_distance=np.zeros(len(self.modes));self.integral=np.zeros((len(self.modes),6));self.last_tick=np.full(len(self.modes),-1,np.int64);self.raw=np.zeros((len(self.modes),13),np.float32)

    def get_state(self):
        return deepcopy(dict(version=self.VERSION,robot=self.spec.config.robot,modes=self.modes,phase=self.phase,path_distance=self.path_distance,integral=self.integral,last_tick=self.last_tick,raw=self.raw))

    def restore(self,state):
        if state['version']!=self.VERSION or state['robot']!=self.spec.config.robot or not np.array_equal(state['modes'],self.modes):raise ValueError('obstacle public binding differs')
        for name in ('phase','integral','last_tick','raw'):setattr(self,name,deepcopy(state[name]))
        self.path_distance=deepcopy(state.get('path_distance',np.zeros(len(self.modes))))

    def reorder(self,indices):
        indices=np.asarray(indices,np.int64)
        for name in ('modes','phase','path_distance','integral','last_tick','raw'):setattr(self,name,getattr(self,name)[indices].copy())

    def reference(self):
        if self.spec.config.robot!='ur5e':return self.routes[self.modes,self.phase].copy()
        values=[]
        for mode,distance,phase in zip(self.modes,self.path_distance,self.phase):
            low=phase;high=phase+1;span=self.route_arcs[mode,high]-self.route_arcs[mode,low]
            fraction=np.clip((distance-self.route_arcs[mode,low])/max(span,1e-9),0.,1.)
            values.append(self.routes[mode,low]+fraction*(self.routes[mode,high]-self.routes[mode,low]))
        return np.asarray(values)

    def act(self,public):
        p=np.asarray(public['tcp_position']);r=np.asarray(public['tcp_rotation']);rp=np.asarray(public['reference_position']);rr=np.asarray(public['reference_rotation']);goal_r=np.asarray(public['goal_rotation']);dt=self.spec.config.control_dt
        tick=np.rint(np.asarray(public['elapsed_seconds']).reshape(-1)/dt).astype(np.int64)
        if len(p)!=len(self.modes) or np.any(tick<self.last_tick) or np.any((self.last_tick>=0)&(tick>self.last_tick+1)):raise ValueError('observe every controltick; reset/reorder explicitly')
        if self.spec.config.robot=='ur5e':
            fresh=tick!=self.last_tick;totals=self.route_arcs[self.modes,-1]
            self.path_distance=np.where(fresh,np.minimum(self.path_distance+.08*dt,totals),self.path_distance)
            self.phase=np.asarray([min(int(np.searchsorted(self.route_arcs[mode,1:],distance,side='right')),3) for mode,distance in zip(self.modes,self.path_distance)])
            target=self.reference();tangent=np.asarray([self.routes[mode,phase+1]-self.routes[mode,phase] for mode,phase in zip(self.modes,self.phase)])
            tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-9);feedforward=np.where(self.path_distance<totals,.08,0.)[:,None]
            angular=np.stack([orientation_error(np,a,b) for a,b in zip(r,goal_r)]);issued=np.stack([orientation_error(np,a,b) for a,b in zip(rr,goal_r)])
            command=np.zeros((len(p),13),np.float32);command[:,:3]=np.clip(feedforward*tangent+4.*(target-rp)+(target-p),-.12,.12);command[:,3:6]=np.clip(3.*issued+3.*angular,-.6,.6);command[:,6:12]=1.
            self.raw=np.where(fresh[:,None],command,self.raw);self.last_tick=tick.copy();actual=self.raw.copy()
            if self.pose_transport is not None:actual=self.pose_transport.apply(actual[:,None],phase=self.phase)[:,0]
            actual=np.clip(actual,-OBSTACLE_BOUNDS,OBSTACLE_BOUNDS);actual[:,12]=0.;return actual.astype(np.float32)
        fresh=tick!=self.last_tick;target=self.reference();advance=fresh&(np.linalg.norm(p-target,axis=-1)<.012)&(self.phase<3)
        self.phase+=advance;self.integral[advance]=0.;target=self.reference()
        angular=np.stack([orientation_error(np,a,b) for a,b in zip(r,goal_r)]);error=np.concatenate((target-p,angular),axis=-1)
        gain,limit=(.8,.04) if self.spec.config.robot=='panda' else (4.,.2)
        self.integral=np.where(fresh[:,None],np.clip(self.integral+gain*dt*error,-limit,limit),self.integral)
        command=np.zeros((len(p),13),np.float32);command[:,:3]=np.clip(4.*(target-rp)+3.*(target-p)+self.integral[:,:3],-.12,.12)
        issued=np.stack([orientation_error(np,a,b) for a,b in zip(rr,goal_r)])
        command[:,3:6]=np.clip(3.*issued+3.*angular+self.integral[:,3:],-.6,.6);command[:,6:12]=1.
        self.raw=np.where(fresh[:,None],command,self.raw);self.last_tick=tick.copy()
        actual=self.raw.copy()
        if self.pose_transport is not None:actual=self.pose_transport.apply(actual[:,None],phase=self.phase)[:,0]
        actual=np.clip(actual,-OBSTACLE_BOUNDS,OBSTACLE_BOUNDS);actual[:,12]=0.
        return actual.astype(np.float32)
