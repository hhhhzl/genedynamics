import numpy as np
import mujoco
from genedynamics.learning.cross_embodiment.contact_query import ContactQueryResult
from genedynamics.learning.cross_embodiment.core.control_math import orientation_error,orientation_distance


class MujocoContactQuery:
    def __init__(self,runner): self.runner=runner
    def build(self,semantic,request,*,previous=None):
        r=self.runner._impl; m=r.model; batch=len(request.sample_ids)
        for x,shape in zip(semantic,((batch,3),(batch,3,3),(batch,6),(batch,),(batch,))):
            if np.shape(x)!=shape: raise ValueError("contact semantic shape mismatch")
        base=r.reset_batch(request); states=[]; diagnostics=[]
        for i,identity in enumerate(request.sample_ids):
            pos,rot,twist,force,phase=(np.asarray(x)[i] for x in semantic)
            valid=all(np.all(np.isfinite(x)) for x in (pos,rot,twist,force,phase))
            valid&=np.allclose(rot.T@rot,np.eye(3),atol=.01) and abs(np.linalg.det(rot)-1)<.01
            valid&=phase in (0,1) and force>=0
            state=r._slice(base,i); data=mujoco.MjData(m)
            q0=state.physics.qpos
            if previous is not None and previous.sample_id[i]==identity and np.isfinite(previous.physics[i].qpos).all(): q0=previous.physics[i].qpos
            data.qpos[:]=q0
            mujoco.mj_forward(m,data)
            if valid:
                for _ in range(96):
                    mujoco.mj_forward(m,data)
                    p,j=np.zeros((3,m.nv)),np.zeros((3,m.nv)); mujoco.mj_jacSite(m,data,p,j,r.sid)
                    jac=np.r_[p,j]
                    error=np.r_[pos-data.site_xpos[r.sid],orientation_error(np,data.site_xmat[r.sid].reshape(3,3),rot)]
                    dq=jac.T@np.linalg.solve(jac@jac.T+.002**2*np.eye(6),error)
                    data.qpos[:]=np.clip(data.qpos+np.clip(dq,-.12,.12),m.jnt_range[:,0]+.002,m.jnt_range[:,1]-.002)
                mujoco.mj_forward(m,data)
                p,j=np.zeros((3,m.nv)),np.zeros((3,m.nv)); mujoco.mj_jacSite(m,data,p,j,r.sid); jac=np.r_[p,j]
                data.qvel[:]=jac.T@np.linalg.solve(jac@jac.T+.002**2*np.eye(6),twist)
                mujoco.mj_forward(m,data)
                pe=np.linalg.norm(data.site_xpos[r.sid]-pos); ae=orientation_distance(np,data.site_xmat[r.sid].reshape(3,3),rot)
                ve=np.linalg.norm(jac@data.qvel-twist)
            else: pe,ae,ve=1.,1.,1.
            wrench,_,contact,forbidden=r._measure(data)
            compatible=phase==0 or contact
            ok=valid and pe<.001 and ae<.01 and ve<.005 and compatible and not forbidden and np.max(np.abs(data.qvel))<3.
            state=state._replace(physics=data,reference_position=data.site_xpos[r.sid].copy(),reference_rotation=data.site_xmat[r.sid].reshape(3,3).copy(),
                measured_force=wrench[2],contact_seen=bool(contact),status=0 if ok else 7)
            states.append(state); diagnostics.append((ok,pe,ae,ve,compatible))
        values=np.array(diagnostics)
        return ContactQueryResult(r._stack(states),values[:,0].astype(bool),values[:,1],values[:,2],values[:,3],values[:,4].astype(bool))
