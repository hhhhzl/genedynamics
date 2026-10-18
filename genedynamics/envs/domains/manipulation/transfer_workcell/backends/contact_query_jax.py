import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx
from mujoco.mjx._src import support
from genedynamics.envs.domains.manipulation.transfer_workcell.backends.jax.execution import _fp32_scope
from ..queries.contact import ContactQueryResult
from genedynamics.core.control.transfer import orientation_error,orientation_distance


class JaxContactQuery:
    def __init__(self,runner):
        self.runner=runner
        self._build=jax.jit(jax.vmap(self._one),device=runner._impl.device)

    def _one(self,semantic,state,q0):
        r=self.runner._impl; pos,rot,twist,force,phase=semantic
        valid=jp.all(jp.isfinite(pos))&jp.all(jp.isfinite(rot))&jp.all(jp.isfinite(twist))&jp.isfinite(force)&(force>=0)
        valid&=jp.max(jp.abs(rot.T@rot-jp.eye(3)))<.01
        valid&=(jp.abs(jp.linalg.det(rot)-1)<.01)&((phase==0)|(phase==1))
        position=jp.where(valid,pos,state.reference_position); rotation=jp.where(valid,rot,state.reference_rotation)
        velocity=jp.where(valid,twist,jp.zeros(6))
        def kinematics(q): return mjx.com_pos(r.model,mjx.kinematics(r.model,r.initial.replace(qpos=q)))
        def jacobian(data):
            p,j=support.jac(r.model,data,data.site_xpos[r.sid],r.body)
            return jp.concatenate((p.T,j.T))
        def iteration(_,q):
            data=kinematics(q); jac=jacobian(data)
            error=jp.concatenate((position-data.site_xpos[r.sid],orientation_error(jp,data.site_xmat[r.sid],rotation)))
            dq=jac.T@jp.linalg.solve(jac@jac.T+.002**2*jp.eye(6),error)
            return jp.clip(q+jp.clip(dq,-.12,.12),r.ranges[:,0]+.002,r.ranges[:,1]-.002)
        q=jax.lax.fori_loop(0,96,iteration,q0); data=kinematics(q); jac=jacobian(data)
        qvel=jac.T@jp.linalg.solve(jac@jac.T+.002**2*jp.eye(6),velocity)
        data=mjx.forward(r.model,r.initial.replace(qpos=q,qvel=qvel))
        wrench,_,touch,forbidden=r._measure(data)
        pe=jp.linalg.norm(data.site_xpos[r.sid]-position); ae=orientation_distance(jp,data.site_xmat[r.sid],rotation)
        ve=jp.linalg.norm(jac@qvel-velocity); compatible=(phase==0)|touch
        ok=valid&(pe<.001)&(ae<.01)&(ve<.005)&compatible&~forbidden&(jp.max(jp.abs(qvel))<3.)
        state=state._replace(physics=data,reference_position=data.site_xpos[r.sid],reference_rotation=data.site_xmat[r.sid],
            measured_force=wrench[2],contact_seen=touch,status=jp.where(ok,0,7))
        return ContactQueryResult(state,ok,pe,ae,ve,compatible)

    @_fp32_scope
    def build(self,semantic,request,*,previous=None):
        r=self.runner._impl; batch=len(request.sample_ids)
        for x,shape in zip(semantic,((batch,3),(batch,3,3),(batch,6),(batch,),(batch,))):
            if np.shape(x)!=shape: raise ValueError("contact semantic shape mismatch")
        base=r.reset_batch(request); q0=base.physics.qpos
        if previous is not None:
            compatible=(previous.sample_id==base.sample_id)&jp.all(jp.isfinite(previous.physics.qpos),axis=1)
            q0=jp.where(compatible[:,None],previous.physics.qpos,q0)
        semantic=jax.tree.map(lambda x:jax.device_put(jp.asarray(x,dtype=jp.float32),r.device),semantic)
        return self._build(semantic,base,q0)
