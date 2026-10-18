"""Ordinary Coulomb-friction feedforward through the public 13D request chart.

Only nominal model friction/Jacobian and current public encoders enter. The
real controller, torques, plant coefficients, hidden carry and geometry remain
unchanged. This is a model-based reference, not a safety certificate.
"""
from copy import deepcopy
import numpy as np
import mujoco
from genedynamics.tasks.transfer.contact import bounded_contact_command
from genedynamics.tasks.transfer.runtime import MechanismSpec
from genedynamics.envs.domains.manipulation.transfer_workcell.backends.contact import MujocoContact
from .guide_feedback import PublicGuideFeedbackPolicy


def friction_wrench(jacobian,desired_twist,frictionloss):
    j=np.asarray(jacobian,np.float64);v=np.asarray(desired_twist,np.float64);mu=np.asarray(frictionloss,np.float64)
    if j.ndim!=2 or j.shape[0]!=6 or v.shape!=(6,) or mu.shape!=(j.shape[1],) or np.any(mu<0) or not np.isfinite(np.r_[j.ravel(),v,mu]).all():raise ValueError('finite6Dpublic model and nonnegative friction required')
    qdot=j.T@np.linalg.solve(j@j.T+1e-6*np.eye(6),v);direction=np.where(np.abs(qdot)>1e-6,np.sign(qdot),0.);tau=mu*direction
    # Ridge least squares J.T*w ~= nominal Coulomb torque. Its nullspace
    # component is unavailable through the shared Cartesian request interface.
    return np.linalg.solve(j@j.T+1e-6*np.eye(6),j@tau)


class PublicGuideFrictionFeedforwardPolicy:
    VERSION='public-T2-nominal-Coulomb-feedforward-v1'
    def __init__(self,spec,*,mode,charge=None):
        if spec.alias!='T2' or spec.config.robot not in {'panda','xarm7'}:raise ValueError('explicit fixedarm T2 model reference')
        self.spec=spec;self.mode=mode;self.charge=charge;self.base=PublicGuideFeedbackPolicy(spec,modes=[mode],lead_xy_m=.008,position_admittance=True);self.backend=None if spec.config.robot=='panda' else MujocoContact(spec.config,'cpu',MechanismSpec());self.irreversible_model_forwards=0;self.reset()
    def reset(self):self.base.reset();self.tick=-1;self.cached=None
    def get_state(self):return deepcopy(dict(version=self.VERSION,robot=self.spec.config.robot,mode=self.mode,base=self.base.get_state(),tick=self.tick,cached=self.cached))
    def restore(self,state):
        if (state['version'],state['robot'],state['mode'])!=(self.VERSION,self.spec.config.robot,self.mode):raise ValueError('model reference binding differs')
        self.base.restore(state['base']);self.tick=state['tick'];self.cached=deepcopy(state['cached'])
    def act(self,public):
        tick=int(np.rint(np.asarray(public['elapsed_seconds']).item()/self.spec.config.control_dt))
        if tick==self.tick:return self.cached.copy()
        if tick!=self.tick+1:raise ValueError('every public control tick required')
        base=self.base.act(public)
        if self.backend is None:self.cached=base.copy();self.tick=tick;return self.cached.copy()
        if len(public['tcp_position'])!=1:raise ValueError('explicit batch1')
        model=self.backend.model;data=mujoco.MjData(model);data.qpos[:]=np.asarray(public['joint_position'])[0];data.qvel[:]=np.asarray(public['joint_velocity'])[0]
        if self.charge:self.charge('nominal_model_forwards',1)
        mujoco.mj_forward(model,data);self.irreversible_model_forwards+=1;jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);mujoco.mj_jacSite(model,data,jp,jr,self.backend.sid);j=np.r_[jp,jr];issued=bounded_contact_command(np,base[0],np.asarray(public['previous_action'])[0],self.spec.config);ff=friction_wrench(j,issued[:6],model.dof_frictionloss);ff[2]=0.
        # Positive task-space damping plus integrated spring map. The angular
        # relation is first-order for the existing small per-tick rotation.
        stiffness=np.asarray([800.]*3+[60.]*3)*np.exp(issued[6:12]);coefficient=self.spec.config.control_dt*stiffness+2*np.sqrt(stiffness);value=base.copy();value[0,:6]+=ff/coefficient;value[0,:3]=np.clip(value[0,:3],-.08,.08);value[0,3:6]=np.clip(value[0,3:6],-.4,.4);value[0,2]=base[0,2];value[0,12]=0.;self.cached=value.astype(np.float32);self.tick=tick;return self.cached.copy()
