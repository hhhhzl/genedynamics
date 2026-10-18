"""Public T3 common geometry teacher with native ACP19/SPD execution.

This is a declared scripted task adaptation, not an author checkpoint or
learned ACP result. Source/nominal qualification precedes supervised data.
"""
from copy import deepcopy
import numpy as np
from genedynamics.core.control.transfer import integrate_rotation
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.bias_feedback import PublicBiasFeedbackSpec
from genedynamics.tasks.transfer.controllers.mating_feedback import PublicMatingFeedbackPolicy
from genedynamics.solvers.single.acp.backends import spatial
from genedynamics.solvers.single.acp.backends.compliance import VirtualTargetEstimator, decode_stiffness_tool

class PublicNativeACPFeedbackTeacher:
    VERSION='public-T3-common-geometry-native-ACP19-teacher-v1'
    def __init__(self,spec,*,mode,feedback,force_recipe):
        if spec.alias!='T3':raise ValueError('explicit T3 fixedarm teacher')
        self.spec=spec;self.mode=int(mode);self.feedback_spec=PublicBiasFeedbackSpec(**feedback);self.force_recipe=deepcopy(force_recipe)
        self.base=PublicMatingFeedbackPolicy.load(spec,None,feedback=self.feedback_spec,modes=[mode],seed=0);self.reset()
    def reset(self):
        self.base.reset(modes=[self.mode]);self.estimator=VirtualTargetEstimator(5000,200,.5,5,dim=3,characteristic_length=.02,vel_tol=999.002);self.integral=0.;self.tick=-1;self.cached=None;self.requests=[]
    def get_state(self):
        return deepcopy(dict(version=self.VERSION,robot=self.spec.config.robot,mode=self.mode,base=self.base.get_state(),estimator=vars(self.estimator),integral=self.integral,tick=self.tick,cached=self.cached,requests=self.requests))
    def restore(self,s):
        if (s['version'],s['robot'],s['mode'])!=(self.VERSION,self.spec.config.robot,self.mode):raise ValueError('teacherbinding mismatch')
        self.base.restore(s['base']);self.estimator.__dict__=deepcopy(s['estimator']);self.integral=s['integral'];self.tick=s['tick'];self.cached=deepcopy(s['cached']);self.requests=deepcopy(s['requests'])
    def act(self,public):
        tick=int(np.rint(np.asarray(public['elapsed_seconds']).item()/self.spec.config.control_dt))
        if tick==self.tick:return deepcopy(self.cached)
        if tick!=self.tick+1:raise ValueError('all public ticks required')
        action=self.base.act(public)[0];phase=int(self.base.observer.memory.phase[0]);p=public['tcp_position'][0];rot=public['tcp_rotation'][0]
        current=np.eye(4);current[:3,:3]=rot;current[:3,3]=p;rt=rot.T
        wrench=np.r_[rt@public['measured_wrench'][0,:3],rt@public['measured_wrench'][0,3:]]
        twist=np.r_[rt@public['tcp_twist'][0,:3],rt@public['tcp_twist'][0,3:]]
        scalar,_,_=self.estimator.update(wrench,twist)
        reference=np.eye(4);reference[:3,3]=public['reference_position'][0]+action[:3]*self.spec.config.control_dt;reference[:3,:3]=integrate_rotation(np,public['reference_rotation'][0],action[3:6],self.spec.config.control_dt);virtual=reference.copy()
        # Native ACP normal-force construction uses the shared *public* phase,
        # since the admittance shared13 policy deliberately requests force0.
        if phase in (0,3):
            error=self.spec.config.desired_force-float(public['measured_wrench'][0,2]);r=self.force_recipe
            self.integral=float(np.clip(self.integral+error*self.spec.config.control_dt,-r['force_integral_limit'],r['force_integral_limit']))
            force=float(np.clip(self.spec.config.desired_force+r['force_feedback_gain']*error+r['force_integral_gain']*self.integral,-r['force_command_limit'],r['force_command_limit']))
            reference[2,3]=current[2,3];virtual[2,3]=reference[2,3]-force/scalar
        else:self.integral=0.
        stiffness=decode_stiffness_tool(reference,virtual,scalar,current)
        self.requests.append(np.r_[spatial.SE3_to_pose9(reference),spatial.SE3_to_pose9(virtual),scalar].astype(np.float32))
        self.cached=dict(reference_pose_world=reference[None,None,None],virtual_target_pose_world=virtual[None,None,None],stiffness_tool=stiffness[None,None,None],waypoint_times_s=np.array([0.]),execution_horizon=1);self.tick=tick
        return deepcopy(self.cached)
