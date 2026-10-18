"""Native and rate-limited ACP demonstrators over the shared task interface."""
from types import SimpleNamespace

import numpy as np

from genedynamics.core.control.transfer import integrate_rotation
from genedynamics.solvers.single.acp.backends import spatial
from genedynamics.solvers.single.acp.backends.compliance import (
    VirtualTargetEstimator,
    decode_stiffness_tool,
)
from genedynamics.tasks.transfer.teachers import PublicTeacher


def world_pose(position, rotation):
    value = np.eye(4)
    value[:3, :3] = rotation
    value[:3, 3] = position
    return value


class SharedTeacher:
    def __init__(self, spec, mode):
        task = SimpleNamespace(spec=spec, config=spec.config)
        self.teacher = PublicTeacher(
            task, mode=mode, recipe="recovery_stiffness"
        )

    def act(self, public):
        return self.teacher.act(public)[None]


class NativeACPTeacher(SharedTeacher):
    """Adapt the public teacher through native anisotropic ACP stiffness."""

    def __init__(self, spec, mode, recipe):
        super().__init__(spec, mode)
        self.spec, self.recipe, self.integral = spec, recipe, 0.
        self.estimator = VirtualTargetEstimator(
            5000, 200, .5, 5, dim=3, characteristic_length=.02, vel_tol=999.002
        )
        self.requests = []

    def act(self, public):
        action = self.teacher.act(public)
        cfg = self.spec.config
        rp = public["reference_position"][0]
        rr = public["reference_rotation"][0]
        current = world_pose(public["tcp_position"][0], public["tcp_rotation"][0])
        rotation_tool = current[:3, :3].T
        wrench = np.r_[
            rotation_tool @ public["measured_wrench"][0, :3],
            rotation_tool @ public["measured_wrench"][0, 3:],
        ]
        twist = np.r_[
            rotation_tool @ public["tcp_twist"][0, :3],
            rotation_tool @ public["tcp_twist"][0, 3:],
        ]
        scalar, _, _ = self.estimator.update(wrench, twist)
        reference = world_pose(
            rp + action[:3] * cfg.control_dt,
            integrate_rotation(np, rr, action[3:6], cfg.control_dt),
        )
        virtual = reference.copy()
        if action[12] > .01:
            error = cfg.desired_force - float(public["measured_wrench"][0, 2])
            recipe = self.recipe
            self.integral = float(
                np.clip(
                    self.integral + error * cfg.control_dt,
                    -recipe["force_integral_limit"],
                    recipe["force_integral_limit"],
                )
            )
            force = np.clip(
                cfg.desired_force
                + recipe["force_feedback_gain"] * error
                + recipe["force_integral_gain"] * self.integral,
                -recipe["force_command_limit"],
                recipe["force_command_limit"],
            )
            reference[2, 3] = current[2, 3]
            virtual[2, 3] = reference[2, 3] - force / scalar
        else:
            self.integral = 0.
        stiffness = decode_stiffness_tool(reference, virtual, scalar, current)
        self.requests.append(
            np.r_[
                spatial.SE3_to_pose9(reference),
                spatial.SE3_to_pose9(virtual),
                scalar,
            ].astype(np.float32)
        )
        return {
            "reference_pose_world": reference[None, None, None],
            "virtual_target_pose_world": virtual[None, None, None],
            "stiffness_tool": stiffness[None, None, None],
            "waypoint_times_s": np.array([0.]),
            "execution_horizon": 1,
        }


class RateLimitedACPTeacher(NativeACPTeacher):
    def __init__(self,spec,mode,recipe):
        super().__init__(spec,mode,recipe)
        self.previous_force=0.;self.previous_scalar=5000.;self.previous_virtual=None
        self.limiter_receipts=[]

    def act(self,public):
        action=self.teacher.act(public);cfg=self.spec.config;dt=cfg.control_dt;recipe=self.recipe
        current=world_pose(public["tcp_position"][0],public["tcp_rotation"][0]);rt=current[:3,:3].T
        wrench=np.r_[rt@public["measured_wrench"][0,:3],rt@public["measured_wrench"][0,3:]]
        twist=np.r_[rt@public["tcp_twist"][0,:3],rt@public["tcp_twist"][0,3:]]
        requested_scalar,_,_=self.estimator.update(wrench,twist)
        log_step=recipe["log_stiffness_rate"]*dt
        scalar=float(np.exp(np.clip(np.log(requested_scalar),np.log(self.previous_scalar)-log_step,np.log(self.previous_scalar)+log_step)))
        reference=world_pose(public["reference_position"][0]+action[:3]*dt,
            integrate_rotation(np,public["reference_rotation"][0],action[3:6],dt));virtual=reference.copy()
        if action[12]>.01:
            error=cfg.desired_force-float(public["measured_wrench"][0,2])
            candidate=float(np.clip(self.integral+error*dt,-recipe["force_integral_limit"],recipe["force_integral_limit"]))
            raw_force=float(np.clip(cfg.desired_force+recipe["force_feedback_gain"]*error+recipe["force_integral_gain"]*candidate,
                -recipe["force_command_limit"],recipe["force_command_limit"]))
            force=float(np.clip(raw_force,self.previous_force-recipe["normal_force_rate"]*dt,self.previous_force+recipe["normal_force_rate"]*dt))
            if abs(force-raw_force)<1e-10:self.integral=candidate
            reference[2,3]=current[2,3];virtual[2,3]=reference[2,3]-force/scalar
        else:
            self.integral=0.;force=0.
        if self.previous_virtual is None:self.previous_virtual=current.copy()
        delta=virtual[:3,3]-self.previous_virtual[:3,3];length=float(np.linalg.norm(delta));maximum=recipe["virtual_linear_speed"]*dt
        virtual[:3,3]=self.previous_virtual[:3,3]+delta*min(1.,maximum/max(length,1e-12))
        # Pose orientation is already integrated at the shared .12rad/s cap;
        # retain the author's native anisotropic stiffness decoding.
        stiffness=decode_stiffness_tool(reference,virtual,scalar,current)
        self.requests.append(np.r_[spatial.SE3_to_pose9(reference),spatial.SE3_to_pose9(virtual),scalar].astype(np.float32))
        self.limiter_receipts.append(dict(force_setpoint=force,scalar_stiffness=scalar,virtual_translation_step=float(np.linalg.norm(virtual[:3,3]-self.previous_virtual[:3,3]))))
        self.previous_force=force;self.previous_scalar=scalar;self.previous_virtual=virtual.copy()
        return dict(reference_pose_world=reference[None,None,None],virtual_target_pose_world=virtual[None,None,None],
            stiffness_tool=stiffness[None,None,None],waypoint_times_s=np.array([0.]),execution_horizon=1)
