"""Contact-preserving public reference for the spatial dual-channel guide.

This task controller is an explicitly selected replacement for the historical
lift route.  It uses the public pose, wrench, and issued reference only.  The
normal-force loop stays active while each requested channel is traversed; its
contact latch is feedback, never the evaluator's private axial-support flag.
"""
from copy import deepcopy

import numpy as np

from genedynamics.core.control.transfer import orientation_error
from genedynamics.tasks.transfer.contact import bounded_contact_command
from .guide_wrench_projection import project_wrench_box_ball


class PublicGuideContactPolicy:
    LEGACY_VERSION = "public-T2-contact-preserving-reference-v1"
    LOW_TRANSIT_LOAD_VERSION = "public-T2-contact-preserving-reference-v2"
    WRENCH_GOVERNOR_VERSION = "public-T2-contact-preserving-reference-v3"
    FILTERED_WRENCH_GOVERNOR_VERSION = "public-T2-contact-preserving-reference-v4"
    RATE_WRENCH_GOVERNOR_VERSION = "public-T2-contact-preserving-reference-v5"
    LOAD_RATE_WRENCH_GOVERNOR_VERSION = "public-T2-contact-preserving-reference-v6"
    FILTERED_NORMAL_GOVERNOR_VERSION = "public-T2-contact-preserving-reference-v7"
    PERSISTENT_NORMAL_GOVERNOR_VERSION = "public-T2-contact-preserving-reference-v8"
    UNILATERAL_TRANSIT_GOVERNOR_VERSION = "public-T2-contact-preserving-reference-v9"
    ANISOTROPIC_TRANSIT_GOVERNOR_VERSION = "public-T2-contact-preserving-reference-v10"
    STAGES = ("establish_contact", "hold_contact", "first_axis", "outlet", "terminal")
    REFERENCE_LEAD_XY_M = .008
    TERMINAL_RADIUS_FRACTION = .75

    def __init__(
        self, spec, *, mode, transit_force_N=None, wrench_governor=False,
        governor_start_fraction=.4, governor_stop_fraction=.6,
        filtered_wrench_governor=True, governor_decay=.9,
        governor_recovery_per_tick=.05, tangent_speed_limit_mps=None,
        normal_force_decay=None, unilateral_transit=False,
    ):
        if spec.alias != "T2" or spec.config.robot not in {"panda", "xarm7"}:
            raise ValueError("contact reference requires fixed-arm T2")
        if spec.config.guide_profile != "spatial_dual_channel" or mode not in (0, 1):
            raise ValueError("contact reference requires spatial guide and mode 0/1")
        if transit_force_N is not None and (
            not np.isfinite(transit_force_N) or not .5 <= transit_force_N <= spec.config.desired_force
        ):
            raise ValueError("transit force must stay within the declared contact load range")
        fractions = (governor_start_fraction, governor_stop_fraction)
        if (not np.isfinite(fractions).all() or not
                0. < governor_start_fraction < governor_stop_fraction < 1.):
            raise ValueError("wrench governor needs a finite fractional safety band")
        if (not np.isfinite([governor_decay, governor_recovery_per_tick]).all()
                or not 0. < governor_decay < 1.
                or not 0. < governor_recovery_per_tick <= 1.):
            raise ValueError("filtered wrench governor needs finite memory and recovery")
        tangent_limits = None
        if tangent_speed_limit_mps is not None:
            tangent_limits = np.asarray(tangent_speed_limit_mps, dtype=float)
            if tangent_limits.ndim == 0:
                tangent_limits = np.repeat(tangent_limits, 2)
            if (tangent_limits.shape != (2,) or not np.isfinite(tangent_limits).all()
                    or np.any(tangent_limits < .005)
                    or np.any(tangent_limits > .02)):
                raise ValueError("tangent speed limit must preserve finite task reachability")
        if (normal_force_decay is not None and (
                not np.isfinite(normal_force_decay)
                or not 0. < normal_force_decay < 1.)):
            raise ValueError("normal force memory must be a strict finite decay")
        self.spec, self.mode = spec, int(mode)
        self.transit_force_N = None if transit_force_N is None else float(transit_force_N)
        self.wrench_governor = bool(wrench_governor)
        self.filtered_wrench_governor = bool(filtered_wrench_governor)
        self.governor_start_fraction = float(governor_start_fraction)
        self.governor_stop_fraction = float(governor_stop_fraction)
        self.governor_decay = float(governor_decay)
        self.governor_recovery_per_tick = float(governor_recovery_per_tick)
        self.tangent_speed_limit_mps = (
            None if tangent_limits is None else tangent_limits.copy()
        )
        self.normal_force_decay = (
            None if normal_force_decay is None else float(normal_force_decay)
        )
        self.unilateral_transit = bool(unilateral_transit)
        if self.wrench_governor:
            if self.unilateral_transit:
                self.version = (
                    self.ANISOTROPIC_TRANSIT_GOVERNOR_VERSION
                    if self.tangent_speed_limit_mps is not None
                    and self.tangent_speed_limit_mps[0] != self.tangent_speed_limit_mps[1]
                    else self.UNILATERAL_TRANSIT_GOVERNOR_VERSION
                )
            elif self.normal_force_decay is not None:
                self.version = (
                    self.PERSISTENT_NORMAL_GOVERNOR_VERSION
                    if self.normal_force_decay >= .9
                    else self.FILTERED_NORMAL_GOVERNOR_VERSION
                )
            elif (self.tangent_speed_limit_mps is not None
                    and self.transit_force_N is not None):
                self.version = self.LOAD_RATE_WRENCH_GOVERNOR_VERSION
            elif self.tangent_speed_limit_mps is not None:
                self.version = self.RATE_WRENCH_GOVERNOR_VERSION
            elif self.filtered_wrench_governor:
                self.version = self.FILTERED_WRENCH_GOVERNOR_VERSION
            else:
                self.version = self.WRENCH_GOVERNOR_VERSION
        elif transit_force_N is None:
            self.version = self.LEGACY_VERSION
        else:
            self.version = self.LOW_TRANSIT_LOAD_VERSION
        self.reset()

    def _tangent_scale(self, wrench):
        """Filtered public-wrench governor below the unchanged hard limits."""
        if not self.wrench_governor or self.stage not in (2, 3):
            self.wrench_utilization = 0.
            self.tangent_scale = 1.
            return 1.
        force = np.asarray(wrench, dtype=float)[:3]
        utilization = max(
            np.linalg.norm(force) / self.spec.config.force_limit,
            np.linalg.norm(force[:2]) / self.spec.config.lateral_force_limit,
        )
        span = self.governor_stop_fraction - self.governor_start_fraction
        if not self.filtered_wrench_governor:
            return float(np.clip(
                (self.governor_stop_fraction - utilization) / span, 0., 1.,
            ))
        self.wrench_utilization = max(
            utilization, self.governor_decay * self.wrench_utilization,
        )
        target = float(np.clip(
            (self.governor_stop_fraction - self.wrench_utilization) / span, 0., 1.,
        ))
        self.tangent_scale = min(
            target, self.tangent_scale + self.governor_recovery_per_tick,
        )
        return self.tangent_scale

    def reset(self):
        self.stage = 0
        self.tick = -1
        self.contact_ticks = 0
        self.first_contact_tick = None
        self.first_outlet_tick = None
        self.compensation = np.zeros((1, 6))
        self.wrench_utilization = 0.
        self.tangent_scale = 1.
        self.normal_force_envelope = 0.
        self.cached = None

    def get_state(self):
        value = dict(
            version=self.version, robot=self.spec.config.robot, mode=self.mode,
            stage=self.stage, tick=self.tick, contact_ticks=self.contact_ticks,
            first_contact_tick=self.first_contact_tick,
            first_outlet_tick=self.first_outlet_tick,
            compensation=self.compensation, cached=self.cached,
        )
        if self.wrench_governor:
            value["wrench_governor"] = dict(
                utilization=self.wrench_utilization, scale=self.tangent_scale,
                normal_force_envelope=self.normal_force_envelope,
            )
        return deepcopy(value)

    def restore(self, state):
        if (state["version"], state["robot"], state["mode"]) != (
            self.version, self.spec.config.robot, self.mode,
        ):
            raise ValueError("guide contact continuation binding differs")
        for name in (
            "stage", "tick", "contact_ticks", "first_contact_tick",
            "first_outlet_tick", "compensation", "cached",
        ):
            setattr(self, name, deepcopy(state[name]))
        if self.wrench_governor:
            governor = state["wrench_governor"]
            self.wrench_utilization = float(governor["utilization"])
            self.tangent_scale = float(governor["scale"])
            self.normal_force_envelope = float(
                governor.get("normal_force_envelope", 0.)
            )

    def _geometric_outlet_reached(self, position):
        return bool(
            position[0, 0] > .455
            if self.mode == 0 else position[0, 1] > -.181
        )

    def _outlet_ready(self, public, position, reference, goal):
        del public, reference, goal
        return self._geometric_outlet_reached(position)

    def act(self, public):
        dt = self.spec.config.control_dt
        elapsed = np.asarray(public["elapsed_seconds"])
        if elapsed.size != 1 or np.asarray(public["tcp_position"]).shape != (1, 3):
            raise ValueError("guide contact reference requires explicit batch 1")
        tick = int(np.rint(elapsed.item() / dt))
        if tick == self.tick:
            return self.cached.copy()
        if tick != self.tick + 1:
            raise ValueError("every public control tick required")
        p = np.asarray(public["tcp_position"])
        r = np.asarray(public["tcp_rotation"])
        rp = np.asarray(public["reference_position"])
        wrench = np.asarray(public["measured_wrench"])[0]
        force = float(wrench[2])
        goal = np.asarray(self.spec.config.goal)
        loaded = force > max(.5, self.spec.config.contact_threshold)
        self.contact_ticks = self.contact_ticks + 1 if loaded else 0
        if loaded and self.first_contact_tick is None:
            self.first_contact_tick = tick
        if self.stage == 0 and loaded:
            self.stage = 1
        if self.stage == 1 and self.contact_ticks >= int(np.ceil(.2 / dt)):
            self.stage = 2
            self.compensation[:] = 0.
        if self.stage == 2 and self._outlet_ready(public, p, rp, goal):
            self.stage = 3
            self.first_outlet_tick = tick
            self.compensation[:] = 0.
        if (self.stage == 3 and np.linalg.norm(p[0, :2] - goal[:2])
                < self.TERMINAL_RADIUS_FRACTION
                * self.spec.config.position_tolerance):
            self.stage = 4
            self.compensation[:] = 0.

        target = np.asarray([[.38, -.20, goal[2]]])
        axis = 0 if self.mode == 0 else 1
        if self.stage >= 2:
            target[:, axis] = goal[axis]
        if self.stage >= 3:
            target[:, :2] = goal[:2]
        desired = np.broadcast_to(np.diag([1., -1., -1.]), r.shape)
        angular = np.stack([orientation_error(np, a, b) for a, b in zip(r, desired)])
        issued_angular = np.stack([
            orientation_error(np, a, b)
            for a, b in zip(public["reference_rotation"], desired)
        ])
        error = np.concatenate((target - p, angular), axis=-1)
        issued = np.concatenate((target - rp, issued_angular), axis=-1)
        gains = np.asarray([6.] * 3 + [4.] * 3)
        limits = np.asarray([.02] * 3 + [.12] * 3)
        raw = gains * error + 2. * issued - np.asarray(public["tcp_twist"]) + self.compensation
        increment = 2. * dt * error
        integrate = (np.abs(raw) < limits) | (raw * increment <= 0.)
        if self.tangent_speed_limit_mps is not None:
            integrate[:, :2] &= (
                (np.abs(raw[:, :2]) < self.tangent_speed_limit_mps)
                | (raw[:, :2] * increment[:, :2] <= 0.)
            )
        tangent_scale = self._tangent_scale(wrench)
        if tangent_scale < 1.:
            integrate[:, :2] &= raw[:, :2] * increment[:, :2] <= 0.
        self.compensation = np.clip(self.compensation + np.where(integrate, increment, 0.), -.04, .04)
        self.compensation[:, 2] = 0.
        velocity = gains * error + 2. * issued - np.asarray(public["tcp_twist"]) + self.compensation
        velocity = np.clip(velocity, -limits, limits)
        if self.tangent_speed_limit_mps is not None:
            velocity[:, :2] = np.clip(
                velocity[:, :2], -self.tangent_speed_limit_mps,
                self.tangent_speed_limit_mps,
            )
        velocity[:, :2] *= tangent_scale

        # The known surface rises in both tangent directions. Its feedforward
        # term accompanies measured-force admittance and never bypasses it.
        slope_velocity = .1 * np.sum(velocity[:, :2], axis=1)
        transit_force = self.spec.config.desired_force if self.transit_force_N is None else self.transit_force_N
        desired_force = transit_force if 1 <= self.stage <= 3 else self.spec.config.desired_force
        effective_force = force
        if self.normal_force_decay is not None and 1 <= self.stage <= 3:
            self.normal_force_envelope = max(
                force, self.normal_force_decay * self.normal_force_envelope,
            )
            effective_force = self.normal_force_envelope
        elif self.stage not in (1, 2, 3):
            self.normal_force_envelope = 0.
        normal_velocity = .004 * (effective_force - desired_force) + slope_velocity
        if self.unilateral_transit and self.stage in (2, 3):
            normal_velocity = np.maximum(normal_velocity, slope_velocity)
        velocity[:, 2] = np.clip(normal_velocity, -.01, .01)
        lead = np.asarray([
            self.REFERENCE_LEAD_XY_M,
            self.REFERENCE_LEAD_XY_M,
            .01,
        ])
        velocity[:, :3] = np.clip(velocity[:, :3], (p - lead - rp) / dt, (p + lead - rp) / dt)
        rotation_lead = np.stack([
            orientation_error(np, a, b)
            for a, b in zip(r, public["reference_rotation"])
        ])
        velocity[:, 3:] = np.clip(velocity[:, 3:], (-.04 - rotation_lead) / dt, (.04 - rotation_lead) / dt)
        velocity = np.clip(velocity, -limits, limits)
        stiffness = np.broadcast_to([1., 1., 0., 1., 1., 1.], (1, 6))
        self.cached = np.concatenate((velocity, stiffness, np.zeros((1, 1))), axis=1).astype(np.float32)
        self.tick = tick
        return self.cached.copy()


class PublicCornerGuidePolicy:
    """Public pose/wrench reference for the semantic right-angle guide."""
    VERSION="public-corner-guided-seating-v1"
    STAGES=("establish_contact","first_segment","corner_settle","second_segment","terminal")

    def __init__(self,spec,*,mode=0,authority_growth_per_control=.02,
                 nominal_lead_m=.008,maximum_contact_speed_mps=.03):
        if (spec.name!="corner_guided_seating" or
                spec.config.guide_profile!="planar_corner_v1" or mode!=0):
            raise ValueError("corner guide policy requires semantic mode 0")
        self.spec,self.mode=spec,int(mode)
        # Frozen T2 response actors read this public-planner field when they
        # construct their task-reference chart.  The corner route uses the
        # same declared surface-force target throughout its transit stages.
        self.transit_force_N=spec.config.desired_force
        self.authority_growth_per_control=float(authority_growth_per_control)
        self.nominal_lead_m=float(nominal_lead_m)
        self.maximum_contact_speed_mps=float(maximum_contact_speed_mps)
        if (not .01<=self.authority_growth_per_control<=.04
                or not .006<=self.nominal_lead_m<=.01
                or not .02<=self.maximum_contact_speed_mps<=.04):
            raise ValueError("corner public policy parameters outside declared T2 support")
        self.reset()

    def reset(self):
        self.stage=0;self.tick=-1;self.contact_ticks=0;self.cached=None;self.stage_entry_tick=0
        self.compensation=np.zeros((1,6))

    def get_state(self):
        return deepcopy(dict(version=self.VERSION,stage=self.stage,tick=self.tick,
                             authority_growth_per_control=self.authority_growth_per_control,
                             nominal_lead_m=self.nominal_lead_m,
                             maximum_contact_speed_mps=self.maximum_contact_speed_mps,
                             contact_ticks=self.contact_ticks,stage_entry_tick=self.stage_entry_tick,compensation=self.compensation,
                             cached=self.cached))

    def restore(self,state):
        binding=(
            state.get("version"),
            state.get("authority_growth_per_control"),
            state.get("nominal_lead_m"),
            state.get("maximum_contact_speed_mps"),
        )
        expected=(
            self.VERSION,
            self.authority_growth_per_control,
            self.nominal_lead_m,
            self.maximum_contact_speed_mps,
        )
        if binding!=expected:raise ValueError("corner guide continuation differs")
        for name in ("stage","tick","contact_ticks","stage_entry_tick","compensation","cached"):
            setattr(self,name,deepcopy(state[name]))

    def reorder(self,indices):
        if np.asarray(indices).tolist()!=[0]:raise ValueError("corner source reference is batch-one")

    def act(self,public):
        dt=self.spec.config.control_dt
        elapsed=np.asarray(public["elapsed_seconds"])
        if elapsed.size!=1 or np.asarray(public["tcp_position"]).shape!=(1,3):
            raise ValueError("corner source reference requires explicit batch one")
        tick=int(np.rint(elapsed.item()/dt))
        if tick==self.tick:return self.cached.copy()
        if tick!=self.tick+1:raise ValueError("every public control tick required")
        p=np.asarray(public["tcp_position"],float);r=np.asarray(public["tcp_rotation"],float)
        rp=np.asarray(public["reference_position"],float)
        force=float(np.asarray(public["measured_wrench"])[0,2])
        loaded=force>max(.5,self.spec.config.contact_threshold)
        self.contact_ticks=self.contact_ticks+1 if loaded else 0
        if self.stage==0 and self.contact_ticks>=round(.2/dt):
            self.stage=1;self.stage_entry_tick=tick
        if self.stage==1 and p[0,0]>=.436 and -.211<=p[0,1]<=-.189:
            self.stage=2;self.stage_entry_tick=tick;self.compensation[:]=0.
        goal=np.asarray(self.spec.config.goal,float)
        if self.stage==2 and tick-self.stage_entry_tick>=round(.3/dt) and p[0,0]>=.439:
            self.stage=3;self.stage_entry_tick=tick;self.compensation[:]=0.
        if self.stage==3 and np.linalg.norm(p[0,:2]-goal[:2])<.75*self.spec.config.position_tolerance:
            self.stage=4;self.stage_entry_tick=tick;self.compensation[:]=0.
        target=np.asarray([[.38,-.20,goal[2]]],float)
        if self.stage in (1,2):target[0,0]=.44
        elif self.stage>=3:target[0,:2]=goal[:2]
        desired=np.broadcast_to(np.diag([1.,-1.,-1.]),r.shape)
        angular=np.stack([orientation_error(np,a,b) for a,b in zip(r,desired)])
        issued_angular=np.stack([orientation_error(np,a,b) for a,b in zip(public["reference_rotation"],desired)])
        error=np.concatenate((target-p,angular),axis=-1)
        issued=np.concatenate((target-rp,issued_angular),axis=-1)
        twist=np.asarray(public["tcp_twist"],float)
        gains=np.asarray([6.,6.,6.,4.,4.,4.])
        tangent_scale=self.maximum_contact_speed_mps/.03
        limits=np.asarray([.0045*tangent_scale,.0045*tangent_scale,.01,.12,.12,.12])
        raw=gains*error+2.*issued-twist+self.compensation
        increment=(self.authority_growth_per_control/.02)*2.*dt*error
        integrate=(np.abs(raw)<limits)|(raw*increment<=0.)
        self.compensation=np.clip(self.compensation+np.where(integrate,increment,0.),-.025,.025)
        self.compensation[:,2]=0.
        velocity=np.clip(gains*error+2.*issued-twist+self.compensation,-limits,limits)
        normal_velocity=.004*(force-self.spec.config.desired_force)
        if self.stage==0 and not loaded:
            normal_velocity=max(normal_velocity,-.003)
        velocity[:,2]=np.clip(normal_velocity,-.006,.006)
        lead=np.asarray([self.nominal_lead_m,self.nominal_lead_m,.01])
        velocity[:,:3]=np.clip(velocity[:,:3],(p-lead-rp)/dt,(p+lead-rp)/dt)
        rotation_lead=np.stack([orientation_error(np,a,b) for a,b in zip(r,public["reference_rotation"])])
        velocity[:,3:]=np.clip(velocity[:,3:],(-.04-rotation_lead)/dt,(.04-rotation_lead)/dt)
        stiffness=np.broadcast_to([.5,.5,0.,1.,1.,1.],(1,6))
        self.cached=np.concatenate((velocity,stiffness,
                                    np.full((1,1),self.spec.config.desired_force)),axis=1).astype(np.float32)
        self.tick=tick
        return self.cached.copy()


def bind_corner_public_planner(actor, spec, *, mode):
    """Install the semantic corner planner under a frozen T2 method actor.

    The learned response, transport, or online optimizer is retained.  The
    right-angle route and contact schedule are equally public task geometry,
    so every admitted method receives this same planner.  This is a declared
    development task adaptation; it does not claim that the original T2
    training covered the corner geometry.
    """
    if spec.name != "corner_guided_seating" or int(mode) != 0:
        raise ValueError("semantic corner-guided seating mode 0 required")
    if not hasattr(actor, "base"):
        raise ValueError("T2 method actor has no task-owned public planner")
    actor.spec = spec
    actor.base = PublicCornerGuidePolicy(spec, mode=mode)
    return PublicCornerGuideMethodExecution(actor)


def adapt_corner_public_policy(actor, spec, *, mode):
    """Map an admitted ASID/AdaptSim T2 gain policy onto corner geometry.

    The original actor must first pass its exact T2 artifact and training
    checks on a construction-only surrogate.  This adapter retains only its
    three declared public policy coordinates; route geometry, force target,
    stage transitions, and evaluation remain task-shared.
    """
    names=("AUTHORITY_GROWTH_PER_CONTROL","NOMINAL_LEAD_M",
           "MAXIMUM_CONTACT_SPEED_MPS")
    if any(not hasattr(actor,name) for name in names):
        raise ValueError("admitted T2 parameter policy required")
    policy=PublicCornerGuidePolicy(
        spec,mode=mode,
        authority_growth_per_control=getattr(actor,names[0]),
        nominal_lead_m=getattr(actor,names[1]),
        maximum_contact_speed_mps=getattr(actor,names[2]),
    )
    return PublicCornerParameterizedExecution(actor,policy)


class PublicCornerParameterizedExecution:
    """Task-adapt an admitted three-parameter T2 policy to the corner route."""

    VERSION="public-corner-parameterized-execution-v1"

    def __init__(self,source_actor,policy):
        self.source_actor=source_actor
        self.policy=policy

    def act(self,public):
        return self.policy.act(public)

    def reset(self):
        return self.policy.reset()

    def get_state(self):
        return {"version":self.VERSION,"policy":self.policy.get_state(),
                "source_actor_type":type(self.source_actor).__name__}


class PublicCornerGuideMethodExecution:
    """Preserve the frozen response chart around the new corner planner."""

    VERSION = "public-corner-guide-method-execution-v1"

    def __init__(self, actor):
        self.actor = actor
        self.base = actor.base

    def act(self, public):
        from genedynamics.solvers.single.cross_ebdm.learning.response import (
            PUBLIC_FIELDS,
        )
        return self.actor.act({key: public[key] for key in PUBLIC_FIELDS})

    def reset(self):
        return self.actor.reset()

    def get_state(self):
        return {
            "version": self.VERSION,
            "actor": self.actor.get_state() if hasattr(self.actor, "get_state") else None,
        }


class PublicPredictiveGuideContactPolicy(PublicGuideContactPolicy):
    """Strict-contact route with a public one-step tangent-wrench budget.

    The historical contact route changes tangent direction as soon as the TCP
    crosses an outlet plane.  A force sample taken at the preceding 20 ms
    control tick cannot see the contact impulse generated during the next ten
    2 ms physics steps.  This controller instead comes to a measured stop at
    the first-axis endpoint before starting the second axis, and constrains the
    next issued reference velocity with the declared nominal impedance law.

    Only public pose, twist, wrench, issued reference, and previous issued
    action enter.  The projection is a controller-side budget, not a physical
    force certificate and not access to evaluator contact truth.
    """

    VERSION = "public-T2-predictive-contact-path-v1"
    TANGENT_WRENCH_BUDGET_N = 6.0
    PREDICTION_NUMERICAL_MARGIN_N = 1e-5
    TANGENT_STIFFNESS_SCALE = 1.0
    TRANSIT_FORCE_N = 5.0
    NORMAL_ADMITTANCE_GAIN = 0.004
    NORMAL_UNLOADING_LIMIT_MPS = 0.010
    NORMAL_FORCE_FILTER_DECAY = None
    TRANSIT_HYBRID_FORCE_N = None
    SURFACE_LEAD_UNLOADING_GAIN = 0.0
    TRANSIT_SPEED_MPS = 0.012
    CORRECTION_SPEED_MPS = 0.004
    REFERENCE_LEAD_XY_M = 0.008
    ENDPOINT_TOLERANCE_M = 0.0015
    ENDPOINT_SPEED_MPS = 0.003

    def __init__(self, spec, *, mode):
        super().__init__(spec, mode=mode)
        self.version = self.VERSION

    def reset(self):
        super().reset()
        self.projection_active_ticks = 0
        self.projection_infeasible_ticks = 0
        self.stage_entry_tick = 0
        self.filtered_normal_force = 0.0

    def get_state(self):
        value = super().get_state()
        value["projection"] = {
            "active_ticks": self.projection_active_ticks,
            "infeasible_ticks": self.projection_infeasible_ticks,
            "stage_entry_tick": self.stage_entry_tick,
            "filtered_normal_force": self.filtered_normal_force,
        }
        return value

    def restore(self, state):
        super().restore(state)
        projection = state["projection"]
        self.projection_active_ticks = int(projection["active_ticks"])
        self.projection_infeasible_ticks = int(projection["infeasible_ticks"])
        self.stage_entry_tick = int(projection.get("stage_entry_tick", 0))
        self.filtered_normal_force = float(
            projection.get("filtered_normal_force", 0.0)
        )

    def _advance_stage(self, p, twist, reference_lead, force, tick):
        loaded = force > max(.5, self.spec.config.contact_threshold)
        self.contact_ticks = self.contact_ticks + 1 if loaded else 0
        if loaded and self.first_contact_tick is None:
            self.first_contact_tick = tick
        if self.stage == 0 and loaded:
            self.stage = 1
            self.stage_entry_tick = tick
        if self.stage == 1 and self.contact_ticks >= int(np.ceil(.2 / self.spec.config.control_dt)):
            self.stage = 2
            self.stage_entry_tick = tick
            self.compensation[:] = 0.
        axis = self.mode
        goal = np.asarray(self.spec.config.goal)
        outlet = p[0] > .455 if self.mode == 0 else p[1] > -.181
        stopped = (
            abs(p[axis] - goal[axis]) <= self.ENDPOINT_TOLERANCE_M
            and abs(twist[axis]) <= self.ENDPOINT_SPEED_MPS
            and abs(reference_lead[axis]) <= 2. * self.ENDPOINT_TOLERANCE_M
        )
        if self.stage == 2 and outlet and stopped:
            self.stage = 3
            self.stage_entry_tick = tick
            self.first_outlet_tick = tick
            self.compensation[:] = 0.
        if self.stage == 3 and np.linalg.norm(p[:2] - goal[:2]) < .75 * self.spec.config.position_tolerance:
            self.stage = 4
            self.stage_entry_tick = tick
            self.compensation[:] = 0.

    def _tangent_wrench_budget(self):
        return self.TANGENT_WRENCH_BUDGET_N

    def _tangent_command(self, public, target):
        dt = self.spec.config.control_dt
        p = np.asarray(public["tcp_position"], dtype=float)[0]
        rp = np.asarray(public["reference_position"], dtype=float)[0]
        twist = np.asarray(public["tcp_twist"], dtype=float)[0, :2]
        previous = np.asarray(public["previous_action"], dtype=float)[0]
        error = target[:2] - p[:2]
        issued_error = target[:2] - rp[:2]
        desired = 4. * error + 1.5 * issued_error - twist

        active_axis = self.mode if self.stage == 2 else 1 - self.mode
        speed = np.full(2, self.CORRECTION_SPEED_MPS)
        if self.stage in (2, 3):
            speed[active_axis] = self.TRANSIT_SPEED_MPS
        elif self.stage == 4:
            speed[:] = .006
        else:
            speed[:] = self.CORRECTION_SPEED_MPS
        rate = .4 * dt
        # ``bounded_contact_command`` applies amplitude clipping before its
        # slew limit.  Transform the requested speed box through that exact
        # public operation so the interval remains ordered even when the
        # previous command lies just outside the new phase's smaller box.
        low = previous[:2] + np.clip(-speed - previous[:2], -rate, rate)
        high = previous[:2] + np.clip(speed - previous[:2], -rate, rate)

        stiffness_scale = self._tangent_stiffness_scale()
        stiffness = 800. * stiffness_scale
        damping = 2. * np.sqrt(stiffness)
        lead = rp[:2] - p[:2]
        offset = stiffness * lead - damping * twist
        coefficient = dt * stiffness + damping
        desired_wrench = offset + coefficient * np.clip(desired, low, high)
        projected_wrench, feasible = project_wrench_box_ball(
            desired_wrench,
            offset + coefficient * low,
            offset + coefficient * high,
            radius=(self._tangent_wrench_budget()
                    - self.PREDICTION_NUMERICAL_MARGIN_N),
        )
        velocity = np.clip((projected_wrench - offset) / coefficient, low, high)
        requested = np.clip(desired, low, high)
        if not np.allclose(velocity, requested, rtol=0., atol=1e-12):
            self.projection_active_ticks += 1
        if not feasible:
            self.projection_infeasible_ticks += 1
        return velocity

    def _tangent_stiffness_scale(self):
        return np.full(2, self.TANGENT_STIFFNESS_SCALE, dtype=float)

    def _tangent_lead_limits(self):
        return np.full(2, self.REFERENCE_LEAD_XY_M, dtype=float)

    def _target_for_stage(self, goal):
        target = np.asarray([.38, -.20, goal[2]], dtype=float)
        if self.stage >= 2:
            target[self.mode] = goal[self.mode]
        if self.stage >= 3:
            target[:2] = goal[:2]
        return target

    def act(self, public):
        dt = self.spec.config.control_dt
        elapsed = np.asarray(public["elapsed_seconds"])
        if (elapsed.size != 1 or np.asarray(public["tcp_position"]).shape != (1, 3)
                or np.asarray(public["previous_action"]).shape != (1, 13)):
            raise ValueError("predictive guide reference requires explicit batch 1 public state")
        tick = int(np.rint(elapsed.item() / dt))
        if tick == self.tick:
            return self.cached.copy()
        if tick != self.tick + 1:
            raise ValueError("every public control tick required")

        p = np.asarray(public["tcp_position"], dtype=float)[0]
        r = np.asarray(public["tcp_rotation"], dtype=float)
        rp = np.asarray(public["reference_position"], dtype=float)[0]
        twist = np.asarray(public["tcp_twist"], dtype=float)[0]
        wrench = np.asarray(public["measured_wrench"], dtype=float)[0]
        self._advance_stage(p, twist, rp - p, float(wrench[2]), tick)

        goal = np.asarray(self.spec.config.goal, dtype=float)
        target = self._target_for_stage(goal)
        tangent = self._tangent_command(public, target)

        desired_rotation = np.broadcast_to(np.diag([1., -1., -1.]), r.shape)
        angular = np.stack([
            orientation_error(np, current, desired)
            for current, desired in zip(r, desired_rotation)
        ])
        issued_angular = np.stack([
            orientation_error(np, current, desired)
            for current, desired in zip(public["reference_rotation"], desired_rotation)
        ])
        angular_velocity = np.clip(
            4. * angular + 2. * issued_angular
            - np.asarray(public["tcp_twist"], dtype=float)[:, 3:],
            -.12, .12,
        )

        slope_velocity = .1 * float(np.sum(tangent))
        measured_normal_force = float(wrench[2])
        if self.NORMAL_FORCE_FILTER_DECAY is not None:
            decay = self.NORMAL_FORCE_FILTER_DECAY
            self.filtered_normal_force = (
                decay * self.filtered_normal_force
                + (1.0 - decay) * measured_normal_force
            )
            measured_normal_force = self.filtered_normal_force
        desired_force = (
            self.TRANSIT_FORCE_N
            if self.stage in (2, 3)
            else self.spec.config.desired_force
        )
        normal_velocity = np.clip(
            self.NORMAL_ADMITTANCE_GAIN * (measured_normal_force - desired_force)
            + slope_velocity,
            -.01, self.NORMAL_UNLOADING_LIMIT_MPS,
        )
        if self.stage in (2, 3) and self.SURFACE_LEAD_UNLOADING_GAIN:
            reference_lead = rp - p
            surface_consistent_z_lead = .1 * float(
                np.sum(reference_lead[:2])
            )
            normal_velocity += np.clip(
                self.SURFACE_LEAD_UNLOADING_GAIN
                * (surface_consistent_z_lead - reference_lead[2]),
                0.0,
                0.008,
            )
            normal_velocity = float(np.clip(normal_velocity, -.01, .01))
        velocity = np.concatenate((tangent, [normal_velocity]))[None]
        lead = np.r_[self._tangent_lead_limits(), .01]
        velocity = np.clip(velocity, (p - lead - rp) / dt, (p + lead - rp) / dt)
        rotation_lead = np.stack([
            orientation_error(np, current, issued)
            for current, issued in zip(r, public["reference_rotation"])
        ])
        angular_velocity = np.clip(
            angular_velocity,
            (-.04 - rotation_lead) / dt,
            (.04 - rotation_lead) / dt,
        )
        tangent_log_stiffness = np.log(self._tangent_stiffness_scale())
        stiffness = np.broadcast_to(
            np.r_[tangent_log_stiffness, 0., 1., 1., 1.],
            (1, 6),
        )
        force = np.zeros((1, 1), dtype=float)
        if self.stage in (2, 3) and self.TRANSIT_HYBRID_FORCE_N is not None:
            force[:] = self.TRANSIT_HYBRID_FORCE_N
        requested = np.concatenate(
            (velocity, angular_velocity, stiffness, force), axis=1,
        ).astype(np.float32)
        # Keep the predictor and executor on the same declared issued command.
        self.cached = bounded_contact_command(
            np, requested[0], np.asarray(public["previous_action"])[0], self.spec.config,
        )[None].astype(np.float32)
        self.tick = tick
        return self.cached.copy()


class PublicProgressQualifiedGuideContactPolicy(
    PublicPredictiveGuideContactPolicy
):
    """Predictive route with the previously qualified 9 N progress budget."""

    VERSION = "public-T2-predictive-progress-qualified-v2"
    TANGENT_WRENCH_BUDGET_N = 9.0


class PublicWrenchGovernedGuideContactPolicy(
    PublicProgressQualifiedGuideContactPolicy
):
    """Make the predictive wrench projection the active tangent governor."""

    VERSION = "public-T2-wrench-governed-contact-v3"
    REFERENCE_LEAD_XY_M = 0.012


class PublicLowLoadWrenchGovernedGuideContactPolicy(
    PublicPredictiveGuideContactPolicy
):
    """Predictive route using the qualified low-load guide traversal regime."""

    VERSION = "public-T2-low-load-wrench-governed-contact-v4"
    TANGENT_WRENCH_BUDGET_N = 12.0
    TANGENT_STIFFNESS_SCALE = float(np.e)
    TRANSIT_FORCE_N = 2.0


class PublicLowLoadPredictiveGuideContactPolicy(
    PublicLowLoadWrenchGovernedGuideContactPolicy
):
    """Low-load traversal with a strict 9 N predictive tangent budget."""

    VERSION = "public-T2-low-load-predictive-contact-v5"
    TANGENT_WRENCH_BUDGET_N = 9.0


class PublicAnisotropicPredictiveGuideContactPolicy(
    PublicPredictiveGuideContactPolicy
):
    """Low-impact transit with stiff transverse guide-channel centering."""

    VERSION = "public-T2-anisotropic-predictive-contact-v6"
    TANGENT_WRENCH_BUDGET_N = 9.0
    TRANSIT_FORCE_N = 2.0

    def _active_tangent_axis(self):
        if self.stage == 2:
            return self.mode
        if self.stage == 3:
            return 1 - self.mode
        return None

    def _tangent_stiffness_scale(self):
        scale = np.full(2, float(np.e), dtype=float)
        active = self._active_tangent_axis()
        if active is None:
            scale[:] = 1.0
        else:
            scale[active] = 1.0
        return scale

    def _tangent_lead_limits(self):
        lead = np.full(2, 0.008, dtype=float)
        active = self._active_tangent_axis()
        if active is not None:
            lead[active] = 0.012
        return lead


class PublicSurfaceConsistentGuideContactPolicy(
    PublicAnisotropicPredictiveGuideContactPolicy
):
    """Keep tangent reference lead consistent with the known rising surface."""

    VERSION = "public-T2-surface-consistent-contact-v7"
    SURFACE_LEAD_UNLOADING_GAIN = 4.0


class PublicChannelCenteredGuideContactPolicy(
    PublicProgressQualifiedGuideContactPolicy
):
    """Low-load low-impact traversal centered in the known narrow channel."""

    VERSION = "public-T2-channel-centered-contact-v8"
    TRANSIT_FORCE_N = 2.0
    TRANSIT_SPEED_MPS = 0.020
    REFERENCE_LEAD_XY_M = 0.012
    MODE0_CHANNEL_CENTER_Y_M = -0.202

    def _target_for_stage(self, goal):
        target = super()._target_for_stage(goal)
        if self.stage == 2 and self.mode == 0:
            target[1] = self.MODE0_CHANNEL_CENTER_Y_M
        return target


class PublicSettledContactRampGuideContactPolicy(
    PublicChannelCenteredGuideContactPolicy
):
    """Ramp tangent effort after each measured contact-route transition."""

    VERSION = "public-T2-settled-contact-ramp-v9"
    INITIAL_TANGENT_WRENCH_BUDGET_N = 4.0
    TANGENT_WRENCH_RAMP_SECONDS = 2.0

    def _tangent_wrench_budget(self):
        if self.stage not in (2, 3):
            return self.TANGENT_WRENCH_BUDGET_N
        elapsed = max(
            0.0,
            (self.tick + 1 - self.stage_entry_tick)
            * self.spec.config.control_dt,
        )
        fraction = min(1.0, elapsed / self.TANGENT_WRENCH_RAMP_SECONDS)
        return (
            self.INITIAL_TANGENT_WRENCH_BUDGET_N
            + fraction * (
                self.TANGENT_WRENCH_BUDGET_N
                - self.INITIAL_TANGENT_WRENCH_BUDGET_N
            )
        )


class PublicLowLoadSettledGuideContactPolicy(
    PublicSettledContactRampGuideContactPolicy
):
    """Settled tangent ramp with a one-newton contact traversal target."""

    VERSION = "public-T2-low-load-settled-contact-v10"
    TRANSIT_FORCE_N = 1.0


class PublicOutletContactPolicy(PublicSettledContactRampGuideContactPolicy):
    """Contact prefix used only until the frozen strict-contact outlet."""

    VERSION = "public-T2-outlet-contact-prefix-v11"
    TRANSIT_FORCE_N = 0.75
    MODE0_CHANNEL_CENTER_Y_M = -0.201


class PublicDampedOutletContactPolicy(PublicOutletContactPolicy):
    """Suppress normal contact loss/reimpact during the outlet prefix."""

    VERSION = "public-T2-damped-outlet-contact-prefix-v12"
    NORMAL_ADMITTANCE_GAIN = 0.001


class PublicBoundedUnloadingOutletContactPolicy(PublicDampedOutletContactPolicy):
    """Prevent normal unloading from breaking established guide contact."""

    VERSION = "public-T2-bounded-unloading-outlet-prefix-v13"
    NORMAL_UNLOADING_LIMIT_MPS = 0.0015


class PublicFilteredNormalOutletContactPolicy(PublicOutletContactPolicy):
    """Filter public normal force before the traversal admittance loop."""

    VERSION = "public-T2-filtered-normal-outlet-prefix-v14"
    NORMAL_FORCE_FILTER_DECAY = 0.9


class PublicHybridNormalOutletContactPolicy(PublicOutletContactPolicy):
    """Use the existing 2 ms hybrid force channel after contact settles."""

    VERSION = "public-T2-hybrid-normal-outlet-prefix-v15"
    TRANSIT_HYBRID_FORCE_N = 0.75


class PublicPostOutletLiftGuideContactPolicy:
    """Contact-qualified first axis followed by the existing public lift route."""

    VERSION = "public-T2-post-outlet-lift-route-v11"

    def __init__(self, spec, *, mode):
        from .guide_lift_route import PublicGuideLiftRoutePolicy

        self.spec = spec
        self.mode = int(mode)
        self.base = PublicOutletContactPolicy(spec, mode=mode)
        self.candidate = PublicGuideLiftRoutePolicy(spec, mode=mode)
        self.reset()

    def reset(self):
        self.base.reset()
        self.candidate.reset()
        self.switched = False
        self.switch_tick = None
        self.tick = -1
        self.cached = None

    def get_state(self):
        return deepcopy({
            "version": self.VERSION,
            "robot": self.spec.config.robot,
            "mode": self.mode,
            "switched": self.switched,
            "switch_tick": self.switch_tick,
            "tick": self.tick,
            "cached": self.cached,
            "base": self.base.get_state(),
            "candidate": self.candidate.get_state(),
        })

    def restore(self, state):
        identity = (state["version"], state["robot"], state["mode"])
        expected = (self.VERSION, self.spec.config.robot, self.mode)
        if identity != expected:
            raise ValueError("post-outlet lift route binding differs")
        self.switched = bool(state["switched"])
        self.switch_tick = state["switch_tick"]
        self.tick = int(state["tick"])
        self.cached = deepcopy(state["cached"])
        self.base.restore(state["base"])
        self.candidate.restore(state["candidate"])

    def act(self, public):
        dt = self.spec.config.control_dt
        tick = int(np.rint(np.asarray(public["elapsed_seconds"]).item() / dt))
        if tick == self.tick:
            return self.cached.copy()
        if tick != self.tick + 1:
            raise ValueError("every public control tick required")
        p = np.asarray(public["tcp_position"], dtype=float)[0]
        outlet = p[0] > .455 if self.mode == 0 else p[1] > -.181
        if not self.switched and outlet:
            self.switched = True
            self.switch_tick = tick
        if self.switched:
            local = dict(public)
            local["elapsed_seconds"] = np.asarray(
                [[(tick - self.switch_tick) * dt]], dtype=float,
            )
            self.cached = self.candidate.act(local)
        else:
            self.cached = self.base.act(public)
        self.tick = tick
        return self.cached.copy()


class PublicDampedPostOutletLiftGuideContactPolicy(
    PublicPostOutletLiftGuideContactPolicy
):
    """Post-outlet lift route with the damped contact-qualified prefix."""

    VERSION = "public-T2-damped-post-outlet-lift-route-v12"

    def __init__(self, spec, *, mode):
        super().__init__(spec, mode=mode)
        self.base = PublicDampedOutletContactPolicy(spec, mode=mode)
        self.reset()


class PublicBoundedUnloadingPostOutletLiftGuideContactPolicy(
    PublicPostOutletLiftGuideContactPolicy
):
    """Post-outlet route with bounded contact-prefix unloading."""

    VERSION = "public-T2-bounded-unloading-post-outlet-route-v13"

    def __init__(self, spec, *, mode):
        super().__init__(spec, mode=mode)
        self.base = PublicBoundedUnloadingOutletContactPolicy(spec, mode=mode)
        self.reset()


class PublicFilteredNormalPostOutletLiftGuideContactPolicy(
    PublicPostOutletLiftGuideContactPolicy
):
    """Post-outlet route with filtered normal-force contact prefix."""

    VERSION = "public-T2-filtered-normal-post-outlet-route-v14"

    def __init__(self, spec, *, mode):
        super().__init__(spec, mode=mode)
        self.base = PublicFilteredNormalOutletContactPolicy(spec, mode=mode)
        self.reset()


class PublicHybridNormalPostOutletLiftGuideContactPolicy(
    PublicPostOutletLiftGuideContactPolicy
):
    """Post-outlet route with a physics-rate normal-force prefix."""

    VERSION = "public-T2-hybrid-normal-post-outlet-route-v15"

    def __init__(self, spec, *, mode):
        super().__init__(spec, mode=mode)
        self.base = PublicHybridNormalOutletContactPolicy(spec, mode=mode)
        self.reset()


class PublicFiniteStateGuideContactMode0Policy(PublicGuideContactPolicy):
    """Low-load mode-0 route qualified with the shared substep state machine.

    The low-level executor owns brake and recontact.  This public route only
    limits the post-midchannel unloading rate relative to the known rising
    guide surface; it uses the public TCP pose and the command it just issued.
    Mode 1 keeps its separately qualified post-outlet route in the task
    binding rather than hiding a mode switch inside this actor.
    """

    VERSION = "public-T2-finite-state-guide-contact-mode0-v2"
    MIDCHANNEL_X_M = 0.440
    MAXIMUM_RELATIVE_UNLOADING_MPS = 0.008

    def __init__(self, spec, *, mode):
        if int(mode) != 0:
            raise ValueError("finite-state guide-contact mode-0 route requires mode 0")
        super().__init__(
            spec, mode=0, transit_force_N=2.0,
            wrench_governor=True,
            tangent_speed_limit_mps=spec.config.guide_mode0_transit_speed_mps,
        )
        self.version = self.VERSION
        self.reset()

    def act(self, public):
        action = super().act(public)
        tcp_x = float(np.asarray(public["tcp_position"])[0, 0])
        if self.stage in (2, 3) and tcp_x >= self.MIDCHANNEL_X_M:
            action = action.copy()
            surface_velocity = .1 * float(action[0, 0] + action[0, 1])
            action[0, 2] = min(
                float(action[0, 2]),
                surface_velocity + self.MAXIMUM_RELATIVE_UNLOADING_MPS,
            )
            self.cached = action.copy()
        return action


class PublicFiniteStatePostOutletLiftMode0Policy(
    PublicPostOutletLiftGuideContactPolicy
):
    """Qualified finite-state mode-0 prefix followed by the public lift route."""

    VERSION = "public-T2-finite-state-post-outlet-lift-mode0-v3"

    def __init__(self, spec, *, mode):
        if int(mode) != 0:
            raise ValueError("finite-state post-outlet lift requires mode 0")
        super().__init__(spec, mode=mode)
        self.base = PublicFiniteStateGuideContactMode0Policy(spec, mode=mode)
        self.reset()


class PublicFiniteStatePostOutletLeadLiftMode0Policy(
    PublicFiniteStatePostOutletLiftMode0Policy
):
    """Finite-state prefix followed by the public lead-governed lift route."""

    VERSION = "public-T2-finite-state-post-outlet-lead-lift-mode0-v1"

    def __init__(self, spec, *, mode):
        from .guide_lead_lift_route import PublicGuideLeadLiftRoutePolicy

        super().__init__(spec, mode=mode)
        self.candidate = PublicGuideLeadLiftRoutePolicy(spec, mode=mode)
        self.reset()


class PublicFiniteStatePostOutletPlanarLeadLiftMode0Policy(
    PublicFiniteStatePostOutletLiftMode0Policy
):
    """Finite-state prefix followed by the planar-lead public lift route."""

    VERSION = "public-T2-finite-state-post-outlet-planar-lead-lift-mode0-v1"

    def __init__(self, spec, *, mode):
        from .guide_lead_lift_route import PublicGuidePlanarLeadLiftRoutePolicy

        super().__init__(spec, mode=mode)
        self.candidate = PublicGuidePlanarLeadLiftRoutePolicy(spec, mode=mode)
        self.reset()


class PublicFiniteStatePostOutletRawPlanarLeadLiftMode0Policy(
    PublicFiniteStatePostOutletLiftMode0Policy
):
    """Finite-state prefix followed by raw lift with only a planar lead bound."""

    VERSION = "public-T2-finite-state-post-outlet-raw-planar-lead-mode0-v1"

    def __init__(self, spec, *, mode):
        from .guide_lead_lift_route import PublicGuideRawPlanarLeadLiftRoutePolicy

        super().__init__(spec, mode=mode)
        self.candidate = PublicGuideRawPlanarLeadLiftRoutePolicy(spec, mode=mode)
        self.reset()


class PublicFiniteStatePostOutletConcurrentLiftMode0Policy(
    PublicFiniteStatePostOutletLiftMode0Policy
):
    """Finite-state prefix followed by the concurrent monotone public route."""

    VERSION = "public-T2-finite-state-post-outlet-concurrent-lift-mode0-v1"

    def __init__(self, spec, *, mode):
        from .guide_lift_route import PublicGuideConcurrentMonotoneLiftRoutePolicy

        super().__init__(spec, mode=mode)
        self.candidate = PublicGuideConcurrentMonotoneLiftRoutePolicy(
            spec, mode=mode,
        )
        self.reset()


class PublicFiniteStatePostOutletConcurrentLiftMode0PolicyV2(
    PublicFiniteStatePostOutletLiftMode0Policy
):
    """Concurrent route using the exact frozen task transition tolerance."""

    VERSION = "public-T2-finite-state-post-outlet-concurrent-lift-mode0-v2"

    def __init__(self, spec, *, mode):
        from .guide_lift_route import PublicGuideConcurrentMonotoneLiftRoutePolicyV2

        super().__init__(spec, mode=mode)
        self.candidate = PublicGuideConcurrentMonotoneLiftRoutePolicyV2(
            spec, mode=mode,
        )
        self.reset()


class PublicFiniteStatePostOutletConcurrentLiftMode0PolicyV3(
    PublicFiniteStatePostOutletLiftMode0Policy
):
    """Concurrent route with bounded feedback during descent and hold."""

    VERSION = "public-T2-finite-state-post-outlet-concurrent-lift-mode0-v3"

    def __init__(self, spec, *, mode):
        from .guide_lift_route import PublicGuideConcurrentMonotoneLiftRoutePolicyV3

        super().__init__(spec, mode=mode)
        self.candidate = PublicGuideConcurrentMonotoneLiftRoutePolicyV3(
            spec, mode=mode,
        )
        self.reset()


class PublicDeadlineAwareGuideContactPolicy:
    """Admit public tangent reference when progress and force agree.

    The qualified finite-state route still owns geometry, normal motion,
    impedance, and force. This wrapper only increases tangent/reference
    progress before the outlet using public pose, issued reference, wrench,
    and time. It never reads evaluator support truth or hidden mechanisms.
    """

    VERSION = "public-T2-deadline-aware-reference-admission-v1"
    NOMINAL_LEAD_M = .008
    MAXIMUM_LEAD_M = {0: .014, 1: .012}
    LEAD_GROWTH_M_PER_TICK = .0001
    LEAD_RECOVERY_M_PER_TICK = .0002
    MAXIMUM_TANGENT_SPEED_MPS = .020
    POST_OUTLET_RESERVE_SECONDS = {0: 7.0, 1: 6.0}
    FORCE_GROWTH_UTILIZATION = .35
    FORCE_HOLD_UTILIZATION = .55
    PROGRESS_EMA_DECAY = .9

    def __init__(self, spec, *, mode, route):
        if spec.alias != "T2" or int(mode) not in (0, 1):
            raise ValueError("deadline-aware reference admission requires T2 mode 0/1")
        self.spec = spec
        self.mode = int(mode)
        self._route = route
        self.reset()

    def __getattr__(self, name):
        route = vars(self).get("_route")
        if route is None:
            raise AttributeError(name)
        return getattr(route, name)

    def reset(self):
        self._route.reset()
        self.tick = -1
        self.cached = None
        self.previous_position_m = None
        self.achieved_speed_mps = 0.
        self.required_speed_mps = 0.
        self.lead_limit_m = self.NOMINAL_LEAD_M
        self.last_selected_speed_mps = 0.
        self.last_force_utilization = 0.
        self.admission_controls = 0
        self.progress_empty_controls = 0

    def get_state(self):
        return deepcopy({
            "version": self.VERSION,
            "robot": self.spec.config.robot,
            "mode": self.mode,
            "tick": self.tick,
            "cached": self.cached,
            "previous_position_m": self.previous_position_m,
            "achieved_speed_mps": self.achieved_speed_mps,
            "required_speed_mps": self.required_speed_mps,
            "lead_limit_m": self.lead_limit_m,
            "last_selected_speed_mps": self.last_selected_speed_mps,
            "last_force_utilization": self.last_force_utilization,
            "admission_controls": self.admission_controls,
            "progress_empty_controls": self.progress_empty_controls,
            "route": self._route.get_state(),
        })

    def restore(self, state):
        if (state["version"], state["robot"], int(state["mode"])) != (
            self.VERSION, self.spec.config.robot, self.mode,
        ):
            raise ValueError("deadline-aware guide binding differs")
        self.tick = int(state["tick"])
        self.cached = deepcopy(state["cached"])
        self.previous_position_m = state["previous_position_m"]
        self.achieved_speed_mps = float(state["achieved_speed_mps"])
        self.required_speed_mps = float(state["required_speed_mps"])
        self.lead_limit_m = float(state["lead_limit_m"])
        self.last_selected_speed_mps = float(state["last_selected_speed_mps"])
        self.last_force_utilization = float(state["last_force_utilization"])
        self.admission_controls = int(state["admission_controls"])
        self.progress_empty_controls = int(state["progress_empty_controls"])
        self._route.restore(state["route"])

    def _contact_prefix(self):
        return self._route.base if hasattr(self._route, "base") else self._route

    def act(self, public):
        dt = self.spec.config.control_dt
        tick = int(np.rint(np.asarray(public["elapsed_seconds"]).item() / dt))
        if tick == self.tick:
            return self.cached.copy()
        if tick != self.tick + 1:
            raise ValueError("every public control tick required")

        p = np.asarray(public["tcp_position"], dtype=float)[0]
        rp = np.asarray(public["reference_position"], dtype=float)[0]
        wrench = np.asarray(public["measured_wrench"], dtype=float)[0, :3]
        axis = self.mode
        if self.previous_position_m is not None:
            measured = (float(p[axis]) - self.previous_position_m) / dt
            self.achieved_speed_mps = (
                self.PROGRESS_EMA_DECAY * self.achieved_speed_mps
                + (1. - self.PROGRESS_EMA_DECAY) * measured
            )
        self.previous_position_m = float(p[axis])

        action = np.asarray(self._route.act(public), dtype=np.float32)
        prefix = self._contact_prefix()
        before_outlet = not bool(getattr(self._route, "switched", False))
        active = before_outlet and int(getattr(prefix, "stage", 0)) >= 2
        if not active:
            self.required_speed_mps = 0.
            self.last_selected_speed_mps = float(action[0, axis])
            self.cached = action.copy()
            self.tick = tick
            return self.cached.copy()

        total_utilization = np.linalg.norm(wrench) / self.spec.config.force_limit
        lateral_utilization = (
            np.linalg.norm(wrench[:2]) / self.spec.config.lateral_force_limit
        )
        utilization = float(max(total_utilization, lateral_utilization))
        self.last_force_utilization = utilization
        outlet = .455 if self.mode == 0 else -.181
        remaining_distance = max(0., outlet - float(p[axis]))
        remaining_seconds = max(
            dt,
            self.spec.config.timeout_seconds
            - float(np.asarray(public["elapsed_seconds"]).item())
            - self.POST_OUTLET_RESERVE_SECONDS[self.mode],
        )
        self.required_speed_mps = min(
            self.MAXIMUM_TANGENT_SPEED_MPS,
            remaining_distance / remaining_seconds,
        )

        maximum_lead = self.MAXIMUM_LEAD_M[self.mode]
        if (utilization <= self.FORCE_GROWTH_UTILIZATION
                and self.achieved_speed_mps < .9 * self.required_speed_mps):
            self.lead_limit_m = min(
                maximum_lead,
                self.lead_limit_m + self.LEAD_GROWTH_M_PER_TICK,
            )
        elif utilization >= self.FORCE_HOLD_UTILIZATION:
            self.lead_limit_m = max(
                self.NOMINAL_LEAD_M,
                self.lead_limit_m - self.LEAD_RECOVERY_M_PER_TICK,
            )

        original_speed = float(action[0, axis])
        base_speed = max(0., original_speed)
        deficit = max(0., self.required_speed_mps - self.achieved_speed_mps)
        requested_speed = min(
            self.MAXIMUM_TANGENT_SPEED_MPS,
            max(base_speed, self.required_speed_mps + .5 * deficit),
        )
        lead_speed = (
            float(p[axis]) + self.lead_limit_m - float(rp[axis])
        ) / dt
        selected_speed = max(base_speed, min(requested_speed, lead_speed))
        if utilization >= self.FORCE_HOLD_UTILIZATION:
            selected_speed = original_speed
        if selected_speed + 1e-6 < self.required_speed_mps:
            self.progress_empty_controls += 1
        if abs(selected_speed - original_speed) > 1e-7:
            delta = selected_speed - original_speed
            action = action.copy()
            action[0, axis] = selected_speed
            action[0, 2] = np.clip(
                float(action[0, 2]) + .1 * delta, -.01, .01,
            )
            self.admission_controls += 1
        self.last_selected_speed_mps = float(action[0, axis])
        self.cached = action.copy()
        self.tick = tick
        return self.cached.copy()


class _ResponseNormalizedGuideRoutePolicy(PublicGuideContactPolicy):
    """Single surface-following route with phase-gated acquisition authority."""

    VERSION = "public-T2-response-normalized-guide-route-v4"
    REFERENCE_LEAD_XY_M = .016
    PRE_OUTLET_MARGIN_M = {0: .010, 1: .007}
    ACQUISITION_SPEED_LIMIT_MPS = .080
    CONTACT_SPEED_FLOOR_MPS = .00675
    POST_OUTLET_NORMAL_GAIN_MPS_PER_N = .008
    POST_OUTLET_NORMAL_LIMIT_MPS = .020
    POST_OUTLET_NORMAL_RAMP_SECONDS = 1.0
    AXIS_SETTLING_SPEED_MPS = .004
    TERMINAL_RADIUS_FRACTION = 1.0

    def __init__(self, spec, *, mode):
        super().__init__(
            spec,
            mode=mode,
            transit_force_N=2.0,
            wrench_governor=True,
            governor_start_fraction=.35,
            governor_stop_fraction=.70,
            filtered_wrench_governor=True,
            governor_decay=.85,
            governor_recovery_per_tick=.04,
            tangent_speed_limit_mps=.020,
        )
        self.version = self.VERSION

    def reset(self):
        super().reset()
        self.axis_endpoint_reached = False

    def get_state(self):
        state = super().get_state()
        state["axis_endpoint_reached"] = self.axis_endpoint_reached
        return state

    def restore(self, state):
        super().restore(state)
        self.axis_endpoint_reached = bool(state["axis_endpoint_reached"])

    def _outlet_ready(self, public, position, reference, goal):
        geometric_outlet = self._geometric_outlet_reached(position)
        if self.mode != 0:
            return geometric_outlet
        axis = self.mode
        tolerance = self.spec.config.position_tolerance
        if geometric_outlet:
            self.axis_endpoint_reached = True
        if not self.axis_endpoint_reached:
            return False
        speed = abs(float(np.asarray(public["tcp_twist"])[0, axis]))
        return bool(
            self.axis_endpoint_reached
            and abs(float(reference[0, axis] - position[0, axis])) <= tolerance
            and speed <= self.AXIS_SETTLING_SPEED_MPS
        )

    def act(self, public):
        action = super().act(public)
        axis = self.mode
        if self.stage in (0, 1):
            outlet = .455 if axis == 0 else -.181
            waypoint = outlet - self.PRE_OUTLET_MARGIN_M[axis]
            position = float(np.asarray(public["tcp_position"])[0, axis])
            reference = float(np.asarray(
                public["reference_position"], dtype=float,
            )[0, axis])
            twist = float(np.asarray(public["tcp_twist"])[0, axis])
            dt = self.spec.config.control_dt
            requested = (
                4. * (waypoint - position)
                + 2. * (waypoint - reference)
                - twist
            )
            reference_room = max(0., (waypoint - reference) / dt)
            selected = min(
                self.ACQUISITION_SPEED_LIMIT_MPS,
                max(0., requested),
                reference_room,
            )
            action = action.copy()
            action[0, axis] = selected
            self.cached = action.copy()
        elif self.stage == 2 and not self._geometric_outlet_reached(
            np.asarray(public["tcp_position"], dtype=float)
        ):
            action = action.copy()
            action[0, axis] = max(
                float(action[0, axis]),
                self.CONTACT_SPEED_FLOOR_MPS * self.tangent_scale,
            )
            self.cached = action.copy()
        if self.stage >= 3:
            action = action.copy()
            measured_force = float(np.asarray(
                public["measured_wrench"], dtype=float,
            )[0, 2])
            surface_velocity = .1 * float(action[0, 0] + action[0, 1])
            outlet_tick = (
                self.tick if self.first_outlet_tick is None
                else self.first_outlet_tick
            )
            outlet_age = max(
                0., (self.tick - outlet_tick) * self.spec.config.control_dt,
            )
            normal_limit = .010 + min(
                1., outlet_age / self.POST_OUTLET_NORMAL_RAMP_SECONDS,
            ) * (self.POST_OUTLET_NORMAL_LIMIT_MPS - .010)
            action[0, 2] = np.clip(
                self.POST_OUTLET_NORMAL_GAIN_MPS_PER_N * (
                    measured_force - self.spec.config.desired_force
                ) + surface_velocity,
                -normal_limit,
                .010,
            )
            self.cached = action.copy()
        return self.cached.copy()


class PublicResponseNormalizedGuideCompletionPolicy:
    """Shared T2 route that raises authority only after measured lag.

    The policy uses public pose, reference, twist, wrench and time. It never
    reads mechanism parameters or evaluator-only contact identity. Mode 0
    remains on the surface route; mode 1 changes to the task-owned clearance
    route only after its public outlet event.
    """

    VERSION = "public-T2-response-normalized-hybrid-force-v50"
    BASE_LOG_STIFFNESS = .25
    AUTHORITY_GROWTH_PER_CONTROL = .02
    AUTHORITY_RELEASE_PER_CONTROL = .04
    PROGRESS_EMA_DECAY = .9
    FORCE_GROWTH_UTILIZATION = .35
    FORCE_RELEASE_UTILIZATION = .55
    MAXIMUM_CONTACT_SPEED_MPS = .030
    MAXIMUM_FREE_SPEED_MPS = .080
    NOMINAL_LEAD_M = .008
    ADDITIONAL_LEAD_M = .016
    COMPLETION_RESERVE_SECONDS = 3.0
    TERMINAL_DWELL_CONTROLS = 15
    TERMINAL_SURFACE_FORCE_TARGET_N = 6.0
    AXIS_TRANSITION_AUTHORITY = .20
    CLEARANCE_SETTLING_SPEED_MPS = .004
    CLEARANCE_TRAVERSE_REQUIRED_SPEED_MPS = .030

    def _clearance_required_speed_limit(self, remaining_planar):
        del remaining_planar
        return self.CLEARANCE_TRAVERSE_REQUIRED_SPEED_MPS

    def __init__(self, spec, *, mode):
        if spec.config.log_stiffness_limit <= 1.:
            raise ValueError(
                "response-normalized completion requires versioned stiffness support"
            )
        if spec.config.tangential_wrench_limit < 16.:
            raise ValueError(
                "response-normalized completion requires versioned tangent support"
            )
        self.spec = spec
        self.mode = int(mode)
        self.route = _ResponseNormalizedGuideRoutePolicy(spec, mode=mode)
        if self.mode == 1:
            from .guide_lift_route import (
                PublicGuideResponseNormalizedCompletionRoutePolicy,
            )
            self.completion = PublicGuideResponseNormalizedCompletionRoutePolicy(
                spec, mode=mode,
            )
        else:
            self.completion = None
        self.reset()

    def reset(self):
        self.route.reset()
        if self.completion is not None:
            self.completion.reset()
        self.tick = -1
        self.cached = None
        self.switched = False
        self.completion_switch_tick = None
        self.authority = 0.
        self.previous_position_m = None
        self.previous_axis = None
        self.achieved_speed_mps = 0.
        self.required_speed_mps = 0.
        self.tracking_lag_m = 0.
        self.force_utilization = 0.
        self.authority_controls = 0
        self.progress_deficit_controls = 0
        self.terminal_pose_active = False
        self.terminal_distance_m = -1.
        self.terminal_orientation_error_rad = -1.
        self.terminal_pose_hold_controls = 0
        self.terminal_capture_controls = 0
        self.terminal_reference_overshoot_m = 0.
        self.clearance_reference_settled = False
        self.clearance_settling_controls = 0
        self.clearance_lift_controls = 0
        self.normal_force_channel_controls = 0
        self._sync_route_diagnostics()

    def _sync_route_diagnostics(self):
        self.stage = self.route.stage
        self.first_contact_tick = self.route.first_contact_tick
        self.first_outlet_tick = self.route.first_outlet_tick
        self.axis_endpoint_reached = self.route.axis_endpoint_reached

    def get_state(self):
        return deepcopy({
            "version": self.VERSION,
            "robot": self.spec.config.robot,
            "mode": self.mode,
            "tick": self.tick,
            "cached": self.cached,
            "switched": self.switched,
            "completion_switch_tick": self.completion_switch_tick,
            "authority": self.authority,
            "previous_position_m": self.previous_position_m,
            "previous_axis": self.previous_axis,
            "achieved_speed_mps": self.achieved_speed_mps,
            "required_speed_mps": self.required_speed_mps,
            "tracking_lag_m": self.tracking_lag_m,
            "force_utilization": self.force_utilization,
            "authority_controls": self.authority_controls,
            "progress_deficit_controls": self.progress_deficit_controls,
            "terminal_pose_active": self.terminal_pose_active,
            "terminal_distance_m": self.terminal_distance_m,
            "terminal_orientation_error_rad": (
                self.terminal_orientation_error_rad
            ),
            "terminal_pose_hold_controls": self.terminal_pose_hold_controls,
            "terminal_capture_controls": self.terminal_capture_controls,
            "terminal_reference_overshoot_m": (
                self.terminal_reference_overshoot_m
            ),
            "clearance_reference_settled": (
                self.clearance_reference_settled
            ),
            "clearance_settling_controls": self.clearance_settling_controls,
            "clearance_lift_controls": self.clearance_lift_controls,
            "normal_force_channel_controls": (
                self.normal_force_channel_controls
            ),
            "route": self.route.get_state(),
            "completion": (
                None if self.completion is None else self.completion.get_state()
            ),
        })

    def restore(self, state):
        identity = (state["version"], state["robot"], int(state["mode"]))
        expected = (self.VERSION, self.spec.config.robot, self.mode)
        if identity != expected:
            raise ValueError("response-normalized guide binding differs")
        self.tick = int(state["tick"])
        self.cached = deepcopy(state["cached"])
        self.switched = bool(state["switched"])
        self.completion_switch_tick = state["completion_switch_tick"]
        self.authority = float(state["authority"])
        previous = state["previous_position_m"]
        self.previous_position_m = (
            None if previous is None else np.asarray(previous, dtype=float)
        )
        axis = state["previous_axis"]
        self.previous_axis = None if axis is None else int(axis)
        for name in (
            "achieved_speed_mps", "required_speed_mps", "tracking_lag_m",
            "force_utilization", "terminal_distance_m",
            "terminal_orientation_error_rad",
            "terminal_reference_overshoot_m",
        ):
            setattr(self, name, float(state[name]))
        self.authority_controls = int(state["authority_controls"])
        self.progress_deficit_controls = int(state["progress_deficit_controls"])
        self.terminal_pose_hold_controls = int(
            state["terminal_pose_hold_controls"]
        )
        self.terminal_capture_controls = int(
            state["terminal_capture_controls"]
        )
        self.terminal_pose_active = bool(state["terminal_pose_active"])
        self.clearance_reference_settled = bool(
            state["clearance_reference_settled"]
        )
        self.clearance_settling_controls = int(
            state["clearance_settling_controls"]
        )
        self.clearance_lift_controls = int(
            state["clearance_lift_controls"]
        )
        self.normal_force_channel_controls = int(
            state["normal_force_channel_controls"]
        )
        self.route.restore(state["route"])
        if self.completion is None:
            if state["completion"] is not None:
                raise ValueError("response-normalized completion route differs")
        else:
            self.completion.restore(state["completion"])
        self._sync_route_diagnostics()

    def _active_axis(self, position):
        if not self.switched and self.route.stage < 3:
            return self.mode
        remaining = np.asarray(self.spec.config.goal[:2]) - position[:2]
        return int(np.argmax(np.abs(remaining)))

    def _phase_target(self, position, axis):
        target = np.asarray(self.spec.config.goal, dtype=float).copy()
        if not self.switched and self.route.stage < 2:
            outlet = .455 if axis == 0 else -.181
            target[:2] = position[:2]
            target[axis] = (
                outlet - _ResponseNormalizedGuideRoutePolicy.PRE_OUTLET_MARGIN_M[axis]
            )
        return target

    def _terminal_pose(self, position, rotation):
        goal = np.asarray(self.spec.config.goal, dtype=float)
        position_error = float(np.linalg.norm(position - goal))
        self.terminal_distance_m = max(
            0., position_error - self.spec.config.position_tolerance,
        )
        desired = np.diag([1., -1., -1.])
        self.terminal_orientation_error_rad = float(np.linalg.norm(
            orientation_error(np, rotation, desired),
        ))
        return (
            self.terminal_distance_m <= 0.
            and self.terminal_orientation_error_rad
            <= self.spec.config.orientation_tolerance
        )

    def act(self, public):
        dt = self.spec.config.control_dt
        tick = int(np.rint(np.asarray(public["elapsed_seconds"]).item() / dt))
        if tick == self.tick:
            return self.cached.copy()
        if tick != self.tick + 1:
            raise ValueError("every public control tick required")

        position = np.asarray(public["tcp_position"], dtype=float)[0]
        rotation = np.asarray(public["tcp_rotation"], dtype=float)[0]
        reference = np.asarray(public["reference_position"], dtype=float)[0]
        twist = np.asarray(public["tcp_twist"], dtype=float)[0]
        wrench = np.asarray(public["measured_wrench"], dtype=float)[0, :3]
        if self.switched:
            local = dict(public)
            local["elapsed_seconds"] = np.asarray([[
                (tick - self.completion_switch_tick) * dt
            ]], dtype=float)
            action = np.asarray(self.completion.act(local), dtype=np.float32)
        else:
            action = np.asarray(self.route.act(public), dtype=np.float32)
            if self.mode == 1 and self.route.stage >= 3:
                self.switched = True
                self.completion_switch_tick = tick
                local = dict(public)
                local["elapsed_seconds"] = np.asarray([[0.]], dtype=float)
                action = np.asarray(self.completion.act(local), dtype=np.float32)
        axis = self._active_axis(position)
        phase_target = self._phase_target(position, axis)
        direction = float(np.sign(phase_target[axis] - position[axis]))
        if direction == 0.:
            direction = 1.

        if self.previous_axis is not None and self.previous_axis != axis:
            self.authority = min(self.authority, self.AXIS_TRANSITION_AUTHORITY)
        if self.previous_position_m is not None and self.previous_axis == axis:
            measured = (
                direction * (position[axis] - self.previous_position_m[axis]) / dt
            )
            self.achieved_speed_mps = (
                self.PROGRESS_EMA_DECAY * self.achieved_speed_mps
                + (1. - self.PROGRESS_EMA_DECAY) * measured
            )
        else:
            self.achieved_speed_mps = 0.
        self.previous_position_m = position.copy()
        self.previous_axis = axis
        surface_completion = self.mode == 0 and self.route.stage >= 3
        clearance_lift = (
            self.switched and self.completion.stage == 0
        )
        clearance_traverse = (
            self.switched and self.completion.stage == 1
        )
        terminal_completion = (
            surface_completion
            or (self.switched and self.completion.stage >= 2)
        )
        self.terminal_pose_active = terminal_completion
        terminal_pose = (
            self._terminal_pose(position, rotation)
            if terminal_completion else False
        )
        if terminal_pose:
            self.terminal_pose_hold_controls += 1
        else:
            self.terminal_pose_hold_controls = 0
        if terminal_completion and not terminal_pose:
            self.terminal_capture_controls += 1

        self.force_utilization = float(max(
            np.linalg.norm(wrench) / self.spec.config.force_limit,
            np.linalg.norm(wrench[:2]) / self.spec.config.lateral_force_limit,
        ))
        if clearance_lift:
            planar_lead = float(np.linalg.norm(reference[:2] - position[:2]))
            planar_speed = float(np.linalg.norm(twist[:2]))
            if (
                not self.clearance_reference_settled
                and planar_lead <= self.spec.config.position_tolerance
                and planar_speed <= self.CLEARANCE_SETTLING_SPEED_MPS
            ):
                self.clearance_reference_settled = True
            action = action.copy()
            action[0, :2] = np.clip(action[0, :2], -.010, .010)
            if self.clearance_reference_settled:
                span = (
                    self.FORCE_RELEASE_UTILIZATION
                    - self.FORCE_GROWTH_UTILIZATION
                )
                lift_scale = float(np.clip(
                    (self.FORCE_RELEASE_UTILIZATION - self.force_utilization)
                    / span,
                    0., 1.,
                ))
                action[0, 2] = np.clip(
                    action[0, 2], 0.,
                    self.MAXIMUM_CONTACT_SPEED_MPS * lift_scale,
                )
                self.clearance_lift_controls += 1
            else:
                action[0, 2] = 0.
                self.clearance_settling_controls += 1
            action[0, 6:8] = self.BASE_LOG_STIFFNESS
            action[0, 8] = self.spec.config.log_stiffness_limit
            self.authority = max(
                0., self.authority - self.AUTHORITY_RELEASE_PER_CONTROL,
            )
            self.required_speed_mps = 0.
            self.tracking_lag_m = planar_lead
            self.cached = action.astype(np.float32)
            self.tick = tick
            self._sync_route_diagnostics()
            return self.cached.copy()

        elapsed = float(np.asarray(public["elapsed_seconds"]).item())
        if clearance_lift:
            remaining_seconds = max(
                dt,
                self.spec.config.timeout_seconds
                - elapsed
                - self.COMPLETION_RESERVE_SECONDS,
            )
            remaining_planar = 0.
        elif terminal_completion:
            completion_reserve = (
                self.COMPLETION_RESERVE_SECONDS
                if surface_completion
                else self.TERMINAL_DWELL_CONTROLS * dt
            )
            remaining_seconds = max(
                dt,
                self.spec.config.timeout_seconds
                - elapsed
                - completion_reserve
                - self.TERMINAL_DWELL_CONTROLS * dt,
            )
            remaining_planar = self.terminal_distance_m
        elif clearance_traverse:
            remaining_seconds = max(
                dt,
                self.spec.config.timeout_seconds
                - elapsed
                - self.COMPLETION_RESERVE_SECONDS,
            )
            remaining_planar = max(
                0.,
                float(np.linalg.norm(
                    phase_target[:2] - position[:2],
                )) - self.spec.config.position_tolerance,
            )
        elif not self.switched and self.route.stage < 2:
            remaining_seconds = max(
                dt,
                self.spec.config.timeout_seconds
                - elapsed
                - self.COMPLETION_RESERVE_SECONDS,
            )
            remaining_planar = abs(phase_target[axis] - position[axis])
        else:
            remaining_seconds = max(
                dt,
                self.spec.config.timeout_seconds
                - elapsed
                - self.COMPLETION_RESERVE_SECONDS,
            )
            remaining_planar = float(np.linalg.norm(
                phase_target[:2] - position[:2], ord=1,
            ))
        required_speed_limit = (
            self._clearance_required_speed_limit(remaining_planar)
            if clearance_traverse else self.MAXIMUM_CONTACT_SPEED_MPS
        )
        self.required_speed_mps = (
            0. if clearance_lift else min(
                required_speed_limit,
                remaining_planar / remaining_seconds,
            )
        )
        self.tracking_lag_m = max(
            0., direction * (reference[axis] - position[axis]),
        )
        deficit = max(0., self.required_speed_mps - self.achieved_speed_mps)
        deficient = (
            deficit > .2 * max(self.required_speed_mps, 1e-6)
            and self.tracking_lag_m >= self.spec.config.position_tolerance
        )
        if (deficient
                and self.force_utilization <= self.FORCE_GROWTH_UTILIZATION):
            self.authority = min(
                1., self.authority + self.AUTHORITY_GROWTH_PER_CONTROL,
            )
            self.authority_controls += 1
            self.progress_deficit_controls += 1
        elif (self.force_utilization >= self.FORCE_RELEASE_UTILIZATION
                or self.achieved_speed_mps >= self.required_speed_mps):
            self.authority = max(
                0., self.authority - self.AUTHORITY_RELEASE_PER_CONTROL,
            )

        free_acquisition = (
            (not self.switched and self.route.stage < 2)
            or clearance_traverse
        )
        speed_limit = (
            self.MAXIMUM_FREE_SPEED_MPS
            if free_acquisition else self.MAXIMUM_CONTACT_SPEED_MPS
        )
        if terminal_completion and remaining_planar < .006:
            speed_limit = min(speed_limit, .010)
        if clearance_traverse:
            speed_limit = min(
                speed_limit,
                max(
                    self.CLEARANCE_SETTLING_SPEED_MPS,
                    4. * remaining_planar,
                ),
            )
        terminal_force_scale = 1.
        if terminal_completion:
            force_span = (
                self.FORCE_RELEASE_UTILIZATION
                - self.FORCE_GROWTH_UTILIZATION
            )
            terminal_force_scale = float(np.clip(
                (self.FORCE_RELEASE_UTILIZATION - self.force_utilization)
                / force_span,
                0., 1.,
            ))
        original_action = float(action[0, axis])
        original = direction * original_action
        requested = max(
            0., original,
            self.required_speed_mps + self.authority * deficit,
        )
        lead_limit = (
            self.NOMINAL_LEAD_M + self.authority * self.ADDITIONAL_LEAD_M
        )
        if clearance_traverse:
            lead_limit = min(
                lead_limit,
                max(
                    self.spec.config.position_tolerance + remaining_planar,
                    self.tracking_lag_m + dt * speed_limit,
                ),
            )
        lead_room = max(
            0., direction * (position[axis] - reference[axis]) / dt
            + lead_limit / dt,
        )
        reference_target = float(phase_target[axis])
        self.terminal_reference_overshoot_m = 0.
        if (terminal_completion and not terminal_pose) or clearance_traverse:
            self.terminal_reference_overshoot_m = min(
                self.tracking_lag_m,
                lead_limit,
            ) * terminal_force_scale
            reference_target += (
                direction * self.terminal_reference_overshoot_m
            )
        remaining_room = max(
            0., direction * (reference_target - reference[axis]) / dt,
        )
        selected = min(speed_limit, requested, lead_room, remaining_room)
        if self.force_utilization >= self.FORCE_RELEASE_UTILIZATION:
            if terminal_completion:
                selected = 0.
            else:
                selected = max(0., original)

        action = action.copy()
        if terminal_pose:
            action[0, :2] = 0.
            selected = 0.
        else:
            action[0, axis] = direction * selected
            if terminal_completion:
                action[0, :2] *= terminal_force_scale
        outlet = .455
        transition_window = (
            self.mode == 0
            and not self.switched
            and self.route.stage == 2
            and (
                position[axis] >= outlet - self.spec.config.position_tolerance
                or self.route.axis_endpoint_reached
            )
            and (
                position[axis] <= outlet
                or reference[axis] - position[axis]
                > self.spec.config.position_tolerance
            )
        )
        if self.route.axis_endpoint_reached and self.route.stage == 2:
            unloading_limit = float(
                self.route.tangent_speed_limit_mps[axis]
            )
            action[0, axis] = np.clip(
                (position[axis] - reference[axis]) / dt,
                -unloading_limit,
                unloading_limit,
            )
        elif (
            transition_window
            and position[axis]
            < self.spec.config.goal[axis] - self.spec.config.position_tolerance
        ):
            damping_room = max(
                0., (
                    position[axis]
                    + self.NOMINAL_LEAD_M
                    + 2. * self.spec.config.position_tolerance
                    - reference[axis]
                ) / dt,
            )
            action[0, axis] = max(
                float(action[0, axis]),
                min(
                    self.route.CONTACT_SPEED_FLOOR_MPS * self.route.tangent_scale,
                    damping_room,
                ),
            )
        if (
            terminal_completion
            and self.force_utilization >= self.FORCE_RELEASE_UTILIZATION
        ):
            action[0, :2] = np.clip(
                (position[:2] - reference[:2]) / dt,
                -.010, .010,
            )
        delta = (
            0. if terminal_pose or terminal_force_scale <= 0.
            else float(action[0, axis]) - original_action
        )
        normal_lower = (
            -self.completion.VERTICAL_SPEED_LIMIT_MPS
            if self.switched else
            -self.route.POST_OUTLET_NORMAL_LIMIT_MPS
            if self.route.stage >= 3 else -.010
        )
        normal_upper = (
            self.completion.VERTICAL_SPEED_LIMIT_MPS
            if clearance_lift else .010
        )
        action[0, 2] = np.clip(
            float(action[0, 2]) + .1 * delta,
            normal_lower, normal_upper,
        )
        if not self.switched:
            # The shared executor already exposes a backend-neutral hybrid
            # normal-force channel.  Use it for contact acquisition and the
            # complete surface route, instead of asking a position spring to
            # both find and hold the guide under unknown delay/payload.  The
            # backend freezes only the normal position reference while this
            # channel is active; tangent reference admission is unchanged.
            action[0, 2] = 0.
            action[0, 12] = (
                self.TERMINAL_SURFACE_FORCE_TARGET_N
                if self.mode == 0 and terminal_pose
                else self.spec.config.desired_force
            )
            self.normal_force_channel_controls += 1
        clearance_descent = (
            self.switched and self.completion.stage == 2
        )
        if clearance_descent:
            planar_distance = max(
                0., float(np.linalg.norm(
                    position[:2] - np.asarray(self.spec.config.goal[:2]),
                )) - self.spec.config.position_tolerance,
            )
            if planar_distance > 0.:
                action[0, 2] = max(
                    float(action[0, 2]),
                    -self.MAXIMUM_CONTACT_SPEED_MPS * terminal_force_scale,
                )
        action[0, 6 + axis] = (
            self.BASE_LOG_STIFFNESS
            + self.authority * (
                self.spec.config.log_stiffness_limit
                - self.BASE_LOG_STIFFNESS
            )
        )
        if transition_window:
            public_lead = max(
                self.spec.config.position_tolerance,
                abs(float(reference[axis] - position[axis])),
            )
            spring_effort_budget = (
                self.FORCE_RELEASE_UTILIZATION
                * self.spec.config.tangential_wrench_limit
            )
            transition_stiffness_limit = np.log(
                spring_effort_budget / (800. * public_lead)
            )
            action[0, 6 + axis] = min(
                float(action[0, 6 + axis]),
                float(np.clip(
                    transition_stiffness_limit,
                    (-self.spec.config.log_stiffness_limit
                     if self.route.axis_endpoint_reached
                     else self.BASE_LOG_STIFFNESS),
                    self.spec.config.log_stiffness_limit,
                )),
            )
        if self.switched:
            action[0, 8] = self.spec.config.log_stiffness_limit
        self.cached = action.astype(np.float32)
        self.tick = tick
        self._sync_route_diagnostics()
        return self.cached.copy()


class PublicCompletionBudgetGuideCompletionPolicy(
    PublicResponseNormalizedGuideCompletionPolicy
):
    """Use deadline-normalized stage-1 progress for v10 handoff liveness.

    The frozen v50 controller caps required stage-1 progress at 30 mm/s.
    V51 raises only that public clearance-traverse requirement; lift, descent,
    force feedback, limits and terminal acceptance remain inherited exactly.
    """

    VERSION = "public-T2-completion-budget-liveness-v51"
    CLEARANCE_TRAVERSE_REQUIRED_SPEED_MPS = .045


class PublicCompletionBudgetCaptureGuideCompletionPolicy(
    PublicCompletionBudgetGuideCompletionPolicy
):
    """Latch the v50 capture ceiling after first planar-set entry.

    Far-field stage-1 motion retains the v51 deadline request.  The first
    public observation inside the frozen planar tolerance irreversibly enters
    capture for the episode, including after a later re-exit.  No method
    identity or hidden mechanism state is read.
    """

    VERSION = "public-T2-completion-budget-terminal-capture-v52"
    CAPTURE_REQUIRED_SPEED_MPS = .030
    CAPTURE_BAND_M = 0.

    def reset(self):
        super().reset()
        self.clearance_capture_latched = False

    def get_state(self):
        state = super().get_state()
        state["clearance_capture_latched"] = self.clearance_capture_latched
        state["clearance_capture_band_m"] = self.CAPTURE_BAND_M
        return state

    def restore(self, state):
        super().restore(state)
        self.clearance_capture_latched = bool(
            state["clearance_capture_latched"]
        )

    def _clearance_required_speed_limit(self, remaining_planar):
        if remaining_planar <= self.CAPTURE_BAND_M:
            self.clearance_capture_latched = True
        return (
            self.CAPTURE_REQUIRED_SPEED_MPS
            if self.clearance_capture_latched
            else self.CLEARANCE_TRAVERSE_REQUIRED_SPEED_MPS
        )


class PublicCompletionBudgetCaptureBandGuideCompletionPolicy(
    PublicCompletionBudgetCaptureGuideCompletionPolicy
):
    """Enter the public low-speed capture regime before terminal-set entry."""

    VERSION = "public-T2-completion-budget-capture-band-v53"
    CAPTURE_BAND_M = .010


class PublicCompletionBudgetPlanarBrakingGuideCompletionPolicy(
    PublicCompletionBudgetCaptureBandGuideCompletionPolicy
):
    """Bound the complete planar common action during terminal descent.

    The inherited terminal speed check uses the full three-dimensional
    terminal distance.  During clearance descent that distance is dominated
    by height, so the planar near-terminal limit is not active even though
    the TCP is already inside the planar task set.  This version separates
    planar distance from vertical completion, constrains both planar
    coordinates, and prevents the issued reference from moving away from the
    public goal.  It does not inspect mechanism or method identity.
    """

    VERSION = "public-T2-completion-budget-planar-braking-v54"
    TERMINAL_PLANAR_SPEED_LIMIT_MPS = .010

    def reset(self):
        super().reset()
        self.terminal_planar_distance_m = -1.
        self.terminal_planar_speed_cap_active = False

    def get_state(self):
        state = super().get_state()
        state.update({
            "terminal_planar_distance_m": self.terminal_planar_distance_m,
            "terminal_planar_speed_cap_active": (
                self.terminal_planar_speed_cap_active
            ),
        })
        return state

    def restore(self, state):
        super().restore(state)
        self.terminal_planar_distance_m = float(
            state["terminal_planar_distance_m"]
        )
        self.terminal_planar_speed_cap_active = bool(
            state["terminal_planar_speed_cap_active"]
        )

    def act(self, public):
        action = np.asarray(super().act(public), dtype=np.float32).copy()
        self.terminal_planar_distance_m = -1.
        self.terminal_planar_speed_cap_active = False
        if not (
            self.mode == 1
            and self.switched
            and int(self.completion.stage) >= 2
        ):
            return action

        position = np.asarray(public["tcp_position"], dtype=float)[0]
        reference = np.asarray(public["reference_position"], dtype=float)[0]
        goal = np.asarray(public["goal_position"], dtype=float)[0]
        self.terminal_planar_distance_m = max(
            0., float(np.linalg.norm(position[:2] - goal[:2]))
            - self.spec.config.position_tolerance,
        )

        # Do not let either issued planar reference move farther from the
        # public goal.  Apply a vector-norm limit after the componentwise
        # monotonic projection so diagonal motion cannot exceed the declared
        # terminal speed.
        reference_room = (
            goal[:2] - reference[:2]
        ) / self.spec.config.control_dt
        planar = np.clip(
            action[0, :2],
            np.minimum(reference_room, 0.),
            np.maximum(reference_room, 0.),
        )
        planar_speed = float(np.linalg.norm(planar))
        if planar_speed > self.TERMINAL_PLANAR_SPEED_LIMIT_MPS:
            planar *= self.TERMINAL_PLANAR_SPEED_LIMIT_MPS / planar_speed
        action[0, :2] = planar
        self.terminal_planar_speed_cap_active = True
        self.cached = action
        return self.cached.copy()


class PublicCompletionBudgetPreDescentSettlingGuideCompletionPolicy(
    PublicCompletionBudgetCaptureBandGuideCompletionPolicy
):
    """Settle public planar execution state before terminal descent.

    V13 applied a permanent planar projection after descent had already been
    admitted.  Recorded evidence shows that this both removed terminal
    liveness and left 8--9 mm of public reference lead at recontact.  This
    version instead inserts one finite, causal barrier at the mode-1 stage-2
    transition.  It holds the reference at clearance until public reference
    lead and TCP speed have remained inside existing task tolerances long
    enough to cover the declared executor-delay envelope.  Once admitted, the
    inherited v12 completion action is used without further projection.
    """

    VERSION = "public-T2-completion-budget-pre-descent-settling-v55"
    PRE_DESCENT_SETTLING_CONTROLS = 4

    def _pre_descent_planar_action(self, position, reference):
        del position, reference
        return np.zeros(2, dtype=float)

    def reset(self):
        super().reset()
        self.pre_descent_settling_active = False
        self.pre_descent_settling_complete = False
        self.pre_descent_settling_controls = 0
        self.pre_descent_settled_controls = 0
        self.pre_descent_reference_lead_norm_m = -1.
        self.pre_descent_planar_speed_mps = -1.

    def get_state(self):
        state = super().get_state()
        state.update({
            "pre_descent_settling_active": self.pre_descent_settling_active,
            "pre_descent_settling_complete": (
                self.pre_descent_settling_complete
            ),
            "pre_descent_settling_controls": (
                self.pre_descent_settling_controls
            ),
            "pre_descent_settled_controls": (
                self.pre_descent_settled_controls
            ),
            "pre_descent_reference_lead_norm_m": (
                self.pre_descent_reference_lead_norm_m
            ),
            "pre_descent_planar_speed_mps": (
                self.pre_descent_planar_speed_mps
            ),
        })
        return state

    def restore(self, state):
        super().restore(state)
        self.pre_descent_settling_active = bool(
            state["pre_descent_settling_active"]
        )
        self.pre_descent_settling_complete = bool(
            state["pre_descent_settling_complete"]
        )
        self.pre_descent_settling_controls = int(
            state["pre_descent_settling_controls"]
        )
        self.pre_descent_settled_controls = int(
            state["pre_descent_settled_controls"]
        )
        self.pre_descent_reference_lead_norm_m = float(
            state["pre_descent_reference_lead_norm_m"]
        )
        self.pre_descent_planar_speed_mps = float(
            state["pre_descent_planar_speed_mps"]
        )

    def act(self, public):
        tick = int(np.rint(
            np.asarray(public["elapsed_seconds"]).item()
            / self.spec.config.control_dt
        ))
        repeated_tick = tick == self.tick
        action = np.asarray(super().act(public), dtype=np.float32).copy()
        if repeated_tick:
            return action

        self.pre_descent_settling_active = False
        if (
            self.mode != 1
            or not self.switched
            or int(self.completion.stage) < 2
            or self.pre_descent_settling_complete
        ):
            return action

        position = np.asarray(public["tcp_position"], dtype=float)[0]
        reference = np.asarray(public["reference_position"], dtype=float)[0]
        twist = np.asarray(public["tcp_twist"], dtype=float)[0]
        self.pre_descent_reference_lead_norm_m = float(np.linalg.norm(
            reference[:2] - position[:2]
        ))
        self.pre_descent_planar_speed_mps = float(np.linalg.norm(twist[:2]))
        settled = (
            self.pre_descent_reference_lead_norm_m
            <= self.spec.config.position_tolerance
            and self.pre_descent_planar_speed_mps
            <= self.CLEARANCE_SETTLING_SPEED_MPS
        )
        self.pre_descent_settled_controls = (
            self.pre_descent_settled_controls + 1 if settled else 0
        )
        self.pre_descent_settling_controls += 1
        self.pre_descent_settling_active = True

        # Four complete frozen requests precede the first admitted descent
        # request.  The next new control tick uses the inherited v12 action.
        action[0, :2] = self._pre_descent_planar_action(
            position, reference,
        )
        action[0, 2] = 0.
        if (
            self.pre_descent_settled_controls
            >= self.PRE_DESCENT_SETTLING_CONTROLS
        ):
            self.pre_descent_settling_complete = True
        self.cached = action
        return self.cached.copy()


class PublicCompletionBudgetReferenceUnloadingGuideCompletionPolicy(
    PublicCompletionBudgetPreDescentSettlingGuideCompletionPolicy
):
    """Unload public planar reference lead during the descent barrier.

    The v14 frozen-reference barrier can settle TCP speed but leaves a static
    3--6 mm reference offset under the low-gain cells.  This version changes
    only the barrier request: while vertical descent remains held, it reuses
    the controller's existing 10 mm/s per-axis reference-unloading envelope
    to move the public reference toward the measured TCP.  Admission and all
    post-barrier v12 behavior remain unchanged.
    """

    VERSION = "public-T2-completion-budget-reference-unloading-v56"
    PRE_DESCENT_REFERENCE_UNLOADING_LIMIT_MPS = .010

    def reset(self):
        super().reset()
        self.pre_descent_reference_unloading_active = False
        self.pre_descent_reference_unloading_speed_mps = 0.

    def get_state(self):
        state = super().get_state()
        state.update({
            "pre_descent_reference_unloading_active": (
                self.pre_descent_reference_unloading_active
            ),
            "pre_descent_reference_unloading_speed_mps": (
                self.pre_descent_reference_unloading_speed_mps
            ),
        })
        return state

    def restore(self, state):
        super().restore(state)
        self.pre_descent_reference_unloading_active = bool(
            state["pre_descent_reference_unloading_active"]
        )
        self.pre_descent_reference_unloading_speed_mps = float(
            state["pre_descent_reference_unloading_speed_mps"]
        )

    def _pre_descent_planar_action(self, position, reference):
        unloading = np.clip(
            (position[:2] - reference[:2]) / self.spec.config.control_dt,
            -self.PRE_DESCENT_REFERENCE_UNLOADING_LIMIT_MPS,
            self.PRE_DESCENT_REFERENCE_UNLOADING_LIMIT_MPS,
        )
        self.pre_descent_reference_unloading_active = True
        self.pre_descent_reference_unloading_speed_mps = float(
            np.linalg.norm(unloading)
        )
        return unloading

    def act(self, public):
        tick = int(np.rint(
            np.asarray(public["elapsed_seconds"]).item()
            / self.spec.config.control_dt
        ))
        if tick != self.tick:
            self.pre_descent_reference_unloading_active = False
            self.pre_descent_reference_unloading_speed_mps = 0.
        return super().act(public)


class PublicCompletionBudgetReferenceDebtUnloadingGuideCompletionPolicy(
    PublicCompletionBudgetReferenceUnloadingGuideCompletionPolicy
):
    """Discharge lead within the inherited completion reserve.

    V15 proves that lead-reducing unloading removes the recontact failures,
    but its force-release envelope consumes 34 controls on the tightest
    request and leaves no terminal dwell margin.  Reference unloading occurs
    at clearance and always points from the issued reference toward the public
    TCP, so this version uses the controller's existing maximum contact speed
    rather than introducing or scanning another numerical limit.
    """

    VERSION = "public-T2-completion-budget-reference-debt-unloading-v57"
    PRE_DESCENT_REFERENCE_UNLOADING_LIMIT_MPS = (
        PublicResponseNormalizedGuideCompletionPolicy.
        MAXIMUM_CONTACT_SPEED_MPS
    )


class PublicCompletionBudgetCausalReferenceUnloadingGuideCompletionPolicy(
    PublicCompletionBudgetReferenceUnloadingGuideCompletionPolicy
):
    """Scale reference unloading by causally observed executor delay.

    V16 shows that a globally faster unloading request increases rather than
    decreases barrier residence under the three-control delay.  The public
    wrapper can reconstruct the backend-bounded request by applying the task's
    shared public limiter to its own selected-action history.  Comparing that
    causal history with the allowed public reference identifies the declared
    one-to-three-control delay without expanding the observation contract or
    reading mechanism labels.  This policy consumes the frozen wrapper
    estimate, keeps v15's safe fallback, and preserves its barrier and
    post-release action.
    """

    VERSION = "public-T2-completion-budget-causal-reference-unloading-v58"
    PUBLIC_REFERENCE_DELAY_CANDIDATES = (1, 2, 3)
    PUBLIC_REFERENCE_DELAY_MINIMUM_SAMPLES = 8
    PUBLIC_REFERENCE_DELAY_EXCITATION_THRESHOLD_MPS = 1e-6
    REFERENCE_UNLOADING_DEBT_MPS_CONTROLS = .030
    REFERENCE_UNLOADING_MAXIMUM_MPS = (
        PublicResponseNormalizedGuideCompletionPolicy.
        MAXIMUM_CONTACT_SPEED_MPS
    )

    def reset(self):
        super().reset()
        self.public_reference_delay_estimate_controls = 0
        self.public_reference_delay_identification_samples = 0
        self.public_reference_delay_score_gap_mps = -1.
        self.pre_descent_causal_unloading_limit_mps = (
            self.PRE_DESCENT_REFERENCE_UNLOADING_LIMIT_MPS
        )

    def get_state(self):
        state = super().get_state()
        state.update({
            "public_reference_delay_estimate_controls": (
                self.public_reference_delay_estimate_controls
            ),
            "public_reference_delay_identification_samples": (
                self.public_reference_delay_identification_samples
            ),
            "public_reference_delay_score_gap_mps": (
                self.public_reference_delay_score_gap_mps
            ),
            "pre_descent_causal_unloading_limit_mps": (
                self.pre_descent_causal_unloading_limit_mps
            ),
        })
        return state

    def restore(self, state):
        super().restore(state)
        self.public_reference_delay_estimate_controls = int(
            state["public_reference_delay_estimate_controls"]
        )
        self.public_reference_delay_identification_samples = int(
            state["public_reference_delay_identification_samples"]
        )
        self.public_reference_delay_score_gap_mps = float(
            state["public_reference_delay_score_gap_mps"]
        )
        self.pre_descent_causal_unloading_limit_mps = float(
            state["pre_descent_causal_unloading_limit_mps"]
        )
        if (
            self.public_reference_delay_estimate_controls
            not in (0, *self.PUBLIC_REFERENCE_DELAY_CANDIDATES)
            or not 0 <= self.public_reference_delay_identification_samples
            <= self.PUBLIC_REFERENCE_DELAY_MINIMUM_SAMPLES
            or not np.isfinite(self.public_reference_delay_score_gap_mps)
        ):
            raise ValueError("causal reference-delay checkpoint differs")

    def set_public_reference_delay(self, estimate, samples, score_gap):
        estimate, samples, score_gap = (
            int(estimate), int(samples), float(score_gap),
        )
        if (
            estimate not in (0, *self.PUBLIC_REFERENCE_DELAY_CANDIDATES)
            or not 0 <= samples <= self.PUBLIC_REFERENCE_DELAY_MINIMUM_SAMPLES
            or not np.isfinite(score_gap)
        ):
            raise ValueError("invalid causal public reference-delay estimate")
        self.public_reference_delay_estimate_controls = estimate
        self.public_reference_delay_identification_samples = samples
        self.public_reference_delay_score_gap_mps = score_gap
        self.pre_descent_causal_unloading_limit_mps = (
            self._causal_reference_unloading_limit()
        )

    def _causal_reference_unloading_limit(self):
        delay = self.public_reference_delay_estimate_controls
        if delay == 0:
            return self.PRE_DESCENT_REFERENCE_UNLOADING_LIMIT_MPS
        return min(
            self.REFERENCE_UNLOADING_MAXIMUM_MPS,
            self.REFERENCE_UNLOADING_DEBT_MPS_CONTROLS / delay,
        )

    def _pre_descent_planar_action(self, position, reference):
        limit = self._causal_reference_unloading_limit()
        unloading = np.clip(
            (position[:2] - reference[:2]) / self.spec.config.control_dt,
            -limit, limit,
        )
        self.pre_descent_causal_unloading_limit_mps = limit
        self.pre_descent_reference_unloading_active = True
        self.pre_descent_reference_unloading_speed_mps = float(
            np.linalg.norm(unloading)
        )
        return unloading


class PublicCompletionBudgetCausalTraverseGuideCompletionPolicy(
    PublicCompletionBudgetCausalReferenceUnloadingGuideCompletionPolicy
):
    """Reserve enough wall-clock time for descent and terminal dwell.

    The v17 interface causally unloads the planar reference before descent,
    but its 45 mm/s clearance traverse reaches the public planar terminal set
    with only about 108 controls left in the adverse T2 payload cells.  A
    36 mm descent plus the frozen 15-control dwell cannot reliably complete
    in that remainder.  This revision changes only the free-space traverse
    request to 60 mm/s, below the existing 80 mm/s public free-space limit.
    The capture band, causal unloading barrier, contact controller, task
    limits and evaluator remain inherited unchanged.
    """

    VERSION = "public-T2-completion-budget-causal-traverse-v59"
    CLEARANCE_TRAVERSE_REQUIRED_SPEED_MPS = .060


def _fixed_finite_state_guide_contact_policy(spec, *, mode):
    """Construct the qualified fixed-lead mode-specific T2 public route.

    This is the single task-level factory shared by the public reference and
    learned/model-based actors.  Keeping the mode split here prevents those
    actors from silently training around a different task scaffold.
    """
    mode = int(mode)
    if mode == 0:
        if spec.config.guide_mode0_post_outlet_route == "concurrent_monotone_lift_v3":
            return PublicFiniteStatePostOutletConcurrentLiftMode0PolicyV3(
                spec, mode=mode,
            )
        if spec.config.guide_mode0_post_outlet_route == "concurrent_monotone_lift_v2":
            return PublicFiniteStatePostOutletConcurrentLiftMode0PolicyV2(
                spec, mode=mode,
            )
        if spec.config.guide_mode0_post_outlet_route == "concurrent_monotone_lift":
            return PublicFiniteStatePostOutletConcurrentLiftMode0Policy(
                spec, mode=mode,
            )
        if spec.config.guide_mode0_post_outlet_route == "raw_planar_lead_lift":
            return PublicFiniteStatePostOutletRawPlanarLeadLiftMode0Policy(
                spec, mode=mode,
            )
        if spec.config.guide_mode0_post_outlet_route == "planar_lead_lift":
            return PublicFiniteStatePostOutletPlanarLeadLiftMode0Policy(
                spec, mode=mode,
            )
        if spec.config.guide_mode0_post_outlet_lead_governed_lift:
            return PublicFiniteStatePostOutletLeadLiftMode0Policy(
                spec, mode=mode,
            )
        if spec.config.guide_mode0_post_outlet_lift:
            return PublicFiniteStatePostOutletLiftMode0Policy(spec, mode=mode)
        return PublicFiniteStateGuideContactMode0Policy(spec, mode=mode)
    if mode == 1:
        return PublicPostOutletLiftGuideContactPolicy(spec, mode=mode)
    raise ValueError("finite-state guide contact requires mode 0 or 1")


def public_finite_state_guide_contact_policy(spec, *, mode):
    """Construct the task-owned route and versioned reference admission."""
    admission = getattr(
        spec.config, "guide_reference_admission", "fixed_lead_v1",
    )
    route = _fixed_finite_state_guide_contact_policy(spec, mode=mode)
    if admission == "fixed_lead_v1":
        return route
    if admission == "deadline_aware_v1":
        return PublicDeadlineAwareGuideContactPolicy(
            spec, mode=mode, route=route,
        )
    if admission == "response_normalized_completion":
        return PublicResponseNormalizedGuideCompletionPolicy(
            spec, mode=mode,
        )
    raise ValueError(f"unknown guide reference admission: {admission}")


class PublicTerminalOwnedGuideActor:
    """Give the public T2 completion policy exclusive late-phase authority.

    The candidate continues to observe every public tick, so its causal state
    and resource accounting remain comparable.  Its request is issued until
    a declared task-owned boundary.  From that tick onward the exact common
    request is issued on all 13 coordinates.  V1 starts at terminal capture;
    V2 starts at the mode-1 outlet-to-clearance completion switch because V1
    preserved unsafe approach state before its later handoff.  V3 retains full
    candidate authority during clearance lift, then continuously scales the
    residual by public planar distance during traverse and reaches exact common
    authority before descent.  V4 keeps that clearance envelope, then admits
    only transport-coordinate residuals whose public one-step terminal-set
    progress is no worse than the common request during descent.  V5 keeps
    the v3 clearance envelope and retains only force-released transport
    residuals throughout descent and terminal dwell.  Task-owned coordinates
    remain exactly common; the relative one-step dominance proxy is removed.
    V7 starts at the public outlet latch and bounds transport residual authority
    by an augmented-state handoff debt: remaining deadline, public reference
    lag and the existing clearance-progress envelope.  Authority is monotone
    after outlet, task-owned coordinates remain exact common, and completion
    stage 2 has exact common authority.
    V8 adds a public clearance-stage residence debt.  It preserves the same
    coordinate ownership and monotone handoff, but prevents a candidate from
    consuming the remaining completion budget while the shared completion
    policy is still waiting to leave clearance stage 0.
    V9 leaves stage-0 behavior unchanged, then bounds stage-1 transport
    authority by a public finite-time completion budget and measured progress
    speed.  It reserves traverse, descent, settling and dwell controls without
    reading the hidden delay, payload, servo gain or method identity.
    V10 retains the v9 authority law and binds the versioned public v51
    completion controller, whose only change is a larger deadline-normalized
    stage-1 required-speed range.
    V14--V17 retain that ownership law and revise only the shared mode-1
    pre-descent transition. V17 identifies the public reference delay from
    the wrapper's reconstructed bounded-action history and selects a
    delay-scaled unloading limit before restoring the unchanged v12 completion
    action.
    """

    TERMINAL_CAPTURE_VERSION = "public-T2-terminal-completion-ownership-v1"
    POST_OUTLET_COMPLETION_VERSION = (
        "public-T2-post-outlet-completion-ownership-v2"
    )
    CLEARANCE_PROGRESS_VERSION = "public-T2-clearance-progress-authority-v3"
    TERMINAL_SET_ADMISSION_VERSION = (
        "public-T2-clearance-terminal-set-admission-v4"
    )
    TERMINAL_TRANSPORT_VERSION = (
        "public-T2-clearance-terminal-transport-v5"
    )
    TERMINAL_STRICT_PROGRESS_VERSION = (
        "public-T2-clearance-terminal-strict-progress-v6"
    )
    HANDOFF_VIABILITY_VERSION = "public-T2-handoff-viability-v7"
    STAGE_DEBT_VIABILITY_VERSION = "public-T2-stage-debt-viability-v8"
    COMPLETION_BUDGET_VIABILITY_VERSION = (
        "public-T2-completion-budget-viability-v9"
    )
    COMPLETION_BUDGET_LIVENESS_VERSION = (
        "public-T2-completion-budget-liveness-v10"
    )
    COMPLETION_BUDGET_CAPTURE_VERSION = (
        "public-T2-completion-budget-terminal-capture-v11"
    )
    COMPLETION_BUDGET_CAPTURE_BAND_VERSION = (
        "public-T2-completion-budget-capture-band-v12"
    )
    COMPLETION_BUDGET_PLANAR_BRAKING_VERSION = (
        "public-T2-completion-budget-planar-braking-v13"
    )
    COMPLETION_BUDGET_PRE_DESCENT_SETTLING_VERSION = (
        "public-T2-completion-budget-pre-descent-settling-v14"
    )
    COMPLETION_BUDGET_REFERENCE_UNLOADING_VERSION = (
        "public-T2-completion-budget-reference-unloading-v15"
    )
    COMPLETION_BUDGET_REFERENCE_DEBT_UNLOADING_VERSION = (
        "public-T2-completion-budget-reference-debt-unloading-v16"
    )
    COMPLETION_BUDGET_CAUSAL_REFERENCE_UNLOADING_VERSION = (
        "public-T2-completion-budget-causal-reference-unloading-v17"
    )
    COMPLETION_BUDGET_CAUSAL_TRAVERSE_VERSION = (
        "public-T2-completion-budget-causal-traverse-v18"
    )
    TERMINAL_BLENDS = (1., .5, .25, 0.)
    TRANSPORT_COORDINATES = (0, 1, 3, 4, 5)
    TASK_OWNED_COORDINATES = (2, 6, 7, 8, 9, 10, 11, 12)
    PROGRESS_TOLERANCE = 1e-12
    CANDIDATE_SELECTED = 0
    COMMON_TERMINAL_SELECTED = 2
    CLEARANCE_BLEND_SELECTED = 3
    TERMINAL_PROGRESS_BLEND_SELECTED = 4
    TERMINAL_TRANSPORT_SELECTED = 5
    HANDOFF_VIABILITY_SELECTED = 6

    def __init__(
        self, candidate, spec, *, mode,
        version=TERMINAL_CAPTURE_VERSION, common=None,
        completion_reserve_controls=150,
        reserve_taper_controls=100,
        reference_lag_full_authority_m=.008,
        reference_lag_zero_authority_m=.012,
        maximum_authority_decay_per_control=.05,
        clearance_debt_full_controls=None,
        clearance_debt_zero_controls=None,
        completion_progress_floor_mps=.020,
        post_traverse_reserve_controls=100,
        completion_budget_taper_controls=60,
        force_speed_full_authority_mps=.010,
        force_speed_zero_authority_mps=.025,
    ):
        if spec.alias != "T2":
            raise ValueError("terminal completion ownership is T2-only")
        if version not in {
            self.TERMINAL_CAPTURE_VERSION,
            self.POST_OUTLET_COMPLETION_VERSION,
            self.CLEARANCE_PROGRESS_VERSION,
            self.TERMINAL_SET_ADMISSION_VERSION,
            self.TERMINAL_TRANSPORT_VERSION,
            self.TERMINAL_STRICT_PROGRESS_VERSION,
            self.HANDOFF_VIABILITY_VERSION,
            self.STAGE_DEBT_VIABILITY_VERSION,
            self.COMPLETION_BUDGET_VIABILITY_VERSION,
            self.COMPLETION_BUDGET_LIVENESS_VERSION,
            self.COMPLETION_BUDGET_CAPTURE_VERSION,
            self.COMPLETION_BUDGET_CAPTURE_BAND_VERSION,
            self.COMPLETION_BUDGET_PLANAR_BRAKING_VERSION,
            self.COMPLETION_BUDGET_PRE_DESCENT_SETTLING_VERSION,
            self.COMPLETION_BUDGET_REFERENCE_UNLOADING_VERSION,
            self.COMPLETION_BUDGET_REFERENCE_DEBT_UNLOADING_VERSION,
            self.COMPLETION_BUDGET_CAUSAL_REFERENCE_UNLOADING_VERSION,
            self.COMPLETION_BUDGET_CAUSAL_TRAVERSE_VERSION,
        }:
            raise ValueError("unknown T2 completion ownership version")
        parameters = {
            "completion_reserve_controls": int(completion_reserve_controls),
            "reserve_taper_controls": int(reserve_taper_controls),
            "reference_lag_full_authority_m": float(
                reference_lag_full_authority_m
            ),
            "reference_lag_zero_authority_m": float(
                reference_lag_zero_authority_m
            ),
            "maximum_authority_decay_per_control": float(
                maximum_authority_decay_per_control
            ),
        }
        if version == self.STAGE_DEBT_VIABILITY_VERSION:
            parameters.update({
                "clearance_debt_full_controls": int(
                    343 if clearance_debt_full_controls is None
                    else clearance_debt_full_controls
                ),
                "clearance_debt_zero_controls": int(
                    371 if clearance_debt_zero_controls is None
                    else clearance_debt_zero_controls
                ),
            })
        elif version in {
            self.COMPLETION_BUDGET_VIABILITY_VERSION,
            self.COMPLETION_BUDGET_LIVENESS_VERSION,
            self.COMPLETION_BUDGET_CAPTURE_VERSION,
            self.COMPLETION_BUDGET_CAPTURE_BAND_VERSION,
            self.COMPLETION_BUDGET_PLANAR_BRAKING_VERSION,
            self.COMPLETION_BUDGET_PRE_DESCENT_SETTLING_VERSION,
            self.COMPLETION_BUDGET_REFERENCE_UNLOADING_VERSION,
            self.COMPLETION_BUDGET_REFERENCE_DEBT_UNLOADING_VERSION,
            self.COMPLETION_BUDGET_CAUSAL_REFERENCE_UNLOADING_VERSION,
            self.COMPLETION_BUDGET_CAUSAL_TRAVERSE_VERSION,
        }:
            if (
                clearance_debt_full_controls is not None
                or clearance_debt_zero_controls is not None
            ):
                raise ValueError("clearance-stage debt parameters require v8")
            parameters.update({
                "completion_progress_floor_mps": float(
                    completion_progress_floor_mps
                ),
                "post_traverse_reserve_controls": int(
                    post_traverse_reserve_controls
                ),
                "completion_budget_taper_controls": int(
                    completion_budget_taper_controls
                ),
                "force_speed_full_authority_mps": float(
                    force_speed_full_authority_mps
                ),
                "force_speed_zero_authority_mps": float(
                    force_speed_zero_authority_mps
                ),
            })
        elif (
            clearance_debt_full_controls is not None
            or clearance_debt_zero_controls is not None
        ):
            raise ValueError("clearance-stage debt parameters require v8")
        if (
            parameters["completion_reserve_controls"] <= 0
            or parameters["reserve_taper_controls"] <= 0
            or parameters["reference_lag_full_authority_m"] < 0.
            or parameters["reference_lag_zero_authority_m"]
            <= parameters["reference_lag_full_authority_m"]
            or not 0. < parameters["maximum_authority_decay_per_control"] <= 1.
            or (
                version == self.STAGE_DEBT_VIABILITY_VERSION
                and (
                    parameters["clearance_debt_full_controls"] < 0
                    or parameters["clearance_debt_zero_controls"]
                    <= parameters["clearance_debt_full_controls"]
                )
            )
            or (
                version in {
                    self.COMPLETION_BUDGET_VIABILITY_VERSION,
                    self.COMPLETION_BUDGET_LIVENESS_VERSION,
                    self.COMPLETION_BUDGET_CAPTURE_VERSION,
                    self.COMPLETION_BUDGET_CAPTURE_BAND_VERSION,
                    self.COMPLETION_BUDGET_PLANAR_BRAKING_VERSION,
                    self.COMPLETION_BUDGET_PRE_DESCENT_SETTLING_VERSION,
                    self.COMPLETION_BUDGET_REFERENCE_UNLOADING_VERSION,
                    self.COMPLETION_BUDGET_REFERENCE_DEBT_UNLOADING_VERSION,
                    self.COMPLETION_BUDGET_CAUSAL_REFERENCE_UNLOADING_VERSION,
                    self.COMPLETION_BUDGET_CAUSAL_TRAVERSE_VERSION,
                }
                and (
                    parameters["completion_progress_floor_mps"] <= 0.
                    or parameters["post_traverse_reserve_controls"] <= 0
                    or parameters["completion_budget_taper_controls"] <= 0
                    or parameters["force_speed_full_authority_mps"] < 0.
                    or parameters["force_speed_zero_authority_mps"]
                    <= parameters["force_speed_full_authority_mps"]
                )
            )
        ):
            raise ValueError("invalid handoff viability parameters")
        if (
            version not in {
                self.HANDOFF_VIABILITY_VERSION,
                self.STAGE_DEBT_VIABILITY_VERSION,
                self.COMPLETION_BUDGET_VIABILITY_VERSION,
                self.COMPLETION_BUDGET_LIVENESS_VERSION,
                self.COMPLETION_BUDGET_CAPTURE_VERSION,
                self.COMPLETION_BUDGET_CAPTURE_BAND_VERSION,
                self.COMPLETION_BUDGET_PLANAR_BRAKING_VERSION,
                self.COMPLETION_BUDGET_PRE_DESCENT_SETTLING_VERSION,
                self.COMPLETION_BUDGET_REFERENCE_UNLOADING_VERSION,
                self.COMPLETION_BUDGET_REFERENCE_DEBT_UNLOADING_VERSION,
                self.COMPLETION_BUDGET_CAUSAL_REFERENCE_UNLOADING_VERSION,
                self.COMPLETION_BUDGET_CAUSAL_TRAVERSE_VERSION,
            }
            and parameters != {
                "completion_reserve_controls": 150,
                "reserve_taper_controls": 100,
                "reference_lag_full_authority_m": .008,
                "reference_lag_zero_authority_m": .012,
                "maximum_authority_decay_per_control": .05,
            }
        ):
            raise ValueError("handoff parameters require the v7 interface")
        if (
            version not in {
                self.COMPLETION_BUDGET_VIABILITY_VERSION,
                self.COMPLETION_BUDGET_LIVENESS_VERSION,
                self.COMPLETION_BUDGET_CAPTURE_VERSION,
                self.COMPLETION_BUDGET_CAPTURE_BAND_VERSION,
                self.COMPLETION_BUDGET_PLANAR_BRAKING_VERSION,
                self.COMPLETION_BUDGET_PRE_DESCENT_SETTLING_VERSION,
                self.COMPLETION_BUDGET_REFERENCE_UNLOADING_VERSION,
                self.COMPLETION_BUDGET_REFERENCE_DEBT_UNLOADING_VERSION,
                self.COMPLETION_BUDGET_CAUSAL_REFERENCE_UNLOADING_VERSION,
                self.COMPLETION_BUDGET_CAUSAL_TRAVERSE_VERSION,
            }
            and (
                completion_progress_floor_mps != .020
                or post_traverse_reserve_controls != 100
                or completion_budget_taper_controls != 60
                or force_speed_full_authority_mps != .010
                or force_speed_zero_authority_mps != .025
            )
        ):
            raise ValueError("completion-budget parameters require v9")
        self.candidate = candidate
        self.common = (
            (
                PublicCompletionBudgetCausalTraverseGuideCompletionPolicy(
                    spec, mode=mode,
                )
                if version == self.COMPLETION_BUDGET_CAUSAL_TRAVERSE_VERSION
                else PublicCompletionBudgetCausalReferenceUnloadingGuideCompletionPolicy(
                    spec, mode=mode,
                )
                if version == self.COMPLETION_BUDGET_CAUSAL_REFERENCE_UNLOADING_VERSION
                else PublicCompletionBudgetReferenceDebtUnloadingGuideCompletionPolicy(
                    spec, mode=mode,
                )
                if version == self.COMPLETION_BUDGET_REFERENCE_DEBT_UNLOADING_VERSION
                else PublicCompletionBudgetReferenceUnloadingGuideCompletionPolicy(
                    spec, mode=mode,
                )
                if version == self.COMPLETION_BUDGET_REFERENCE_UNLOADING_VERSION
                else PublicCompletionBudgetPreDescentSettlingGuideCompletionPolicy(
                    spec, mode=mode,
                )
                if version == self.COMPLETION_BUDGET_PRE_DESCENT_SETTLING_VERSION
                else PublicCompletionBudgetPlanarBrakingGuideCompletionPolicy(
                    spec, mode=mode,
                )
                if version == self.COMPLETION_BUDGET_PLANAR_BRAKING_VERSION
                else PublicCompletionBudgetCaptureBandGuideCompletionPolicy(
                    spec, mode=mode,
                )
                if version == self.COMPLETION_BUDGET_CAPTURE_BAND_VERSION
                else PublicCompletionBudgetCaptureGuideCompletionPolicy(
                    spec, mode=mode,
                )
                if version == self.COMPLETION_BUDGET_CAPTURE_VERSION
                else PublicCompletionBudgetGuideCompletionPolicy(
                    spec, mode=mode,
                )
                if version == self.COMPLETION_BUDGET_LIVENESS_VERSION
                else public_finite_state_guide_contact_policy(spec, mode=mode)
            )
            if common is None else common
        )
        self.spec = spec
        self.mode = int(mode)
        self.version = str(version)
        self.handoff_parameters = parameters
        self.reset_wrapper_state()

    def reset_wrapper_state(self):
        self.last_common_action = None
        self.last_candidate_action = None
        self.last_selected_action = None
        self.last_selector_code = self.CANDIDATE_SELECTED
        self.terminal_ownership_active = False
        self.terminal_ownership_controls = 0
        self.terminal_ownership_activations = 0
        self.candidate_authority_fraction = 1.0
        self.terminal_admission_controls = 0
        self.terminal_common_fallback_controls = 0
        self.terminal_set_controls = 0
        self.terminal_transport_controls = 0
        self.terminal_force_fallback_controls = 0
        self.terminal_blend_counts = np.zeros(
            len(self.TERMINAL_BLENDS), dtype=np.int64,
        )
        self.handoff_authority_cap = 1.0
        self.handoff_deadline_scale = 1.0
        self.handoff_reference_lag_scale = 1.0
        self.handoff_progress_scale = 1.0
        self.handoff_clearance_debt_scale = 1.0
        self.handoff_completion_budget_scale = 1.0
        self.handoff_force_speed_scale = 1.0
        self.handoff_clearance_lift_controls = 0
        self.handoff_completion_budget_controls = 0
        self.handoff_planar_remaining_m = 0.0
        self.handoff_progress_speed_mps = 0.0
        self.handoff_remaining_controls = self.spec.config.timeout_steps
        self.handoff_reference_lag_m = 0.0
        self.handoff_viability_controls = 0
        self.handoff_exact_common_controls = 0
        if self.version in {
            self.COMPLETION_BUDGET_CAUSAL_REFERENCE_UNLOADING_VERSION,
            self.COMPLETION_BUDGET_CAUSAL_TRAVERSE_VERSION,
        }:
            self.causal_reference_tick = -1
            self.causal_reference_previous_position = None
            self.causal_bounded_action_history = []
            self.causal_last_bounded_action = np.zeros(13, dtype=np.float32)
            self.causal_reference_delay_squared_errors = np.zeros(
                len(self.common.PUBLIC_REFERENCE_DELAY_CANDIDATES),
                dtype=float,
            )
            self.public_reference_delay_estimate_controls = 0
            self.public_reference_delay_identification_samples = 0
            self.public_reference_delay_score_gap_mps = -1.

    def reset(self):
        if hasattr(self.candidate, "reset"):
            self.candidate.reset()
        self.common.reset()
        self.reset_wrapper_state()

    def get_state(self):
        if not hasattr(self.candidate, "get_state"):
            raise TypeError("terminal-owned candidate is not checkpointable")
        return deepcopy({
            "version": self.version,
            "mode": self.mode,
            "candidate": self.candidate.get_state(),
            "common": self.common.get_state(),
            "last_common_action": self.last_common_action,
            "last_candidate_action": self.last_candidate_action,
            "last_selected_action": self.last_selected_action,
            "last_selector_code": self.last_selector_code,
            "terminal_ownership_active": self.terminal_ownership_active,
            "terminal_ownership_controls": self.terminal_ownership_controls,
            "terminal_ownership_activations": self.terminal_ownership_activations,
            "candidate_authority_fraction": self.candidate_authority_fraction,
            "terminal_admission_controls": self.terminal_admission_controls,
            "terminal_common_fallback_controls": (
                self.terminal_common_fallback_controls
            ),
            "terminal_set_controls": self.terminal_set_controls,
            "terminal_transport_controls": self.terminal_transport_controls,
            "terminal_force_fallback_controls": (
                self.terminal_force_fallback_controls
            ),
            "terminal_blend_counts": self.terminal_blend_counts,
            "handoff_parameters": deepcopy(self.handoff_parameters),
            "handoff_authority_cap": self.handoff_authority_cap,
            "handoff_deadline_scale": self.handoff_deadline_scale,
            "handoff_reference_lag_scale": self.handoff_reference_lag_scale,
            "handoff_progress_scale": self.handoff_progress_scale,
            "handoff_clearance_debt_scale": (
                self.handoff_clearance_debt_scale
            ),
            "handoff_completion_budget_scale": (
                self.handoff_completion_budget_scale
            ),
            "handoff_force_speed_scale": self.handoff_force_speed_scale,
            "handoff_clearance_lift_controls": (
                self.handoff_clearance_lift_controls
            ),
            "handoff_completion_budget_controls": (
                self.handoff_completion_budget_controls
            ),
            "handoff_planar_remaining_m": self.handoff_planar_remaining_m,
            "handoff_progress_speed_mps": self.handoff_progress_speed_mps,
            "handoff_remaining_controls": self.handoff_remaining_controls,
            "handoff_reference_lag_m": self.handoff_reference_lag_m,
            "handoff_viability_controls": self.handoff_viability_controls,
            "handoff_exact_common_controls": (
                self.handoff_exact_common_controls
            ),
            **({
                "causal_reference_delay": {
                    "tick": self.causal_reference_tick,
                    "previous_position": (
                        None
                        if self.causal_reference_previous_position is None
                        else self.causal_reference_previous_position.copy()
                    ),
                    "bounded_action_history": [
                        value.copy()
                        for value in self.causal_bounded_action_history
                    ],
                    "last_bounded_action": (
                        self.causal_last_bounded_action.copy()
                    ),
                    "squared_errors": (
                        self.causal_reference_delay_squared_errors.copy()
                    ),
                    "estimate_controls": (
                        self.public_reference_delay_estimate_controls
                    ),
                    "identification_samples": (
                        self.public_reference_delay_identification_samples
                    ),
                    "score_gap_mps": (
                        self.public_reference_delay_score_gap_mps
                    ),
                },
            } if self.version in {
                self.COMPLETION_BUDGET_CAUSAL_REFERENCE_UNLOADING_VERSION,
                self.COMPLETION_BUDGET_CAUSAL_TRAVERSE_VERSION,
            } else {}),
        })

    def restore(self, state):
        if (state.get("version"), int(state.get("mode", -1))) != (
            self.version, self.mode,
        ):
            raise ValueError("terminal completion ownership binding differs")
        if state.get("handoff_parameters", self.handoff_parameters) != (
            self.handoff_parameters
        ):
            raise ValueError("handoff viability parameters differ")
        if not hasattr(self.candidate, "restore"):
            raise TypeError("terminal-owned candidate is not checkpointable")
        self.candidate.restore(state["candidate"])
        self.common.restore(state["common"])
        self.last_common_action = deepcopy(state["last_common_action"])
        self.last_candidate_action = deepcopy(state["last_candidate_action"])
        self.last_selected_action = deepcopy(state["last_selected_action"])
        self.last_selector_code = int(state["last_selector_code"])
        self.terminal_ownership_active = bool(
            state["terminal_ownership_active"]
        )
        self.terminal_ownership_controls = int(
            state["terminal_ownership_controls"]
        )
        self.terminal_ownership_activations = int(
            state["terminal_ownership_activations"]
        )
        self.candidate_authority_fraction = float(
            state["candidate_authority_fraction"]
        )
        self.terminal_admission_controls = int(
            state["terminal_admission_controls"]
        )
        self.terminal_common_fallback_controls = int(
            state["terminal_common_fallback_controls"]
        )
        self.terminal_set_controls = int(state["terminal_set_controls"])
        self.terminal_transport_controls = int(
            state.get("terminal_transport_controls", 0)
        )
        self.terminal_force_fallback_controls = int(
            state.get("terminal_force_fallback_controls", 0)
        )
        self.terminal_blend_counts = np.asarray(
            state["terminal_blend_counts"], dtype=np.int64,
        ).copy()
        self.handoff_authority_cap = float(
            state.get("handoff_authority_cap", 1.)
        )
        self.handoff_deadline_scale = float(
            state.get("handoff_deadline_scale", 1.)
        )
        self.handoff_reference_lag_scale = float(
            state.get("handoff_reference_lag_scale", 1.)
        )
        self.handoff_progress_scale = float(
            state.get("handoff_progress_scale", 1.)
        )
        self.handoff_clearance_debt_scale = float(
            state.get("handoff_clearance_debt_scale", 1.)
        )
        self.handoff_completion_budget_scale = float(
            state.get("handoff_completion_budget_scale", 1.)
        )
        self.handoff_force_speed_scale = float(
            state.get("handoff_force_speed_scale", 1.)
        )
        self.handoff_clearance_lift_controls = int(
            state.get("handoff_clearance_lift_controls", 0)
        )
        self.handoff_completion_budget_controls = int(
            state.get("handoff_completion_budget_controls", 0)
        )
        self.handoff_planar_remaining_m = float(
            state.get("handoff_planar_remaining_m", 0.)
        )
        self.handoff_progress_speed_mps = float(
            state.get("handoff_progress_speed_mps", 0.)
        )
        self.handoff_remaining_controls = int(
            state.get("handoff_remaining_controls", self.spec.config.timeout_steps)
        )
        self.handoff_reference_lag_m = float(
            state.get("handoff_reference_lag_m", 0.)
        )
        self.handoff_viability_controls = int(
            state.get("handoff_viability_controls", 0)
        )
        self.handoff_exact_common_controls = int(
            state.get("handoff_exact_common_controls", 0)
        )
        if self.version in {
            self.COMPLETION_BUDGET_CAUSAL_REFERENCE_UNLOADING_VERSION,
            self.COMPLETION_BUDGET_CAUSAL_TRAVERSE_VERSION,
        }:
            causal = state["causal_reference_delay"]
            self.causal_reference_tick = int(causal["tick"])
            previous = causal["previous_position"]
            self.causal_reference_previous_position = (
                None if previous is None
                else np.asarray(previous, dtype=float).copy()
            )
            self.causal_bounded_action_history = [
                np.asarray(value, dtype=float).copy()
                for value in causal["bounded_action_history"]
            ]
            self.causal_last_bounded_action = np.asarray(
                causal["last_bounded_action"], dtype=np.float32,
            ).copy()
            self.causal_reference_delay_squared_errors = np.asarray(
                causal["squared_errors"], dtype=float,
            ).copy()
            self.public_reference_delay_estimate_controls = int(
                causal["estimate_controls"]
            )
            self.public_reference_delay_identification_samples = int(
                causal["identification_samples"]
            )
            self.public_reference_delay_score_gap_mps = float(
                causal["score_gap_mps"]
            )
            maximum_history = (
                max(self.common.PUBLIC_REFERENCE_DELAY_CANDIDATES) + 1
            )
            if (
                self.causal_reference_delay_squared_errors.shape
                != (len(self.common.PUBLIC_REFERENCE_DELAY_CANDIDATES),)
                or self.causal_last_bounded_action.shape != (13,)
                or any(
                    value.shape != (2,)
                    for value in self.causal_bounded_action_history
                )
                or len(self.causal_bounded_action_history) > maximum_history
                or self.public_reference_delay_estimate_controls
                not in (0, *self.common.PUBLIC_REFERENCE_DELAY_CANDIDATES)
                or not 0 <= (
                    self.public_reference_delay_identification_samples
                ) <= self.common.PUBLIC_REFERENCE_DELAY_MINIMUM_SAMPLES
                or not np.isfinite(
                    self.public_reference_delay_score_gap_mps
                )
            ):
                raise ValueError("causal wrapper checkpoint differs")
            common_identity = (
                self.common.public_reference_delay_estimate_controls,
                self.common.public_reference_delay_identification_samples,
                self.common.public_reference_delay_score_gap_mps,
            )
            if common_identity != (
                self.public_reference_delay_estimate_controls,
                self.public_reference_delay_identification_samples,
                self.public_reference_delay_score_gap_mps,
            ):
                raise ValueError("causal wrapper/common checkpoint differs")

    def _observe_causal_reference_delay(self, public):
        if (
            self.version not in {
                self.COMPLETION_BUDGET_CAUSAL_REFERENCE_UNLOADING_VERSION,
                self.COMPLETION_BUDGET_CAUSAL_TRAVERSE_VERSION,
            }
            or self.mode != 1
        ):
            return False
        tick = int(np.rint(
            np.asarray(public["elapsed_seconds"]).item()
            / self.spec.config.control_dt
        ))
        if tick == self.causal_reference_tick:
            return False
        if self.causal_reference_tick >= 0 and tick != (
            self.causal_reference_tick + 1
        ):
            raise ValueError("every public control tick required")

        reference = np.asarray(
            public["reference_position"], dtype=float,
        )[0, :2]
        previous_reference = self.causal_reference_previous_position
        if (
            np.isfinite(reference).all()
            and previous_reference is not None
            and self.public_reference_delay_estimate_controls == 0
            and self.public_reference_delay_identification_samples
            < self.common.PUBLIC_REFERENCE_DELAY_MINIMUM_SAMPLES
            and len(self.causal_bounded_action_history)
            >= max(self.common.PUBLIC_REFERENCE_DELAY_CANDIDATES) + 1
        ):
            candidates = np.stack([
                self.causal_bounded_action_history[-(delay + 1)]
                for delay in self.common.PUBLIC_REFERENCE_DELAY_CANDIDATES
            ])
            separation = max(
                float(np.linalg.norm(left - right))
                for left in candidates for right in candidates
            )
            if separation >= (
                self.common.PUBLIC_REFERENCE_DELAY_EXCITATION_THRESHOLD_MPS
            ):
                observed_velocity = (
                    reference - previous_reference
                ) / self.spec.config.control_dt
                self.causal_reference_delay_squared_errors += np.sum(
                    np.square(candidates - observed_velocity[None]), axis=1,
                )
                self.public_reference_delay_identification_samples += 1
                if self.public_reference_delay_identification_samples == (
                    self.common.PUBLIC_REFERENCE_DELAY_MINIMUM_SAMPLES
                ):
                    scores = np.sqrt(
                        self.causal_reference_delay_squared_errors
                        / self.public_reference_delay_identification_samples
                    )
                    if np.isfinite(scores).all():
                        ordering = np.argsort(scores, kind="stable")
                        self.public_reference_delay_score_gap_mps = float(
                            scores[ordering[1]] - scores[ordering[0]]
                        )
                        if self.public_reference_delay_score_gap_mps > 0.:
                            self.public_reference_delay_estimate_controls = int(
                                self.common.PUBLIC_REFERENCE_DELAY_CANDIDATES[
                                    int(ordering[0])
                                ]
                            )
        if np.isfinite(reference).all():
            self.causal_reference_previous_position = reference.copy()
        self.causal_reference_tick = tick
        self.common.set_public_reference_delay(
            self.public_reference_delay_estimate_controls,
            self.public_reference_delay_identification_samples,
            self.public_reference_delay_score_gap_mps,
        )
        return True

    def _append_causal_bounded_action(self, selected):
        bounded = bounded_contact_command(
            np,
            np.asarray(selected[0], dtype=np.float32),
            self.causal_last_bounded_action,
            self.spec.config,
        )
        self.causal_last_bounded_action = np.asarray(
            bounded, dtype=np.float32,
        ).copy()
        self.causal_bounded_action_history.append(
            self.causal_last_bounded_action[:2].astype(float, copy=True)
        )
        maximum_history = max(
            self.common.PUBLIC_REFERENCE_DELAY_CANDIDATES,
        ) + 1
        self.causal_bounded_action_history = (
            self.causal_bounded_action_history[-maximum_history:]
        )

    def _clearance_progress_authority(self, public):
        stage = int(self.common.completion.stage)
        if stage <= 0:
            return 1.0
        if stage >= 2:
            return 0.0
        initial = np.asarray(self.common.completion.initial, dtype=float)[0, :2]
        goal = np.asarray(self.spec.config.goal, dtype=float)[:2]
        position = np.asarray(public["tcp_position"], dtype=float)[0, :2]
        start_distance = float(np.linalg.norm(initial - goal))
        if start_distance <= self.spec.config.position_tolerance:
            return 0.0
        remaining = float(np.linalg.norm(position - goal))
        return float(np.clip(remaining / start_distance, 0.0, 1.0))

    def _handoff_viability_authority(self, public):
        self.handoff_viability_controls += 1
        elapsed = float(np.asarray(public["elapsed_seconds"]).reshape(-1)[0])
        tick = int(np.rint(elapsed / self.spec.config.control_dt))
        self.handoff_remaining_controls = max(
            0, self.spec.config.timeout_steps - tick,
        )
        reference = np.asarray(public["reference_position"], dtype=float)[0]
        position = np.asarray(public["tcp_position"], dtype=float)[0]
        self.handoff_reference_lag_m = float(np.linalg.norm(reference - position))
        parameters = self.handoff_parameters
        self.handoff_deadline_scale = float(np.clip(
            (
                self.handoff_remaining_controls
                - parameters["completion_reserve_controls"]
            ) / parameters["reserve_taper_controls"],
            0., 1.,
        ))
        self.handoff_reference_lag_scale = float(np.clip(
            (
                parameters["reference_lag_zero_authority_m"]
                - self.handoff_reference_lag_m
            ) / (
                parameters["reference_lag_zero_authority_m"]
                - parameters["reference_lag_full_authority_m"]
            ),
            0., 1.,
        ))
        self.handoff_progress_scale = self._clearance_progress_authority(public)
        stage = int(self.common.completion.stage)
        self.handoff_clearance_lift_controls = int(
            getattr(self.common, "clearance_lift_controls", 0)
        )
        self.handoff_clearance_debt_scale = 1.0
        if (
            self.version == self.STAGE_DEBT_VIABILITY_VERSION
            and stage <= 0
        ):
            self.handoff_clearance_debt_scale = float(np.clip(
                (
                    parameters["clearance_debt_zero_controls"]
                    - self.handoff_clearance_lift_controls
                ) / (
                    parameters["clearance_debt_zero_controls"]
                    - parameters["clearance_debt_full_controls"]
                ),
                0., 1.,
            ))
        self.handoff_completion_budget_scale = 1.0
        self.handoff_force_speed_scale = 1.0
        self.handoff_completion_budget_controls = 0
        self.handoff_planar_remaining_m = 0.0
        self.handoff_progress_speed_mps = max(
            0., float(getattr(self.common, "achieved_speed_mps", 0.)),
        )
        if self.version in {
            self.COMPLETION_BUDGET_VIABILITY_VERSION,
            self.COMPLETION_BUDGET_LIVENESS_VERSION,
            self.COMPLETION_BUDGET_CAPTURE_VERSION,
            self.COMPLETION_BUDGET_CAPTURE_BAND_VERSION,
            self.COMPLETION_BUDGET_PLANAR_BRAKING_VERSION,
            self.COMPLETION_BUDGET_PRE_DESCENT_SETTLING_VERSION,
            self.COMPLETION_BUDGET_REFERENCE_UNLOADING_VERSION,
            self.COMPLETION_BUDGET_REFERENCE_DEBT_UNLOADING_VERSION,
            self.COMPLETION_BUDGET_CAUSAL_REFERENCE_UNLOADING_VERSION,
            self.COMPLETION_BUDGET_CAUSAL_TRAVERSE_VERSION,
        }:
            if stage == 1:
                goal = np.asarray(self.spec.config.goal, dtype=float)[:2]
                planar = np.asarray(public["tcp_position"], dtype=float)[0, :2]
                self.handoff_planar_remaining_m = max(
                    0., float(np.linalg.norm(goal - planar))
                    - self.spec.config.position_tolerance,
                )
                traverse_controls = int(np.ceil(
                    self.handoff_planar_remaining_m
                    / (
                        parameters["completion_progress_floor_mps"]
                        * self.spec.config.control_dt
                    )
                ))
                self.handoff_completion_budget_controls = (
                    traverse_controls
                    + parameters["post_traverse_reserve_controls"]
                )
                self.handoff_completion_budget_scale = float(np.clip(
                    (
                        self.handoff_remaining_controls
                        - self.handoff_completion_budget_controls
                    ) / parameters["completion_budget_taper_controls"],
                    0., 1.,
                ))
                self.handoff_force_speed_scale = float(np.clip(
                    (
                        parameters["force_speed_zero_authority_mps"]
                        - self.handoff_progress_speed_mps
                    ) / (
                        parameters["force_speed_zero_authority_mps"]
                        - parameters["force_speed_full_authority_mps"]
                    ),
                    0., 1.,
                ))
            elif stage >= 2:
                self.handoff_completion_budget_scale = 0.0
        if stage >= 2:
            self.handoff_authority_cap = 0.0
        else:
            target = min(
                self.handoff_deadline_scale,
                self.handoff_reference_lag_scale,
                self.handoff_progress_scale,
                self.handoff_clearance_debt_scale,
                self.handoff_completion_budget_scale,
                self.handoff_force_speed_scale,
            )
            rate_limited = max(
                target,
                self.handoff_authority_cap
                - parameters["maximum_authority_decay_per_control"],
            )
            self.handoff_authority_cap = min(
                self.handoff_authority_cap,
                rate_limited,
                self.handoff_progress_scale,
            )
        if self.handoff_authority_cap <= 1e-7:
            self.handoff_authority_cap = 0.0
            self.handoff_exact_common_controls += 1
        return self.handoff_authority_cap

    def _candidate_authority(self, public):
        if self.version not in {
            self.CLEARANCE_PROGRESS_VERSION,
            self.TERMINAL_SET_ADMISSION_VERSION,
            self.TERMINAL_TRANSPORT_VERSION,
            self.TERMINAL_STRICT_PROGRESS_VERSION,
            self.HANDOFF_VIABILITY_VERSION,
            self.STAGE_DEBT_VIABILITY_VERSION,
            self.COMPLETION_BUDGET_VIABILITY_VERSION,
            self.COMPLETION_BUDGET_LIVENESS_VERSION,
            self.COMPLETION_BUDGET_CAPTURE_VERSION,
            self.COMPLETION_BUDGET_CAPTURE_BAND_VERSION,
            self.COMPLETION_BUDGET_PLANAR_BRAKING_VERSION,
            self.COMPLETION_BUDGET_PRE_DESCENT_SETTLING_VERSION,
            self.COMPLETION_BUDGET_REFERENCE_UNLOADING_VERSION,
            self.COMPLETION_BUDGET_REFERENCE_DEBT_UNLOADING_VERSION,
            self.COMPLETION_BUDGET_CAUSAL_REFERENCE_UNLOADING_VERSION,
            self.COMPLETION_BUDGET_CAUSAL_TRAVERSE_VERSION,
        }:
            return float(not (
                self.common.terminal_pose_active
                if self.version == self.TERMINAL_CAPTURE_VERSION
                else (
                    self.common.switched
                    if self.mode == 1 else self.common.terminal_pose_active
                )
            ))
        if self.mode != 1 or not self.common.switched:
            return 1.0
        if self.version in {
            self.HANDOFF_VIABILITY_VERSION,
            self.STAGE_DEBT_VIABILITY_VERSION,
            self.COMPLETION_BUDGET_VIABILITY_VERSION,
            self.COMPLETION_BUDGET_LIVENESS_VERSION,
            self.COMPLETION_BUDGET_CAPTURE_VERSION,
            self.COMPLETION_BUDGET_CAPTURE_BAND_VERSION,
            self.COMPLETION_BUDGET_PLANAR_BRAKING_VERSION,
            self.COMPLETION_BUDGET_PRE_DESCENT_SETTLING_VERSION,
            self.COMPLETION_BUDGET_REFERENCE_UNLOADING_VERSION,
            self.COMPLETION_BUDGET_REFERENCE_DEBT_UNLOADING_VERSION,
            self.COMPLETION_BUDGET_CAUSAL_REFERENCE_UNLOADING_VERSION,
            self.COMPLETION_BUDGET_CAUSAL_TRAVERSE_VERSION,
        }:
            return self._handoff_viability_authority(public)
        return self._clearance_progress_authority(public)

    def _terminal_set_distances(self, public, action):
        dt = self.spec.config.control_dt
        position = np.asarray(public["tcp_position"], dtype=float)[0]
        goal = np.asarray(public["goal_position"], dtype=float)[0]
        rotation = np.asarray(public["tcp_rotation"], dtype=float)[0]
        goal_rotation = np.asarray(public["goal_rotation"], dtype=float)[0]
        position_error = goal - position
        rotation_error = orientation_error(np, rotation, goal_rotation)
        predicted_position_error = position_error - dt * action[0, :3]
        predicted_rotation_error = rotation_error - dt * action[0, 3:6]
        return (
            max(
                0., float(np.linalg.norm(predicted_position_error))
                - self.spec.config.position_tolerance,
            ),
            max(
                0., float(np.linalg.norm(predicted_rotation_error))
                - self.spec.config.orientation_tolerance,
            ),
            bool(
                np.linalg.norm(position_error)
                <= self.spec.config.position_tolerance
                and np.linalg.norm(rotation_error)
                <= self.spec.config.orientation_tolerance
            ),
        )

    def _terminal_force_scale(self):
        utilization = float(getattr(self.common, "force_utilization", 0.))
        lower = PublicResponseNormalizedGuideCompletionPolicy.FORCE_GROWTH_UTILIZATION
        upper = PublicResponseNormalizedGuideCompletionPolicy.FORCE_RELEASE_UTILIZATION
        return float(np.clip(
            (upper - utilization) / (upper - lower), 0., 1.,
        ))

    def _terminal_set_admission(self, public, candidate, common):
        common_position, common_rotation, inside = self._terminal_set_distances(
            public, common,
        )
        if inside:
            self.terminal_set_controls += 1
            self.terminal_common_fallback_controls += 1
            self.terminal_blend_counts[-1] += 1
            return common.copy(), 0.0, self.COMMON_TERMINAL_SELECTED

        force_scale = self._terminal_force_scale()
        residual = np.zeros_like(common)
        coordinates = list(self.TRANSPORT_COORDINATES)
        residual[:, coordinates] = candidate[:, coordinates] - common[:, coordinates]
        for index, fraction in enumerate(self.TERMINAL_BLENDS):
            admitted_fraction = float(fraction * force_scale)
            trial = common + admitted_fraction * residual
            position_distance, rotation_distance, _ = self._terminal_set_distances(
                public, trial,
            )
            if (
                position_distance
                <= common_position + self.PROGRESS_TOLERANCE
                and rotation_distance
                <= common_rotation + self.PROGRESS_TOLERANCE
                and (
                    self.version != self.TERMINAL_STRICT_PROGRESS_VERSION
                    or position_distance
                    < common_position - self.PROGRESS_TOLERANCE
                    or rotation_distance
                    < common_rotation - self.PROGRESS_TOLERANCE
                )
            ):
                self.terminal_blend_counts[index] += 1
                if admitted_fraction > 1e-7:
                    self.terminal_admission_controls += 1
                    code = self.TERMINAL_PROGRESS_BLEND_SELECTED
                else:
                    self.terminal_common_fallback_controls += 1
                    code = self.COMMON_TERMINAL_SELECTED
                return trial.astype(np.float32), admitted_fraction, code
        if self.version == self.TERMINAL_STRICT_PROGRESS_VERSION:
            self.terminal_blend_counts[-1] += 1
            self.terminal_common_fallback_controls += 1
            return common.copy(), 0.0, self.COMMON_TERMINAL_SELECTED
        raise RuntimeError("zero terminal residual must preserve common progress")

    def _terminal_transport(self, public, candidate, common):
        _, _, inside = self._terminal_set_distances(public, common)
        if inside:
            self.terminal_set_controls += 1
        force_scale = self._terminal_force_scale()
        residual = np.zeros_like(common)
        coordinates = list(self.TRANSPORT_COORDINATES)
        residual[:, coordinates] = (
            candidate[:, coordinates] - common[:, coordinates]
        )
        selected = common + force_scale * residual
        if force_scale > 1e-7:
            self.terminal_transport_controls += 1
            code = self.TERMINAL_TRANSPORT_SELECTED
        else:
            self.terminal_force_fallback_controls += 1
            code = self.COMMON_TERMINAL_SELECTED
        return selected.astype(np.float32), force_scale, code

    def act(self, public):
        causal_new_tick = self._observe_causal_reference_delay(public)
        candidate = np.asarray(self.candidate.act(public), dtype=np.float32)
        common = np.asarray(self.common.act(public), dtype=np.float32)
        if candidate.shape != (1, 13) or common.shape != (1, 13):
            raise ValueError("terminal ownership requires batch-one shared13 requests")
        if not np.isfinite(candidate).all() or not np.isfinite(common).all():
            raise ValueError("terminal ownership received a nonfinite request")
        terminal_admission = bool(
            self.version in {
                self.TERMINAL_SET_ADMISSION_VERSION,
                self.TERMINAL_STRICT_PROGRESS_VERSION,
            }
            and self.mode == 1
            and self.common.switched
            and int(self.common.completion.stage) >= 2
        )
        terminal_transport = bool(
            self.version == self.TERMINAL_TRANSPORT_VERSION
            and self.mode == 1
            and self.common.switched
            and int(self.common.completion.stage) >= 2
        )
        handoff_viability = bool(
            self.version in {
                self.HANDOFF_VIABILITY_VERSION,
                self.STAGE_DEBT_VIABILITY_VERSION,
                self.COMPLETION_BUDGET_VIABILITY_VERSION,
                self.COMPLETION_BUDGET_LIVENESS_VERSION,
                self.COMPLETION_BUDGET_CAPTURE_VERSION,
                self.COMPLETION_BUDGET_CAPTURE_BAND_VERSION,
                self.COMPLETION_BUDGET_PLANAR_BRAKING_VERSION,
                self.COMPLETION_BUDGET_PRE_DESCENT_SETTLING_VERSION,
                self.COMPLETION_BUDGET_REFERENCE_UNLOADING_VERSION,
                self.COMPLETION_BUDGET_REFERENCE_DEBT_UNLOADING_VERSION,
                self.COMPLETION_BUDGET_CAUSAL_REFERENCE_UNLOADING_VERSION,
                self.COMPLETION_BUDGET_CAUSAL_TRAVERSE_VERSION,
            }
            and self.mode == 1
            and self.common.switched
        )
        if terminal_transport:
            selected, authority, code = self._terminal_transport(
                public, candidate, common,
            )
        elif terminal_admission:
            selected, authority, code = self._terminal_set_admission(
                public, candidate, common,
            )
        elif handoff_viability:
            authority = self._candidate_authority(public)
            residual = np.zeros_like(common)
            coordinates = list(self.TRANSPORT_COORDINATES)
            residual[:, coordinates] = (
                candidate[:, coordinates] - common[:, coordinates]
            )
            selected = (common + authority * residual).astype(np.float32)
            code = (
                self.COMMON_TERMINAL_SELECTED
                if authority <= 1e-7 else self.HANDOFF_VIABILITY_SELECTED
            )
        else:
            authority = self._candidate_authority(public)
            selected = None
            code = None
        active = authority < 1.0 - 1e-7
        if active and not self.terminal_ownership_active:
            self.terminal_ownership_activations += 1
        self.terminal_ownership_active = active
        self.candidate_authority_fraction = authority
        if terminal_admission or terminal_transport or handoff_viability:
            self.terminal_ownership_controls += 1
            self.last_selector_code = code
        elif authority <= 1e-7:
            selected = common.copy()
            self.terminal_ownership_controls += 1
            self.last_selector_code = self.COMMON_TERMINAL_SELECTED
        elif active:
            selected = common + authority * (candidate - common)
            selected = selected.astype(np.float32)
            self.terminal_ownership_controls += 1
            self.last_selector_code = self.CLEARANCE_BLEND_SELECTED
        else:
            selected = candidate.copy()
            self.last_selector_code = self.CANDIDATE_SELECTED
        self.last_candidate_action = candidate.copy()
        self.last_common_action = common.copy()
        self.last_selected_action = selected.copy()
        if causal_new_tick:
            self._append_causal_bounded_action(selected)
        return selected

    def __getattr__(self, name):
        # Resource accounting delegates to the candidate.  Diagnostics use
        # vars(self), so candidate counters remain nested rather than copied.
        return getattr(self.candidate, name)
