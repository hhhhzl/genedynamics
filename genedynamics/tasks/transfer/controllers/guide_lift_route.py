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
    VERSION='public-T2-knowngeometry-lift-traverse-descend-v2'
    HORIZONTAL_SPEED_LIMIT_MPS=.06
    VERTICAL_SPEED_LIMIT_MPS=.06
    FREE_DESCENT_SPEED_MPS=.06
    CONTACT_BAND_M=.002
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
        if (self.stage==2 and np.linalg.norm(p[0,:2]-goal[:2])<=tol
                and np.linalg.norm(np.asarray(public['tcp_twist'])[0,:2])<=.004):self.stage=3
        target=self.initial.copy();target[:,2]=CLEAR_Z
        if self.stage>=1:target[:,axis]=goal[axis]
        if self.stage>=2:target[:,:2]=goal[:2]
        if self.stage==3:target[:,2]=goal[2]
        desired=np.broadcast_to(np.diag([1.,-1.,-1.]),r.shape);angular=np.stack([orientation_error(np,a,b) for a,b in zip(r,desired)]);issued_angular=np.stack([orientation_error(np,a,b) for a,b in zip(public['reference_rotation'],desired)]);error=np.concatenate((target-p,angular),-1);issued=np.concatenate((target-rp,issued_angular),-1);self.compensation=np.clip(self.compensation+2.*dt*error,-.04,.04)
        if self.stage==3:self.compensation[:,2]=0.
        velocity=np.asarray([6.]*3+[4.]*3)*error+2.*issued-np.asarray(public['tcp_twist'])+self.compensation;velocity[:,:2]=np.clip(velocity[:,:2],-self.HORIZONTAL_SPEED_LIMIT_MPS,self.HORIZONTAL_SPEED_LIMIT_MPS);velocity[:,2]=np.clip(velocity[:,2],-self.VERTICAL_SPEED_LIMIT_MPS,self.VERTICAL_SPEED_LIMIT_MPS);velocity[:,3:]=np.clip(velocity[:,3:],-.12,.12);stiffness=np.ones((1,6));force=np.zeros((1,1))
        if self.stage==2 and np.linalg.norm(p[0,:2]-goal[:2])<.012:
            self.compensation[:,:2]=0.;velocity[:,:2]=np.clip(4.*(goal[:2]-p[:,:2])-np.asarray(public['tcp_twist'])[:,:2],-.01,.01)
        if self.stage==3:
            surface_height=.586+.1*(p[:,0]-.38)+.1*(p[:,1]+.20);clearance=p[:,2]-surface_height;contact_velocity=np.clip(.004*(np.asarray(public['measured_wrench'])[:,2]-5.),-.01,.01);free_velocity=-np.clip(4.*clearance,.01,self.FREE_DESCENT_SPEED_MPS);velocity[:,2]=np.where(clearance>self.CONTACT_BAND_M,free_velocity,contact_velocity);stiffness[:,2]=0.;velocity[:,:2]=np.clip(velocity[:,:2],(p[:,:2]-.008-rp[:,:2])/dt,(p[:,:2]+.008-rp[:,:2])/dt);velocity[:,:2]=np.clip(velocity[:,:2],-self.HORIZONTAL_SPEED_LIMIT_MPS,self.HORIZONTAL_SPEED_LIMIT_MPS)
        self.cached=np.concatenate((velocity,stiffness,force),-1).astype(np.float32);self.tick=tick;return self.cached.copy()


class PublicGuideConcurrentMonotoneLiftRoutePolicy(PublicGuideLiftRoutePolicy):
    """Traverse and lift together, then descend without reference overshoot."""

    VERSION = "public-T2-concurrent-monotone-lift-route-v1"
    PLANAR_LEAD_M = np.asarray([.008, .008])
    TRANSITION_TOLERANCE_SCALE = .75
    CLAMP_PLANAR_TO_TARGET_DURING_DESCENT = True
    TRAVERSE_DURING_CLEARANCE = True
    CLEARANCE_FIRST = False

    @staticmethod
    def _toward_target(velocity, reference, target, dt):
        remaining = (target - reference) / dt
        return np.clip(
            velocity, np.minimum(remaining, 0.), np.maximum(remaining, 0.),
        )

    def _ready_to_descend(self, position, goal, tolerance):
        return (
            position[0, 2] >= CLEAR_Z - .002
            and np.linalg.norm(position[0, :2] - goal[:2]) <= tolerance
        )

    def act(self, public):
        tick = int(np.rint(
            np.asarray(public["elapsed_seconds"]).item()
            / self.spec.config.control_dt
        ))
        if tick == self.tick:
            return self.cached.copy()
        if tick != self.tick + 1:
            raise ValueError("every public control tick required")
        if self.base is not None:
            self.cached = self.base.act(public).copy()
            self.tick = tick
            return self.cached.copy()
        if len(public["tcp_position"]) != 1:
            raise ValueError("explicit batch1")

        p = np.asarray(public["tcp_position"], dtype=float)
        r = np.asarray(public["tcp_rotation"], dtype=float)
        rp = np.asarray(public["reference_position"], dtype=float)
        rr = np.asarray(public["reference_rotation"], dtype=float)
        twist = np.asarray(public["tcp_twist"], dtype=float)
        dt = self.spec.config.control_dt
        goal = np.asarray(self.spec.config.goal, dtype=float)
        if self.initial is None:
            self.initial = p.copy()

        tolerance = (
            self.TRANSITION_TOLERANCE_SCALE
            * self.spec.config.position_tolerance
        )
        if self.CLEARANCE_FIRST:
            if self.stage == 0 and p[0, 2] >= CLEAR_Z - .002:
                self.stage = 1
            if (
                self.stage == 1
                and np.linalg.norm(p[0, :2] - goal[:2]) <= tolerance
                and np.linalg.norm(twist[0, :2]) <= .004
            ):
                self.stage = 2
        elif self.stage == 0 and self._ready_to_descend(p, goal, tolerance):
            self.stage = 1

        target = np.broadcast_to(goal, p.shape).copy()
        if self.CLEARANCE_FIRST and self.stage < 2:
            target[:, 2] = CLEAR_Z
            if self.stage == 0:
                target[:, :2] = self.initial[:, :2]
        elif self.stage == 0:
            target[:, 2] = CLEAR_Z
            if not self.TRAVERSE_DURING_CLEARANCE:
                target[:, :2] = self.initial[:, :2]
        desired_rotation = np.broadcast_to(np.diag([1., -1., -1.]), r.shape)
        angular_error = np.stack([
            orientation_error(np, actual, desired)
            for actual, desired in zip(r, desired_rotation)
        ])
        issued_angular_error = np.stack([
            orientation_error(np, actual, desired)
            for actual, desired in zip(rr, desired_rotation)
        ])
        translation = (
            6. * (target - p) + 2. * (target - rp) - twist[:, :3]
        )
        limits = np.asarray([
            self.HORIZONTAL_SPEED_LIMIT_MPS,
            self.HORIZONTAL_SPEED_LIMIT_MPS,
            self.VERTICAL_SPEED_LIMIT_MPS,
        ])
        translation = np.clip(translation, -limits, limits)
        if self.stage == 0 or (self.CLEARANCE_FIRST and self.stage == 1):
            translation = self._toward_target(translation, rp, target, dt)
        elif self.CLAMP_PLANAR_TO_TARGET_DURING_DESCENT:
            translation[:, :2] = self._toward_target(
                translation[:, :2], rp[:, :2], target[:, :2], dt,
            )
        planar_lead = self.PLANAR_LEAD_M
        translation[:, :2] = np.clip(
            translation[:, :2],
            (p[:, :2] - planar_lead - rp[:, :2]) / dt,
            (p[:, :2] + planar_lead - rp[:, :2]) / dt,
        )
        rotation = (
            4. * angular_error + 2. * issued_angular_error - twist[:, 3:]
        )
        rotation = np.clip(rotation, -.12, .12)
        stiffness = np.ones((1, 6))
        force = np.zeros((1, 1))

        descending = self.stage == (2 if self.CLEARANCE_FIRST else 1)
        if descending:
            surface_height = (
                .586 + .1 * (p[:, 0] - .38) + .1 * (p[:, 1] + .20)
            )
            clearance = p[:, 2] - surface_height
            contact_velocity = np.clip(
                .004 * (np.asarray(public["measured_wrench"])[:, 2] - 5.),
                -.01, .01,
            )
            free_velocity = -np.clip(
                4. * clearance, .01, self.FREE_DESCENT_SPEED_MPS,
            )
            translation[:, 2] = np.where(
                clearance > self.CONTACT_BAND_M,
                free_velocity,
                contact_velocity,
            )
            stiffness[:, 2] = 0.

        self.cached = np.concatenate(
            (translation, rotation, stiffness, force), axis=-1,
        ).astype(np.float32)
        self.tick = tick
        return self.cached.copy()


class PublicGuideConcurrentMonotoneLiftRoutePolicyV2(
    PublicGuideConcurrentMonotoneLiftRoutePolicy
):
    """Use the frozen task tolerance for the free-space stage transition."""

    VERSION = "public-T2-concurrent-monotone-lift-route-v2"
    TRANSITION_TOLERANCE_SCALE = 1.


class PublicGuideConcurrentMonotoneLiftRoutePolicyV3(
    PublicGuideConcurrentMonotoneLiftRoutePolicyV2
):
    """Keep monotone free travel, then permit bounded terminal feedback."""

    VERSION = "public-T2-concurrent-monotone-lift-route-v3"
    CLAMP_PLANAR_TO_TARGET_DURING_DESCENT = False


class PublicGuideBoundedClearanceCompletionPolicy(
    PublicGuideConcurrentMonotoneLiftRoutePolicyV3
):
    """Descend once planar completion has finite public surface clearance."""

    VERSION = "public-T2-bounded-clearance-completion-route-v4"
    MINIMUM_SURFACE_CLEARANCE_M = .006

    def _ready_to_descend(self, position, goal, tolerance):
        surface_height = (
            .586
            + .1 * (position[0, 0] - .38)
            + .1 * (position[0, 1] + .20)
        )
        return (
            position[0, 2]
            >= surface_height + self.MINIMUM_SURFACE_CLEARANCE_M
            and np.linalg.norm(position[0, :2] - goal[:2]) <= tolerance
        )


class PublicGuideResponseNormalizedCompletionRoutePolicy(
    PublicGuideBoundedClearanceCompletionPolicy
):
    """Lift in place, traverse at clearance, then descend at the goal."""

    VERSION = "public-T2-response-normalized-clearance-traverse-route-v3"
    TRAVERSE_DURING_CLEARANCE = False
    CLEARANCE_FIRST = True

    def _ready_to_descend(self, position, goal, tolerance):
        del goal, tolerance
        return position[0, 2] >= CLEAR_Z - .002
