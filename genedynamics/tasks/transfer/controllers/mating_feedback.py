"""Public T3/T4 mating feedback binding; physical task identity stays explicit."""
from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.bias_feedback import PublicBiasFeedbackPolicy
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.liveness import SafeSettledRetreatReleaseMixin


class PublicMatingFeedbackPolicy(PublicBiasFeedbackPolicy):
    """Reuse the qualified public recovery geometry on the actual T4 body.

    Only the observer template uses T3's recovery clock. This class neither
    creates a simulator nor accesses WBC/actuator carry or axial-support truth.
    T4 physical reachability and body stability require separate qualification.
    """
    def __init__(self,spec,**kwargs):
        if spec.alias not in {'T3','T4'} or not spec.config.insertion:
            raise ValueError('matched mating T3/T4 public feedback binding required')
        self.physical_task_spec=spec
        super().__init__(replace(spec,alias='T3'),**kwargs)

    def reset(self,**kwargs):
        self.spec=replace(self.physical_task_spec,alias='T3')
        try:super().reset(**kwargs)
        finally:self.spec=self.physical_task_spec


class PublicLivenessMatingFeedbackPolicy(SafeSettledRetreatReleaseMixin,PublicMatingFeedbackPolicy):
    """P5 T3/T4 binding with the safe-settled retreat release."""

    VERSION='public-mating-feedback-safe-settled-v1'


class PublicEarlyContactMatingFeedbackPolicy(PublicLivenessMatingFeedbackPolicy):
    """Development-only public release before the T3 loaded recovery impulse.

    A height crossing starts the already defined phase-1 withdrawal. Until the
    TCP clears the jam region, the actor asks only for bounded upward motion;
    it has no access to the simulator's substep peak, delay, or actuator carry.
    The old liveness actor and all frozen checkpoints remain unchanged.
    """

    VERSION = 'public-mating-feedback-early-contact-release-v1'
    RELEASE_Z_M = .5965
    CLEAR_Z_M = .604
    RETREAT_SPEED_MPS = .008

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.early_release_seen = np.zeros(len(self._reset_modes), dtype=bool)
        self.early_guard_active = np.zeros(len(self._reset_modes), dtype=bool)

    def get_state(self):
        state = super().get_state()
        state['early_contact_release'] = deepcopy({
            'seen': self.early_release_seen,
            'active': self.early_guard_active,
        })
        return state

    def restore(self, state):
        value = deepcopy(state)
        guard = value.pop('early_contact_release')
        super().restore(value)
        self.early_release_seen = np.asarray(guard['seen'], dtype=bool).copy()
        self.early_guard_active = np.asarray(guard['active'], dtype=bool).copy()

    def reorder(self, indices):
        super().reorder(indices)
        self.early_release_seen = self.early_release_seen[indices].copy()
        self.early_guard_active = self.early_guard_active[indices].copy()

    def act(self, public):
        if self.spec.alias != 'T3':
            raise ValueError('early-contact release is a T3-only development actor')
        command = super().act(public).copy()
        height = np.asarray(public['tcp_position'])[:, 2]
        memory = self.observer.memory
        release = (memory.phase == 0) & (height < self.RELEASE_Z_M)
        self.early_release_seen |= release
        if np.any(release):
            self.observer.memory = memory._replace(
                phase=np.where(release, 1, memory.phase),
                tracking_integral=np.where(
                    release[:, None], 0., memory.tracking_integral,
                ),
            )
        self.early_guard_active = (self.observer.memory.phase == 1) & (height < self.CLEAR_Z_M)
        command[self.early_guard_active, :6] = [0., 0., self.RETREAT_SPEED_MPS, 0., 0., 0.]
        command[self.early_guard_active, 12] = 0.
        return command


class PublicReferenceUnloadingMatingFeedbackPolicy(PublicEarlyContactMatingFeedbackPolicy):
    """Development-only T3 early withdrawal with public planar reference unloading.

    The reference is a public observation. Four control periods cover the
    declared three-control delay plus one observation; the bounded request
    moves the reference toward the measured TCP during guarded withdrawal.
    """

    VERSION = 'public-mating-feedback-reference-unloading-v1'
    UNLOAD_SECONDS = .08
    MAX_PLANAR_SPEED_MPS = .025

    def act(self, public):
        command = super().act(public)
        lead = (np.asarray(public['reference_position'])[:, :2]
                - np.asarray(public['tcp_position'])[:, :2])
        unload = np.clip(-lead / self.UNLOAD_SECONDS,
                         -self.MAX_PLANAR_SPEED_MPS, self.MAX_PLANAR_SPEED_MPS)
        command[self.early_guard_active, :2] = unload[self.early_guard_active]
        return command


class PublicMatchedReferenceUnloadingMatingFeedbackPolicy(
        PublicReferenceUnloadingMatingFeedbackPolicy):
    """Reference unloading inside the original public actor's planar bound."""

    VERSION = 'public-mating-feedback-matched-reference-unloading-v1'
    MAX_PLANAR_SPEED_MPS = .02


@dataclass(frozen=True)
class PublicPlanarBodyIntentSpec:
    """Finite public base motion used to restore T4 arm reachability."""

    velocity_world: tuple[float, float, float] = (.005, .005, 0.)
    start_seconds: float = 0.
    duration_seconds: float = 6.

    def __post_init__(self):
        velocity = np.asarray(self.velocity_world, dtype=float)
        if (velocity.shape != (3,) or not np.isfinite(velocity).all()
                or np.any(np.abs(velocity[:2]) > .01)
                or abs(velocity[2]) > .03):
            raise ValueError("planar body intent exceeds declared velocity support")
        if (not np.isfinite([self.start_seconds, self.duration_seconds]).all()
                or self.start_seconds < 0 or not 0 < self.duration_seconds <= 8.):
            raise ValueError("finite body-intent timing outside declared support")


class PublicPlanarBodyMatingFeedbackPolicy(PublicLivenessMatingFeedbackPolicy):
    """T4 public feedback plus a finite world-planar base velocity request."""

    VERSION = 'public-mating-feedback-planar-body-v1'

    def __init__(self, spec, *, body_intent=None, **kwargs):
        if spec.alias != 'T4' or not spec.config.mobile:
            raise ValueError('planar body mating feedback qualifies mobile T4 only')
        self.body_intent = body_intent or PublicPlanarBodyIntentSpec()
        if not isinstance(self.body_intent, PublicPlanarBodyIntentSpec):
            raise TypeError('explicit public planar body-intent specification required')
        super().__init__(spec, **kwargs)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.last_body_twist = np.zeros((len(self._reset_modes), 3), dtype=np.float32)

    def body_twist(self, public):
        elapsed = np.asarray(public['elapsed_seconds'], dtype=float).reshape(-1)
        start = self.body_intent.start_seconds
        active = (elapsed >= start) & (elapsed < start + self.body_intent.duration_seconds)
        velocity = np.asarray(self.body_intent.velocity_world, dtype=np.float32)
        self.last_body_twist = np.where(active[:, None], velocity[None], 0.).astype(np.float32)
        return self.last_body_twist.copy()

    def act_with_body(self, public):
        return self.act(public), self.body_twist(public)
