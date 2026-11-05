"""Known tool/guide envelope and same source/control interface."""
import numpy as np
import pytest
from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.tasks.transfer.controllers.guide_lift_route import (
    CLEAR_Z,
    PublicGuideConcurrentMonotoneLiftRoutePolicy,
    PublicGuideConcurrentMonotoneLiftRoutePolicyV2,
    PublicGuideConcurrentMonotoneLiftRoutePolicyV3,
    PublicGuideResponseNormalizedCompletionRoutePolicy,
    PublicGuideLiftRoutePolicy,
)
from genedynamics.tasks.transfer.controllers.guide_feedback import PublicGuideFeedbackPolicy
from test.transfer_methods.test_public_observed_response import record


def test_geometry_clearance_source_identity_route_order_and_state():
    assert np.isclose(CLEAR_Z-.106-.524,.004)
    p,_=record();spec=task_spec('T2',robot='panda');a=PublicGuideFeedbackPolicy(spec,modes=[0],lead_xy_m=.008,position_admittance=True);b=PublicGuideLiftRoutePolicy(spec,mode=0)
    for i in range(10):
        public={k:v[i:i+1].copy() for k,v in p.items()};public['elapsed_seconds'][:]=i*.02;np.testing.assert_array_equal(a.act(public),b.act(public))
    spec=task_spec('T2',robot='xarm7')
    for mode,axis in [(0,0),(1,1)]:
        actor=PublicGuideLiftRoutePolicy(spec,mode=mode);public={k:v[:1].copy() for k,v in p.items()};public['elapsed_seconds'][:]=0;public['tcp_position'][:]=[.38,-.20,.605];public['reference_position'][:]=public['tcp_position'];state=actor.get_state();action=actor.act(public);assert actor.stage==0 and action[0,2]>0 and action[0,12]==0;actor.restore(state);np.testing.assert_array_equal(actor.act(public),action)
        assert actor.HORIZONTAL_SPEED_LIMIT_MPS==.06 and actor.VERTICAL_SPEED_LIMIT_MPS==.06 and actor.FREE_DESCENT_SPEED_MPS==.06
        public['elapsed_seconds'][:]=.02;public['tcp_position'][:,2]=CLEAR_Z;public['reference_position'][:]=public['tcp_position'];action=actor.act(public);assert actor.stage==1 and action[0,axis]>0


def test_concurrent_route_moves_all_required_axes_and_then_descends():
    values, _ = record()
    spec = task_spec("T2", robot="xarm7")
    actor = PublicGuideConcurrentMonotoneLiftRoutePolicy(spec, mode=0)
    public = {key: value[:1].copy() for key, value in values.items()}
    public["elapsed_seconds"][:] = 0.
    public["tcp_position"][:] = [.456, -.20, .594]
    public["reference_position"][:] = public["tcp_position"]
    public["tcp_twist"][:] = 0.
    before = actor.get_state()
    action = actor.act(public)
    assert actor.stage == 0
    assert np.all(action[0, :3] > 0.)
    actor.restore(before)
    np.testing.assert_array_equal(actor.act(public), action)

    public["elapsed_seconds"][:] = .02
    public["tcp_position"][:] = [*spec.config.goal[:2], CLEAR_Z]
    public["reference_position"][:] = public["tcp_position"]
    action = actor.act(public)
    assert actor.stage == 1
    assert action[0, 2] < 0.
    assert action[0, 8] == 0.


def test_concurrent_route_cannot_issue_past_its_free_space_target():
    values, _ = record()
    spec = task_spec("T2", robot="xarm7")
    actor = PublicGuideConcurrentMonotoneLiftRoutePolicy(spec, mode=0)
    public = {key: value[:1].copy() for key, value in values.items()}
    target = np.asarray([*spec.config.goal[:2], CLEAR_Z])
    public["elapsed_seconds"][:] = 0.
    public["tcp_position"][:] = target - np.asarray([.01, .01, .001])
    public["reference_position"][:] = target - np.asarray([.0001, .0001, .0001])
    public["tcp_twist"][:] = 0.
    action = actor.act(public)
    issued = public["reference_position"] + .02 * action[:, :3]
    assert np.all(issued <= target + 1e-8)
    assert issued[0, 2] >= public["reference_position"][0, 2] - 1e-8


def test_v2_concurrent_route_uses_the_frozen_task_tolerance():
    values, _ = record()
    spec = task_spec("T2", robot="xarm7")
    actor = PublicGuideConcurrentMonotoneLiftRoutePolicyV2(spec, mode=0)
    public = {key: value[:1].copy() for key, value in values.items()}
    goal = np.asarray(spec.config.goal)
    public["elapsed_seconds"][:] = 0.
    public["tcp_position"][:] = [goal[0] - .0015, goal[1] + .0006, CLEAR_Z - .001]
    public["reference_position"][:] = [goal[0], goal[1], CLEAR_Z]
    public["tcp_twist"][:] = 0.
    action = actor.act(public)
    assert np.linalg.norm(public["tcp_position"][0, :2] - goal[:2]) > .75 * spec.config.position_tolerance
    assert actor.stage == 1
    assert action[0, 2] < 0.


def test_v3_descent_retains_bounded_planar_feedback_after_reference_hits_goal():
    values, _ = record()
    spec = task_spec("T2", robot="xarm7")
    actor = PublicGuideConcurrentMonotoneLiftRoutePolicyV3(spec, mode=0)
    public = {key: value[:1].copy() for key, value in values.items()}
    goal = np.asarray(spec.config.goal)
    public["elapsed_seconds"][:] = 0.
    public["tcp_position"][:] = [goal[0] + .003, goal[1] - .001, CLEAR_Z]
    public["reference_position"][:] = [goal[0], goal[1], CLEAR_Z]
    public["tcp_twist"][:] = 0.
    actor.stage = 1
    action = actor.act(public)
    assert action[0, 0] < 0.
    assert action[0, 1] > 0.
    issued_lead = public["reference_position"][0, :2] + .02 * action[0, :2] - public["tcp_position"][0, :2]
    assert np.all(np.abs(issued_lead) <= .008 + 1e-8)


def test_response_normalized_route_descends_only_after_full_clearance():
    values, _ = record()
    spec = task_spec("T2", robot="xarm7")
    actor = PublicGuideResponseNormalizedCompletionRoutePolicy(spec, mode=1)
    public = {key: value[:1].copy() for key, value in values.items()}
    goal = np.asarray(spec.config.goal)
    public["elapsed_seconds"][:] = 0.
    public["tcp_position"][:] = [goal[0] - .02, goal[1] - .01, CLEAR_Z]
    public["reference_position"][:] = public["tcp_position"]
    public["tcp_twist"][:] = 0.
    action = actor.act(public)
    assert actor.stage == 1
    assert action[0, 0] > 0.
    assert action[0, 2] == pytest.approx(0.)

    public["elapsed_seconds"][:] = .02
    public["tcp_position"][:] = [*goal[:2], CLEAR_Z]
    public["reference_position"][:] = public["tcp_position"]
    action = actor.act(public)
    assert actor.stage == 2
    assert action[0, 2] < 0.

    actor.reset()
    public["elapsed_seconds"][:] = 0.
    public["tcp_position"][:] = [goal[0] - .02, goal[1] - .01, CLEAR_Z - .003]
    public["reference_position"][:] = public["tcp_position"]
    action = actor.act(public)
    assert actor.stage == 0
    np.testing.assert_array_equal(action[0, :2], 0.)
    assert action[0, 2] > 0.
