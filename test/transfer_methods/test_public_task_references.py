"""Knownbody path aliases and public task-force/controller-wrench constraints."""
import numpy as np
from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.tasks.transfer.contact import bounded_contact_command
from genedynamics.tasks.transfer.controllers.obstacle_feedback import PublicObstacleFeedbackPolicy
from genedynamics.tasks.transfer.controllers.obstacle_path import PublicObstaclePathPolicy
from genedynamics.tasks.transfer.controllers.guide_wrench_projection import PublicGuideWrenchProjectionPolicy as PublicGuideWrenchSchedulePolicy,project_wrench_box_ball
from test.transfer_methods.test_public_observed_response import record


def test_planned_waypoints_leave_Panda_and_side_mode_numerically_identical():
    p,_=record()
    for body,mode in [('panda',0),('panda',1),('xarm7',1)]:
        spec=task_spec('T1',robot=body);original=PublicObstacleFeedbackPolicy(spec,modes=[mode]);planned=PublicObstaclePathPolicy(spec,modes=[mode],path=[.58,.68,.64,.585])
        for tick in range(20):
            public={k:v[:1].copy() for k,v in p.items()};public['elapsed_seconds'][:]=tick*.02;public['tcp_position'][:]=original.routes[mode,min(tick//4,3)]+[.001,0,0]
            expected=original.act(public);before=planned.get_state();value=planned.act(public);after=planned.get_state();planned.restore(before);np.testing.assert_array_equal(value,planned.act(public));planned.restore(after);np.testing.assert_array_equal(value,expected)


def test_guide_unloads_only_motion_and_budget_uses_known_next_issued_stiffness():
    p,_=record();spec=task_spec('T2',robot='xarm7');actor=PublicGuideWrenchSchedulePolicy(spec,modes=[0])
    for tick in range(11):
        public={k:v[:1].copy() for k,v in p.items()};public['elapsed_seconds'][:]=tick*.02;public['tcp_position'][:]=[.38,-.20,.596];public['reference_position'][:]=public['tcp_position']+[.008,.008,0];public['tcp_twist'][:]=0;public['previous_action'][:]=0;public['measured_wrench'][:]=[0,0,5,0,0,0];public['tcp_rotation'][:]=np.diag([1.,-1.,-1.]);public['reference_rotation'][:]=public['tcp_rotation'];public['goal_rotation'][:]=public['tcp_rotation'];value=actor.act(public)
        if tick==0:assert value[0,2]==0
    assert actor.observer.memory.phase[0]==1 and np.isclose(value[0,2],.01)
    issued=bounded_contact_command(np,value[0],public['previous_action'][0],spec.config);k=800*np.exp(issued[6:8]);d=2*np.sqrt(k);w=k*(public['reference_position'][0,:2]-public['tcp_position'][0,:2]+.02*issued[:2])+d*(issued[:2]-public['tcp_twist'][0,:2]);assert np.linalg.norm(w)<=9.001
    public['elapsed_seconds'][:]=.22;public['tcp_position'][0,:2]=spec.config.goal[:2];assert actor.act(public)[0,2]==0


def test_exact_wrench_projection_has_active_axis_and_handles_empty_intersection():
    value,feasible=project_wrench_box_ball([7.,6.4],[6.,6.32],[12.,12.]);assert feasible
    np.testing.assert_allclose(value,[np.sqrt(81.-6.32**2),6.32],atol=1e-10,rtol=0)
    impossible,feasible=project_wrench_box_ball([10.,10.],[7.,7.],[12.,12.]);assert not feasible
    np.testing.assert_array_equal(impossible,[7.,7.])
