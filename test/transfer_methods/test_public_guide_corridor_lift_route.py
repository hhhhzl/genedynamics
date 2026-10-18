"""Outlet clearance uses the actual public tip pose, not a task success label."""
import numpy as np
from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.tasks.transfer.controllers.guide_lift_route import CLEAR_Z
from genedynamics.tasks.transfer.controllers.guide_corridor_lift_route import PublicGuideCorridorLiftRoutePolicy
from test.transfer_methods.test_public_observed_response import record
def test_known_tip_corridor_allows_early_descent_and_rejects_ridge():
    p,_=record()
    for x,expected in [(.44,2),(.444,3)]:
        a=PublicGuideCorridorLiftRoutePolicy(task_spec('T2',robot='xarm7'),mode=0);row={k:v[:1].copy() for k,v in p.items()};row['elapsed_seconds'][:]=0;row['tcp_position'][:]=[.38,-.2,CLEAR_Z];row['reference_position'][:]=row['tcp_position'];row['tcp_rotation'][:]=np.diag([1.,-1.,-1.]);row['reference_rotation'][:]=row['tcp_rotation'];row['tcp_twist'][:]=0;a.act(row)
        row['elapsed_seconds'][:]=.02;row['tcp_position'][:]=[x,-.18,CLEAR_Z];row['reference_position'][:]=row['tcp_position'];row['measured_wrench'][:]=0;a.stage=2;state=a.get_state();command=a.act(row)
        assert a.stage==expected
        if expected==3:assert command[0,2]<0 and np.linalg.norm(row['tcp_position'][0,:2]-np.array([.46,-.18]))>.002
        a.restore(state);np.testing.assert_array_equal(command,a.act(row))
