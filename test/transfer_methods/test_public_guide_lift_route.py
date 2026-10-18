"""Known tool/guide envelope and same source/control interface."""
import numpy as np
from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.tasks.transfer.controllers.guide_lift_route import PublicGuideLiftRoutePolicy,CLEAR_Z
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
        public['elapsed_seconds'][:]=.02;public['tcp_position'][:,2]=CLEAR_Z;public['reference_position'][:]=public['tcp_position'];action=actor.act(public);assert actor.stage==1 and action[0,axis]>0
