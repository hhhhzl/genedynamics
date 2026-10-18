"""Conditional integration preserves source actions and avoids saturation windup."""
import numpy as np
from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.tasks.transfer.controllers.guide_lift_route import PublicGuideLiftRoutePolicy
from genedynamics.tasks.transfer.controllers.guide_antiwindup_lift_route import PublicGuideAntiWindupLiftRoutePolicy
from test.transfer_methods.test_public_observed_response import record
def test_source_identity_conditional_integrator_and_continuation():
    p,_=record()
    for mode in (0,1):
        a=PublicGuideLiftRoutePolicy(task_spec('T2',robot='panda'),mode=mode);b=PublicGuideAntiWindupLiftRoutePolicy(task_spec('T2',robot='panda'),mode=mode)
        for t in range(10):
            row={k:v[t:t+1].copy() for k,v in p.items()};row['elapsed_seconds'][:]=t*.02;np.testing.assert_array_equal(a.act(row),b.act(row))
    a=PublicGuideAntiWindupLiftRoutePolicy(task_spec('T2',robot='xarm7'),mode=0)
    row={k:v[:1].copy() for k,v in p.items()};row['elapsed_seconds'][:]=0;row['tcp_position'][:]=[.38,-.2,.605];row['reference_position'][:]=row['tcp_position'];row['tcp_twist'][:]=0
    before=a.get_state();value=a.act(row);assert a.compensation[0,2]==0 and value[0,2]==np.float32(.02)
    a.restore(before);np.testing.assert_array_equal(value,a.act(row))
