"""Lead limits bound issued intent across all free-air ticks."""
import numpy as np
from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.tasks.transfer.controllers.guide_lift_route import PublicGuideLiftRoutePolicy
from genedynamics.tasks.transfer.controllers.guide_lead_lift_route import PublicGuideLeadLiftRoutePolicy
from test.transfer_methods.test_public_observed_response import record
def test_lead_bounds_from_consistent_public_reference_and_source_identity():
    p,_=record()
    for mode in (0,1):
        a=PublicGuideLiftRoutePolicy(task_spec('T2',robot='panda'),mode=mode);b=PublicGuideLeadLiftRoutePolicy(task_spec('T2',robot='panda'),mode=mode)
        for t in range(10):
            row={k:v[t:t+1].copy() for k,v in p.items()};row['elapsed_seconds'][:]=t*.02;np.testing.assert_array_equal(a.act(row),b.act(row))
    actor=PublicGuideLeadLiftRoutePolicy(task_spec('T2',robot='xarm7'),mode=0);row={k:v[:1].copy() for k,v in p.items()};row['tcp_position'][:]=[.38,-.2,.605];row['reference_position'][:]=row['tcp_position'];row['tcp_twist'][:]=0
    for tick in range(50):
        row['elapsed_seconds'][:]=tick*.02;state=actor.get_state();command=actor.act(row);actor.restore(state);np.testing.assert_array_equal(command,actor.act(row));row['reference_position']+=command[:,:3]*.02
        assert np.all(np.abs(row['reference_position']-row['tcp_position'])<=np.array([.008,.008,.01])+1e-6)
