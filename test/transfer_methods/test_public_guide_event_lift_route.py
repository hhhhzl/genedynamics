"""Frozen mode-order events permit simultaneous remaining free-air travel."""
import numpy as np
from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.tasks.transfer.controllers.guide_lift_route import PublicGuideLiftRoutePolicy,CLEAR_Z
from genedynamics.tasks.transfer.controllers.guide_event_lift_route import PublicGuideEventLiftRoutePolicy
from test.transfer_methods.test_public_observed_response import record
def test_source_exact_first_trace_anchor_event_progress_and_continuation():
    p,_=record();spec=task_spec('T2',robot='panda');a=PublicGuideLiftRoutePolicy(spec,mode=0);b=PublicGuideEventLiftRoutePolicy(spec,mode=0)
    for i in range(10):
        public={k:v[i:i+1].copy() for k,v in p.items()};public['elapsed_seconds'][:]=i*.02;np.testing.assert_array_equal(a.act(public),b.act(public))
    for mode,axis,delta in [(0,0,.018),(1,1,.011)]:
        old=PublicGuideLiftRoutePolicy(task_spec('T2',robot='xarm7'),mode=mode);new=PublicGuideEventLiftRoutePolicy(task_spec('T2',robot='xarm7'),mode=mode);public={k:v[:1].copy() for k,v in p.items()};public['elapsed_seconds'][:]=0;public['tcp_position'][:]=[.38,-.2,.605];public['reference_position'][:]=public['tcp_position'];old.act(public);new.act(public);assert new.mode_anchor is None
        public['elapsed_seconds'][:]=.02;public['tcp_position'][:,2]=CLEAR_Z;public['reference_position'][:]=public['tcp_position'];old.act(public);new.act(public);assert old.stage==new.stage==1
        public['elapsed_seconds'][:]=.04;public['tcp_position'][0,axis]+=delta;public['reference_position'][:]=public['tcp_position'];state=new.get_state();command=new.act(public);old.act(public);assert new.stage==2 and old.stage==1 and command[0,1-axis]>0
        new.restore(state);np.testing.assert_array_equal(command,new.act(public));again=new.get_state();np.testing.assert_array_equal(command,new.act(public));assert new.stage==again['stage']
