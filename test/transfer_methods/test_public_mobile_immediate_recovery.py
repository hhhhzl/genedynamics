"""Known-jam direct withdrawal preserves Panda and causal continuation."""
import numpy as np
from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.bias_feedback import PublicBiasFeedbackSpec
from genedynamics.tasks.transfer.controllers.mating_feedback import PublicMatingFeedbackPolicy
from genedynamics.tasks.transfer.controllers.mobile_recovery import PublicMobileImmediateRecoveryPolicy
from test.transfer_methods.test_public_observed_response import record


def test_source_exact_and_mobile_initial_force_off_with_upward_request():
    p,_=record();fb=PublicBiasFeedbackSpec(insertion_force_scale=1.,lift_before_lateral=True,normal_control='position_admittance',compensation_gain=2.)
    spec=task_spec('T3',robot='panda');a=PublicMatingFeedbackPolicy.load(spec,None,feedback=fb,modes=[0]);b=PublicMobileImmediateRecoveryPolicy.load(spec,None,feedback=fb,modes=[0])
    for i in range(10):
        public={k:v[i:i+1].copy() for k,v in p.items()};public['elapsed_seconds'][:]=i*.02;np.testing.assert_array_equal(a.act(public),b.act(public))
    spec=task_spec('T4',robot='go2_z1');a=PublicMobileImmediateRecoveryPolicy.load(spec,None,feedback=fb,modes=[0]);public={k:v[:1].copy() for k,v in p.items()};public['tcp_position'][:]=[.45,-.2,.597];public['reference_position'][:]=public['tcp_position'];public['elapsed_seconds'][:]=0;s=a.get_state();value=a.act(public);assert a.observer.memory.phase[0]==1 and value[0,12]==0 and value[0,2]>0;a.restore(s);np.testing.assert_array_equal(a.act(public),value)
