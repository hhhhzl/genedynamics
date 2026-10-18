"""T2 route reference and ordinary source-free MPC, not a transport benefit test."""
import numpy as np
from types import SimpleNamespace
from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.solvers.single.cross_ebdm.learning.response import PublicResponseSpec
from genedynamics.solvers.single.cross_ebdm.learning.task_response import PublicTaskIssuedResponseEnsemble
from genedynamics.solvers.single.cross_ebdm.learning.task_transport import PublicTaskResidualTransport
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import PublicTransportSpec
from genedynamics.tasks.transfer.controllers.cross_ebdm.guide_lift import PublicLiftGuideMPCActor,PublicLiftGuideNoiseActor
from genedynamics.tasks.transfer.controllers.cross_ebdm.arm_noise import PublicArmTaskNoiseActor
from genedynamics.tasks.transfer.controllers.guide_lift_route import CLEAR_Z
from test.transfer_methods.test_public_observed_response import record
def test_source_noise_exact_lift_reference_and_no_source_gate_on_ordinary_mpc():
    p,_=record();source=task_spec('T2',robot='panda');a=PublicArmTaskNoiseActor(source,mode=0,seed=2);b=PublicLiftGuideNoiseActor(source,mode=0,seed=2)
    for i in range(10):
        public={k:v[i:i+1].copy() for k,v in p.items()};public['elapsed_seconds'][:]=i*.02;np.testing.assert_array_equal(a.act(public),b.act(public))
    s=PublicResponseSpec(history_len=4,horizon=10,hidden=(8,),members=1);policy=PublicTaskResidualTransport(s,PublicTransportSpec(residual_scale=.025,hidden=(8,)))
    def query(public,phase,modes):return dict(features=np.zeros((1,s.feature_dim),np.float32),commands=np.zeros((1,s.horizon,13),np.float32)),np.array([99.]),np.array([0])
    bank=SimpleNamespace(task='T2',spec=s,query=query);model=PublicTaskIssuedResponseEnsemble(s);actor=PublicLiftGuideMPCActor(task_spec('T2',robot='xarm7'),policy,bank,mode=0,response=model)
    public={k:v[:1].copy() for k,v in p.items()};public['elapsed_seconds'][:]=0;public['tcp_position'][:]=[.38,-.2,.605];public['reference_position'][:]=public['tcp_position'];before=actor.get_state();batch,base=actor.prepare(public)
    assert batch['phase'][0]==0 and np.isclose(batch['task_reference'][0,2],CLEAR_Z) and batch['task_reference'][0,12]==0
    actor.restore(before);value=actor.act(public);assert actor.fallback_controls==0 and actor.irreversible_model_calls==6 and actor.irreversible_action_updates==5 and value[0,12]==0
    actor.restore(before);np.testing.assert_array_equal(value,actor.act(public))
