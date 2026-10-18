"""A source-free planner must execute every lift stage with an unusable bank."""
from types import SimpleNamespace
import numpy as np
from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.solvers.single.cross_ebdm.learning.response import PublicResponseSpec
from genedynamics.solvers.single.cross_ebdm.learning.task_response import PublicTaskIssuedResponseEnsemble
from genedynamics.solvers.single.cross_ebdm.learning.task_transport import PublicTaskResidualTransport
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import PublicTransportSpec
from genedynamics.solvers.single.response_mpc.guide import PublicSourceFreeLiftGuideMPCActor
from genedynamics.tasks.transfer.controllers.guide_lift_route import CLEAR_Z
from test.transfer_methods.test_public_observed_response import record
def test_all_lift_stages_and_continuation_without_source_queries():
    public,_=record();s=PublicResponseSpec(history_len=4,horizon=10,hidden=(8,),members=1)
    def forbidden(*args):raise AssertionError('ordinary MPC queried source')
    bank=SimpleNamespace(task='T2',spec=s,query=forbidden);policy=PublicTaskResidualTransport(s,PublicTransportSpec(residual_scale=.025,hidden=(8,)));model=PublicTaskIssuedResponseEnsemble(s)
    for mode in (0,1):
        actor=PublicSourceFreeLiftGuideMPCActor(task_spec('T2',robot='xarm7'),policy,bank,mode=mode,response=model)
        poses=[[.38,-.20,.605],[.38,-.20,CLEAR_Z],[.46 if mode==0 else .38,-.18 if mode else -.20,CLEAR_Z],[.46,-.18,CLEAR_Z]]
        for t,pos in enumerate(poses):
            row={k:v[:1].copy() for k,v in public.items()};row['elapsed_seconds'][:]=t*.02;row['tcp_position'][:]=pos;row['reference_position'][:]=pos
            before=actor.get_state();command=actor.act(row);after=actor.get_state();actor.restore(before);np.testing.assert_array_equal(command,actor.act(row));actor.restore(after)
            assert actor.base.stage==t and actor.fallback_controls==0
        assert actor.irreversible_model_calls==48 and actor.irreversible_action_updates==40
