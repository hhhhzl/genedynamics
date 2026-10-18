"""MPC must preserve public continuation and frozen model parameters."""
import numpy as np
import torch
from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.bias_feedback import PublicBiasFeedbackSpec
from genedynamics.solvers.single.cross_ebdm.learning.issued_response import PublicIssuedResponseEnsemble
from genedynamics.solvers.single.cross_ebdm.learning.local_transport import PublicLocalResidualTransport
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import PublicTransportSpec
from genedynamics.solvers.single.response_mpc.contact import PublicResponseMPCActor
from test.transfer_methods.test_public_dense_execution import bank


def test_mpc_public_resume_is_exact_and_model_weights_never_update(tmp_path):
    torch.set_num_threads(1);b,p=bank(tmp_path);response=PublicIssuedResponseEnsemble(b.spec);policy=PublicLocalResidualTransport(b.spec,PublicTransportSpec(residual_scale=.025,hidden=(8,)))
    actor=PublicResponseMPCActor(task_spec('T3',robot='panda'),policy,b,PublicBiasFeedbackSpec(compensation_gain=2.,guard_prediction_seconds=0.),response=response)
    weights={k:v.clone() for k,v in response.state_dict().items()};before=actor.get_state();row={k:v[:1] for k,v in p.items()};a=actor.act(row);actor.restore(before);again=actor.act(row)
    np.testing.assert_array_equal(a,again);assert actor.irreversible_model_calls==12 and actor.irreversible_action_updates==10
    assert all(torch.equal(v,response.state_dict()[k]) for k,v in weights.items()) and not any(p.requires_grad for p in response.parameters())
    assert not any(k in vars(actor) for k in ('task','mechanism','physics')) and len(actor.inference_seconds)==2
