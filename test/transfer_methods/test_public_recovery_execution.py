"""Causal contact/withdrawal gate and training command mask agree."""
import numpy as np
import torch
from genedynamics.tasks.transfer.controllers.recovery_execution import PublicRecoveryGate
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.bias_feedback import PublicBiasFeedbackSpec
from genedynamics.solvers.single.cross_ebdm.learning.response import PublicResponseEnsemble,PublicResponseSpec
from genedynamics.solvers.single.cross_ebdm.learning.recovery_transport import PublicRecoveryTransportTrainer,effective_commands,recovery_policy_commands,recovery_task_proxy
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import PublicResidualTransport,PublicTransportSpec
from test.transfer_methods.test_public_dense_transport import training_batch
def test_public_gate_cold_contact_conservative_withdrawal_and_gradients():
    gate=PublicRecoveryGate()
    def p(t,z,force):return dict(elapsed_seconds=np.array([[t*.02]]),tcp_position=np.array([[.45,-.2,z]]),measured_wrench=np.array([[0,0,force,0,0,0]]))
    assert not gate.observe(p(0,.597,5.))
    assert not gate.contact_seen
    assert not gate.observe(p(1,.596,1.))
    assert gate.contact_seen
    before=gate.get_state();assert not gate.observe(p(2,.6091,0));gate.restore(before)
    assert gate.observe(p(2,.6093,0));assert gate.observe(p(2,.6093,0))
    assert gate.observe(p(3,.590,5.))
    candidate=torch.randn(2,10,13,requires_grad=True);common=torch.ones(2,13);batch=dict(recovery_enabled=torch.tensor([False,True]),common_action=common)
    commands=effective_commands(candidate,batch);torch.testing.assert_close(commands[0],common[0,None].expand(10,13));torch.testing.assert_close(commands[1],candidate[1]);commands.sum().backward()
    assert torch.count_nonzero(candidate.grad[0])==0 and torch.all(candidate.grad[1]==1)


def test_recovery_objective_and_validation_ignore_disabled_candidates():
    torch.manual_seed(128)
    spec=PublicResponseSpec(history_len=1,horizon=2,maximum_joints=2,members=2,hidden=(8,))
    response=PublicResponseEnsemble(spec)
    feedback=PublicBiasFeedbackSpec()
    batch=training_batch(spec)
    batch['recovery_enabled']=torch.zeros(3,dtype=torch.bool)
    batch['common_action']=torch.randn(3,13)
    first=PublicResidualTransport(spec,PublicTransportSpec(hidden=(16,)))
    second=PublicResidualTransport.from_checkpoint(first.checkpoint())
    with torch.no_grad():
        second.network[-1].bias.fill_(.4)
    _,first_commands=recovery_policy_commands(first,feedback,batch)
    _,second_commands=recovery_policy_commands(second,feedback,batch)
    torch.testing.assert_close(first_commands,second_commands,rtol=0,atol=0)
    torch.testing.assert_close(recovery_task_proxy(first,response,feedback,batch),recovery_task_proxy(second,response,feedback,batch),rtol=0,atol=0)
    first_loss,first_metrics=PublicRecoveryTransportTrainer(first,response,feedback).objective(batch,response_weight=1.)
    second_loss,second_metrics=PublicRecoveryTransportTrainer(second,response,feedback).objective(batch,response_weight=1.)
    torch.testing.assert_close(first_loss,second_loss,rtol=0,atol=0)
    for key in first_metrics:
        torch.testing.assert_close(first_metrics[key],second_metrics[key],rtol=0,atol=0)


def test_recovery_objective_only_uses_candidates_when_enabled():
    torch.manual_seed(129)
    spec=PublicResponseSpec(history_len=1,horizon=2,maximum_joints=2,members=2,hidden=(8,))
    feedback=PublicBiasFeedbackSpec()
    batch=training_batch(spec)
    batch['recovery_enabled']=torch.tensor([False,True,True])
    batch['common_action']=torch.randn(3,13)
    first=PublicResidualTransport(spec,PublicTransportSpec(hidden=(16,)))
    second=PublicResidualTransport.from_checkpoint(first.checkpoint())
    with torch.no_grad():
        second.network[-1].bias.fill_(.4)
    _,first_commands=recovery_policy_commands(first,feedback,batch)
    _,second_commands=recovery_policy_commands(second,feedback,batch)
    torch.testing.assert_close(first_commands[0],second_commands[0],rtol=0,atol=0)
    assert not torch.equal(first_commands[1:],second_commands[1:])
