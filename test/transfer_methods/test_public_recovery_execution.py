"""Causal contact/withdrawal gate and training command mask agree."""
import numpy as np
import pytest
import torch
from genedynamics.tasks.transfer.controllers.recovery_execution import PublicRecoveryGate
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.bias_feedback import PublicBiasFeedbackSpec
from genedynamics.solvers.single.cross_ebdm.learning.response import PublicResponseEnsemble,PublicResponseSpec
from genedynamics.solvers.single.cross_ebdm.learning.recovery_transport import PublicRecoveryTransportTrainer,effective_commands,recovery_policy_commands,recovery_task_proxy
from genedynamics.solvers.single.cross_ebdm.learning.gated_transport import (
    PublicCompletionProgressSpec,
    PublicFiniteTimeProgressSpec,
    PublicPairedProgressSpec,
    completion_progress_loss,
    finite_time_progress_loss,
    paired_finite_time_progress_loss,
    public_completion_progress,
    public_finite_time_progress,
    recenter_saturated_transport_output,
    select_completion_compatible_checkpoint,
)
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


def test_public_output_recenter_caps_logits_without_special_forward_path():
    torch.manual_seed(130)
    spec = PublicResponseSpec(
        history_len=1, horizon=2, maximum_joints=2, members=2, hidden=(8,),
    )
    feedback = PublicBiasFeedbackSpec()
    batch = training_batch(spec)
    batch["recovery_enabled"] = torch.ones(3, dtype=torch.bool)
    batch["common_action"] = torch.zeros(3, 13)
    policy = PublicResidualTransport(spec, PublicTransportSpec(hidden=(16,)))
    with torch.no_grad():
        policy.network[-1].weight.zero_()
        policy.network[-1].bias.fill_(15.)
    report = recenter_saturated_transport_output(
        policy, batch, feedback, maximum_median_absolute_logit=3.,
    )
    assert report["version"] == "public_output_row_recenter_v1"
    assert report["scaled_outputs"] == 26
    assert report["median_absolute_logit_before_max"] == 15
    assert report["median_absolute_logit_after_max"] <= 3 + 1e-5
    assert report["minimum_output_scale"] == pytest.approx(.2)
    assert report["normalized_command_change_max"] > 0
    _, executed = recovery_policy_commands(policy, feedback, batch)
    assert torch.isfinite(executed).all()


def _completion_batch():
    current_rotation = torch.eye(3).repeat(3, 1, 1)
    goal_rotation = current_rotation.clone()
    task_reference = torch.cat((
        torch.tensor([[.44, -.20, .58], [.44, -.20, .58], [.44, -.20, .58]]),
        goal_rotation.reshape(3, 9),
        torch.full((3, 1), 5.),
    ), dim=1)
    return {
        "recovery_enabled": torch.tensor([False, True, True]),
        "common_action": torch.tensor([
            [.5, 0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 1., 0.],
            [.5, 0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 1., 0.],
        ]),
        "current": {
            "position": torch.tensor([
                [.40, -.20, .58], [.40, -.20, .58], [.44, -.20, .58],
            ]),
            "rotation": current_rotation,
            "wrench": torch.zeros(3, 6),
        },
        "task_reference": task_reference,
        "completion_reference": task_reference[:, :12].clone(),
    }


def test_finite_time_progress_is_phase3_public_and_reserves_terminal_hold():
    batch = _completion_batch()
    batch["common_action"][1, 0] = 0.
    batch["decoder"] = {
        "phase": torch.tensor([2, 3, 3]),
        "compensation": torch.zeros(3, 6),
        "clear_count": torch.zeros(3, dtype=torch.int64),
        "retreating": torch.zeros(3, dtype=torch.bool),
        "waiting": torch.zeros(3, dtype=torch.bool),
        "position": batch["current"]["position"].clone(),
        "reference_position": batch["current"]["position"].clone(),
        "rotation_lead": torch.zeros(3, 3),
        "wrench": torch.zeros(3, 6),
    }
    batch["remaining_controls"] = torch.tensor([20, 20, 200])
    batch["base_proto"] = torch.zeros(3, 2, 13)
    candidate = torch.zeros(3, 2, 13, requires_grad=True)
    with torch.no_grad():
        candidate[:, :, 6:12] = batch["common_action"][:, None, 6:12]
        candidate[0, :, 0] = -1.
        candidate[1, :, 0] = .005
        candidate[2, :, 0] = .5
    spec = PublicFiniteTimeProgressSpec()
    feedback = PublicBiasFeedbackSpec()
    values = public_finite_time_progress(candidate, batch, spec, feedback)
    assert values["enabled"].tolist() == [False, True, True]
    assert values["terminal"].tolist() == [False, False, True]
    assert values["linear_progress_required"][1] > values["linear_progress_common"][1]
    loss, metrics = finite_time_progress_loss(candidate, batch, spec, feedback)
    assert loss > 0
    assert metrics["finite_time_progress_loss"] > 0
    assert metrics["finite_time_terminal_loss"] > 0
    loss.backward()
    assert torch.count_nonzero(candidate.grad[0]) == 0
    assert torch.count_nonzero(candidate.grad[1, 0, :3]) > 0
    assert torch.count_nonzero(candidate.grad[2, 0, :6]) > 0
    assert torch.count_nonzero(candidate.grad[:, 1]) == 0


def test_finite_time_terminal_residual_is_not_diluted_by_nonterminal_states():
    batch = _completion_batch()
    batch["decoder"] = {
        "phase": torch.tensor([3, 3, 3]),
        "compensation": torch.zeros(3, 6),
        "clear_count": torch.zeros(3, dtype=torch.int64),
        "retreating": torch.zeros(3, dtype=torch.bool),
        "waiting": torch.zeros(3, dtype=torch.bool),
        "position": batch["current"]["position"].clone(),
        "reference_position": batch["current"]["position"].clone(),
        "rotation_lead": torch.zeros(3, 3),
        "wrench": torch.zeros(3, 6),
    }
    batch["remaining_controls"] = torch.full((3,), 100)
    batch["base_proto"] = torch.zeros(3, 2, 13)
    candidate = batch["common_action"][:, None, :].repeat(1, 2, 1).clone()
    candidate[2, 0, 0] = .5
    spec = PublicFiniteTimeProgressSpec()
    feedback = PublicBiasFeedbackSpec()
    _, all_enabled = finite_time_progress_loss(candidate, batch, spec, feedback)
    batch["recovery_enabled"][:2] = False
    _, terminal_only = finite_time_progress_loss(candidate, batch, spec, feedback)
    assert all_enabled["finite_time_terminal_loss"] == pytest.approx(
        terminal_only["finite_time_terminal_loss"]
    )


def test_paired_progress_tail_penalizes_each_public_mode_lower_tail():
    batch = _completion_batch()
    batch["current"]["position"][2, 0] = .40
    batch["decoder"] = {
        "phase": torch.full((3,), 3),
        "compensation": torch.zeros(3, 6),
        "clear_count": torch.zeros(3, dtype=torch.int64),
        "retreating": torch.zeros(3, dtype=torch.bool),
        "waiting": torch.zeros(3, dtype=torch.bool),
        "position": batch["current"]["position"].clone(),
        "reference_position": batch["current"]["position"].clone(),
        "rotation_lead": torch.zeros(3, 3),
        "wrench": torch.zeros(3, 6),
    }
    batch["remaining_controls"] = torch.full((3,), 100)
    batch["requested_mode"] = torch.tensor([0, 0, 1])
    batch["base_proto"] = torch.zeros(3, 2, 13)
    task_only = batch["common_action"][:, None, :].repeat(1, 2, 1)
    candidate = task_only.clone().requires_grad_(True)
    with torch.no_grad():
        candidate[1, 0, 0] -= .25
        candidate[2, 0, 0] -= .5
    finite_time = PublicFiniteTimeProgressSpec()
    feedback = PublicBiasFeedbackSpec()
    mean_loss, mean_metrics = paired_finite_time_progress_loss(
        candidate, task_only, batch, finite_time,
        PublicPairedProgressSpec(), feedback,
    )
    tail_loss, tail_metrics = paired_finite_time_progress_loss(
        candidate, task_only, batch, finite_time,
        PublicPairedProgressSpec(tail_fraction=.5, tail_weight=4.), feedback,
    )
    assert mean_metrics["paired_progress_tail_loss"] == 0
    assert tail_metrics["paired_progress_tail_loss"] > 0
    assert tail_metrics["paired_progress_tail_modes"] == 2
    assert tail_loss > mean_loss
    tail_loss.backward()
    assert torch.count_nonzero(candidate.grad[:, 0, :3]) > 0
    assert torch.count_nonzero(candidate.grad[:, 1]) == 0


def test_completion_progress_loss_uses_only_enabled_public_states_and_terminal_residual():
    batch = _completion_batch()
    candidate = torch.zeros(3, 2, 13, requires_grad=True)
    with torch.no_grad():
        candidate[:, :, 6:12] = batch["common_action"][:, None, 6:12]
        candidate[0, :, 0] = -1.
        candidate[1, :, 0] = -.5
        candidate[2, :, 0] = .5
    spec = PublicCompletionProgressSpec()
    values = public_completion_progress(candidate, batch, spec)
    assert values["linear_deficit"][1] > 0
    assert values["terminal"].tolist() == [False, False, True]
    assert not values["admissible"].any()
    loss, metrics = completion_progress_loss(candidate, batch, spec)
    assert loss > 0
    assert metrics["completion_progress_loss"] > 0
    assert metrics["completion_terminal_loss"] > 0
    loss.backward()
    assert torch.count_nonzero(candidate.grad[0]) == 0
    assert torch.count_nonzero(candidate.grad[1, 0, :3]) > 0
    assert torch.count_nonzero(candidate.grad[2, 0, :6]) > 0
    assert torch.count_nonzero(candidate.grad[:, 1]) == 0


def test_completion_progress_accepts_common_relative_useful_candidate():
    batch = _completion_batch()
    batch["recovery_enabled"][-1] = False
    candidate = batch["common_action"][:, None, :].repeat(1, 2, 1)
    candidate[1, 0, 0] = .75
    values = public_completion_progress(
        candidate, batch, PublicCompletionProgressSpec(minimum_candidate_delta=1e-3),
    )
    assert values["admissible"].tolist() == [False, True, False]
    assert values["useful"].tolist() == [False, True, False]
    loss, metrics = completion_progress_loss(
        candidate, batch, PublicCompletionProgressSpec(minimum_candidate_delta=1e-3),
    )
    assert loss == 0
    assert metrics["completion_useful_fraction"] == 1


def test_completion_tail_loss_targets_worst_nonterminal_deficits():
    batch = _completion_batch()
    batch["recovery_enabled"][:] = True
    batch["current"]["position"][2, 0] = .40
    candidate = batch["common_action"][:, None, :].repeat(1, 2, 1)
    candidate[0, 0, 0] = .49
    candidate[1, 0, 0] = .25
    candidate[2, 0, 0] = -.5
    plain, plain_metrics = completion_progress_loss(
        candidate, batch, PublicCompletionProgressSpec(),
    )
    tailed, tail_metrics = completion_progress_loss(
        candidate, batch,
        PublicCompletionProgressSpec(tail_fraction=1 / 3, tail_weight=2.),
    )
    assert plain_metrics["completion_tail_progress_loss"] == 0
    assert tail_metrics["completion_tail_progress_loss"] > plain_metrics["completion_progress_loss"]
    assert tailed > plain


def test_completion_checkpoint_selection_requires_task_progress_and_utility():
    accepted = select_completion_compatible_checkpoint(
        candidate_proxy=1.01, task_only_proxy=1., task_tolerance=.05,
        linear_deficit_p95=5e-8, angular_deficit_p95=8e-8,
        progress_tolerance=1e-7, useful_fraction=.3,
        minimum_useful_fraction=.25, candidate="candidate", task_only="task-only",
    )
    assert accepted["accepted"] and accepted["selected"] == "candidate"
    for field, value in (
        ("candidate_proxy", 1.06),
        ("linear_deficit_p95", 2e-7),
        ("angular_deficit_p95", 2e-7),
        ("useful_fraction", .24),
    ):
        kwargs = dict(
            candidate_proxy=1.01, task_only_proxy=1., task_tolerance=.05,
            linear_deficit_p95=5e-8, angular_deficit_p95=8e-8,
            progress_tolerance=1e-7, useful_fraction=.3,
            minimum_useful_fraction=.25,
            candidate="candidate", task_only="task-only",
        )
        kwargs[field] = value
        rejected = select_completion_compatible_checkpoint(**kwargs)
        assert not rejected["accepted"] and rejected["selected"] == "task-only"
