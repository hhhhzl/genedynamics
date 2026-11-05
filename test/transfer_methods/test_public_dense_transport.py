"""Differentiable decoder parity, bounded dense outputs and exact continuation."""
import numpy as np
import torch

from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.bias_feedback import PublicBiasFeedbackPolicy,PublicBiasFeedbackSpec
from genedynamics.solvers.single.cross_ebdm.learning.response import PublicResponseSpec,PublicResponseEnsemble
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import PublicTransportSpec,PublicResidualTransport,PublicTransportTrainer,decoder_state,decoder_context,decode_pose_transport,BOUNDS
from test.transfer_methods.test_fada_task_training import _equal
from genedynamics.tasks.transfer.controllers.cross_ebdm.dense import PublicAdaptiveStepTaskDominantDenseTransportActor,PublicCompletionValueEnsembleRobustDenseTransportActor,PublicExpandedEnsembleRobustAdaptiveStepTaskDominantDenseTransportActor,PublicModeConditionedAdaptiveStepDenseTransportActor,PublicModeConditionedCompletionValueDenseTransportActor,PublicModeSpecificCalibratedAuthorityDenseTransportActor,PublicModeSpecificRecordedCompletionDenseTransportActor,PublicModeSpecificSupportedPhaseCappedExpansionDenseTransportActor,PublicModeSpecificSupportedPhaseConvexDenseTransportActor,PublicPhaseScheduledAdaptiveStepDenseTransportActor,PublicTransientPhaseAdaptiveStepDenseTransportActor


class Capture:
    def apply(self,value,phase):self.value=value.copy();return value


def observation(x=.44):
    r=np.diag([1.,-1.,-1.])[None]
    return dict(tcp_position=np.array([[x,-.2,.611]],np.float32),tcp_rotation=r,tcp_twist=np.zeros((1,6)),measured_wrench=np.array([[0,0,3,0,0,0.]]),
        reference_position=np.array([[x+.001,-.199,.612]]),reference_rotation=r.copy(),previous_action=np.zeros((1,13)),elapsed_seconds=np.array([[.02]]))


def test_differentiable_pose_decoder_matches_public_actor_all_phases_and_guard_modes():
    for phase in range(4):
        for x in (.44,.442):
            actor=PublicBiasFeedbackPolicy(task_spec('T3',robot='panda'),feedback=PublicBiasFeedbackSpec(insertion_force_scale=1.,lift_before_lateral=True,guard_prediction_seconds=0.))
            actor.observer.memory=actor.observer.memory._replace(phase=np.array([phase]));actor.compensation[:]=[.001,-.001,.0002,.002,-.001,.0004]
            capture=Capture();actor.pose_transport=capture;p=observation(x);expected=actor.act(p)
            state={k:torch.as_tensor(v) for k,v in decoder_state(actor,p).items()}
            actual=decode_pose_transport(torch.as_tensor(capture.value,dtype=torch.float64),state,actor.pose_feedback)[:,0].numpy()
            np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-8)


def training_batch(spec,n=3):
    zeros=lambda *shape:torch.zeros(shape);state=dict(phase=torch.tensor([0,2,3]),clear_count=torch.ones(n,dtype=torch.long),compensation=zeros(n,6),retreating=torch.zeros(n,dtype=torch.bool),waiting=torch.zeros(n,dtype=torch.bool),
        position=torch.tensor([[.44,-.2,.611]]).repeat(n,1),reference_position=torch.tensor([[.44,-.2,.611]]).repeat(n,1),rotation_lead=zeros(n,3),wrench=zeros(n,6))
    reference=torch.cat((torch.tensor([[.44,-.2,.58]]).repeat(n,1),torch.diag(torch.tensor([1.,-1.,-1.])).reshape(1,9).repeat(n,1),torch.full((n,1),5.)),dim=1)
    return dict(target_features=torch.randn(n,spec.feature_dim)*.01,source_features=torch.randn(n,spec.feature_dim)*.01,source_commands=zeros(n,spec.horizon,13),base_proto=zeros(n,spec.horizon,13),decoder=state,
        current=dict(position=state['position'],rotation=torch.diag(torch.tensor([1.,-1.,-1.])).repeat(n,1,1),wrench=state['wrench']),task_reference=reference)


def test_dense_transport_full_box_identity_initialization_and_checkpoint():
    torch.manual_seed(123);s=PublicResponseSpec(history_len=1,horizon=2,maximum_joints=2,members=2,hidden=(8,));policy=PublicResidualTransport(s,PublicTransportSpec(hidden=(16,)));b=training_batch(s)
    context=decoder_context(b['decoder'],PublicBiasFeedbackSpec())
    value=policy(b['target_features'],b['source_features'],b['source_commands'],b['base_proto'],b['decoder']['phase'],context);assert torch.equal(value,b['base_proto'])
    with torch.no_grad():policy.network[-1].bias.fill_(100.)
    value=policy(b['target_features'],b['source_features'],b['source_commands'],b['base_proto'],b['decoder']['phase'],context);assert torch.all(value<=1) and torch.all(value>=-1) and torch.all(value[...,12]>=0)
    loaded=PublicResidualTransport.from_checkpoint(policy.checkpoint());assert torch.equal(value,loaded(b['target_features'],b['source_features'],b['source_commands'],b['base_proto'],b['decoder']['phase'],context))


def test_response_frozen_during_transport_and_optimizer_rng_resume_exact():
    torch.set_num_threads(1);torch.manual_seed(124);s=PublicResponseSpec(history_len=1,horizon=2,maximum_joints=2,members=2,hidden=(8,));response=PublicResponseEnsemble(s);original={k:v.clone() for k,v in response.state_dict().items()}
    policy=PublicResidualTransport(s,PublicTransportSpec(hidden=(16,)));trainer=PublicTransportTrainer(policy,response,PublicBiasFeedbackSpec());b=training_batch(s)
    trainer.step(b,response_weight=0.);state=trainer.state();expected=trainer.step(b,response_weight=1.);weights=trainer.state();trainer.restore(state)
    assert trainer.step(b,response_weight=1.)==expected and _equal(weights,trainer.state())
    assert _equal(original,response.state_dict()) and all(p.grad is None for p in response.parameters())


def test_unused_torch_CPU_rng_does_not_change_dense_objective_update():
    torch.set_num_threads(1);torch.manual_seed(125);s=PublicResponseSpec(history_len=1,horizon=2,maximum_joints=2,members=2,hidden=(8,));response=PublicResponseEnsemble(s)
    policy=PublicResidualTransport(s,PublicTransportSpec(hidden=(16,)));trainer=PublicTransportTrainer(policy,response,PublicBiasFeedbackSpec());b=training_batch(s);state=trainer.state()
    torch.manual_seed(126);first=trainer.step(b);weights=trainer.state();trainer.restore(state);torch.manual_seed(127);second=trainer.step(b);other=trainer.state()
    assert first==second and _equal(weights['checkpoint'],other['checkpoint']) and _equal(weights['optimizer'],other['optimizer'])
    assert not torch.equal(weights['rng']['torch_cpu'],other['rng']['torch_cpu'])


def test_transient_phase_transport_gives_terminal_control_to_task_only():
    actor = object.__new__(PublicTransientPhaseAdaptiveStepDenseTransportActor)
    candidates = torch.tensor([True, False, True, False])
    for phase in (0, 1, 2):
        assert torch.equal(
            actor._candidate_admission_mask(candidates, phase), candidates,
        )
    assert not actor._candidate_admission_mask(candidates, 3).any()


def test_phase_scheduled_transport_has_bounded_phase_and_mode_authority():
    actor = object.__new__(PublicPhaseScheduledAdaptiveStepDenseTransportActor)
    candidates = torch.ones(7, dtype=torch.bool)
    actor.mode = 0
    assert not actor._candidate_admission_mask(candidates, 0)[0]
    assert actor._candidate_admission_mask(candidates, 2).all()
    assert torch.equal(
        actor._candidate_admission_mask(candidates, 3),
        torch.tensor([False, True, True, True, True, True, True]),
    )
    actor.mode = 1
    assert torch.equal(
        actor._candidate_admission_mask(candidates, 3),
        torch.tensor([False, False, True, True, True, True, True]),
    )


def test_mode_conditioned_transport_freezes_the_two_predeclared_branches():
    assert issubclass(
        PublicModeConditionedCompletionValueDenseTransportActor,
        PublicCompletionValueEnsembleRobustDenseTransportActor,
    )
    assert issubclass(
        PublicModeConditionedAdaptiveStepDenseTransportActor,
        PublicAdaptiveStepTaskDominantDenseTransportActor,
    )
    assert (
        PublicModeConditionedCompletionValueDenseTransportActor.VERSION
        == PublicModeConditionedAdaptiveStepDenseTransportActor.VERSION
        == 'public-T3-mode-conditioned-finite-time-dominance-v14'
    )


def test_mode_specific_recorded_completion_uses_robust_public_admission():
    assert issubclass(
        PublicModeSpecificRecordedCompletionDenseTransportActor,
        PublicExpandedEnsembleRobustAdaptiveStepTaskDominantDenseTransportActor,
    )
    assert (
        PublicModeSpecificRecordedCompletionDenseTransportActor.VERSION
        == 'public-T3-mode-specific-recorded-completion-dominance-v15'
    )


def test_mode_specific_supported_phase_convex_mask():
    actor = object.__new__(
        PublicModeSpecificSupportedPhaseConvexDenseTransportActor
    )
    actor.mode = 1
    admitted = torch.ones(len(actor.BLENDS), dtype=torch.bool)
    assert not actor._candidate_admission_mask(admitted, 0).any()
    assert not actor._candidate_admission_mask(admitted, 3).any()
    expected = torch.tensor([False, False, False, True, True, True, True])
    assert torch.equal(actor._candidate_admission_mask(admitted, 1), expected)
    actor.mode = 0
    assert torch.equal(actor._candidate_admission_mask(admitted, 3), admitted)
    assert (
        actor.VERSION
        == 'public-T3-mode-specific-supported-phase-convex-dominance-v16'
    )


def test_mode_specific_supported_phase_capped_expansion_mask():
    actor = object.__new__(
        PublicModeSpecificSupportedPhaseCappedExpansionDenseTransportActor
    )
    actor.mode = 1
    admitted = torch.ones(len(actor.BLENDS), dtype=torch.bool)
    expected = torch.tensor([False, False, True, True, True, True, True])
    assert torch.equal(actor._candidate_admission_mask(admitted, 2), expected)
    assert not actor._candidate_admission_mask(admitted, 3).any()
    assert (
        actor.VERSION
        == 'public-T3-mode-specific-supported-phase-capped-expansion-dominance-v17'
    )


def test_mode_specific_calibrated_authority_mask():
    actor = object.__new__(
        PublicModeSpecificCalibratedAuthorityDenseTransportActor
    )
    actor.mode = 1
    admitted = torch.ones(len(actor.BLENDS), dtype=torch.bool)
    for maximum, expected in (
        (0., torch.tensor([False, False, False, False, False, False, True])),
        (1., torch.tensor([False, False, False, True, True, True, True])),
        (8., torch.ones(len(actor.BLENDS), dtype=torch.bool)),
    ):
        actor.MODE_ONE_MAX_BLEND = maximum
        assert torch.equal(actor._candidate_admission_mask(admitted, 2), expected)
        assert not actor._candidate_admission_mask(admitted, 3).any()
    assert (
        actor.VERSION
        == 'public-T3-mode-specific-calibrated-authority-dominance-v18'
    )
