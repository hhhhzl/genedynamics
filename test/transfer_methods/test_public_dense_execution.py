"""Legal source queries and public actor continuation/reorder/partial reset."""
import hashlib
import numpy as np
import pytest
import torch

from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.bias_feedback import PublicBiasFeedbackPolicy,PublicBiasFeedbackSpec
from genedynamics.solvers.single.cross_ebdm.learning.response import PublicResponseSpec
from genedynamics.solvers.single.cross_ebdm.learning.issued_response import (
    PublicIssuedResponseEnsemble,
)
from genedynamics.solvers.single.cross_ebdm.learning.completion_value import (
    PublicCompletionValueEnsemble,
    PublicCompletionValueSpec,
)
from genedynamics.solvers.single.cross_ebdm.learning.skill_bank import PublicSkillBank
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import PublicResidualTransport,PublicTransportSpec
from genedynamics.tasks.transfer.controllers.cross_ebdm.dense import (
    IndependentWorldDenseSolver,
    PublicAdaptiveStepTaskDominantDenseTransportActor,
    PublicDenseTransportActor,
    PublicEnsembleRobustAdaptiveStepTaskDominantDenseTransportActor,
    PublicExpandedEnsembleRobustAdaptiveStepTaskDominantDenseTransportActor,
    PublicBoundedEndpointEnsembleRobustDenseTransportActor,
    PublicCappedExpansionEnsembleRobustDenseTransportActor,
    PublicHorizonConsistentEnsembleRobustDenseTransportActor,
    PublicValidatedBridgeEnsembleRobustDenseTransportActor,
    PublicCompletionValueEnsembleRobustDenseTransportActor,
    PublicExpandedAdaptiveStepTaskDominantDenseTransportActor,
)
from test.transfer_methods.test_public_observed_response import record


def bank(tmp_path):
    p,a=record();f=tmp_path/'source.npz';np.savez_compressed(f,**{'public.'+k:v for k,v in p.items()},actions=a)
    spec=PublicResponseSpec(history_len=3,horizon=2,maximum_joints=2,members=2,hidden=(8,))
    return PublicSkillBank([dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),mode=0)],spec),p


def test_source_query_never_copies_target_joints_and_rejects_private_inputs(tmp_path):
    b,p=bank(tmp_path);row={k:v[:1].copy() for k,v in p.items()};first,_,i=b.query(row,[0],[0]);row['joint_position']+=100;second,_,j=b.query(row,[0],[0])
    assert np.array_equal(i,j) and all(np.array_equal(first[k],second[k]) for k in first)
    with pytest.raises(ValueError,match='public query'):b.query(dict(row,physics=np.zeros(1)),[0],[0])
    with pytest.raises(ValueError,match='absent'):b.query(row,[2],[0])


def test_dense_identity_decoder_and_resume_are_exact_on_public_inputs(tmp_path):
    torch.set_num_threads(1);b,p=bank(tmp_path);policy=PublicResidualTransport(b.spec,PublicTransportSpec(hidden=(8,)));fb=PublicBiasFeedbackSpec(guard_prediction_seconds=0.)
    actor=PublicDenseTransportActor(task_spec('T3',robot='panda'),policy,b,fb,mode=0);base=PublicBiasFeedbackPolicy(task_spec('T3',robot='panda'),feedback=fb)
    for i in range(8):
        row={k:v[i:i+1] for k,v in p.items()};before=actor.get_state();value=actor.act(row);actor.restore(before);np.testing.assert_array_equal(actor.act(row),value)
        np.testing.assert_allclose(value,base.act(row),rtol=0,atol=2e-8)
    assert not any(k in vars(actor) for k in ('task','mechanism','physics')) and actor.irreversible_network_calls==16


def test_four_world_reorder_and_partial_reset_preserve_other_histories(tmp_path):
    torch.set_num_threads(1);b,p=bank(tmp_path);policy=PublicResidualTransport(b.spec,PublicTransportSpec(hidden=(8,)));fb=PublicBiasFeedbackSpec(guard_prediction_seconds=0.);spec=task_spec('T3',robot='panda')
    solver=IndependentWorldDenseSolver([PublicDenseTransportActor(spec,policy,b,fb,mode=0,seed=i) for i in range(4)])
    rows=lambda tick:{k:np.repeat(v[tick:tick+1],4,axis=0) for k,v in p.items()}
    solver.act(rows(0));state=solver.get_state();expected=solver.act(rows(1));solver.restore(state);solver.reorder([2,0,3,1]);np.testing.assert_array_equal(solver.act(rows(1)),expected[[2,0,3,1]])
    solver.reset([True,False,False,False]);next_rows=rows(2)
    for k,v in next_rows.items():v[0]=p[k][0]
    solver.act(next_rows);assert [a.last_tick for a in solver.actors]==[0,2,2,2]


def test_adaptive_transport_step_keeps_task_only_as_exact_fallback(tmp_path):
    torch.set_num_threads(1)
    b, public = bank(tmp_path)
    transport = PublicResidualTransport(b.spec, PublicTransportSpec(hidden=(8,)))
    task_only = PublicResidualTransport.from_checkpoint(transport.checkpoint())
    response = PublicIssuedResponseEnsemble(b.spec)
    feedback = PublicBiasFeedbackSpec(guard_prediction_seconds=0.)
    actor = PublicAdaptiveStepTaskDominantDenseTransportActor(
        task_spec('T3', robot='panda'), transport, b,
        task_only_policy=task_only, response=response, feedback=feedback, mode=0,
    )
    row = {key: value[:1].copy() for key, value in public.items()}

    before = actor.get_state()
    first = actor.act(row)
    assert actor.transport_admission_controls == 0
    assert actor.task_only_fallback_controls == 1
    assert actor.last_task_dominance_blend == 0.
    assert actor.last_normalized_progress_gain == 0.
    np.testing.assert_array_equal(actor.task_dominance_blend_counts, [0, 0, 0, 0, 0, 1])

    actor.restore(before)
    np.testing.assert_array_equal(actor.act(row), first)
    assert PublicExpandedAdaptiveStepTaskDominantDenseTransportActor.BLENDS == (
        8., 4., 2., 1., .5, .25, 0.,
    )


def test_ensemble_robust_step_requires_every_response_member_to_predict_gain():
    actor = PublicEnsembleRobustAdaptiveStepTaskDominantDenseTransportActor
    linear = torch.tensor([2e-5, 0.])
    angular = torch.tensor([0., 0.])

    disagreement = torch.tensor([
        [0.1, 0.2],
        [0.3, 0.2],
    ])
    *_, worst_gain, admissible = actor._robust_candidate_statistics(
        disagreement, linear, angular, task_index=1,
    )
    assert float(worst_gain[0]) < 0.
    assert admissible.tolist() == [False, False]

    unanimous = torch.tensor([
        [0.1, 0.2],
        [0.15, 0.2],
    ])
    *_, worst_gain, admissible = actor._robust_candidate_statistics(
        unanimous, linear, angular, task_index=1,
    )
    assert float(worst_gain[0]) > 0.
    assert admissible.tolist() == [True, False]
    assert PublicExpandedEnsembleRobustAdaptiveStepTaskDominantDenseTransportActor.BLENDS == (
        8., 4., 2., 1., .5, .25, 0.,
    )


def test_bounded_endpoint_is_the_largest_feasible_direction_step():
    actor = object.__new__(PublicBoundedEndpointEnsembleRobustDenseTransportActor)
    task_only = torch.zeros(1, 2, 13)
    transport = task_only.clone()
    transport[..., 0] = 0.25
    transport[..., 1] = -0.5
    transport[..., 12] = 0.1

    fractions = actor._blend_fractions(transport, task_only).flatten()

    assert float(fractions[0]) == pytest.approx(2.)
    assert fractions[1:].tolist() == [8., 4., 2., 1., .5, .25, 0.]
    endpoint = task_only + fractions[0] * (transport - task_only)
    assert bool((endpoint[..., :12] >= -1.).all())
    assert bool((endpoint[..., :12] <= 1.).all())
    assert bool((endpoint[..., 12:] >= 0.).all())
    assert bool((endpoint[..., 12:] <= 1.).all())
    assert float(endpoint[..., 1].min()) == pytest.approx(-1.)


def test_capped_expansion_is_one_bounded_step_beyond_the_v6_grid():
    assert PublicCappedExpansionEnsembleRobustDenseTransportActor.BLENDS == (
        12., 8., 4., 2., 1., .5, .25, 0.,
    )


def test_horizon_consistent_actor_checkpoints_the_admitted_prefix(tmp_path):
    torch.set_num_threads(1)
    b, _ = bank(tmp_path)
    transport = PublicResidualTransport(b.spec, PublicTransportSpec(hidden=(8,)))
    task_only = PublicResidualTransport.from_checkpoint(transport.checkpoint())
    response = PublicIssuedResponseEnsemble(b.spec)
    feedback = PublicBiasFeedbackSpec(guard_prediction_seconds=0.)
    kwargs = dict(
        task_only_policy=task_only, response=response, feedback=feedback, mode=0,
    )
    actor = PublicHorizonConsistentEnsembleRobustDenseTransportActor(
        task_spec('T3', robot='panda'), transport, b, **kwargs,
    )
    selected = torch.zeros(1, b.spec.horizon, 13)
    actor._record_selected_sequence(
        selected, 8., {'phase': torch.tensor([2])},
    )
    assert actor.horizon_commitment_remaining == b.spec.horizon - 1

    restored = PublicHorizonConsistentEnsembleRobustDenseTransportActor(
        task_spec('T3', robot='panda'),
        PublicResidualTransport.from_checkpoint(transport.checkpoint()), b,
        **kwargs,
    )
    restored.restore(actor.get_state())
    assert restored.horizon_commitment_remaining == b.spec.horizon - 1
    assert restored.committed_phase == 2
    assert restored.committed_blend == 8.


def test_validated_bridge_uses_fit_scale_and_one_public_history(tmp_path):
    torch.set_num_threads(1)
    b, _ = bank(tmp_path)
    transport = PublicResidualTransport(b.spec, PublicTransportSpec(hidden=(8,)))
    actor = PublicValidatedBridgeEnsembleRobustDenseTransportActor(
        task_spec('T3', robot='panda'), transport, b,
        task_only_policy=PublicResidualTransport.from_checkpoint(
            transport.checkpoint()
        ),
        response=PublicIssuedResponseEnsemble(b.spec),
        feedback=PublicBiasFeedbackSpec(guard_prediction_seconds=0.), mode=0,
    )
    selected = torch.zeros(1, b.spec.horizon, 13)
    actor._record_selected_sequence(
        selected, 8., {'phase': torch.tensor([2])},
    )
    assert actor.TASK_TOLERANCE == pytest.approx(.031)
    assert actor.horizon_commitment_remaining == min(
        b.spec.horizon - 1, b.spec.history_len - 1,
    )


def test_completion_value_actor_requires_calibrated_gain_and_public_endpoint(tmp_path):
    torch.set_num_threads(1)
    b, public = bank(tmp_path)
    transport = PublicResidualTransport(b.spec, PublicTransportSpec(hidden=(8,)))
    value = PublicCompletionValueEnsemble(PublicCompletionValueSpec(
        hidden=(8,), members=2, maximum_remaining_controls=10,
    ))
    actor = PublicCompletionValueEnsembleRobustDenseTransportActor(
        task_spec('T3', robot='panda'), transport, b,
        task_only_policy=PublicResidualTransport.from_checkpoint(
            transport.checkpoint()
        ),
        response=PublicIssuedResponseEnsemble(b.spec), completion_value=value,
        minimum_admission_gain_controls=2.,
        feedback=PublicBiasFeedbackSpec(guard_prediction_seconds=0.), mode=0,
    )
    batch, _ = actor.prepare({key: value[:1] for key, value in public.items()})
    assert set(batch['current']) == {
        'position', 'rotation', 'twist', 'wrench', 'goal_position',
        'goal_rotation', 'controller_feedback',
    }
    scores = torch.tensor([[.4, .6], [.4, .6]])
    linear = torch.tensor([2e-5, 0.])
    angular = torch.zeros(2)
    *_, worst_gain, admissible = actor._candidate_statistics(
        scores, linear, angular, task_index=1,
    )
    assert float(worst_gain[0]) == pytest.approx(.2)
    assert admissible.tolist() == [True, False]
    actor.minimum_admission_gain_normalized = .21
    *_, admissible = actor._candidate_statistics(
        scores, linear, angular, task_index=1,
    )
    assert admissible.tolist() == [False, False]
