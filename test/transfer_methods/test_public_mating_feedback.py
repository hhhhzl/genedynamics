"""T4 binding retains physical identity and public continuation semantics."""
from dataclasses import replace
import numpy as np
from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.bias_feedback import PublicBiasFeedbackPolicy,PublicBiasFeedbackSpec
from genedynamics.tasks.transfer.controllers.mating_feedback import PublicMatingFeedbackPolicy
from test.transfer_methods.test_public_observed_response import record


def test_t3_binding_preserves_original_public_actions():
    p,_=record();spec=task_spec('T3',robot='panda');fb=PublicBiasFeedbackSpec(compensation_gain=2.,guard_prediction_seconds=0.)
    bound=PublicMatingFeedbackPolicy(spec,feedback=fb);original=PublicBiasFeedbackPolicy(spec,feedback=fb)
    for tick in range(8):
        row={k:v[tick:tick+1] for k,v in p.items()};np.testing.assert_array_equal(bound.act(row),original.act(row))
    bound.reset();original.reset();np.testing.assert_array_equal(bound.act({k:v[:1] for k,v in p.items()}),original.act({k:v[:1] for k,v in p.items()}))


def test_t4_identity_reset_reorder_and_continuation_remain_explicit():
    p,_=record();spec=task_spec('T4',robot='go2_z1');fb=PublicBiasFeedbackSpec(compensation_gain=2.,guard_prediction_seconds=0.)
    bound=PublicMatingFeedbackPolicy(spec,feedback=fb,modes=[0,1]);template=PublicBiasFeedbackPolicy(replace(spec,alias='T3'),feedback=fb,modes=[0,1])
    row=lambda tick:{k:np.repeat(v[tick:tick+1],2,axis=0) for k,v in p.items()}
    np.testing.assert_array_equal(bound.act(row(0)),template.act(row(0)));state=bound.get_state();a=bound.act(row(1));bound.restore(state);np.testing.assert_array_equal(bound.act(row(1)),a)
    bound.restore(state);bound.reorder([1,0]);np.testing.assert_array_equal(bound.act(row(1)),a[[1,0]])
    bound.reset(modes=[1,0]);assert bound.spec.alias=='T4' and bound.spec.config.robot=='go2_z1' and bound.spec.config.mobile
    assert bound.observer.spec.alias=='T3' and not any(k in vars(bound) for k in ('task','mechanism','physics'))
