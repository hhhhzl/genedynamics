"""Legal source queries and public actor continuation/reorder/partial reset."""
import hashlib
import numpy as np
import pytest
import torch

from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.bias_feedback import PublicBiasFeedbackPolicy,PublicBiasFeedbackSpec
from genedynamics.solvers.single.cross_ebdm.learning.response import PublicResponseSpec
from genedynamics.solvers.single.cross_ebdm.learning.skill_bank import PublicSkillBank
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import PublicResidualTransport,PublicTransportSpec
from genedynamics.tasks.transfer.controllers.cross_ebdm.dense import (
    IndependentWorldDenseSolver,
    PublicDenseTransportActor,
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
