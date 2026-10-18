"""Known limiter equivalence, nonidentifiable requests and strict model identity."""
import numpy as np
import pytest
import torch

from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.tasks.transfer.contact import bounded_contact_command
from genedynamics.solvers.single.cross_ebdm.learning.response import PublicResponseSpec,PublicResponseEnsemble,public_frame,features
from genedynamics.solvers.single.cross_ebdm.learning.issued_response import project_contact_requests,PublicIssuedResponseEnsemble
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import BOUNDS
from test.transfer_methods.test_public_observed_response import record


def test_known_limiter_matches_numpy_physical_command_chart_without_delay():
    rng=np.random.default_rng(42);request=rng.uniform(-1,1,(2,10,13)).astype(np.float32);request[...,12]=np.abs(request[...,12]);previous=np.zeros((2,13),np.float32)
    actual=project_contact_requests(torch.from_numpy(request),torch.from_numpy(previous)).numpy()*BOUNDS
    expected=[]
    for row in request:
        p=np.zeros(13,np.float32);values=[]
        for command in row:
            p=bounded_contact_command(np,command*BOUNDS,p,task_spec('T3',robot='panda').config);values.append(p.copy())
        expected.append(values)
    np.testing.assert_allclose(actual,np.asarray(expected),rtol=0,atol=1e-6)


def test_force5_and25_have_identical_first200ms_issued_effect_from_zero():
    commands=torch.zeros(1,10,13);commands[...,12]=5./25.;other=commands.clone();other[...,12]=1.
    torch.testing.assert_close(project_contact_requests(commands,torch.zeros(1,13)),project_contact_requests(other,torch.zeros(1,13)),rtol=0,atol=0)
    other.requires_grad_();value=project_contact_requests(other,torch.zeros(1,13)).sum();value.backward();assert torch.count_nonzero(other.grad[...,12])==0


def test_raw_force_enable_bit_is_preserved_when_issued_commands_coincide():
    torch.manual_seed(43);s=PublicResponseSpec(history_len=1,horizon=10,maximum_joints=2,members=2,hidden=(8,));p,a=record();p['previous_action'][:,12]=5.;f=features(public_frame(p,'panda',s),[0],'panda','T3',[0,0,0],s)
    low=torch.zeros(1,10,13);low[...,12]=.09/25.;high=low.clone();high[...,12]=.11/25.;previous=torch.zeros(1,13);previous[:,12]=.2
    assert torch.equal(project_contact_requests(low,previous),project_contact_requests(high,previous))
    model=PublicIssuedResponseEnsemble(s);assert not torch.equal(model(torch.from_numpy(f),low),model(torch.from_numpy(f),high))


def test_issued_response_is_invariant_to_equivalent_requests_and_checkpoint_chart():
    torch.set_num_threads(1);s=PublicResponseSpec(history_len=1,horizon=10,maximum_joints=2,members=2,hidden=(8,));p,a=record();f=features(public_frame(p,'panda',s),[0],'panda','T3',[0,0,0],s)
    model=PublicIssuedResponseEnsemble(s);commands=torch.zeros(1,10,13);commands[...,12]=.2;other=commands.clone();other[...,12]=1.
    assert torch.equal(model(torch.from_numpy(f),commands),model(torch.from_numpy(f),other))
    loaded=PublicIssuedResponseEnsemble.from_checkpoint(model.checkpoint());assert torch.equal(model(torch.from_numpy(f),commands),loaded(torch.from_numpy(f),commands))
    with pytest.raises(ValueError):PublicResponseEnsemble.from_checkpoint(model.checkpoint())
