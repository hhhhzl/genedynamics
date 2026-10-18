"""Local class cannot invent force loading in source force-off phases."""
import numpy as np
import pytest
import torch
from genedynamics.solvers.single.cross_ebdm.learning.response import PublicResponseSpec
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import PublicTransportSpec,BOUNDS
from genedynamics.solvers.single.cross_ebdm.learning.local_transport import PublicLocalResidualTransport,PublicLocalNoiseTransport


def test_local_policy_preserves_off_modes_after_learning_and_roundtrip():
    r=PublicResponseSpec(history_len=1,horizon=2,maximum_joints=1,members=1,hidden=(8,));p=PublicLocalResidualTransport(r,PublicTransportSpec(residual_scale=.025,hidden=(8,)))
    with torch.no_grad():p.network[-1].bias.fill_(100.)
    f=torch.zeros((1,r.feature_dim));base=torch.zeros((1,2,13));base[:,1,12]=.2;x=(f,f,base,base,torch.zeros(1,dtype=torch.long),torch.zeros((1,9)));value=p(*x)
    assert value[0,0,12]==0 and value[0,1,12]>.2 and float((value-base).abs().max())<=.025001
    q=PublicLocalResidualTransport.from_checkpoint(p.checkpoint());torch.testing.assert_close(value,q(*x),rtol=0,atol=0)
    with pytest.raises(ValueError,match='maximum'):PublicLocalResidualTransport(r,PublicTransportSpec(residual_scale=1.))


def test_excitation_chunks_and_irreversible_sequence_resume():
    transport=PublicLocalNoiseTransport(seed=1);raw=np.zeros((1,1,13));raw[...,6:12]=1.;values=[]
    for tick in range(12):
        before=transport.get_state();value=transport.apply(raw,[2]);after=transport.get_state();transport.restore(before);np.testing.assert_array_equal(value,transport.apply(raw,[2]));transport.restore(after);values.append(value)
        assert value[0,0,12]==0 and np.max(np.abs((value-raw)/BOUNDS))<=.025001
    np.testing.assert_array_equal(values[0],values[9]);assert not np.array_equal(values[9],values[10])
