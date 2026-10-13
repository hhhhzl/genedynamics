"""Real Go2+Z1 stance dynamics, explicit QP carry and failure bookkeeping."""
import numpy as np


def test_floating_stance_wbc_carry_and_partial_reset():
    from genedynamics.learning.cross_embodiment.contact import ContactRunner,ContactConfig,ContactCommands
    from genedynamics.learning.cross_embodiment.contact_data import contact_carry
    from genedynamics.learning.cross_embodiment import RandomRequest
    r=ContactRunner(config=ContactConfig(environment='mobile_keyed_insertion_recovery',robot='go2_z1',mobile=True))
    request=RandomRequest(6,(60,61)); state=r.reset_batch(request)
    assert r._impl.model.nq==25 and r._impl.model.nv==24
    command=np.zeros((2,30,13),np.float32); body=np.zeros((2,30,3),np.float32);body[...,0]=.005
    result=r.collect_chunk(state,ContactCommands(command,body))
    assert np.asarray(result.state.status).tolist()==[0,0]
    assert (result.trace.wbc_residual<.01).all()
    assert (result.trace.support_count[:,-1]>=3).all()
    assert (result.trace.base_position[:,-1,2]>.2).all()
    carry=contact_carry(result.state,r);assert carry.shape==(2,263)
    assert np.linalg.norm(carry[0,210:262])>0  # explicit QP warm start, not fresh zeros
    reset=r.reset_batch(request,previous=result.state,mask=np.array([True,False]))
    np.testing.assert_array_equal(reset.wbc_warm[1],result.state.wbc_warm[1])
    np.testing.assert_array_equal(reset.foot_anchors[1],result.state.foot_anchors[1])
    assert not reset.wbc_warm[0].any()
    failed=r.collect_chunk(result.state,np.full((2,2,13),np.nan,np.float32))
    assert (failed.state.status==6).all() and failed.trace.terminal.sum()==2
    assert not failed.trace.valid[:,1].any()
    # A failed optimization is a physical/control failure, never a silent zero
    # command rollout counted as success; subsequent padding freezes the body.
    r._impl._qp=lambda data,state,ref,rot,command,body,warm:(np.zeros(r._impl.model.nu),warm,1.)
    qp_failed=r.collect_chunk(result.state,np.zeros((2,2,13),np.float32))
    assert (qp_failed.state.status==11).all() and qp_failed.trace.terminal.sum()==2
    assert not qp_failed.trace.valid[:,1].any()
