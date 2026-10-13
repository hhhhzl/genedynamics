import numpy as np
import pytest
from genedynamics.learning.cross_embodiment.contact import ContactConfig,ContactRunner
from genedynamics.learning.cross_embodiment import RandomRequest


def test_force_channel_carry_failure_and_partial_reset():
    runner=ContactRunner(config=ContactConfig())
    request=RandomRequest(1,(31,32)); initial=runner.reset_batch(request)
    commands=np.zeros((2,3,13),np.float32); commands[...,12]=5.
    result=runner.collect_chunk(initial,commands)
    assert result.trace.command[0,-1,12]>0
    assert np.max(result.trace.normal_force)==0  # still above plane, not command force
    assert np.max(result.state.force_integral)==0  # no force windup before contact
    reset=runner.reset_batch(request,previous=result.state,mask=np.array([True,False]))
    assert reset.steps.tolist()==[0,3]
    np.testing.assert_array_equal(reset.physics[1].qpos,result.state.physics[1].qpos)
    # Declare a previously established contact which has just been lost in air.
    loss=result.state._replace(contact_seen=np.ones(2,bool),loss_count=np.ones(2,int)*14)
    failed=runner.collect_chunk(loss,commands[:,:2])
    assert failed.state.status.tolist()==[8,8]
    assert failed.trace.terminal.sum()==2
    assert not failed.trace.valid[:,1].any()
    np.testing.assert_allclose(failed.state.physics[0].qpos,runner.collect_chunk(loss,commands[:,:1]).state.physics[0].qpos)


def test_contact_jax_reference_short_parity():
    reference=ContactRunner(); compiled=ContactRunner(backend="jax")
    request=RandomRequest(2,(41,)); a=reference.reset_batch(request); b=compiled.reset_batch(request)
    commands=np.zeros((1,10,13),np.float32); commands[...,12]=5.
    # Warm/compile shape once, then run identical chunks; includes load impact.
    for _ in range(30):
        x=reference.collect_chunk(a,commands); y=compiled.collect_chunk(b,commands); compiled.synchronize(y)
        a,b=x.state,y.state
        np.testing.assert_allclose(x.trace.tcp_position,y.trace.tcp_position,atol=.001)
        np.testing.assert_allclose(x.trace.normal_force,y.trace.normal_force,atol=2.,rtol=.1)
    assert x.trace.allowed_contact.any()
    assert np.isfinite(np.asarray(y.trace.force_peak)).all()


def test_normal_projection_and_antiwindup_math():
    from genedynamics.learning.cross_embodiment.contact import hybrid_wrench
    config=ContactConfig(); command=np.zeros(13); command[12]=5.
    pos=np.zeros(3); ref=np.array([.01,.02,100.])
    wrench=hybrid_wrench(np,pos,np.eye(3),np.zeros(6),ref,np.eye(3),command,0.,0.,config,True)
    assert wrench[0]>0 and wrench[1]>0
    np.testing.assert_allclose(wrench[2],-5.25)  # z position error projected away
    command[12]=0.
    released=hybrid_wrench(np,pos,np.eye(3),np.zeros(6),ref,np.eye(3),command,0.,0.,config)
    assert released[2]>1000  # position control restored on release


def test_contact_physics_timestep_refinement():
    # Refine physics only, preserving the 50Hz skill/controller contract.
    rows=[]
    for dt in (.002,.001):
        runner=ContactRunner(config=ContactConfig(physics_dt=dt))
        state=runner.reset_batch(RandomRequest(9,(91,)))
        command=np.zeros((1,300,13),np.float32);command[...,12]=5.
        result=runner.collect_chunk(state,command)
        assert result.state.status[0]==0 and result.trace.allowed_contact.any()
        rows.append(result.trace)
    np.testing.assert_allclose(rows[0].tcp_position[:,-1],rows[1].tcp_position[:,-1],atol=.001)
    np.testing.assert_allclose(rows[0].normal_force[:,-25:].mean(),rows[1].normal_force[:,-25:].mean(),atol=1.)
    assert abs(rows[0].force_peak.max()-rows[1].force_peak.max())<3.
