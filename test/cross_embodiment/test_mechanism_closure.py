"""Independent physical and portable-law checks for the research closure."""
from dataclasses import replace
import numpy as np
import pytest


@pytest.mark.parametrize("backend",["jax","torch","native"])
def test_finite_response_joint_sampling_and_portable_artifact(tmp_path,monkeypatch,backend):
    from genedynamics.learning.cross_embodiment.finite_response import FiniteResponseLaw
    from scripts.cross_embodiment.build_native import main
    import subprocess,sys
    library=tmp_path/"response.so"
    subprocess.run([sys.executable,"-m","scripts.cross_embodiment.build_native","--output",str(library)],check=True,capture_output=True)
    monkeypatch.setenv("CROSS_EMBODIMENT_NATIVE_LIBRARY",str(library))
    # Perfect anti-correlation is lost by independent marginal sampling, while
    # all channel means would still be identical. Test the actual joint draw.
    table=np.array([[[0.,1.],[1.,0.]],[[2.,4.],[4.,2.]]],np.float32)
    law=FiniteResponseLaw.fit(table,backend="native");law.save(tmp_path/"law")
    cases=np.array([0,0,1,1]);uniform=np.array([0.,.99,.1,.6],np.float32)
    expected=np.array([[0.,1.],[1.,0.],[2.,4.],[4.,2.]])
    loaded=FiniteResponseLaw.load(tmp_path/"law",backend=backend)
    result=loaded.sample(cases,uniform)
    if backend=="torch":result=result.detach().numpy()
    np.testing.assert_allclose(result,expected,atol=1e-6)
    for ids,u in (([2],[.1]),([.5],[.1]),([0],[np.nan])):
        with pytest.raises(ValueError):loaded.sample(np.array(ids),np.array(u))
    if backend=="jax":
        import jax
        import jax.numpy as jnp
        assert np.isnan(jax.jit(loaded.sample)(jnp.array([2]),jnp.array([.1]))).all()
    with pytest.raises(ValueError):FiniteResponseLaw.fit(np.full((2,2,3),np.nan))


def test_spatial_guide_has_rising_floor_and_real_route_obstacle():
    import mujoco
    from genedynamics.learning.cross_embodiment.contact import ContactRunner,spatial_guide_config
    from genedynamics.learning.cross_embodiment import RandomRequest
    from genedynamics.envs.domains.manipulation.cross_embodiment.workcell import solve_ik,rotation
    runner=ContactRunner(config=spatial_guide_config());impl=runner._impl;model=impl.model
    data=runner.reset_batch(RandomRequest(1,(12,))).physics[0]
    floor=model.geom("guide_surface").id;ridge=model.geom("guide_ridge").id;peg=model.geom("contact_peg").id
    assert abs(data.geom_xmat[floor].reshape(3,3)[2,0])>.09
    assert abs(data.geom_xmat[floor].reshape(3,3)[2,1])>.09
    def penetrates(position):
        data.qpos[:]=solve_ik(model,impl.sid,impl.scene.start_qpos,np.array(position),rotation())
        mujoco.mj_forward(model,data)
        return any(set((c.geom1,c.geom2))=={peg,ridge} and c.dist<-.001 for c in data.contact)
    assert penetrates([.42,-.19,.591]) # diagonal shortcut runs through ridge
    assert not penetrates([.42,-.20,.590])
    assert not penetrates([.42,-.18,.592])
    with pytest.raises(NotImplementedError):ContactRunner(config=spatial_guide_config(),backend="jax")


def test_feedback_noise_is_replayable_and_does_not_corrupt_physical_state():
    from genedynamics.learning.cross_embodiment.contact import ContactRunner,mating_contact_config
    from genedynamics.learning.cross_embodiment import RandomRequest
    from genedynamics.learning.cross_embodiment.contact_behavior import behavior_reset
    from scripts.cross_embodiment.cpu_mechanism_validation import policy_step
    runner=ContactRunner(config=mating_contact_config());state=runner.reset_batch(RandomRequest(1,(2,)),initial="jammed")
    original=state.physics[0].qpos.copy();memory=behavior_reset(1,[0])
    a=policy_step(runner,state,memory,seed=13,observation_noise=[.00002,.0001,.03])
    b=policy_step(runner,state,memory,seed=13,observation_noise=[.00002,.0001,.03])
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(state.physics[0].qpos,original)
    with pytest.raises(ValueError):policy_step(runner,state,memory,observation_noise=[-1.,0.,0.])


def test_floating_shared_impedance_and_integral_are_explicit_carry():
    from genedynamics.learning.cross_embodiment.contact import ContactRunner,mating_contact_config
    from genedynamics.learning.cross_embodiment.contact_data import contact_carry
    from genedynamics.learning.cross_embodiment import RandomRequest
    config=replace(mating_contact_config("xarm7"),robot="go2_z1",mobile=True,environment="mobile_keyed_insertion_recovery",reset_task_noise=0.)
    runner=ContactRunner(config=config);state=runner.reset_batch(RandomRequest(12,(24,)))
    assert runner._impl.qp_stride==2 and runner.describe()["wbc_dt"]==.004
    fine=ContactRunner(config=replace(config,physics_dt=.001))
    assert fine._impl.qp_stride==4 and fine.describe()["wbc_dt"]==.004
    with pytest.raises(ValueError,match="fixed 4 ms"):
        ContactRunner(config=replace(config,physics_dt=.005))
    one=runner._impl._slice(state,0);n=runner._impl.model.nv+12
    assert one.wbc_warm.shape==(n+6,) and contact_carry(state,runner).shape==(1,263)
    raw=np.zeros(13);raw[5]=.05
    args=(one.physics,one,one.reference_position,one.reference_rotation)
    a=runner._impl._qp(*args,raw,one.body_reference,one.wbc_warm[:n])[0]
    raw[6:12]=1.
    b=runner._impl._qp(*args,raw,one.body_reference,one.wbc_warm[:n])[0]
    assert np.linalg.norm(a-b)>1e-4
    result=runner.collect_chunk(state,np.broadcast_to(raw,(1,10,13)).copy())
    assert np.isfinite(contact_carry(result.state,runner)).all()
    assert np.linalg.norm(result.state.wbc_warm[0,-6:])>0
    reset=runner.reset_batch(RandomRequest(13,(25,)),previous=result.state,mask=np.array([True]))
    assert not reset.wbc_warm.any()


def test_outer_compression_preserves_original_artifact_checksum(tmp_path):
    import gzip
    from genedynamics.learning.cross_embodiment.core.artifacts import save_bundle,load_bundle
    save_bundle(tmp_path,{"observation":np.arange(21).reshape(3,7)},dict(role="test"))
    arrays,metadata=load_bundle(tmp_path);source=tmp_path/metadata["tensor_file"]
    original=source.read_bytes();source.with_suffix(".npz.gz").write_bytes(gzip.compress(original));source.unlink()
    again,after=load_bundle(tmp_path)
    assert after==metadata
    np.testing.assert_array_equal(again["observation"],arrays["observation"])


def test_mobile_probe_preserves_body_command_history(tmp_path):
    from typing import NamedTuple
    from genedynamics.learning.cross_embodiment.calibration import HiddenTargetEnv
    from genedynamics.learning.cross_embodiment.core.budget import ProbeBudget
    from genedynamics.learning.cross_embodiment.contact import ContactCommands
    from genedynamics.learning.cross_embodiment import RandomRequest
    class State(NamedTuple):status:object
    class Trace(NamedTuple):command:object
    class Result(NamedTuple):state:object;trace:object
    class Runner:
        def reset_batch(self,*a,**kw):return State(np.array([0]))
        def collect_chunk(self,state,command):return Result(state,Trace(command.skill))
        def synchronize(self,value):pass
    class Observer:
        def observe(self,*a):return None
    gate=HiddenTargetEnv(Runner(),Observer(),ProbeBudget(1,str(tmp_path/"ledger.json")))
    def policy(observation,state,index):return ContactCommands(np.full((1,2,13),index),np.full((1,2,3),index+.25))
    record=gate.probe_policy(RandomRequest(1,(2,)),policy,chunks=2)
    assert isinstance(record.commands,ContactCommands)
    np.testing.assert_array_equal(record.commands.body_twist[0,:,0],[.25,.25,1.25,1.25])
    assert len(gate.budget.attempts)==1


def test_floating_seat_loss_requires_established_seat_not_lip_brush():
    from types import SimpleNamespace
    from genedynamics.learning.cross_embodiment.contact import mating_contact_config,contact_termination
    from genedynamics.envs.domains.manipulation.cross_embodiment.workcell import rotation
    cfg=mating_contact_config();state=SimpleNamespace(last_command=np.r_[np.zeros(12),5.],contact_seen=True,loss_count=0,hold_count=0,steps=50)
    args=(np,state,np.array([.44,-.2,.581]),0.,1.,False,True,False,cfg)
    assert contact_termination(*args,rotation=rotation(),axial_contact=False,seated_history=False)[2]==0
    assert contact_termination(*args,rotation=rotation(),axial_contact=False,seated_history=True)[2]==1


def test_mobile_probe_restore_never_resets_or_steps_target(tmp_path,monkeypatch):
    from genedynamics.learning.cross_embodiment.contact import ContactRunner,mating_contact_config
    from genedynamics.learning.cross_embodiment import RandomRequest
    from genedynamics.learning.cross_embodiment.core.artifacts import save_bundle
    from scripts.cross_embodiment.validate_mobile_mechanism import restore_probe
    cfg=replace(mating_contact_config(),robot="go2_z1",mobile=True,environment="mobile_keyed_insertion_recovery",reset_task_noise=0.)
    runner=ContactRunner(config=cfg);state=runner.reset_batch(RandomRequest(15,(30,)))
    commands=np.zeros((1,2,13),np.float32);trace=runner.collect_chunk(state,commands).trace
    save_bundle(tmp_path,dict(qpos=state.physics[0].qpos,qvel=state.physics[0].qvel,skill=commands,body=np.zeros((1,2,3)),
        **{"initial_"+n:getattr(state,n) for n in state._fields if n!="physics"},
        **{"trace_"+n:getattr(trace,n) for n in trace._fields}),dict(purpose="validation"))
    def forbidden(*args,**kwargs):raise AssertionError("restoring evidence must not execute a reset or physical step")
    monkeypatch.setattr(ContactRunner,"reset_batch",forbidden)
    monkeypatch.setattr(type(runner._impl),"_step",forbidden)
    restored=restore_probe(cfg,tmp_path,3)
    np.testing.assert_array_equal(restored.initial_state.physics[0].qpos,state.physics[0].qpos)
    np.testing.assert_array_equal(restored.result.trace.tcp_position,trace.tcp_position)
    assert restored.purpose=="validation"
