"""Research-critical regressions: feedback, support, budget and provider parity."""
import numpy as np
import pytest


def test_stopped_source_can_correct_orientation_and_preserve_release():
    import jax
    import jax.numpy as jp
    from genedynamics.learning.cross_embodiment.contact_behavior import reference_commands
    from genedynamics.envs.domains.manipulation.cross_embodiment.workcell import rotation
    p=np.array([[.44,-.2,.614]],np.float32); actual=rotation(.1)[None].astype(np.float32); goal=rotation()[None].astype(np.float32)
    reference=np.array([[.44,-.2,.614,0.,0.]],np.float32)
    expected=reference_commands(reference,p,goal,p,actual)
    compiled=jax.jit(lambda r:reference_commands(jp.asarray(reference),jp.asarray(p),jp.asarray(goal),jp.asarray(p),r,xp=jp))
    np.testing.assert_allclose(expected,compiled(jp.asarray(actual)),atol=2e-6)
    assert expected[0,0,5]<-.01 and not expected[...,12].any()
    gradient=jax.jacrev(compiled)(jp.asarray(actual));assert np.isfinite(gradient).all()


def test_phase_waits_for_contact_and_resets_tracking_integral():
    from genedynamics.learning.cross_embodiment.contact_behavior import behavior_reset,advance_behavior
    from genedynamics.envs.domains.manipulation.cross_embodiment.workcell import rotation
    state=behavior_reset(1,[0]);p=np.array([[.45,-.2,.596]]);r=rotation(.12)[None]
    for _ in range(50):state=advance_behavior(state,p,r,np.array([0.]))
    assert state.phase[0]==0
    state=advance_behavior(state,p,r,np.array([2.]));assert state.phase[0]==0
    state=advance_behavior(state,p,r,np.array([2.]));assert state.phase[0]==1
    assert np.max(np.abs(state.tracking_integral))<.02


def test_strict_acceptance_and_intentional_release():
    from genedynamics.learning.cross_embodiment.contact import ContactRunner,research_contact_config,contact_termination
    from genedynamics.learning.cross_embodiment import RandomRequest
    from genedynamics.envs.domains.manipulation.cross_embodiment.workcell import rotation
    config=research_contact_config();runner=ContactRunner(config=config)
    state=runner.reset_batch(RandomRequest(1,(1,)))
    # Termination is backend-neutral; a lightweight scalar carry isolates the
    # boundary cases from simulator numerics.
    from types import SimpleNamespace
    carry=SimpleNamespace(last_command=np.r_[np.zeros(12),5.],contact_seen=True,loss_count=14,hold_count=14,steps=40)
    evaluate=lambda r,**kw:contact_termination(np,carry,np.array(config.goal),5.,5.,False,True,False,config,rotation=r,**kw)
    assert evaluate(rotation())[0]==1
    assert evaluate(rotation(np.deg2rad(3)))[0]==0
    assert evaluate(rotation(),lateral_peak=11.)[0]==9
    carry.last_command=np.zeros(13)
    result=contact_termination(np,carry,np.array([.45,-.2,.60]),0.,0.,False,True,False,config,rotation=rotation())
    assert result[0]==0 and not result[-1]
    # Raw release is immediate intent although the applied force ramps down.
    loaded=state._replace(contact_seen=np.ones(1,bool),loss_count=np.array([14]),last_command=np.tile(np.r_[np.zeros(12),5.],(1,1)))
    result=runner.collect_chunk(loaded,np.zeros((1,1,13),np.float32))
    assert result.trace.command[0,0,12]>0 and result.state.status[0]!=8


def test_portable_affine_transport_and_phase_reference(tmp_path,monkeypatch):
    import subprocess,shutil
    from pathlib import Path
    import jax.numpy as jp
    import torch
    from genedynamics.learning.cross_embodiment.reference_transport import ReferenceTransport
    from genedynamics.learning.cross_embodiment.contact_behavior import PhaseGaussianPrior,behavior_reset,BehaviorState
    from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
    compiler=shutil.which("g++") or shutil.which("clang++");assert compiler
    source=Path(__file__).resolve().parents[2]/"genedynamics/learning/cross_embodiment/backends/native/ensemble.cpp"
    library=tmp_path/"models.so";subprocess.run([compiler,"-std=c++17","-O2","-fPIC","-shared",str(source),"-o",str(library)],check=True)
    monkeypatch.setenv("CROSS_EMBODIMENT_NATIVE_LIBRARY",str(library))
    model=ReferenceTransport.from_scale(1.4);model.save(tmp_path/"transport")
    commands=np.zeros((2,10,13),np.float32);commands[...,12]=5.
    expected=np.asarray(model.apply(commands))
    for backend in ("torch","native"):
        other=ReferenceTransport.load(tmp_path/"transport",backend=backend)
        np.testing.assert_allclose(other.apply(commands),expected,atol=1e-6)
    phased=ReferenceTransport.from_parameters(2.,(-.01,.005));phased.save(tmp_path/"phased")
    phase=np.array([0,3]);expected=np.asarray(phased.apply(commands,phase))
    assert expected[0,0,12]==5. and expected[1,0,12]==10.
    np.testing.assert_allclose(expected[1,0,:2],[-.01,.005])
    import jax
    compiled=jax.jit(lambda u,p:phased.apply(u,p))
    np.testing.assert_allclose(compiled(jp.asarray(commands),jp.asarray(phase)),expected,atol=1e-6)
    for backend in ("torch","native"):
        other=ReferenceTransport.load(tmp_path/"phased",backend=backend)
        np.testing.assert_allclose(other.apply(commands,phase),expected,atol=1e-6)
    # Source release must remain exactly zero even with deliberately large noise.
    from genedynamics.learning.cross_embodiment.models import LearnedModel,ModelSpec
    from genedynamics.learning.cross_embodiment.learning_data import Normalization
    table=np.tile([.44,-.2,.614,0.,0.],(8,1)).astype(np.float32);table[6:,4]=5.
    net=LearnedModel(ModelSpec(8,5,hidden=()));norm=lambda n:Normalization(np.zeros(n,np.float32),np.ones(n,np.float32))
    prior=FrozenRegressor(net,((jp.asarray(table[None]),jp.zeros((1,5))),),norm(8),norm(5),{"learned_role":"phase_source_prior","reference_std":[.0001]*4+[.005]})
    prior.save(tmp_path/"prior");memory=behavior_reset(2,[0,1]);noise=np.ones((2,5),np.float32)*10
    reference=np.asarray(PhaseGaussianPrior(prior).reference(memory,noise));assert not reference[:,4].any()
    for backend in ("torch","native"):
        other=PhaseGaussianPrior(FrozenRegressor.load(tmp_path/"prior",backend=backend))
        m=BehaviorState(*(torch.as_tensor(x) for x in memory)) if backend=="torch" else memory
        n=torch.as_tensor(noise) if backend=="torch" else noise
        np.testing.assert_allclose(other.reference(m,n),reference,atol=1e-6)


def test_new_transport_objective_and_context_physical_bounds(tmp_path):
    # Existing conformance exercise now also covers the new task objective and
    # a nonzero correction at an exactly stopped source axis.
    from test.cross_embodiment.test_portability import test_transport_and_context_conformance
    test_transport_and_context_conformance(tmp_path,True,revision=True)


def test_actual_probe_resume_and_observed_initial_state(tmp_path):
    from genedynamics.learning.cross_embodiment.contact import ContactConfig
    from genedynamics.learning.cross_embodiment import MechanismSpec
    from scripts.cross_embodiment.calibrate_contact_twin import collect_probes
    from genedynamics.learning.cross_embodiment.structured_response import validate_twin
    config=ContactConfig(environment="keyed_insertion_recovery",robot="xarm7",reset_task_noise=.002)
    records,budget=collect_probes(config,MechanismSpec(.8,2,.5),tmp_path)
    assert len(budget["attempts"])==5
    assert sum(np.count_nonzero(r.result.trace.normal_force>.3) for r in records)>0
    restored,other=collect_probes(config,MechanismSpec(.8,2,.5),tmp_path)
    assert other==budget
    for a,b in zip(records,restored):np.testing.assert_array_equal(a.result.trace.normal_force,b.result.trace.normal_force)
    report=validate_twin(config,MechanismSpec(.8,2,.5),restored[3:]);assert report["force_rmse_N"]<1e-8
    with pytest.raises(ValueError,match="contract mismatch"):collect_probes(config,MechanismSpec(.9,2,.5),tmp_path)
