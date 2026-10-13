"""Cross-provider and replay contracts for learned control on CPU."""
import numpy as np
import pytest


@pytest.mark.parametrize("contact",[False,True])
def test_transport_and_context_conformance(tmp_path,contact,revision=False):
    import jax
    import jax.numpy as jp
    import torch
    from genedynamics.learning.cross_embodiment.models import LearnedModel,ModelSpec,TrainState
    from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
    from genedynamics.learning.cross_embodiment.learning_data import Normalization
    from genedynamics.learning.cross_embodiment.transport import StochasticTransport,TransportSpec,TransportBatch
    from genedynamics.learning.cross_embodiment.calibration import fit_context
    d,h,body=22,2,3 if contact else 2
    active=13 if contact else 12
    model=LearnedModel(ModelSpec(d+h*13+body+3,d+3,hidden=(8,),ensemble=3))
    state=model.initialize(21)
    norm=lambda n:Normalization(np.zeros(n,np.float32),np.ones(n,np.float32))
    meta={"layout":{"carry_dim":d,"semantic_slice":[0,18],"context_dim":3}}
    if revision:meta["context_bounds"]=[[-.8,.6],[0.,1.],[-.3,1.]]
    if contact: meta.update(active_command_dim=13,command_bounds=[.08]*3+[.4]*3+[1.]*6+[25.],risk_kind="contact"); meta["layout"]["force_index"]=20
    frozen=FrozenRegressor(model,state.parameters,norm(model.spec.input_dim),norm(model.spec.output_dim),meta)
    frozen.save(tmp_path/"response")
    other=FrozenRegressor.load(tmp_path/"response",backend="torch")
    spec=TransportSpec(horizon=h,microbatch_size=2,motion_residual_floor=1. if revision else 0.,task_weight=1. if revision else 0.)
    a=StochasticTransport(frozen,spec=spec); initial=a.initialize(4)
    a.save(tmp_path/"transport",initial.parameters)
    b,p=StochasticTransport.load(other,tmp_path/"transport",backend="torch")
    zeros=tuple(tuple(torch.zeros_like(v) for v in layer) for layer in p)
    otherstate=TrainState(p,zeros,zeros,torch.tensor(0))
    rng=np.random.default_rng(10); count=5
    command=rng.uniform(-.01,.01,(count,h,13)).astype(np.float32); command[...,12]=5. if contact else 0.
    if revision:command[...,:6]=0.
    fields=[rng.normal(0,.01,(count,d)).astype(np.float32),rng.normal(0,.01,(count,d)).astype(np.float32),command,
        rng.normal(0,.1,(count,3)).astype(np.float32),np.array([1,1,1,0,1],np.float32),
        np.zeros((count,body),np.float32),np.ones((count,body),np.float32)]
    if revision:fields.append(rng.normal(0,.01,(count,13)).astype(np.float32))
    batch=TransportBatch(*(jp.asarray(v) for v in fields)); tb=TransportBatch(*(torch.as_tensor(v) for v in fields))
    noise=rng.normal(size=(count,h,active)).astype(np.float32)
    np.testing.assert_allclose(a.apply(initial.parameters,batch,jp.asarray(noise)),b.apply(p,tb,torch.as_tensor(noise)).numpy(),atol=2e-6)
    if revision:assert np.abs(np.asarray(a.apply(initial.parameters,batch,jp.asarray(noise)))[...,:6]).max()>1e-7
    sa,la=a.train_step(initial,batch,jp.asarray(noise)); sb,lb=b.train_step(otherstate,tb,torch.as_tensor(noise))
    np.testing.assert_allclose(la,lb.numpy(),rtol=3e-5,atol=3e-5)
    for x,y in zip(jax.tree.leaves(sa.parameters),[v for layer in sb.parameters for v in layer]):
        np.testing.assert_allclose(x,y.numpy(),atol=4e-6)
    base=rng.normal(0,.1,(5,model.spec.input_dim-3)).astype(np.float32)
    targets=rng.normal(0,.01,(5,model.spec.output_dim)).astype(np.float32)
    weights=np.array([1,1,0,1,0],np.float32)
    ca,ha=fit_context(frozen,base,targets,weights,updates=10)
    cb,hb=fit_context(other,base,targets,weights,updates=10)
    np.testing.assert_allclose(ca,cb.numpy(),atol=3e-6)
    np.testing.assert_allclose(ha,hb.numpy(),atol=3e-6)
    if revision:assert ca[1]>=0. and np.all(np.asarray(ca)>=np.array([-.8,0.,-.3])) and np.all(np.asarray(ca)<=np.array([.6,1.,1.]))


def test_random_stream_identity_and_permutation():
    import jax.numpy as jp
    from genedynamics.learning.cross_embodiment.random_streams import jax_normal,jax_modes,RandomStream
    ids=jp.array([1,7,25],jp.uint32); permutation=jp.array([2,0,1])
    a=jax_normal(6,ids,8,RandomStream.PRIOR_SKILL,(10,13))
    b=jax_normal(6,ids[permutation],8,RandomStream.PRIOR_SKILL,(10,13))
    np.testing.assert_array_equal(a[permutation],b)
    np.testing.assert_array_equal(a[1:2],jax_normal(6,ids[1:2],8,RandomStream.PRIOR_SKILL,(10,13)))
    assert not np.array_equal(a,jax_normal(6,ids,8,RandomStream.TRANSPORT,(10,13)))
    p=jp.array([.4,.6]); np.testing.assert_array_equal(jax_modes(4,ids,p)[permutation],jax_modes(4,ids[permutation],p))


def test_actual_fit_resume_and_data_guard(tmp_path):
    import jax
    from genedynamics.learning.cross_embodiment.learned import fit_regressor
    from genedynamics.learning.cross_embodiment.models import ModelSpec
    x=np.arange(24,dtype=np.float32).reshape(12,2)/10; y=x.sum(-1,keepdims=True)
    train=np.arange(12)<8; validation=~train; spec=ModelSpec(2,1,hidden=(4,))
    kwargs=dict(spec=spec,update_chunk=3,batch_size=4)
    full,_,_=fit_regressor(x,y,train,validation,updates=6,**kwargs)
    fit_regressor(x,y,train,validation,updates=3,checkpoint_path=tmp_path,**kwargs)
    resumed,_,_=fit_regressor(x,y,train,validation,updates=6,checkpoint_path=tmp_path,resume=True,**kwargs)
    for a,b in zip(jax.tree.leaves(full.parameters),jax.tree.leaves(resumed.parameters)):np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError,match="contract mismatch"):
        fit_regressor(x+1,y,train,validation,updates=9,checkpoint_path=tmp_path,resume=True,**kwargs)


def test_autoregressive_response_uses_updated_full_carry():
    import jax
    import jax.numpy as jp
    from genedynamics.learning.cross_embodiment.models import LearnedModel,ModelSpec
    from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
    from genedynamics.learning.cross_embodiment.learning_data import Normalization
    from genedynamics.learning.cross_embodiment.backends.jax.response import response_rollout
    d=18; model=LearnedModel(ModelSpec(d+13+2+3,d+3,hidden=(),ensemble=2))
    w=jp.zeros((2,36,21)).at[:,0,0].set(1.); bias=jp.zeros((2,21)).at[:,-3].set(.005)
    norm=lambda n:Normalization(np.zeros(n,np.float32),np.ones(n,np.float32))
    response=FrozenRegressor(model,((w,bias),),norm(36),norm(21),{"layout":{"semantic_slice":[0,18]}})
    initial=jp.zeros((1,d)).at[:,0].set(1.)
    commands=jp.zeros((1,3,1,13)); body=jp.zeros((1,2)); context=jp.zeros((1,3))
    run=jax.jit(lambda c:response_rollout(response,c,commands,body,context))
    states,risks,terminal=run(initial)
    np.testing.assert_allclose(states[0,0,:,0],[2,4,8])
    rotations=np.asarray(states)[...,3:12].reshape(2,1,3,3,3)
    np.testing.assert_allclose(np.swapaxes(rotations,-1,-2)@rotations,np.broadcast_to(np.eye(3),rotations.shape),atol=1e-6)
    assert not np.asarray(terminal).any()
    derivative=jax.grad(lambda c:run(c)[0][0,0,-1,0])(initial)
    np.testing.assert_allclose(derivative[0,0],8.)


def test_atomic_artifact_keeps_previous_checkpoint_on_interruption(tmp_path,monkeypatch):
    from genedynamics.learning.cross_embodiment.core import artifacts
    artifacts.save_bundle(tmp_path,{"x":np.array([1.],np.float32)},{"generation":1})
    replace=artifacts.os.replace
    def interrupted(source,target):
        if str(target).endswith('metadata.json'):raise OSError('simulated interruption')
        return replace(source,target)
    monkeypatch.setattr(artifacts.os,'replace',interrupted)
    with pytest.raises(OSError):artifacts.save_bundle(tmp_path,{"x":np.array([2.],np.float32)},{"generation":2})
    data,meta=artifacts.load_bundle(tmp_path)
    np.testing.assert_array_equal(data['x'],[1.]); assert meta['generation']==1


def test_frozen_contact_prior_torch_and_native_replay(tmp_path,monkeypatch):
    import shutil
    import subprocess
    from pathlib import Path
    import jax.numpy as jp
    import torch
    from genedynamics.learning.cross_embodiment.models import LearnedModel,ModelSpec
    from genedynamics.learning.cross_embodiment.learned import FrozenRegressor,GaussianSkillPrior
    from genedynamics.learning.cross_embodiment.learning_data import Normalization
    compiler=shutil.which('g++') or shutil.which('clang++')
    if compiler is None:pytest.skip('native compiler unavailable')
    source=Path(__file__).resolve().parents[2]/'genedynamics/learning/cross_embodiment/backends/native/ensemble.cpp'
    library=tmp_path/'prior.so';subprocess.run([compiler,'-std=c++17','-O2','-fPIC','-shared',str(source),'-o',str(library)],check=True)
    monkeypatch.setenv('CROSS_EMBODIMENT_NATIVE_LIBRARY',str(library))
    h=2; model=LearnedModel(ModelSpec(33,h*16,hidden=(8,))); state=model.initialize(8)
    norm=lambda n:Normalization(np.zeros(n,np.float32),np.ones(n,np.float32))
    regressor=FrozenRegressor(model,state.parameters,norm(33),norm(h*16),{
        'learned_role':'source_prior','horizon':h,'command_dim':13,'command_bounds':[.08]*3+[.4]*3+[1.]*6+[25.],
        'gaussian_std':[.01]*(h*13),'mode_probabilities':[.5,.5],'reference_feedback_gain':1.,'control_dt':.02})
    regressor.save(tmp_path/'prior')
    x=np.zeros((3,31),np.float32); x[:,:3]=[.4,-.2,.6]; mode=np.array([0,1,0],np.int32)
    noise=np.random.default_rng(5).normal(size=(3,h,13)).astype(np.float32)
    expected=GaussianSkillPrior(regressor).sample(jp.asarray(x),jp.asarray(mode),jp.asarray(noise))
    other=GaussianSkillPrior(FrozenRegressor.load(tmp_path/'prior',backend='torch'))
    actual=other.sample(torch.as_tensor(x),torch.as_tensor(mode),torch.as_tensor(noise)).numpy()
    np.testing.assert_allclose(expected,actual,atol=3e-6)
    native=GaussianSkillPrior(FrozenRegressor.load(tmp_path/'prior',backend='native'))
    np.testing.assert_allclose(expected,native.sample(x,mode,noise),atol=3e-6)


def test_actual_probe_resume_never_replays_charged_world(tmp_path):
    import jax.numpy as jp
    from genedynamics.learning.cross_embodiment.models import LearnedModel,ModelSpec
    from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
    from genedynamics.learning.cross_embodiment.learning_data import Normalization
    from genedynamics.learning.cross_embodiment.contact import ContactConfig
    from genedynamics.learning.cross_embodiment import MechanismSpec
    from scripts.cross_embodiment.cpu_contact_learned_loop import calibrate
    from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
    model=LearnedModel(ModelSpec(333,200,hidden=(4,),ensemble=1)); state=model.initialize(1)
    norm=lambda n:Normalization(np.zeros(n,np.float32),np.ones(n,np.float32))
    response=FrozenRegressor(model,state.parameters,norm(333),norm(200),{'layout':{'context_dim':3,'carry_dim':197}})
    response.save(tmp_path/'response');response=FrozenRegressor.load(tmp_path/'response')
    a,first=calibrate(response,ContactConfig(),MechanismSpec(),2,tmp_path/'run')
    b,second=calibrate(response,ContactConfig(),MechanismSpec(),2,tmp_path/'run',resume=True)
    assert len(first['ledger']['attempts'])==len(second['ledger']['attempts'])==2
    np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError,match='contract mismatch'):
        calibrate(response,ContactConfig(),MechanismSpec(.9,1,.6),2,tmp_path/'run',resume=True)
