"""Numerical learning/provider contracts, not assertions about research success."""
import numpy as np
import pytest


def test_jax_torch_forward_grad_and_adam(tmp_path):
    import jax
    import jax.numpy as jnp
    import torch
    from genedynamics.solvers.single.cross_ebdm.learning.models import LearnedModel,ModelSpec,RegressionBatch,TrainState
    spec=ModelSpec(3,2,hidden=(5,),ensemble=2)
    model=LearnedModel(spec); state=model.initialize(4)
    model.save(tmp_path/"model",state.parameters)
    other,p,_=LearnedModel.load(tmp_path/"model",backend="torch")
    x=np.random.default_rng(3).normal(size=(7,3)).astype(np.float32)
    y=np.ones((7,2),np.float32); weights=np.array([1,1,1,1,0,0,1],np.float32)
    prediction=np.asarray(model.predict(state.parameters,x))
    np.testing.assert_allclose(prediction,other.predict(p,x).detach().numpy(),atol=2e-6)
    cotangent=np.ones_like(prediction)
    gradient=np.asarray(model.input_gradient(state.parameters,x,cotangent))
    np.testing.assert_allclose(gradient,other.input_gradient(p,x,cotangent).numpy(),atol=2e-6)
    # Directional finite differences independently check input gradients.
    direction=np.random.default_rng(8).normal(size=x.shape).astype(np.float32)
    eps=.001
    numeric=(np.asarray(model.predict(state.parameters,x+eps*direction)).sum()-
             np.asarray(model.predict(state.parameters,x-eps*direction)).sum())/(2*eps)
    np.testing.assert_allclose(np.sum(gradient*direction),numeric,rtol=.002,atol=.002)
    zero=tuple(tuple(torch.zeros_like(v) for v in layer) for layer in p)
    torch_state=TrainState(p,zero,zero,torch.tensor(0))
    a,loss=model.train_step(state,RegressionBatch(jnp.asarray(x),jnp.asarray(y),jnp.asarray(weights)))
    b,other_loss=other.train_step(torch_state,RegressionBatch(x,y,weights))
    np.testing.assert_allclose(loss,other_loss.numpy(),atol=2e-6)
    for v,w in zip(jax.tree.leaves(a.parameters),[v for layer in b.parameters for v in layer]):
        np.testing.assert_allclose(v,w.numpy(),atol=2e-6)
    unchanged,_=model.train_step(a,RegressionBatch(jnp.asarray(x),jnp.asarray(y),jnp.zeros(7)))
    for v,w in zip(jax.tree.leaves(a),jax.tree.leaves(unchanged)):
        np.testing.assert_array_equal(v,w)


def test_global_cvar_microbatch_ties_and_gradient():
    import jax
    import jax.numpy as jnp
    from genedynamics.solvers.single.cross_ebdm.learning.backends.jax.risk import cvar_weights,microbatch_value_and_grad
    risk=jnp.array([3.,3.,1.,99.]); valid=jnp.array([1,1,1,0])
    np.testing.assert_allclose(cvar_weights(risk,valid,.5),[.5,.5,0,0])
    np.testing.assert_array_equal(cvar_weights(risk,jnp.zeros(4),.5),np.zeros(4))
    x=jnp.array([[1.,2.],[2.,1.],[3.,-1.],[0.,4.],[2.,2.]])
    valid=jnp.array([1.,1.,1.,0.,1.]); p=jnp.array([.3,.2])
    def loss(p,x):
        r=(x@p)**2
        return r*.2,jnp.stack((r,r*.8),axis=-1)
    reference=None
    for microbatch in (1,2,3,5):
        value,gradient=jax.jit(lambda p:microbatch_value_and_grad(loss,p,x,valid,
            microbatch_size=microbatch,tail_fraction=.35,multiplier=.7))(p)
        if reference is None: reference=value,gradient
        np.testing.assert_allclose(value,reference[0],atol=1e-6)
        np.testing.assert_allclose(gradient,reference[1],atol=1e-6)


def test_persistent_budget_failure_and_seal(tmp_path):
    from genedynamics.solvers.single.cross_ebdm.learning.probe_budget import ProbeBudget
    from genedynamics.solvers.single.cross_ebdm.learning.calibration import HiddenTargetEnv
    path=tmp_path/"ledger.json"
    budget=ProbeBudget(2,str(path)); index=budget.reserve("fit")
    assert ProbeBudget(2,str(path)).attempts[index]["outcome"]=="pending"
    budget.complete(index,"failed")
    class Broken:
        def reset_batch(self,request): raise RuntimeError("physical reset failed")
    gate=HiddenTargetEnv(Broken(),None,budget)
    with pytest.raises(RuntimeError,match="physical reset failed"): gate.probe(None,None)
    assert ProbeBudget(2,str(path)).attempts[1]["outcome"]=="failed"
    with pytest.raises(RuntimeError,match="exhausted"): gate.probe(None,None)
    gate.seal()
    with pytest.raises(RuntimeError,match="sealed"): gate.probe(None,None)


def test_checkpoint_exact_resume(tmp_path):
    import jax
    import jax.numpy as jnp
    from genedynamics.solvers.single.cross_ebdm.learning.models import LearnedModel,ModelSpec,RegressionBatch
    from genedynamics.solvers.single.cross_ebdm.learning.probe_budget import ProbeBudget
    from genedynamics.solvers.single.cross_ebdm.learning.checkpoint import save_training_checkpoint,load_training_checkpoint
    model=LearnedModel(ModelSpec(2,1,hidden=(4,))); state=model.initialize(9)
    batch=RegressionBatch(jnp.ones((3,2)),jnp.ones((3,1)),jnp.ones(3))
    state,_=model.train_step(state,batch)
    budget=ProbeBudget(10); budget.reserve("validation")
    save_training_checkpoint(tmp_path,state,context=np.zeros(3),multipliers=np.ones(2),rng_seed=7,rng_counter=12,budget=budget)
    restored,_,_,meta=load_training_checkpoint(tmp_path,model)
    assert meta["probe_budget"]["attempts"][0]["outcome"]=="pending"
    a,_=model.train_step(state,batch); b,_=model.train_step(restored,batch)
    for v,w in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
        np.testing.assert_array_equal(v,w)


def test_native_whole_batch_portable_inference(tmp_path,monkeypatch):
    import shutil
    import subprocess
    from pathlib import Path
    from genedynamics.solvers.single.cross_ebdm.learning.models import LearnedModel,ModelSpec
    compiler=shutil.which("g++") or shutil.which("clang++")
    if compiler is None: pytest.skip("native compiler unavailable")
    source=Path(__file__).resolve().parents[2]/"genedynamics/solvers/single/cross_ebdm/learning/backends/native/ensemble.cpp"
    library=tmp_path/"model.so"
    subprocess.run([compiler,"-std=c++17","-O2","-fPIC","-shared",str(source),"-o",str(library)],check=True)
    monkeypatch.setenv("CROSS_EBDM_NATIVE_LIBRARY",str(library))
    model=LearnedModel(ModelSpec(3,2,hidden=(7,5),ensemble=3)); state=model.initialize(8)
    model.save(tmp_path/"weights",state.parameters)
    native,p,_=LearnedModel.load(tmp_path/"weights",backend="native")
    inputs=np.arange(18,dtype=np.float32).reshape(6,3)/10
    np.testing.assert_allclose(model.predict(state.parameters,inputs),native.predict(p,inputs),atol=1e-6)
    np.testing.assert_allclose(native.predict(p,inputs[::-1]),native.predict(p,inputs)[:,::-1],atol=1e-6)
    with pytest.raises(ValueError): native.predict(p,np.ones((2,4)))
    with pytest.raises(NotImplementedError): native.train_step(None,None)
