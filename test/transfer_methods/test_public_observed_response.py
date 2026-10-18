"""No future/private leakage and exact ensemble/optimizer/RNG continuation."""
from copy import deepcopy
import numpy as np
import pytest
import torch

from genedynamics.solvers.single.cross_ebdm.learning.response import PublicResponseSpec,PublicResponseEnsemble,PublicResponseTrainer,public_frame,response_windows,rotation_log
from test.transfer_methods.test_fada_task_training import _equal


def record(n=12):
    t=np.arange(n+1);p=np.tile([.44,-.2,.6],(n+1,1));p[:,0]+=.001*t;r=np.tile(np.diag([1.,-1.,-1.]),(n+1,1,1));zeros=lambda d:np.zeros((n+1,d),np.float32)
    public=dict(joint_position=zeros(2),joint_velocity=zeros(2),tcp_position=p,tcp_rotation=r,tcp_twist=zeros(6),measured_wrench=zeros(6),
        goal_position=p.copy(),goal_rotation=r.copy(),previous_action=zeros(13),previous_request=zeros(13),reference_position=p.copy(),reference_rotation=r.copy(),
        elapsed_seconds=t[:,None]*.02,body_position=zeros(3),body_twist=zeros(6),controller_feedback=zeros(2))
    public['measured_wrench'][:,2]=5.;requests=np.zeros((n,13),np.float32);requests[:,0]=.05
    return public,requests


def test_windows_use_only_past_sensors_and_actions_not_future_or_terminal_labels():
    spec=PublicResponseSpec(history_len=3,horizon=2,maximum_joints=2,members=2,hidden=(8,));p,a=record()
    first=response_windows(p,a,body='panda',task='T2',calibration_context=[0,0,0],spec=spec)
    changed=deepcopy(p);changed['tcp_position'][2:,0]+=1.;changed['reference_position'][2:,0]+=2.
    second=response_windows(changed,a,body='panda',task='T2',calibration_context=[0,0,0],spec=spec)
    assert np.array_equal(first['features'][:2],second['features'][:2]) and not np.array_equal(first['targets'][0],second['targets'][0])
    assert first['features'].shape==(11,spec.feature_dim) and first['commands'].shape==(11,2,13)
    np.testing.assert_allclose(first['targets'][:,0],.2,atol=1e-6)
    np.testing.assert_allclose(np.expm1(first['targets'][:,-2]),5.,atol=1e-6)
    with pytest.raises(ValueError,match='private'):public_frame(dict(p,physics=np.ones(1)),'panda',spec)
    changed=deepcopy(p);changed['joint_position']=np.zeros((13,3))
    with pytest.raises(ValueError,match='padding'):public_frame(changed,'panda',spec)


def test_public_rotation_log_and_response_schema_do_not_accept_private_checkpoints():
    theta=.2;c,s=np.cos(theta),np.sin(theta);r=np.asarray([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    np.testing.assert_allclose(rotation_log(r),[0,0,theta],atol=1e-7)
    with pytest.raises(ValueError,match='near-pi'):rotation_log(np.diag([1.,-1.,-1.]))
    with pytest.raises(ValueError,match='public response'):PublicResponseEnsemble.from_checkpoint(dict(version='legacy-response',private_state=True))


def test_ensemble_training_bootstrap_and_optimizer_rng_resume_exact():
    torch.set_num_threads(1);torch.manual_seed(1234);spec=PublicResponseSpec(history_len=3,horizon=2,maximum_joints=2,members=2,hidden=(16,),learning_rate=.003)
    p,a=record();raw=response_windows(p,a,body='panda',task='T2',calibration_context=[0,0,0],spec=spec);batch={k:torch.from_numpy(raw[k]) for k in ('features','commands','targets')}
    model=PublicResponseEnsemble(spec);trainer=PublicResponseTrainer(model)
    with torch.no_grad():before=float(((model(batch['features'],batch['commands'])-batch['targets'])**2).mean())
    for _ in range(30):assert np.isfinite(trainer.step(batch))
    with torch.no_grad():after=float(((model(batch['features'],batch['commands'])-batch['targets'])**2).mean())
    assert after<before/2
    state=trainer.state();expected=trainer.step(batch);weights=trainer.state();trainer.restore(state)
    assert trainer.step(batch)==expected and _equal(weights,trainer.state())
    payload=model.checkpoint();loaded=PublicResponseEnsemble.from_checkpoint(payload)
    assert torch.equal(model(batch['features'],batch['commands']),loaded(batch['features'],batch['commands']))
