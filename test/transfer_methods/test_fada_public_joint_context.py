"""Public-only currentencoder chart and nativefuture-chart initialization parity."""
import numpy as np
import torch
from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.experiments.plugins.methods.transfer.fada.joint import joint_context,lift_joint_checkpoint,JOINT_CONTEXT_DIM
from genedynamics.solvers.single.fada import FADASolver
from genedynamics.solvers.single.fada.learning.task_training import released_model_config
from genedynamics.solvers.single.fada.backends.model import PlannerIDMPolicy
from test.transfer_methods.test_public_observed_response import record


def test_current_joint_features_do_not_claim_joint_correspondence_or_accept_private_fields():
    p,_=record();p={k:v[:2].copy() for k,v in p.items()};a=joint_context(p,'panda');assert a.shape==(2,132);np.testing.assert_array_equal(a[:,32:34],1.);np.testing.assert_array_equal(a[:,34:64],0.);np.testing.assert_array_equal(a[:,-4:],[[1,0,0,0]]*2)
    changed={k:v.copy() for k,v in p.items()};changed['joint_position'][:,0]+=.1;b=joint_context(changed,'panda');np.testing.assert_allclose(b[:,0]-a[:,0],.1/np.pi,rtol=1e-5);np.testing.assert_array_equal(b[:,1:],a[:,1:])
    import pytest
    with pytest.raises(ValueError):joint_context(dict(p,mechanism=np.ones((2,1))),'panda')


def test_lift_preserves_taskspace_planner_and_native_inverse_can_observe_current_joints():
    torch.set_num_threads(1);torch.manual_seed(113000);cfg=released_model_config(35,cmd_dim=67);model=PlannerIDMPolicy(**cfg);payload=dict(cfg=cfg,obs_dim=35,act_dim=13,cmd_dim=67,model_state_dict=model.state_dict(),normalization=dict(io_enabled=True),obs_norm_stats=dict(obs_mean=torch.zeros(35),obs_std=torch.ones(35)),action_norm_stats=dict(action_mean=torch.zeros(13),action_std=torch.ones(13)),command_norm_stats=dict(command_mean=torch.zeros(67),command_std=torch.ones(67)),genedynamics_task_adapter=dict(task_name='keyed_insertion_recovery',action_decoder=dict(kind='continuous13D')));lift=lift_joint_checkpoint(payload);old=FADASolver.from_checkpoint(payload);new=FADASolver.from_checkpoint(lift);obs=torch.randn(2,30,35);act=torch.randn(2,30,13);cmd=torch.randn(2,67);extra=torch.randn(2,JOINT_CONTEXT_DIM);mask=torch.ones(2,30,dtype=torch.bool)
    a,fa=old.predict(obs,act,cmd,history_valid_mask=mask,return_obs=True);b,fb=new.predict(obs,act,torch.cat((cmd,extra),-1),history_valid_mask=mask,return_obs=True);torch.testing.assert_close(fa,fb,atol=2e-6,rtol=1e-5);zero=torch.zeros_like(extra);a0,f0=new.predict(obs,act,torch.cat((cmd,zero),-1),history_valid_mask=mask,return_obs=True);torch.testing.assert_close(a,a0,atol=2e-6,rtol=1e-5);assert torch.max(torch.abs(a0-b))>1e-5;assert lift['cmd_dim']==199 and payload['cmd_dim']==67;assert lift['command_norm_stats']['command_mean'].numel()==199;assert lift['model_state_dict']['planner.cmd_embed.weight'][:,67:].count_nonzero()==0
