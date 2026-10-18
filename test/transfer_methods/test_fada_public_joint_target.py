"""Zero feature extension preserves pretrained function; full native IDM learns it."""
import torch
from genedynamics.solvers.single.fada.learning.task_training import released_model_config
from genedynamics.solvers.single.fada.backends.model import PlannerIDMPolicy
from genedynamics.solvers.single.fada import FADASolver
from genedynamics.experiments.plugins.methods.transfer.fada.joint_target import lift_target_joint_checkpoint
from genedynamics.solvers.single.fada.learning.trainer import FADAIDMAdapter,IDMAdaptationConfig


def test_source_preserving_target_extension_and_full_native_current_encoder_gradient():
    torch.set_num_threads(1);torch.manual_seed(129000);cfg=released_model_config(35,cmd_dim=67);cfg['idm_use_current_command_for_history']=True;model=PlannerIDMPolicy(**cfg);payload=dict(cfg=cfg,obs_dim=35,act_dim=13,cmd_dim=67,model_state_dict=model.state_dict(),normalization=dict(io_enabled=True),obs_norm_stats=dict(obs_mean=torch.zeros(35),obs_std=torch.ones(35)),action_norm_stats=dict(action_mean=torch.zeros(13),action_std=torch.ones(13)),command_norm_stats=dict(command_mean=torch.zeros(67),command_std=torch.ones(67)),genedynamics_task_adapter=dict(task_name='keyed_insertion_recovery',action_decoder=dict(kind='continuous13D')));lift=lift_target_joint_checkpoint(payload);old=FADASolver.from_checkpoint(payload);new=FADASolver.from_checkpoint(lift);obs=torch.randn(2,30,35);act=torch.randn(2,30,13);cmd=torch.randn(2,67);extra=torch.randn(2,132);aug=torch.cat((cmd,extra),-1);mask=torch.ones(2,30,dtype=torch.bool);a,fa=old.predict(obs,act,cmd,history_valid_mask=mask,return_obs=True);b,fb=new.predict(obs,act,aug,history_valid_mask=mask,return_obs=True);torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5);torch.testing.assert_close(fa,fb,atol=2e-6,rtol=1e-5)
    adapter=FADAIDMAdapter(lift,config=IDMAdaptationConfig(finetune_method='full'));adapter.step(dict(history_obs=obs,history_act=act,current_command=aug,history_valid_mask=mask,future_observations=torch.randn(2,6,35),target_actions=torch.randn(2,6,13)));trained=adapter.export_checkpoint()['model_state_dict'];assert trained['idm.history_cmd_embed.weight'][:,67:].count_nonzero()>0
    for k,v in lift['model_state_dict'].items():
        if k.startswith('planner.'):assert torch.equal(v,trained[k])
    assert payload['cmd_dim']==67 and adapter.method=='full'
