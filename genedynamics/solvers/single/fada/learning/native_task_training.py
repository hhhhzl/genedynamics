"""Native FADA separate-source passes on the declared portable task chart.

The historical task trainer remains available for reproducing old artifacts.
This adapter fixes planner target routing and uses verbatim upstream update
methods. It does not import Isaac or claim native humanoid task reproduction.
"""
from copy import deepcopy
from types import SimpleNamespace
import torch
from .native_training_passes import NativeTrainingPasses
from .task_training import TaskPlannerIDMTrainer,sample_dataset

ROUTING_VERSION='native-oracle-planner-separate-idm-v1'


class SourceBatchRouter(NativeTrainingPasses):
    def __init__(self,batch_size,offline_ratio,online_real_ratio):
        self.cfg=SimpleNamespace(batch_size=batch_size,mixed_batch_offline_ratio=offline_ratio,online_trajectory_batch_ratio=online_real_ratio,
            use_generalized_idm=True,planner_suboptimal_batch_ratio=0.,idm_suboptimal_batch_ratio=0.,warmup_planner_suboptimal_batch_ratio=0.,warmup_idm_suboptimal_batch_ratio=0.,suboptimal_expert_batch_ratio=.5)
        self.planner_suboptimal_enabled=False;self.suboptimal_enabled=False


def sample_native_source_batches(dataset,real_data,shadows,batch_size,*,offline_ratio=.2,online_real_ratio=.5):
    if not 0<offline_ratio<1 or not 0<online_real_ratio<1:raise ValueError('strictly interior declared source ratios required')
    router=SourceBatchRouter(batch_size,offline_ratio,online_real_ratio);mode='mixed' if real_data is not None and shadows else 'offline';pc=router._planner_batch_counts(mode=mode);ic=router._idm_batch_counts(mode=mode)
    def shadow(n):
        ids=torch.randint(len(shadows),(n,)).tolist();return {k:torch.stack([shadows[i][k] for i in ids]) for k in shadows[0]}
    def cat(parts):return parts[0] if len(parts)==1 else {k:torch.cat([p[k] for p in parts]) for k in parts[0]}
    # Pinned trainer samples planner first, then IDM sources in native order.
    planner_parts=[]
    if pc['optimal']:planner_parts.append(sample_dataset(dataset,pc['optimal']))
    if pc['online']:planner_parts.append(shadow(pc['online']))
    planner=cat(planner_parts);idm={}
    if ic['optimal']:idm['optimal']=sample_dataset(dataset,ic['optimal'])
    if ic['online']:idm['online']=shadow(ic['online'])
    if ic['online_trajectory']:idm['online_trajectory']=sample_dataset(real_data,ic['online_trajectory'])
    return dict(planner=planner,idm_sources=idm,planner_counts=pc,idm_counts=ic,mode=mode,routing_version=ROUTING_VERSION)


class NativeRoutedTaskPlannerIDMTrainer(TaskPlannerIDMTrainer,NativeTrainingPasses):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.student_model=self.model;self.student_policy=self.model;self.optimizer=self.planner_optimizer
        self.lr_scheduler=None;self.idm_lr_scheduler=None;self._pending=None;self.last_source_counts=None;self.act_dim=self.config['act_dim']
        self.cfg=SimpleNamespace(grad_clip=1.,planner_obs_loss_coef=self.planner_obs_loss_coef,idm_action_loss_coef=1.,idm_action_loss_horizon_gamma=0.,idm_planner_pass_action_loss_coef=1.,
            idm_use_teacher_forcing=True,idm_teacher_forcing_ratio=1.,idm_detach_planner_future_obs=True,planner_predict_delta=self.config['planner_predict_delta'],
            pred_horizon=self.config['pred_horizon'],augment_history_mask_ratio=0.,augment_future_mask_ratio=0.,augment_action_noise_level=0.)

    @property
    def train_step_count(self):return self.steps
    @train_step_count.setter
    def train_step_count(self,value):self.steps=value

    def _use_io_normalization(self):return False  # Task chart normalized once before native passes.
    def _build_augment_noise_scales(self):return None  # No native humanoid term names in the35D task view.
    def _obs_term_slices(self):return []

    def _batch_to_tensors(self,batch):
        return batch['history_obs'],batch['history_act'],batch['current_command'],batch['target_actions'],batch['future_observations'],batch['history_valid_mask']
    def _sample_planner_batch(self,*,mode):return self._pending['planner'],self._pending['planner_counts']
    def _sample_idm_batches_by_source(self,*,mode):return self._pending['idm_sources'],self._pending['idm_counts']

    def step(self,bundle):
        if bundle['routing_version']!=ROUTING_VERSION or bundle['planner_counts']['online_trajectory'] or set(bundle['idm_sources'])-{'optimal','online','online_trajectory'}:raise ValueError('oracle-only planner/native IDM sources required')
        self._pending=dict(bundle,planner=self.normalize(bundle['planner']),idm_sources={k:self.normalize(v) for k,v in bundle['idm_sources'].items()})
        try:
            _,counts,metrics=self._train_step_separate_pass(mode=bundle['mode']);self.last_source_counts=counts
            return dict(idm_action_loss=metrics['idm_action_loss'],planner_via_idm_action_loss=metrics['idm_planner_pass_action_loss'],
                planner_observation_loss=metrics['planner_future_obs_loss'],planner_total_loss=metrics['planner_loss_total']+metrics['idm_planner_pass_action_loss_weighted'])
        finally:self._pending=None

    def checkpoint(self):
        payload=super().checkpoint();payload['genedynamics_task_adapter'].update(source_batch_routing=ROUTING_VERSION,
            source_training='pinned native separate-source IDM TF1 then oracle-only planner TF0 and native observation forward; two AdamW',
            source_augmentation='disabled explicitly for portable task view; upstream dropout and update call graph retained')
        return payload

    def restore_training_state(self,state):
        if state['checkpoint'].get('genedynamics_task_adapter',{}).get('source_batch_routing')!=ROUTING_VERSION:raise ValueError('native source routing differs; legacy states are historical, not silently migrated')
        super().restore_training_state(state)
