"""Optional source future supervision must preserve IDM pass and exact resume."""
from copy import deepcopy
import pytest
import torch

from genedynamics.solvers.single.fada.learning.task_training import TaskPlannerIDMTrainer
from test.transfer_methods.test_fada_task_training import _trainer_and_data,_equal


def test_future_supervision_changes_planner_without_changing_first_idm_update():
    torch.set_num_threads(1);torch.manual_seed(33200)
    _,_,zero,data=_trainer_and_data();positive=TaskPlannerIDMTrainer(zero.config,zero.statistics,planner_obs_loss_coef=.1)
    positive.model.load_state_dict(zero.model.state_dict());batch={k:torch.stack([data[i][k] for i in (0,1)]) for k in data[0]}
    torch.manual_seed(33201);m0=zero.step(batch);torch.manual_seed(33201);m1=positive.step(batch)
    assert m0['idm_action_loss']==m1['idm_action_loss']
    assert _equal(zero.model.idm.state_dict(),positive.model.idm.state_dict())
    assert not _equal(zero.model.planner.state_dict(),positive.model.planner.state_dict())
    assert m1['planner_observation_loss']>0
    assert m1['planner_total_loss']==pytest.approx(m1['planner_via_idm_action_loss']+.1*m1['planner_observation_loss'],rel=1e-6)
    assert m0['planner_observation_loss']==0 and m0['planner_total_loss']==m0['planner_via_idm_action_loss']


def test_nonzero_future_loss_and_optimizer_rng_resume_exact_and_mismatch_rejected():
    torch.set_num_threads(1);torch.manual_seed(33202)
    _,_,base,data=_trainer_and_data();trainer=TaskPlannerIDMTrainer(base.config,base.statistics,planner_obs_loss_coef=.1)
    batch={k:torch.stack([data[i][k] for i in (0,1)]) for k in data[0]};trainer.step(batch);state=trainer.get_training_state()
    expected=trainer.step(batch);weights=trainer.get_training_state();trainer.restore_training_state(state)
    assert trainer.step(batch)==expected and _equal(trainer.get_training_state(),weights)
    with pytest.raises(ValueError,match='coefficient'):base.restore_training_state(state)
    legacy=base.get_training_state();legacy['checkpoint']['genedynamics_task_adapter'].pop('planner_obs_loss_coef');base.restore_training_state(legacy)


def test_future_loss_coefficient_rejects_nonfinite_and_negative_values():
    _,_,base,_=_trainer_and_data()
    for coefficient in (-.1,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='coefficient'):TaskPlannerIDMTrainer(base.config,base.statistics,planner_obs_loss_coef=coefficient)
