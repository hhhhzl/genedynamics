"""Restore both sides of an actual upstream meter broadcast transition."""
from copy import deepcopy
import pytest
import torch
from genedynamics.solvers.single.human2sim2robot.learning.training import source_learner
from genedynamics.solvers.single.human2sim2robot.learning.training_v2 import prepare_meter_restore

@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_upstream_dynamic_meter_restore_both_directions(device):
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    source_learner()
    from human2sim2robot.ppo.utils.torch_ext import AverageMeter
    meter=AverageMeter((1,),100).to(device);initial=deepcopy(meter.state_dict());meter.update(torch.tensor([[[2.]],[[4.]]],device=device));later=deepcopy(meter.state_dict())
    assert initial['mean'].shape==(1,) and later['mean'].shape==(1,1)
    prepare_meter_restore(meter,initial);meter.load_state_dict(initial);meter.current_size=0
    meter.update(torch.tensor([[[2.]],[[4.]]],device=device));assert torch.equal(meter.mean,later['mean'])
    prepare_meter_restore(meter,initial);meter.load_state_dict(initial);prepare_meter_restore(meter,later);meter.load_state_dict(later);assert torch.equal(meter.mean,later['mean'])
    with pytest.raises(ValueError):prepare_meter_restore(meter,dict(mean=torch.zeros(2,device=device)))
