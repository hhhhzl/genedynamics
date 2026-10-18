"""Restore native dynamic AverageMeter buffers without changing PPO math.

Upstream AverageMeter.update may broadcast `mean` from [1] to [1,1] after an
episode ends. Returning to an earlier checkpoint must restore its exact shape
before strict load_state_dict. Original training.py stays sealed for old runs.
"""
import torch
from .training import build_learner,capture_training_state,train_epoch
from .training import restore_training_state as legacy_restore_training_state

def prepare_meter_restore(meter,state):
    if set(state)!={'mean'} or set(meter._buffers)!={'mean'}:raise ValueError('only pinned native AverageMeter.mean may change shape')
    saved=state['mean'];current=meter._buffers['mean']
    if saved.numel()!=current.numel() or saved.dtype!=current.dtype:raise ValueError('native meter scalar content/type changed')
    if saved.shape!=current.shape:meter.mean=torch.empty_like(saved,device=current.device)

def restore_training_state(agent,payload):
    for name,meter in payload['meters'].items():prepare_meter_restore(getattr(agent,name),meter['state'])
    legacy_restore_training_state(agent,payload)
