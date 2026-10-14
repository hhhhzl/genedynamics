"""Source-derived FADA target adaptation; oracle PPO/DAgger remain upstream."""

from .adaptation import configure_idm_finetune_method, inject_lora_into_idm_backbone, _run_epoch
from .data import Trajectory, TrajectoryWindowDataset
from .trainer import FADAIDMAdapter, IDMAdaptationConfig

run_idm_epoch = _run_epoch

__all__ = ["configure_idm_finetune_method", "inject_lora_into_idm_backbone", "run_idm_epoch", "Trajectory", "TrajectoryWindowDataset", "FADAIDMAdapter", "IDMAdaptationConfig"]
