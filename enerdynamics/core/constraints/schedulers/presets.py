"""
Preset scheduler configurations.

This module provides preset scheduler configurations for common use cases:
- Paper configurations
- Ablation study configurations
- Common parameter schedules
"""

from typing import Dict, Any
import numpy as np

from enerdynamics.core.constraints.schedulers.base import Scheduler
from enerdynamics.core.constraints.schedulers.cosine_anneal import CosineAnnealScheduler
from enerdynamics.core.constraints.schedulers.dual_anneal import DualAnnealScheduler
from enerdynamics.core.constraints.schedulers.adaptive_gate import AdaptiveGateScheduler
from enerdynamics.core.constraints.core.types import ScheduleState, ScheduleParams


def create_preset_scheduler(
    preset_name: str,
    **kwargs
) -> Scheduler:
    """
    Create scheduler from preset configuration.
    
    Args:
        preset_name: Name of preset ("soft_to_hard", "aggressive", "conservative", etc.)
        **kwargs: Override preset parameters
        
    Returns:
        Scheduler instance
    """
    presets = {
        "soft_to_hard": {
            "scheduler_type": "cosine_anneal",
            "margin_start": 0.5,
            "margin_end": 0.1,
            "rho_start": 0.1,
            "rho_end": 10.0,
            "qp_gate_start": False,
            "qp_gate_end": True,
            "qp_prob_start": 0.0,
            "qp_prob_end": 1.0,
        },
        "aggressive": {
            "scheduler_type": "cosine_anneal",
            "margin_start": 0.3,
            "margin_end": 0.05,
            "rho_start": 1.0,
            "rho_end": 20.0,
            "qp_gate_start": True,
            "qp_gate_end": True,
            "qp_prob_start": 0.5,
            "qp_prob_end": 1.0,
        },
        "conservative": {
            "scheduler_type": "cosine_anneal",
            "margin_start": 0.8,
            "margin_end": 0.2,
            "rho_start": 0.05,
            "rho_end": 5.0,
            "qp_gate_start": False,
            "qp_gate_end": True,
            "qp_prob_start": 0.0,
            "qp_prob_end": 0.8,
        },
        "adaptive": {
            "scheduler_type": "adaptive_gate",
            "margin_start": 0.5,
            "margin_end": 0.1,
            "rho_start": 0.1,
            "rho_end": 10.0,
            "violation_threshold": 0.1,
            "ess_threshold": 0.5,
            "snr_threshold": 1.0,
        },
        "dual_anneal": {
            "scheduler_type": "dual_anneal",
            "target_feasible_rate_start": 0.5,
            "target_feasible_rate_end": 1.0,
            "margin_start": 0.5,
            "margin_end": 0.1,
            "rho_start": 0.1,
            "rho_end": 10.0,
        },
    }
    
    if preset_name not in presets:
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available presets: {list(presets.keys())}"
        )
    
    # Get preset config
    config = presets[preset_name].copy()
    
    # Override with kwargs
    config.update(kwargs)
    
    # Create scheduler
    scheduler_type = config.pop("scheduler_type")
    
    if scheduler_type == "cosine_anneal":
        return CosineAnnealScheduler(**config)
    elif scheduler_type == "dual_anneal":
        return DualAnnealScheduler(**config)
    elif scheduler_type == "adaptive_gate":
        return AdaptiveGateScheduler(**config)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# Convenience functions for common presets
def soft_to_hard_scheduler(**kwargs) -> CosineAnnealScheduler:
    """Create soft-to-hard scheduler (default preset)."""
    return create_preset_scheduler("soft_to_hard", **kwargs)


def aggressive_scheduler(**kwargs) -> CosineAnnealScheduler:
    """Create aggressive scheduler."""
    return create_preset_scheduler("aggressive", **kwargs)


def conservative_scheduler(**kwargs) -> CosineAnnealScheduler:
    """Create conservative scheduler."""
    return create_preset_scheduler("conservative", **kwargs)


def adaptive_scheduler(**kwargs) -> AdaptiveGateScheduler:
    """Create adaptive scheduler."""
    return create_preset_scheduler("adaptive", **kwargs)


def dual_anneal_scheduler(**kwargs) -> DualAnnealScheduler:
    """Create dual annealing scheduler."""
    return create_preset_scheduler("dual_anneal", **kwargs)

