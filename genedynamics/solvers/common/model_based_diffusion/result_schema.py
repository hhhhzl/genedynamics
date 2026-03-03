"""
Lightweight result schema for model-based diffusion solvers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class DiffusionTrace(TypedDict, total=False):
    diffusion_actions_traj: Any
    diffusion_sampled_actions: Any


class ModelBasedDiffusionResult(TypedDict, total=False):
    # Required by experiment extraction
    states: Any
    actions: Any

    # Candidate/multirun outputs
    candidate_states: List[Any]
    candidate_actions: List[Any]
    candidate_costs: Any
    best_idx: int
    mode_strategy: str
    multirun_keys: Any
    multirun_diffusion_data: List[DiffusionTrace]

    # Optional traces/diagnostics
    reward_history: Any
    diffusion_actions_traj: Any
    diffusion_sampled_actions: Any


ResultDict = Dict[str, Any]
MaybeResultDict = Optional[ResultDict]

