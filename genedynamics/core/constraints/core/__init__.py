"""
Core infrastructure for high-performance constraint system.

This module provides:
- BackendArray: Unified array interface for zero-copy backend switching
- UnifiedRegistry: Multi-backend registry system
- HighPerformanceConstraintPipeline: JIT-compiled, batched constraint pipeline
- Types: Core data structures (ScheduleState, ScheduleParams, etc.)
- Cache: Intelligent caching system
- Stats: Statistical extraction utilities
"""

from .array_interface import BackendArray, BackendType, ensure_backend
from .registry import (
    UnifiedRegistry,
    get_registry,
    register,
)
from .types import (
    ScheduleState,
    ScheduleParams,
    ConvexConstraint,
    ConstraintStats,
    OperatorInfo,
)
from .cache import (
    CacheKey,
    ConstraintCache,
    ParamCache,
)
from .stats import (
    compute_violation,
    compute_min_sdf,
    compute_feasible_rate,
    compute_ess,
    compute_proj_displacement,
    extract_stats,
    extract_stats_batch,
    topK_selection,
    topL_selection,
)
from .pipeline import (
    PipelineConfig,
    HighPerformanceConstraintPipeline,
)
from .performance import (
    PerformanceProfiler,
    benchmark_pipeline,
    compare_backends,
    profile_memory,
)
from .utils import (
    topK_selection,
    topL_selection,
    select_active_constraints,
    select_active_time_steps,
    batch_trajectories,
    unbatch_trajectories,
    pad_trajectory,
    ensure_backend_array,
    stack_trajectories,
)

__all__ = [
    # Array interface
    "BackendArray",
    "BackendType",
    "ensure_backend",
    # Registry
    "UnifiedRegistry",
    "get_registry",
    "register",
    # Types
    "ScheduleState",
    "ScheduleParams",
    "ConvexConstraint",
    "ConstraintStats",
    "OperatorInfo",
    # Cache
    "CacheKey",
    "ConstraintCache",
    "ParamCache",
    # Stats
    "compute_violation",
    "compute_min_sdf",
    "compute_feasible_rate",
    "compute_ess",
    "compute_proj_displacement",
    "extract_stats",
    "extract_stats_batch",
    "topK_selection",
    "topL_selection",
    # Pipeline
    "PipelineConfig",
    "HighPerformanceConstraintPipeline",
    # Performance
    "PerformanceProfiler",
    "benchmark_pipeline",
    "compare_backends",
    "profile_memory",
    # Utils
    "topK_selection",
    "topL_selection",
    "select_active_constraints",
    "select_active_time_steps",
    "batch_trajectories",
    "unbatch_trajectories",
    "pad_trajectory",
    "ensure_backend_array",
    "stack_trajectories",
]

