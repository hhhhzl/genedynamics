"""
High-Performance Multi-Backend Constraint System.

Legacy constraint components have been removed. Use the new architecture:
  from genedynamics.core.constraints.core import HighPerformanceConstraintPipeline
"""

from genedynamics.core.constraints.core import (
    BackendArray,
    BackendType,
    UnifiedRegistry,
    get_registry,
    register,
    ScheduleState,
    ScheduleParams,
    ConvexConstraint,
    ConstraintStats,
    OperatorInfo,
    CacheKey,
    ConstraintCache,
    ParamCache,
    PipelineConfig,
    HighPerformanceConstraintPipeline,
)

__all__ = [
    "BackendArray",
    "BackendType",
    "UnifiedRegistry",
    "get_registry",
    "register",
    "ScheduleState",
    "ScheduleParams",
    "ConvexConstraint",
    "ConstraintStats",
    "OperatorInfo",
    "CacheKey",
    "ConstraintCache",
    "ParamCache",
    "PipelineConfig",
    "HighPerformanceConstraintPipeline",
]
