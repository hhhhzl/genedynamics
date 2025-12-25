"""
Type definitions for the constraint system.

This module provides all data structures used throughout the constraint system:
- ScheduleState: State for constraint scheduling
- ScheduleParams: Scheduled parameters (output of scheduler)
- ConvexConstraint: Convex constraint representation
- ConstraintStats: Statistics extracted from rollouts

These types are used across all constraint system modules for type safety
and clear interfaces.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class ScheduleState:
    """
    State for constraint scheduling.
    
    This represents the current state in the diffusion/optimization process,
    used by schedulers to determine appropriate parameters.
    
    Attributes:
        k: Current diffusion step (0 = final, K = initial noise)
        K: Total diffusion steps
        t: Current time step in trajectory (optional)
        H: Horizon length (optional)
    """
    k: int  # Current diffusion step
    K: int  # Total diffusion steps
    t: Optional[int] = None  # Current time step (optional)
    H: Optional[int] = None  # Horizon length (optional)
    
    def __post_init__(self):
        """Validate state."""
        if self.k < 0:
            raise ValueError(f"k must be non-negative, got {self.k}")
        if self.K <= 0:
            raise ValueError(f"K must be positive, got {self.K}")
        if self.k > self.K:
            raise ValueError(f"k ({self.k}) must be <= K ({self.K})")
    
    @property
    def progress(self) -> float:
        """Get progress ratio (0 = final, 1 = initial)."""
        if self.K == 0:
            return 0.0
        return self.k / self.K
    
    def __hash__(self) -> int:
        """Hash for use in dictionaries/caches."""
        return hash((self.k, self.K, self.t, self.H))


@dataclass
class ScheduleParams:
    """
    Scheduled parameters (output of scheduler).
    
    These parameters control constraint enforcement behavior and are
    typically computed by schedulers based on ScheduleState.
    
    Attributes:
        margin: Safety margin for constraints (e.g., obstacle clearance)
        rho: Slack penalty weight for slack-QP
        topK: Top-K active constraints to consider
        topL: Top-L time steps to repair
        qp_gate: Whether to apply QP at this step
        qp_prob: Probability of applying QP (for stochastic gating)
        _extra: Additional parameters (extensible)
    """
    margin: float = 0.0
    rho: float = 1.0
    topK: Optional[int] = None
    topL: Optional[int] = None
    qp_gate: bool = True
    qp_prob: float = 1.0
    _extra: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get parameter value (supports both direct and extra params)."""
        if hasattr(self, key) and not key.startswith('_'):
            return getattr(self, key)
        return self._extra.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set parameter value (supports both direct and extra params)."""
        if hasattr(self, key) and not key.startswith('_'):
            setattr(self, key, value)
        else:
            self._extra[key] = value
    
    def __hash__(self) -> int:
        """Hash for use in dictionaries/caches."""
        return hash((
            self.margin,
            self.rho,
            self.topK,
            self.topL,
            self.qp_gate,
            self.qp_prob,
            tuple(sorted(self._extra.items()))
        ))


@dataclass
class ConvexConstraint:
    """
    Convex constraint representation (A x >= b).
    
    This is the unified representation for all convex constraints used
    throughout the constraint system. It can represent:
    - Linear inequalities: A x >= b
    - Box constraints: l <= x <= u (encoded as A x >= b)
    - Polytope constraints: Multiple linear inequalities
    
    Attributes:
        A: Constraint matrix (backend array: numpy, JAX, PyTorch, etc.)
        b: Constraint vector (backend array)
        meta: Additional metadata (constraint type, per-step flag, etc.)
    """
    A: Any  # Constraint matrix (backend array)
    b: Any  # Constraint vector (backend array)
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def is_per_step(self) -> bool:
        """Check if this is a per-step constraint."""
        return self.meta.get("per_step", False)
    
    def get_constraint_type(self) -> str:
        """Get constraint type from metadata."""
        return self.meta.get("type", "linear")
    
    def __repr__(self) -> str:
        """String representation."""
        A_shape = getattr(self.A, 'shape', 'unknown')
        b_shape = getattr(self.b, 'shape', 'unknown')
        return f"ConvexConstraint(A.shape={A_shape}, b.shape={b_shape}, meta={self.meta})"


@dataclass
class ConstraintStats:
    """
    Statistics extracted from constraint evaluation/rollouts.
    
    These statistics are used by schedulers for adaptive parameter tuning
    and by the pipeline for performance monitoring.
    
    Attributes:
        violation: Maximum constraint violation
        min_sdf: Minimum signed distance function value
        feasible_rate: Fraction of feasible trajectories
        ess: Effective sample size (for importance sampling)
        proj_displacement: Average projection displacement
        qp_time: Time spent in QP solving
        qp_count: Number of QP solves
    """
    violation: float = 0.0
    min_sdf: float = float('inf')
    feasible_rate: float = 1.0
    ess: Optional[float] = None
    proj_displacement: float = 0.0
    qp_time: float = 0.0
    qp_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "violation": self.violation,
            "min_sdf": self.min_sdf,
            "feasible_rate": self.feasible_rate,
            "ess": self.ess,
            "proj_displacement": self.proj_displacement,
            "qp_time": self.qp_time,
            "qp_count": self.qp_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstraintStats":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class OperatorInfo:
    """
    Information returned by operators after constraint enforcement.
    
    This provides feedback about the operator's execution, useful for:
    - Performance monitoring
    - Adaptive scheduling
    - Debugging
    
    Attributes:
        success: Whether operator succeeded
        violation_before: Constraint violation before operator
        violation_after: Constraint violation after operator
        iterations: Number of iterations (for iterative operators)
        time: Execution time
        extra: Additional operator-specific info
    """
    success: bool = True
    violation_before: float = 0.0
    violation_after: float = 0.0
    iterations: int = 0
    time: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "violation_before": self.violation_before,
            "violation_after": self.violation_after,
            "iterations": self.iterations,
            "time": self.time,
            "extra": self.extra,
        }

