"""
Base classes for CFS projection backend implementations.

This module defines the interface that all backend-specific CFS projection
implementations must follow. Each backend (NumPy, JAX, PyTorch) implements
this interface to provide optimized projection functionality.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Optional, Tuple, List
import numpy as np

from genedynamics.envs.obstacles.base import ObstacleManager


class CFSProjectionBackend(Protocol):
    """
    Protocol for CFS projection backend implementations.
    
    This defines the interface that all backend implementations must provide.
    Backend-specific classes implement these methods using their native
    operations (NumPy, JAX, PyTorch, etc.).
    """
    
    @abstractmethod
    def project_batch(
        self,
        positions: np.ndarray,
        clearance: float,
        step: Optional[int],
        obstacles: ObstacleManager,
        max_iterations: int,
        convergence_tol: float,
        max_constraints_per_point: int,
        constraint_margin: float,
        use_trajectory_qp: bool,
        smoothness_weight: float,
    ) -> np.ndarray:
        """
        Batch project multiple points onto CFS using linearized constraints and QP.
        
        This is the core projection operation that all backends must implement.
        The implementation should use backend-specific optimizations (e.g., JIT
        compilation for JAX, GPU acceleration, etc.).
        
        Args:
            positions: Points to project, shape (N, dim)
            clearance: Minimum clearance required
            step: Current step (for logging/debugging)
            obstacles: ObstacleManager containing obstacles
            max_iterations: Maximum iterations for iterative projection
            convergence_tol: Convergence tolerance
            max_constraints_per_point: Maximum constraints per point
            constraint_margin: Margin for constraint selection
            use_trajectory_qp: Whether to use trajectory-level QP
            smoothness_weight: Smoothness regularization weight for trajectory QP
            
        Returns:
            Projected positions, shape (N, dim)
        """
        ...
    
    @abstractmethod
    def solve_projection_qp(
        self,
        x0: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """
        Solve pointwise projection QP: min 0.5||x - x0||^2 s.t. A x >= b.
        
        This is used for pointwise projection when trajectory QP is not used
        or not available. Different backends may use different QP solvers.
        
        Args:
            x0: Reference point, shape (dim,)
            A: Constraint matrix, shape (m, dim)
            b: Constraint vector, shape (m,)
            
        Returns:
            Projected point, shape (dim,)
        """
        ...


class CFSProjectionBase(ABC):
    """
    Base class for CFS projection backend implementations.
    
    This provides common functionality shared by all backend implementations,
    while allowing backend-specific optimizations in the abstract methods.
    
    Subclasses should implement:
    - project_batch(): Main batch projection method
    - solve_projection_qp(): Pointwise QP solver
    """
    
    def __init__(self, obstacles: ObstacleManager, **config):
        """
        Initialize backend implementation.
        
        Args:
            obstacles: ObstacleManager containing obstacles
            **config: Backend-specific configuration
        """
        self.obstacles = obstacles
        self.config = config
    
    @abstractmethod
    def project_batch(
        self,
        positions: np.ndarray,
        clearance: float,
        step: Optional[int],
        **kwargs
    ) -> np.ndarray:
        """
        Batch project multiple points onto CFS.
        
        Args:
            positions: Points to project, shape (N, dim)
            clearance: Minimum clearance required
            step: Current step (optional)
            **kwargs: Additional parameters (max_iterations, convergence_tol, etc.)
            
        Returns:
            Projected positions, shape (N, dim)
        """
        pass
    
    @abstractmethod
    def solve_projection_qp(
        self,
        x0: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """
        Solve pointwise projection QP.
        
        Args:
            x0: Reference point, shape (dim,)
            A: Constraint matrix, shape (m, dim)
            b: Constraint vector, shape (m,)
            
        Returns:
            Projected point, shape (dim,)
        """
        pass
    
    def _finite_difference_gradient(
        self,
        obstacle,
        x: np.ndarray,
        eps: float = 1e-4
    ) -> np.ndarray:
        """
        Numerically approximate ∇sdf(x) with central differences.
        
        This is a fallback method when obstacle doesn't provide a gradient.
        Backend implementations may override this to use backend-specific
        operations (e.g., JAX automatic differentiation).
        
        Args:
            obstacle: Obstacle object with sdf() method
            x: Point to evaluate, shape (dim,)
            eps: Step size for finite differences
            
        Returns:
            Gradient, shape (dim,)
        """
        x = np.asarray(x, dtype=np.float32).flatten()
        dim = x.shape[0]
        grad = np.zeros((dim,), dtype=np.float32)
        for k in range(dim):
            xp = x.copy()
            xm = x.copy()
            xp[k] += eps
            xm[k] -= eps
            dp = obstacle.sdf(xp)
            dm = obstacle.sdf(xm)
            dp = float(np.asarray(dp).item() if hasattr(dp, "item") else dp)
            dm = float(np.asarray(dm).item() if hasattr(dm, "item") else dm)
            grad[k] = (dp - dm) / (2.0 * eps)
        return grad
    
    @staticmethod
    def _is_feasible(A: np.ndarray, b: np.ndarray, x: np.ndarray, tol: float = 1e-7) -> bool:
        """
        Check if point satisfies constraints: A x >= b within tolerance.
        
        Args:
            A: Constraint matrix, shape (m, dim)
            b: Constraint vector, shape (m,)
            x: Point to check, shape (dim,)
            tol: Tolerance
            
        Returns:
            True if all constraints satisfied
        """
        if A.size == 0:
            return True
        lhs = A @ x
        return bool(np.all(lhs + tol >= b))

