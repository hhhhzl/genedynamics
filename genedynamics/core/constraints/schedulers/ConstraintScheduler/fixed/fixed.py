"""
Fixed constraint scheduler.

All constraint parameters are constant regardless of diffusion step.
Useful for ablation studies, deterministic behavior, and simple baselines.
"""

from typing import Optional, Dict, Any
import numpy as np

from genedynamics.core.constraints.schedulers.ConstraintScheduler.base import ConstraintScheduler
from genedynamics.core.constraints.core.types import ScheduleState
from genedynamics.core.constraints.core.registry import get_registry


class FixedConstraintScheduler(ConstraintScheduler):
    """
    Fixed constraint scheduler.
    
    All constraint parameters are constant regardless of diffusion step.
    This scheduler provides deterministic behavior and is useful for:
    - Ablation studies
    - Baseline comparisons
    - Simple fixed-schedule experiments
    - Testing and debugging
    
    Performance: O(1) - constant time parameter generation.
    """
    
    def __init__(
        self,
        rho: float = 10.0,
        topK: Optional[int] = None,
        eps: float = 1e-4,
        I_QP: int = 10,
        qp_gate: bool = True,
        qp_prob: float = 1.0,
        margin: float = 0.0,
        backend: str = "numpy",
        _skip_backend_lookup: bool = False,
        **kwargs
    ):
        """
        Initialize fixed constraint scheduler.
        
        Args:
            rho: Fixed slack penalty weight
            topK: Fixed top-K constraints (None = no limit, use all constraints)
            eps: Fixed solver tolerance
            I_QP: Fixed maximum QP iterations
            qp_gate: Whether to always apply QP (True = always, False = never)
            qp_prob: QP invocation probability (1.0 = always, 0.0 = never)
            margin: Fixed safety margin for constraints
            backend: Backend to use ("numpy", "jax") - for future JAX support
            _skip_backend_lookup: Internal flag to prevent recursion
            **kwargs: Additional parameters stored in _extra
        """
        # Store parameters as attributes for fast access
        self.rho = float(rho)
        self.topK = topK if topK is None else int(topK)
        self.eps = float(eps)
        self.I_QP = int(I_QP)
        self.qp_gate = bool(qp_gate)
        self.qp_prob = float(qp_prob)
        self.margin = float(margin)
        self.extra = kwargs
        self.backend = backend
        
        # Pre-compute parameter dict for fast access (performance optimization)
        self._cached_params = {
            "rho": self.rho,
            "topK": self.topK,
            "eps": self.eps,
            "I_QP": self.I_QP,
            "qp_gate": self.qp_gate,
            "qp_prob": self.qp_prob,
            "margin": self.margin,
            "_extra": self.extra,
        }
        
        # Get backend implementation from registry (for future JAX support)
        if not _skip_backend_lookup:
            registry = get_registry()
            impl_class = registry.get("scheduler", "fixed_constraint", backend)
            if impl_class is not None and impl_class != FixedConstraintScheduler:
                self._backend_impl = impl_class(
                    rho=rho, topK=topK, eps=eps, I_QP=I_QP,
                    qp_gate=qp_gate, qp_prob=qp_prob, margin=margin,
                    backend=backend, _skip_backend_lookup=True, **kwargs
                )
            else:
                self._backend_impl = None
        else:
            self._backend_impl = None
    
    def constraint_params(self, state: ScheduleState) -> Dict[str, Any]:
        """
        Generate fixed constraint parameters.
        
        This method returns constant parameters regardless of state,
        optimized for performance by using pre-computed cache.
        
        Args:
            state: Current schedule state (ignored for fixed scheduler)
            
        Returns:
            Dictionary with fixed constraint parameters
        """
        # Use backend implementation if available
        if self._backend_impl is not None:
            return self._backend_impl.constraint_params(state)
        
        # Return pre-computed parameters (O(1) performance)
        return self._cached_params.copy()

