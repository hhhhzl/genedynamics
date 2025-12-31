"""
Composite scheduler: Combines multiple constraint and diffusion schedulers.

Allows flexible combination of multiple schedulers with various merge strategies:
- Chain: Apply schedulers sequentially
- Merge: Combine parameters from all schedulers
- Weighted: Weighted average of parameters
- Priority: Use first non-None value
- Max: Maximum value across schedulers
- Min: Minimum value across schedulers
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Union
import numpy as np

from enerdynamics.core.constraints.schedulers.base import Scheduler
from enerdynamics.core.constraints.schedulers.ConstraintScheduler.base import ConstraintScheduler
from enerdynamics.core.constraints.schedulers.DiffusionScheduler.base import DiffusionScheduler
from enerdynamics.core.constraints.core.types import ScheduleState, ScheduleParams
from enerdynamics.core.constraints.core.registry import register


class MergeStrategy(Enum):
    """Merge strategy for combining parameters from multiple schedulers."""
    CHAIN = "chain"  # Apply schedulers sequentially
    MERGE = "merge"  # Combine all parameters (last wins for conflicts)
    WEIGHTED = "weighted"  # Weighted average
    PRIORITY = "priority"  # First non-None value
    MAX = "max"  # Maximum value
    MIN = "min"  # Minimum value


@register("scheduler", "composite", "numpy")
class CompositeScheduler(Scheduler):
    """
    Composite scheduler that combines multiple constraint and diffusion schedulers.
    
    This scheduler allows flexible combination of:
    - Multiple constraint schedulers (list)
    - Multiple diffusion schedulers (list)
    
    Each type of scheduler can use different merge strategies to combine
    parameters from multiple schedulers.
    
    Performance: O(N) where N is number of schedulers (typically small).
    """
    
    def __init__(
        self,
        constraint_schedulers: Optional[List[Union[Scheduler, ConstraintScheduler]]] = None,
        diffusion_schedulers: Optional[List[Union[Scheduler, DiffusionScheduler]]] = None,
        constraint_merge_strategy: MergeStrategy = MergeStrategy.MERGE,
        diffusion_merge_strategy: MergeStrategy = MergeStrategy.MERGE,
        constraint_weights: Optional[List[float]] = None,
        diffusion_weights: Optional[List[float]] = None,
        **kwargs
    ):
        """
        Initialize composite scheduler.
        
        Args:
            constraint_schedulers: List of constraint schedulers (can be empty)
            diffusion_schedulers: List of diffusion schedulers (can be empty)
            constraint_merge_strategy: How to merge constraint parameters
            diffusion_merge_strategy: How to merge diffusion parameters
            constraint_weights: Weights for weighted merge (normalized automatically)
            diffusion_weights: Weights for weighted merge (normalized automatically)
            **kwargs: Additional parameters
        """
        # Auto-adapt schedulers to appropriate interfaces
        self.constraint_schedulers = [
            self._adapt_constraint_scheduler(s) 
            for s in (constraint_schedulers or [])
        ]
        self.diffusion_schedulers = [
            self._adapt_diffusion_scheduler(s)
            for s in (diffusion_schedulers or [])
        ]
        
        # Store merge strategies
        self.constraint_merge_strategy = constraint_merge_strategy
        self.diffusion_merge_strategy = diffusion_merge_strategy
        
        # Normalize weights if provided
        if constraint_weights is not None:
            if len(constraint_weights) != len(self.constraint_schedulers):
                raise ValueError(
                    f"constraint_weights length ({len(constraint_weights)}) "
                    f"must match constraint_schedulers length ({len(self.constraint_schedulers)})"
                )
            weights_sum = sum(constraint_weights)
            if weights_sum > 0:
                self.constraint_weights = [w / weights_sum for w in constraint_weights]
            else:
                self.constraint_weights = [1.0 / len(constraint_weights)] * len(constraint_weights)
        else:
            self.constraint_weights = None
        
        if diffusion_weights is not None:
            if len(diffusion_weights) != len(self.diffusion_schedulers):
                raise ValueError(
                    f"diffusion_weights length ({len(diffusion_weights)}) "
                    f"must match diffusion_schedulers length ({len(self.diffusion_schedulers)})"
                )
            weights_sum = sum(diffusion_weights)
            if weights_sum > 0:
                self.diffusion_weights = [w / weights_sum for w in diffusion_weights]
            else:
                self.diffusion_weights = [1.0 / len(diffusion_weights)] * len(diffusion_weights)
        else:
            self.diffusion_weights = None
        
        self.kwargs = kwargs
    
    def _adapt_constraint_scheduler(self, scheduler: Scheduler) -> Any:
        """
        Adapt scheduler to ConstraintScheduler interface if needed.
        
        If scheduler has constraint_params() method, use it directly.
        Otherwise, try to extract constraint parameters from params().
        
        Args:
            scheduler: Scheduler instance
            
        Returns:
            Adapted scheduler (or original if already compatible)
        """
        # Already a ConstraintScheduler
        if isinstance(scheduler, ConstraintScheduler):
            return scheduler
        
        # Has constraint_params method
        if hasattr(scheduler, 'constraint_params'):
            return scheduler
        
        # Try to use as-is (will extract from params() in _get_constraint_params)
        return scheduler
    
    def _adapt_diffusion_scheduler(self, scheduler: Scheduler) -> Any:
        """
        Adapt scheduler to DiffusionScheduler interface if needed.
        
        If scheduler has diffusion_params() method, use it directly.
        Otherwise, try to extract diffusion parameters from params().
        
        Args:
            scheduler: Scheduler instance
            
        Returns:
            Adapted scheduler (or original if already compatible)
        """
        # Already a DiffusionScheduler
        if isinstance(scheduler, DiffusionScheduler):
            return scheduler
        
        # Has diffusion_params method
        if hasattr(scheduler, 'diffusion_params'):
            return scheduler
        
        # Try to use as-is (will extract from params() in _get_diffusion_params)
        return scheduler
    
    def _get_constraint_params(self, scheduler: Any, state: ScheduleState) -> Dict[str, Any]:
        """
        Extract constraint parameters from a scheduler.
        
        Args:
            scheduler: Scheduler instance
            state: Current schedule state
            
        Returns:
            Dictionary with constraint parameters
        """
        # Try constraint_params method
        if hasattr(scheduler, 'constraint_params'):
            return scheduler.constraint_params(state)
        
        # Fallback: extract from params()
        params = scheduler.params(state)
        return {
            "rho": params.rho,
            "topK": params.topK,
            "topL": params.topL,
            "eps": params._extra.get("eps", 1e-4),
            "I_QP": params._extra.get("I_QP", 10),
            "qp_gate": params.qp_gate,
            "qp_prob": params.qp_prob,
            "margin": params.margin,
            "_extra": params._extra,
        }
    
    def _get_diffusion_params(self, scheduler: Any, state: ScheduleState) -> Dict[str, Any]:
        """
        Extract diffusion parameters from a scheduler.
        
        Args:
            scheduler: Scheduler instance
            state: Current schedule state
            
        Returns:
            Dictionary with diffusion parameters
        """
        # Try diffusion_params method
        if hasattr(scheduler, 'diffusion_params'):
            return scheduler.diffusion_params(state)
        
        # Fallback: extract from params()
        params = scheduler.params(state)
        return {
            "M_k": params._extra.get("M_k"),
            "T_k": params._extra.get("T_k"),
            "s_k": params._extra.get("s_k"),
            **{k: v for k, v in params._extra.items() 
               if k not in ["M_k", "T_k", "s_k", "eps", "I_QP"]},
        }
    
    def params(self, state: ScheduleState) -> ScheduleParams:
        """
        Generate combined schedule parameters.
        
        Combines parameters from all constraint and diffusion schedulers
        using the specified merge strategies.
        
        Args:
            state: Current schedule state
            
        Returns:
            Combined ScheduleParams
        """
        # Get constraint parameters
        constraint_params = self._merge_constraint_params(state)
        
        # Get diffusion parameters
        diffusion_params = self._merge_diffusion_params(state)
        
        # Combine into ScheduleParams
        return ScheduleParams(
            margin=constraint_params.get("margin", 0.0),
            rho=constraint_params.get("rho", 1.0),
            topK=constraint_params.get("topK"),
            topL=constraint_params.get("topL"),
            qp_gate=constraint_params.get("qp_gate", True),
            qp_prob=constraint_params.get("qp_prob", 1.0),
            _extra={
                "eps": constraint_params.get("eps", 1e-4),
                "I_QP": constraint_params.get("I_QP", 10),
                "M_k": diffusion_params.get("M_k"),
                "T_k": diffusion_params.get("T_k"),
                "s_k": diffusion_params.get("s_k"),
                **constraint_params.get("_extra", {}),
                **{k: v for k, v in diffusion_params.items() 
                   if k not in ["M_k", "T_k", "s_k"]},
            }
        )
    
    def _merge_constraint_params(self, state: ScheduleState) -> Dict[str, Any]:
        """
        Merge constraint parameters from all constraint schedulers.
        
        Args:
            state: Current schedule state
            
        Returns:
            Merged constraint parameters dictionary
        """
        if not self.constraint_schedulers:
            # Default values if no constraint schedulers
            return {
                "rho": 1.0,
                "topK": None,
                "topL": None,
                "eps": 1e-4,
                "I_QP": 10,
                "qp_gate": True,
                "qp_prob": 1.0,
                "margin": 0.0,
                "_extra": {},
            }
        
        # Get parameters from all schedulers
        all_params = [
            self._get_constraint_params(sched, state)
            for sched in self.constraint_schedulers
        ]
        
        # Merge based on strategy
        if self.constraint_merge_strategy == MergeStrategy.CHAIN:
            return self._chain_merge(all_params)
        elif self.constraint_merge_strategy == MergeStrategy.MERGE:
            return self._merge_merge(all_params)
        elif self.constraint_merge_strategy == MergeStrategy.WEIGHTED:
            return self._weighted_merge(all_params, self.constraint_weights)
        elif self.constraint_merge_strategy == MergeStrategy.PRIORITY:
            return self._priority_merge(all_params)
        elif self.constraint_merge_strategy == MergeStrategy.MAX:
            return self._max_merge(all_params)
        elif self.constraint_merge_strategy == MergeStrategy.MIN:
            return self._min_merge(all_params)
        else:
            raise ValueError(f"Unknown merge strategy: {self.constraint_merge_strategy}")
    
    def _merge_diffusion_params(self, state: ScheduleState) -> Dict[str, Any]:
        """
        Merge diffusion parameters from all diffusion schedulers.
        
        Args:
            state: Current schedule state
            
        Returns:
            Merged diffusion parameters dictionary
        """
        if not self.diffusion_schedulers:
            # Default values if no diffusion schedulers
            return {
                "M_k": 64,
                "T_k": 0.5,
                "s_k": None,
            }
        
        # Get parameters from all schedulers
        all_params = [
            self._get_diffusion_params(sched, state)
            for sched in self.diffusion_schedulers
        ]
        
        # Merge based on strategy
        if self.diffusion_merge_strategy == MergeStrategy.CHAIN:
            return self._chain_merge(all_params)
        elif self.diffusion_merge_strategy == MergeStrategy.MERGE:
            return self._merge_merge(all_params)
        elif self.diffusion_merge_strategy == MergeStrategy.WEIGHTED:
            return self._weighted_merge(all_params, self.diffusion_weights)
        elif self.diffusion_merge_strategy == MergeStrategy.PRIORITY:
            return self._priority_merge(all_params)
        elif self.diffusion_merge_strategy == MergeStrategy.MAX:
            return self._max_merge(all_params)
        elif self.diffusion_merge_strategy == MergeStrategy.MIN:
            return self._min_merge(all_params)
        else:
            raise ValueError(f"Unknown merge strategy: {self.diffusion_merge_strategy}")
    
    def _chain_merge(self, all_params: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Chain merge: Apply schedulers sequentially (last wins).
        
        Args:
            all_params: List of parameter dictionaries
            
        Returns:
            Merged parameters (last non-None value wins)
        """
        result = {}
        for params in all_params:
            for key, value in params.items():
                if value is not None or key not in result:
                    result[key] = value
        return result
    
    def _merge_merge(self, all_params: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge merge: Combine all parameters (last wins for conflicts).
        
        Args:
            all_params: List of parameter dictionaries
            
        Returns:
            Merged parameters (last value wins)
        """
        result = {}
        for params in all_params:
            result.update(params)
        return result
    
    def _weighted_merge(
        self,
        all_params: List[Dict[str, Any]],
        weights: Optional[List[float]]
    ) -> Dict[str, Any]:
        """
        Weighted merge: Weighted average of numeric parameters.
        
        Args:
            all_params: List of parameter dictionaries
            weights: Normalized weights (must match length of all_params)
            
        Returns:
            Weighted average parameters
        """
        if weights is None:
            weights = [1.0 / len(all_params)] * len(all_params)
        
        result = {}
        
        # Collect all keys
        all_keys = set()
        for params in all_params:
            all_keys.update(params.keys())
        
        # Weighted average for numeric values, last non-None for others
        for key in all_keys:
            numeric_values = []
            numeric_weights = []
            
            for i, params in enumerate(all_params):
                value = params.get(key)
                if value is not None:
                    if isinstance(value, (int, float, np.number)):
                        numeric_values.append(float(value))
                        numeric_weights.append(weights[i])
                    else:
                        # Non-numeric: use last value
                        result[key] = value
            
            if numeric_values:
                # Weighted average
                result[key] = sum(v * w for v, w in zip(numeric_values, numeric_weights))
        
        return result
    
    def _priority_merge(self, all_params: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Priority merge: First non-None value wins.
        
        Args:
            all_params: List of parameter dictionaries
            
        Returns:
            Merged parameters (first non-None value wins)
        """
        result = {}
        for params in all_params:
            for key, value in params.items():
                if key not in result and value is not None:
                    result[key] = value
        return result
    
    def _max_merge(self, all_params: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Max merge: Maximum value across schedulers.
        
        Args:
            all_params: List of parameter dictionaries
            
        Returns:
            Merged parameters (max value for numeric, last for others)
        """
        result = {}
        all_keys = set()
        for params in all_params:
            all_keys.update(params.keys())
        
        for key in all_keys:
            numeric_values = []
            non_numeric_values = []
            
            for params in all_params:
                value = params.get(key)
                if value is not None:
                    if isinstance(value, (int, float, np.number)):
                        numeric_values.append(float(value))
                    else:
                        non_numeric_values.append(value)
            
            if numeric_values:
                result[key] = max(numeric_values)
            elif non_numeric_values:
                result[key] = non_numeric_values[-1]  # Last non-numeric
        
        return result
    
    def _min_merge(self, all_params: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Min merge: Minimum value across schedulers.
        
        Args:
            all_params: List of parameter dictionaries
            
        Returns:
            Merged parameters (min value for numeric, first for others)
        """
        result = {}
        all_keys = set()
        for params in all_params:
            all_keys.update(params.keys())
        
        for key in all_keys:
            numeric_values = []
            non_numeric_values = []
            
            for params in all_params:
                value = params.get(key)
                if value is not None:
                    if isinstance(value, (int, float, np.number)):
                        numeric_values.append(float(value))
                    else:
                        non_numeric_values.append(value)
            
            if numeric_values:
                result[key] = min(numeric_values)
            elif non_numeric_values:
                result[key] = non_numeric_values[0]  # First non-numeric
        
        return result
    
    def update(self, state: ScheduleState, feedback: Dict[str, Any]) -> None:
        """
        Update all schedulers with feedback.
        
        Args:
            state: Current schedule state
            feedback: Feedback dictionary
        """
        # Update constraint schedulers
        for sched in self.constraint_schedulers:
            if hasattr(sched, 'update'):
                sched.update(state, feedback)
        
        # Update diffusion schedulers
        for sched in self.diffusion_schedulers:
            if hasattr(sched, 'update'):
                sched.update(state, feedback)
    
    def reset(self) -> None:
        """Reset all schedulers (for new optimization run)."""
        for sched in self.constraint_schedulers:
            if hasattr(sched, 'reset'):
                sched.reset()
        
        for sched in self.diffusion_schedulers:
            if hasattr(sched, 'reset'):
                sched.reset()

