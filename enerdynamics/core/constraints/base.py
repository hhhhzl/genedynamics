import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union, Tuple, TYPE_CHECKING, Any
from enerdynamics.core.types import Trajectory, State, Action

if TYPE_CHECKING:
    from enerdynamics.core.constraints.schedule import ConstraintScheduleManager

Array = np.ndarray


def project_box(x: Array, lower: Array, upper: Array) -> Array:
    """project to box constraint [lower, upper]"""
    return np.minimum(np.maximum(x, lower), upper)


def soft_box_energy(x: Array, lower: Array, upper: Array, alpha: float = 10.0) -> float:
    """soft energy for box constraint: linear/square penalty if out of bounds"""
    below = np.maximum(0.0, lower - x)
    above = np.maximum(0.0, x - upper)
    return float(alpha * (np.sum(below ** 2) + np.sum(above ** 2)))


# ============================================================================
# Soft and Hard Constraint Abstractions
# ============================================================================

class SoftConstraint(ABC):
    """
    Soft constraint: differentiable penalty that shapes preferences.
    
    Contributes to E_soft(τ) = (1/λ)J(τ) + S(τ), where S(τ) aggregates
    all soft constraint penalties. Should be differentiable for gradient-based optimization.
    
    Usage modes:
    1. Standalone: Only soft constraints (no hard enforcement)
    2. Combined: Soft + hard constraints (soft guides, hard enforces)
    3. With schedule: Soft constraints that can be gradually replaced by hard constraints
    """
    
    @abstractmethod
    def evaluate(self, trajectory: Trajectory) -> float:
        """
        Evaluate soft constraint penalty.
        
        Args:
            trajectory: Trajectory to evaluate
            
        Returns:
            Penalty value (non-negative, 0 if constraint satisfied)
        """
        pass
    
    def evaluate_batch(self, trajectories: List[Trajectory]) -> np.ndarray:
        """
        Evaluate soft constraint for multiple trajectories (optional optimization).
        
        Default implementation calls evaluate() in a loop. Override for batch processing.
        
        Args:
            trajectories: List of trajectories
            
        Returns:
            Array of penalty values
        """
        return np.array([self.evaluate(traj) for traj in trajectories])


class HardConstraint(ABC):
    """
    Hard constraint: must be strictly satisfied.
    
    Defined by feasible set F = {τ: c_r(τ) ≤ 0, ∀r}. Hard constraints can:
    1. Be checked (is_feasible) for feasibility
    2. Be projected (project) onto feasible set
    3. Optionally use a schedule (step-dependent relaxation)
    
    Usage modes:
    1. Standalone: Only hard constraints (strict enforcement throughout)
    2. Combined: Hard + soft constraints (soft guides, hard enforces)
    3. With schedule: Gradually tighten hard constraints (soft → hard transition)
    """
    
    @abstractmethod
    def is_feasible(self, trajectory: Trajectory) -> bool:
        """
        Check if trajectory satisfies constraint.
        
        Args:
            trajectory: Trajectory to check
            
        Returns:
            True if feasible, False otherwise
        """
        pass
    
    @abstractmethod
    def violations(self, trajectory: Trajectory) -> np.ndarray:
        """
        Return constraint violations.
        
        For constraint c_r(τ) ≤ 0, returns max(0, c_r(τ)) for each constraint.
        
        Args:
            trajectory: Trajectory to evaluate
            
        Returns:
            Array of violations (non-negative, 0 if satisfied)
        """
        pass
    
    def project(
        self, 
        trajectory: Trajectory, 
        step: Optional[int] = None,
        total_steps: Optional[int] = None
    ) -> Trajectory:
        """
        Project trajectory onto feasible set.
        
        This is the feasibility operator Π_k in the theory. Can be called with
        step information for scheduled relaxation (F_K ⊇ F_{K-1} ⊇ ... ⊇ F_0).
        
        Args:
            trajectory: Trajectory to project
            step: Current diffusion step (optional, for scheduling)
            total_steps: Total diffusion steps (optional, for scheduling)
            
        Returns:
            Projected trajectory (feasible)
        """
        # Default: no-op (subclasses should override)
        return trajectory
    
    def is_feasible_batch(self, trajectories: List[Trajectory]) -> np.ndarray:
        """
        Check feasibility for multiple trajectories.
        
        Default implementation calls is_feasible() in a loop.
        
        Args:
            trajectories: List of trajectories
            
        Returns:
            Boolean array indicating feasibility
        """
        return np.array([self.is_feasible(traj) for traj in trajectories])


class FeasibilityOperator(ABC):
    """
    Plug-and-play feasibility operator for hard constraint projection.
    
    This implements Π_k: R^d → F_k in the EDOC theory. Can be:
    1. Step-independent: Same projection at all steps
    2. Step-dependent: Relaxed early, strict late (soft → hard schedule)
    
    A FeasibilityOperator typically wraps one or more HardConstraints and
    implements efficient projection (e.g., CFS-QP for obstacles).
    """
    
    @abstractmethod
    def project(
        self, 
        trajectory: Trajectory,
        step: Optional[int] = None,
        total_steps: Optional[int] = None
    ) -> Trajectory:
        """
        Project trajectory onto feasible set F_k.
        
        Args:
            trajectory: Trajectory to project
            step: Current diffusion step k (0 = final, K = initial noise)
            total_steps: Total diffusion steps K
            
        Returns:
            Projected trajectory ∈ F_k
        """
        pass
    
    def should_apply(self, step: Optional[int] = None, total_steps: Optional[int] = None) -> bool:
        """
        Check if projection should be applied at current step.
        
        Can be used to implement late-stage hard enforcement (only project
        near convergence). Default: always apply.
        
        Args:
            step: Current diffusion step
            total_steps: Total diffusion steps
            
        Returns:
            True if projection should be applied
        """
        return True


class ActionFilterOperator(ABC):
    """
    Hard constraint operator that enforces feasibility by filtering actions during rollout.

    This enables CBF-style safety filters (or other action-space projections) to be integrated
    into the same soft→hard scheduling architecture, without forcing state-space projection.
    """

    @abstractmethod
    def filter_actions_numpy(
        self,
        x0: State,
        actions: np.ndarray,
        *,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> np.ndarray:
        """Filter a single action sequence (numpy)."""
        raise NotImplementedError

    def should_apply(self, step: Optional[int] = None, total_steps: Optional[int] = None) -> bool:
        """Optional late-stage gating (default: always apply when hard is active)."""
        return True

    # Optional accelerators (used by solvers to build compiled rollouts)
    def make_jax_filter(self):
        """Return a JAX-callable (state, action, hard_clearance)->action_safe or None."""
        return None

    def make_torch_filter(self):
        """Return a torch-callable (state, action, hard_clearance)->action_safe or None."""
        return None


class ConstraintManager:
    """
    Manager for combining soft and hard constraints flexibly.
    
    Supports all combinations:
    1. Soft only: soft_constraints only
    2. Hard only: hard_constraints + feasibility_operator
    3. Soft + Hard: both, with optional scheduling
    4. Scheduled transitions: soft → hard, hard → soft, etc.
    """
    
    def __init__(
        self,
        soft_constraints: Optional[List[SoftConstraint]] = None,
        hard_constraints: Optional[List[HardConstraint]] = None,
        feasibility_operator: Optional[FeasibilityOperator] = None,
        action_filter_operator: Optional[ActionFilterOperator] = None,
        schedule_manager: Optional["ConstraintScheduleManager"] = None,
    ):
        """
        Initialize constraint manager.
        
        Args:
            soft_constraints: List of soft constraints (optional)
            hard_constraints: List of hard constraints (optional)
            feasibility_operator: Feasibility operator for projection (optional)
            schedule_manager: ConstraintScheduleManager for scheduling (optional)
        """
        self.soft_constraints = soft_constraints or []
        self.hard_constraints = hard_constraints or []
        self.feasibility_operator = feasibility_operator
        self.action_filter_operator = action_filter_operator
        self.schedule_manager = schedule_manager
        
        # Link schedule_manager to constraints that support it
        if self.schedule_manager is not None:
            for constraint in self.soft_constraints:
                if hasattr(constraint, 'schedule_manager'):
                    constraint.schedule_manager = self.schedule_manager
            for constraint in self.hard_constraints:
                if hasattr(constraint, 'schedule_manager'):
                    constraint.schedule_manager = self.schedule_manager
            if self.feasibility_operator is not None and hasattr(self.feasibility_operator, 'schedule_manager'):
                self.feasibility_operator.schedule_manager = self.schedule_manager
            if self.action_filter_operator is not None and hasattr(self.action_filter_operator, 'schedule_manager'):
                self.action_filter_operator.schedule_manager = self.schedule_manager
        
        # Validate: if hard constraints exist, should have projection mechanism (either state projection or action filter)
        if self.hard_constraints and not self.feasibility_operator and self.action_filter_operator is None:
            # Try to create default operator from hard constraints
            if len(self.hard_constraints) == 1:
                # Single constraint: use its project method directly
                pass  # Will use individual project methods
            else:
                # Multiple constraints: recommend using a FeasibilityOperator
                import warnings
                warnings.warn(
                    "Multiple hard constraints without FeasibilityOperator. "
                    "Projection may not handle all constraints correctly."
                )

    def filter_actions(
        self,
        x0: State,
        actions: np.ndarray,
        *,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        Filter an action sequence with the configured ActionFilterOperator (if any).

        This is intended for CBF-style hard constraints applied during rollout.
        Scheduling is handled consistently with other hard constraints.
        """
        if self.action_filter_operator is None:
            return np.asarray(actions, dtype=np.float32)

        if self.schedule_manager is not None:
            if not self.schedule_manager.is_hard_active(step, total_steps):
                return np.asarray(actions, dtype=np.float32)

        if not self.action_filter_operator.should_apply(step, total_steps):
            return np.asarray(actions, dtype=np.float32)

        return self.action_filter_operator.filter_actions_numpy(
            x0, np.asarray(actions, dtype=np.float32), step=step, total_steps=total_steps
        )

    def get_jax_action_filter(self):
        """Return the action filter callable for JAX, if provided by the operator."""
        if self.action_filter_operator is None:
            return None
        return self.action_filter_operator.make_jax_filter()

    def get_torch_action_filter(self):
        """Return the action filter callable for Torch, if provided by the operator."""
        if self.action_filter_operator is None:
            return None
        return self.action_filter_operator.make_torch_filter()
    
    def compute_soft_energy(
        self, 
        trajectory: Trajectory,
        step: Optional[int] = None,
        total_steps: Optional[int] = None
    ) -> float:
        """
        Compute total soft constraint penalty S(τ).
        
        Args:
            trajectory: Trajectory to evaluate
            step: Current step (optional, for scheduling)
            total_steps: Total steps (optional, for scheduling)
            
        Returns:
            Total soft penalty
        """
        # Check if soft constraints are active
        if self.schedule_manager is not None:
            if not self.schedule_manager.is_soft_active(step, total_steps):
                return 0.0
        
        total = 0.0
        for constraint in self.soft_constraints:
            # Call evaluate with step info if constraint supports it
            # Check if constraint has schedule_manager or accepts step/total_steps
            if hasattr(constraint, 'schedule_manager') and constraint.schedule_manager is not None:
                # Constraint has its own schedule_manager, let it handle scheduling
                energy = constraint.evaluate(trajectory, step=step, total_steps=total_steps)
            elif hasattr(constraint, 'evaluate') and self.schedule_manager is not None:
                # Check if evaluate accepts step/total_steps (for obstacle constraints)
                try:
                    energy = constraint.evaluate(trajectory, step=step, total_steps=total_steps)
                except TypeError:
                    # Fallback: constraint doesn't support step/total_steps
                    energy = constraint.evaluate(trajectory)
                    # Apply scheduled weight if constraint has alpha
                    if hasattr(constraint, 'alpha'):
                        scheduled_alpha = self.schedule_manager.get_soft_alpha(
                            default=constraint.alpha,
                            step=step,
                            total_steps=total_steps
                        )
                        weight = scheduled_alpha / constraint.alpha
                        energy = weight * energy
            else:
                # Standard evaluate call
                energy = constraint.evaluate(trajectory)
            
            total += energy
        return total
    
    def compute_soft_energy_batch(
        self,
        trajectories: List[Trajectory],
        step: Optional[int] = None,
        total_steps: Optional[int] = None
    ) -> np.ndarray:
        """
        Batch compute soft constraint penalties for multiple trajectories.
        
        Optimized version that processes all trajectories at once when possible.
        
        Args:
            trajectories: List of trajectories to evaluate
            step: Current step (optional, for scheduling)
            total_steps: Total steps (optional, for scheduling)
            
        Returns:
            Array of total soft penalties, shape (len(trajectories),)
        """
        # Check if soft constraints are active
        if self.schedule_manager is not None:
            if not self.schedule_manager.is_soft_active(step, total_steps):
                return np.zeros(len(trajectories), dtype=np.float32)
        
        # Initialize total energies
        total_energies = np.zeros(len(trajectories), dtype=np.float32)
        
        for constraint in self.soft_constraints:
            # Check if constraint supports batch evaluation
            if hasattr(constraint, 'evaluate_batch'):
                # Use batch evaluation
                try:
                    energies = constraint.evaluate_batch(
                        trajectories, step=step, total_steps=total_steps
                    )
                    total_energies += energies.astype(np.float32)
                    continue
                except (TypeError, AttributeError):
                    # Fallback to individual evaluation if batch fails
                    pass
            
            # Individual evaluation (fallback or for constraints without batch support)
            for idx, trajectory in enumerate(trajectories):
                if hasattr(constraint, 'schedule_manager') and constraint.schedule_manager is not None:
                    energy = constraint.evaluate(trajectory, step=step, total_steps=total_steps)
                elif hasattr(constraint, 'evaluate') and self.schedule_manager is not None:
                    try:
                        energy = constraint.evaluate(trajectory, step=step, total_steps=total_steps)
                    except TypeError:
                        energy = constraint.evaluate(trajectory)
                        if hasattr(constraint, 'alpha'):
                            scheduled_alpha = self.schedule_manager.get_soft_alpha(
                                default=constraint.alpha,
                                step=step,
                                total_steps=total_steps
                            )
                            weight = scheduled_alpha / constraint.alpha
                            energy = weight * energy
                else:
                    energy = constraint.evaluate(trajectory)
                
                total_energies[idx] += float(energy)
        
        return total_energies
    
    def check_hard_feasibility(
        self, 
        trajectory: Trajectory,
        step: Optional[int] = None,
        total_steps: Optional[int] = None
    ) -> bool:
        """
        Check if trajectory satisfies all hard constraints.
        
        Args:
            trajectory: Trajectory to check
            step: Current step (optional, for scheduling)
            total_steps: Total steps (optional, for scheduling)
            
        Returns:
            True if all constraints satisfied
        """
        # Check if hard constraints are active
        if self.schedule_manager is not None:
            if not self.schedule_manager.is_hard_active(step, total_steps):
                return True  # If not active, consider as feasible
        
        for constraint in self.hard_constraints:
            if not constraint.is_feasible(trajectory):
                return False
        return True
    
    def project_hard(
        self,
        trajectory: Trajectory,
        step: Optional[int] = None,
        total_steps: Optional[int] = None
    ) -> Trajectory:
        """
        Project trajectory onto hard feasible set.
        
        Args:
            trajectory: Trajectory to project
            step: Current diffusion step (optional)
            total_steps: Total diffusion steps (optional)
            
        Returns:
            Projected trajectory
        """
        # Check if hard constraints should be active
        if self.schedule_manager is not None:
            if not self.schedule_manager.is_hard_active(step, total_steps):
                return trajectory  # If not active, return as-is
        
        # If no hard constraints, return as-is
        if not self.hard_constraints:
            return trajectory
        
        # Use feasibility operator if available
        if self.feasibility_operator is not None:
            if self.feasibility_operator.should_apply(step, total_steps):
                return self.feasibility_operator.project(trajectory, step, total_steps)
        
        # Otherwise, apply individual constraints sequentially
        projected = trajectory
        for constraint in self.hard_constraints:
            projected = constraint.project(projected, step, total_steps)
        
        return projected
    
    def has_soft(self) -> bool:
        """Check if any soft constraints are defined."""
        return len(self.soft_constraints) > 0
    
    def has_hard(self) -> bool:
        """Check if any hard constraints are defined."""
        return (len(self.hard_constraints) > 0) or (self.feasibility_operator is not None) or (self.action_filter_operator is not None)