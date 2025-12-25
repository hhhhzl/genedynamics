"""
Common constraint creation utilities.

This module provides factory functions for creating constraint managers
with various configurations, used across different experiment types.
"""

from typing import Optional, Dict, Any, List
import numpy as np

from enerdynamics.core.constraints import (
    ConstraintManager,
    ObstacleSoftConstraint,
    ObstacleHardConstraint,
    CFSProjection,  # Deprecated: Use new architecture (CFSConvexifier + PerStepQPFilter)
    ConstraintScheduleManager,
    HardConstraint,
)
from enerdynamics.core.types import Trajectory
from enerdynamics.envs.obstacles.base import ObstacleManager


class AccelerationConstraint(HardConstraint):
    """
    Hard constraint on acceleration (control input).
    
    Ensures |u| ≤ u_max for all actions in trajectory.
    """
    
    def __init__(self, u_max: float = 1.0):
        """
        Initialize acceleration constraint.
        
        Args:
            u_max: Maximum acceleration magnitude
        """
        self.u_max = float(u_max)
    
    def is_feasible(self, trajectory: Trajectory) -> bool:
        """Check if all actions satisfy |u| ≤ u_max."""
        for action in trajectory.actions:
            action_np = np.asarray(action, dtype=np.float32)
            if np.linalg.norm(action_np) > self.u_max:
                return False
        return True
    
    def violations(self, trajectory: Trajectory) -> np.ndarray:
        """Return violations: max(0, |u| - u_max) for each action."""
        violations = []
        for action in trajectory.actions:
            action_np = np.asarray(action, dtype=np.float32)
            violation = max(0.0, np.linalg.norm(action_np) - self.u_max)
            violations.append(violation)
        return np.array(violations, dtype=np.float32)
    
    def project(
        self,
        trajectory: Trajectory,
        step: Optional[int] = None,
        total_steps: Optional[int] = None
    ) -> Trajectory:
        """Project actions to satisfy |u| ≤ u_max."""
        projected_actions = []
        for action in trajectory.actions:
            action_np = np.asarray(action, dtype=np.float32)
            norm = np.linalg.norm(action_np)
            if norm > self.u_max:
                action_np = action_np / norm * self.u_max
            projected_actions.append(action_np)
        
        return Trajectory(states=trajectory.states, actions=projected_actions)


class SpeedConstraint(HardConstraint):
    """
    Hard constraint on single-integrator action (interpreted as velocity).
    
    Enforces max_i |u_i| <= u_max (L-infinity bound).
    """
    
    def __init__(self, u_max: float = 1.0):
        """
        Initialize speed constraint.
        
        Args:
            u_max: Maximum speed magnitude (L-infinity)
        """
        self.u_max = float(u_max)
    
    def is_feasible(self, trajectory: Trajectory) -> bool:
        """Check if all actions satisfy max_i |u_i| <= u_max."""
        for action in trajectory.actions:
            a = np.asarray(action, dtype=np.float32)
            if float(np.max(np.abs(a))) > self.u_max + 1e-8:
                return False
        return True
    
    def violations(self, trajectory: Trajectory) -> np.ndarray:
        """Return violations: max(0, max_i |u_i| - u_max) for each action."""
        violations = []
        for action in trajectory.actions:
            a = np.asarray(action, dtype=np.float32)
            violations.append(max(0.0, float(np.max(np.abs(a)) - self.u_max)))
        return np.asarray(violations, dtype=np.float32)
    
    def project(
        self,
        trajectory: Trajectory,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> Trajectory:
        """Project actions to satisfy max_i |u_i| <= u_max."""
        projected_actions = []
        for action in trajectory.actions:
            a = np.asarray(action, dtype=np.float32)
            a = np.clip(a, -self.u_max, self.u_max)
            projected_actions.append(a)
        return Trajectory(states=trajectory.states, actions=projected_actions)


def create_constraint_manager(
    obstacles: ObstacleManager,
    level: int,
    env: Any,
    config: Dict[str, Any],
    backend_name: str = "jax"
) -> Optional[ConstraintManager]:
    """
    Create constraint manager with configuration.
    
    Args:
        obstacles: Obstacle manager instance
        level: Obstacle level (0 means no obstacles)
        env: Environment instance (for getting control limits)
        config: Constraint configuration dictionary with keys:
            - soft_constraint: Dict with 'enabled', 'alpha', 'beta'
            - hard_constraint: Dict with 'enabled', 'clearance'
            - schedule: Dict with schedule parameters
            - cfs: Dict with CFS projection parameters
            - action_constraint_type: 'acceleration' or 'speed'
        backend_name: Computational backend name
        
    Returns:
        ConstraintManager instance or None if level==0 and no constraints
    """
    if level == 0 or len(obstacles) == 0:
        # No obstacles, but may still have action constraints
        action_constraint_type = config.get('action_constraint_type', None)
        if action_constraint_type is None:
            return None
        
        hard_constraints = []
        if action_constraint_type == 'acceleration':
            u_max = getattr(env, 'control_limit', 1.0)
            hard_constraints.append(AccelerationConstraint(u_max=u_max))
        elif action_constraint_type == 'speed':
            u_max = getattr(env, 'control_limit', 1.0)
            hard_constraints.append(SpeedConstraint(u_max=u_max))
        
        return ConstraintManager(
            soft_constraints=[],
            hard_constraints=hard_constraints,
            feasibility_operator=None,
            action_filter_operator=None,
            schedule_manager=None,
        )
    
    # Soft constraint
    soft_constraint = None
    soft_config = config.get('soft_constraint', {})
    if soft_config.get('enabled', True):
        soft_constraint = ObstacleSoftConstraint(
            obstacles=obstacles,
            alpha=float(soft_config.get('alpha', 1.0)),
            beta=float(soft_config.get('beta', 10.0))
        )
    
    # Hard constraint
    hard_constraint = None
    hard_config = config.get('hard_constraint', {})
    if hard_config.get('enabled', True):
        hard_constraint = ObstacleHardConstraint(
            obstacles=obstacles,
            clearance=float(hard_config.get('clearance', 0.1))
        )
    
    # Schedule manager
    schedule_manager = None
    schedule_config = config.get('schedule', {})
    if schedule_config.get('enabled', True) and (soft_constraint or hard_constraint):
        schedule_type = schedule_config.get('type', 'soft_to_hard')
        if schedule_type == 'soft_to_hard':
            schedule_manager = ConstraintScheduleManager.create_soft_to_hard(
                soft_alpha_start=float(schedule_config.get('soft_alpha_start', 1.0)),
                soft_alpha_end=float(schedule_config.get('soft_alpha_end', 0.0)),
                hard_clearance_start=float(schedule_config.get('hard_clearance_start', 0.5)),
                hard_clearance_end=float(schedule_config.get('hard_clearance_end', 0.1)),
                schedule_type=schedule_config.get('schedule_type', 'linear'),
                reverse_mode=schedule_config.get('reverse_mode', True),
            )
            if soft_constraint is not None:
                soft_constraint.schedule_manager = schedule_manager
            if hard_constraint is not None:
                hard_constraint.schedule_manager = schedule_manager
    
    # Feasibility operator (CFS)
    feasibility_op = None
    cfs_config = config.get('cfs', {})
    if cfs_config.get('enabled', True):
        force_python = (backend_name == "numpy")
        feasibility_op = CFSProjection(
            obstacles=obstacles,
            schedule_manager=schedule_manager,
            use_late_stage_only=cfs_config.get('use_late_stage_only', True),
            late_stage_ratio=float(cfs_config.get('late_stage_ratio', 0.2)),
            use_trajectory_qp=cfs_config.get('use_trajectory_qp', True),
            smoothness_weight=float(cfs_config.get('smoothness_weight', 0.0)),
            reconstruct_velocity=cfs_config.get('reconstruct_velocity', True),
            velocity_dt=getattr(env, 'dt', 0.1) if cfs_config.get('reconstruct_velocity', True) else None,
            max_iterations=int(cfs_config.get('max_iterations', 15)),
            force_python_backend=force_python,
        )
    
    # Action constraints
    hard_constraints = []
    if soft_constraint is not None:
        # soft_constraint is handled via soft_constraints list
        pass
    if hard_constraint is not None:
        hard_constraints.append(hard_constraint)
    
    action_constraint_type = config.get('action_constraint_type', None)
    if action_constraint_type == 'acceleration':
        u_max = getattr(env, 'control_limit', 1.0)
        hard_constraints.append(AccelerationConstraint(u_max=u_max))
    elif action_constraint_type == 'speed':
        u_max = getattr(env, 'control_limit', 1.0)
        hard_constraints.append(SpeedConstraint(u_max=u_max))
    
    soft_constraints = [soft_constraint] if soft_constraint is not None else []
    
    return ConstraintManager(
        soft_constraints=soft_constraints,
        hard_constraints=hard_constraints,
        feasibility_operator=feasibility_op,
        action_filter_operator=None,
        schedule_manager=schedule_manager,
    )

