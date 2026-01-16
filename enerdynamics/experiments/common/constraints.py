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
from enerdynamics.core.constraints.core import (
    HighPerformanceConstraintPipeline,
    PipelineConfig,
)
from enerdynamics.core.constraints.schedulers import (
    CosineAnnealScheduler,
    CompositeScheduler,
    MergeStrategy,
    FixedConstraintScheduler,
    FixedDiffusionScheduler,
    DualControlConstraintScheduler,
)

# Import to trigger registration of all components
from enerdynamics.core.constraints.convexify import cfs  
from enerdynamics.core.constraints.operators import qp  
from enerdynamics.core.constraints.solvers import jaxopt_osqp_solver  
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
    backend_name: str = "jax",
    obstacle_config: Optional[Dict[str, Any]] = None,
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
        obstacle_config: Optional obstacle configuration dictionary (for getting robot_radius)
        
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
        
        # Get robot_radius from obstacle_config if available
        robot_radius = 0.0
        if obstacle_config is not None:
            robot_radius = float(obstacle_config.get('robot_radius', 0.0))
        # Also check if config contains obstacle_config (for backward compatibility)
        elif 'obstacle_config' in config:
            robot_radius = float(config['obstacle_config'].get('robot_radius', 0.0))
        
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
            robot_radius=robot_radius,
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


def create_constraint_pipeline(
    obstacles: ObstacleManager,
    level: int,
    env: Any,
    config: Dict[str, Any],
    backend_name: str = "jax",
    obstacle_config: Optional[Dict[str, Any]] = None,
    method_params: Optional[Dict[str, Any]] = None,
) -> Optional[HighPerformanceConstraintPipeline]:
    """
    Create constraint pipeline with new architecture.
    
    This replaces create_constraint_manager with the new high-performance pipeline.
    
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
        backend_name: Computational backend name ("numpy" or "jax")
        obstacle_config: Optional obstacle configuration dictionary (for getting robot_radius)
        method_params: Optional method parameters dictionary. Can contain:
            - cfs_use_trajectory_qp: bool, whether to use full trajectory QP (True) or per-step (False)
            - cfs_max_constraints_per_point: int, max constraints per point
            - cfs_constraint_margin: float, constraint margin
        
    Returns:
        HighPerformanceConstraintPipeline instance or None if level==0 and no constraints
    """
    if level == 0 or len(obstacles) == 0:
        # No obstacles, no pipeline needed
        # Action constraints are handled separately in the solver
        return None
    
    # Get CFS configuration
    cfs_config = config.get('cfs', {})
    if not cfs_config.get('enabled', True):
        return None
    
    # Get robot_radius from obstacle_config if available
    robot_radius = 0.0
    if obstacle_config is not None:
        robot_radius = float(obstacle_config.get('robot_radius', 0.0))
    elif 'obstacle_config' in config:
        robot_radius = float(config['obstacle_config'].get('robot_radius', 0.0))
    
    # Get schedule configuration
    schedule_config = config.get('schedule', {})
    
    # Create scheduler
    scheduler = None
    if schedule_config.get('enabled', True):
        schedule_type = schedule_config.get('type', 'soft_to_hard')
        if schedule_type == 'soft_to_hard':
            # Map legacy schedule parameters to new scheduler
            # Legacy: soft_alpha_start/end, hard_clearance_start/end
            # New: margin_start/end (maps to hard_clearance), rho_start/end (for slack penalty)
            scheduler = CosineAnnealScheduler(
                margin_start=float(schedule_config.get('hard_clearance_start', 0.5)),
                margin_end=float(schedule_config.get('hard_clearance_end', 0.1)),
                rho_start=float(schedule_config.get('rho_start', 0.1)),
                rho_end=float(schedule_config.get('rho_end', 10.0)),
            )
        else:
            # Default scheduler
            scheduler = CosineAnnealScheduler(
                margin_start=0.5,
                margin_end=0.1,
                rho_start=0.1,
                rho_end=10.0,
            )
    else:
        # Fixed parameters (no scheduling)
        hard_config = config.get('hard_constraint', {})
        clearance = float(hard_config.get('clearance', 0.1))
        scheduler = CosineAnnealScheduler(
            margin_start=clearance,
            margin_end=clearance,
            rho_start=1.0,
            rho_end=1.0,
        )
    
    # Create pipeline configuration
    # Disable JIT for now to test JAX backend without JIT compilation issues
    pipeline_config = PipelineConfig(
        backend=backend_name,
        use_jit=False,  # Disable JIT for testing (can enable later)
        use_batch=True,
        cache_constraints=True,
        cache_params=True,
        verbose=False,
    )
    
    # Determine operator based on CFS config
    # CFS constraints are on states (positions), not actions, so we need to use
    # a state projection operator instead of per-step QP filter
    # Priority: method_params > constraint_config > default
    if method_params is not None and 'cfs_use_trajectory_qp' in method_params:
        use_trajectory_qp = bool(method_params.get('cfs_use_trajectory_qp', False))
    else:
        use_trajectory_qp = cfs_config.get('use_trajectory_qp', False)
    
    if use_trajectory_qp:
        operator_name = "traj_qp"
    else:
        # Use projection operator for state constraints (CFS projects states, not actions)
        operator_name = "projection"
    
    # Create pipeline
    # Note: Action constraints (SpeedConstraint, AccelerationConstraint) are handled
    # separately in the solver, not in the pipeline
    pipeline = HighPerformanceConstraintPipeline(
        convexifier_name="cfs",
        operator_name=operator_name,
        scheduler_name="cosine_anneal",
        config=pipeline_config,
        obstacles=obstacles,
        position_extractor=None,  # Use default
        max_constraints_per_point=int(
            method_params.get('cfs_max_constraints_per_point') if (method_params is not None and 'cfs_max_constraints_per_point' in method_params)
            else cfs_config.get('max_constraints_per_point', 8)
        ),
        constraint_margin=float(
            method_params.get('cfs_constraint_margin') if (method_params is not None and 'cfs_constraint_margin' in method_params)
            else cfs_config.get('constraint_margin', 0.25)
        ),
        robot_radius=robot_radius,
        # Operator parameters
        project_states=True,  # CFS projects states (positions)
        project_actions=False,  # CFS doesn't project actions
        use_slack=True,  # Use slack-QP by default (if using traj_qp)
        solver_backend=backend_name,
        # JAX convexifier parameters
        use_jit=False,  # Disable JIT for testing (can enable later)
        # Scheduler parameters (passed to scheduler if it needs them)
        margin_start=scheduler.margin_start,
        margin_end=scheduler.margin_end,
        rho_start=scheduler.rho_start,
        rho_end=scheduler.rho_end,
    )
    
    # Override scheduler with the one we created
    pipeline.scheduler = scheduler
    
    return pipeline


def create_scheduler_from_config(
    config: Dict[str, Any],
    backend_name: str = "numpy",
    method_params: Optional[Dict[str, Any]] = None,
    obstacle_config: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """
    Create scheduler from configuration dictionary.
    
    Supports creating CompositeScheduler with constraint and diffusion schedulers.
    
    Args:
        config: Scheduler configuration dictionary with keys:
            - type: "composite" (required for CompositeScheduler)
            - constraint_schedulers: List of constraint scheduler configs
            - diffusion_schedulers: List of diffusion scheduler configs
            - constraint_merge_strategy: Merge strategy for constraint schedulers
            - diffusion_merge_strategy: Merge strategy for diffusion schedulers
            - constraint_weights: Optional weights for weighted merge
            - diffusion_weights: Optional weights for weighted merge
        backend_name: Computational backend name
        
    Returns:
        Scheduler instance (CompositeScheduler or single scheduler)
    """
    scheduler_type = config.get('type', 'composite')
    
    if scheduler_type == 'composite':
        # Create constraint schedulers
        constraint_schedulers = []
        constraint_configs = config.get('constraint_schedulers', [])
        for cs_config in constraint_configs:
            cs_type = cs_config.get('type', 'fixed')
            if cs_type == 'fixed':
                scheduler = FixedConstraintScheduler(
                    rho=float(cs_config.get('rho', 10.0)),
                    topK=cs_config.get('topK'),
                    eps=float(cs_config.get('eps', 1e-4)),
                    I_QP=int(cs_config.get('I_QP', 10)),
                    qp_gate=cs_config.get('qp_gate', True),
                    qp_prob=float(cs_config.get('qp_prob', 1.0)),
                    margin=float(cs_config.get('margin', 0.0)),
                    backend=backend_name,
                )
                constraint_schedulers.append(scheduler)
            elif cs_type == 'dual_control':
                # Generate betas for dual_control scheduler if needed
                betas = None
                if 'betas' in cs_config:
                    betas = np.asarray(cs_config['betas'], dtype=np.float32)
                elif method_params is not None:
                    # Generate betas from method_params
                    action_diffuse_steps = method_params.get('action_diffuse_steps', 100)
                    action_beta0 = method_params.get('action_beta0', 1e-4)
                    action_betaT = method_params.get('action_betaT', 1e-2)
                    betas = np.linspace(action_beta0, action_betaT, action_diffuse_steps, dtype=np.float32)
                
                # Fix B3: Get robot_radius from obstacle_config or config
                robot_radius_sched = 0.0
                if obstacle_config is not None:
                    robot_radius_sched = float(obstacle_config.get('robot_radius', 0.0))
                elif 'obstacle_config' in config:
                    robot_radius_sched = float(config['obstacle_config'].get('robot_radius', 0.0))
                
                if robot_radius_sched <= 0:
                    # Fallback: try to get from cs_config or use default
                    robot_radius_sched = float(cs_config.get('robot_radius', 0.05))
                
                scheduler = DualControlConstraintScheduler(
                    betas=betas,
                    # Fix B3: Pass robot_radius to scheduler
                    robot_radius=robot_radius_sched,
                    # Dual variable parameters
                    lambda_con_min=float(cs_config.get('lambda_con_min', -2.0)),
                    lambda_con_max=float(cs_config.get('lambda_con_max', 2.0)),
                    lambda_con_init=float(cs_config.get('lambda_con_init', 0.0)),
                    eta_con_base=float(cs_config.get('eta_con_base', 0.1)),
                    eta_con_decay=float(cs_config.get('eta_con_decay', 0.99)),
                    # Target feasibility schedule
                    q_star_min=float(cs_config.get('q_star_min', 0.3)),
                    q_star_max=float(cs_config.get('q_star_max', 0.95)),
                    p_q=float(cs_config.get('p_q', 1.0)),
                    # Constraint mapping parameters
                    rho_min=float(cs_config.get('rho_min', 0.1)),
                    rho_max=float(cs_config.get('rho_max', 100.0)),
                    a_rho=float(cs_config.get('a_rho', 1.0)),
                    b_rho=float(cs_config.get('b_rho', 0.0)),
                    a_g=float(cs_config.get('a_g', 1.0)),
                    b_g=float(cs_config.get('b_g', 0.0)),
                    # Constraint scheduling parameters
                    K_min=int(cs_config.get('K_min', 5)),
                    K_max=int(cs_config.get('K_max', 50)),
                    eps_min=float(cs_config.get('eps_min', 1e-4)),
                    eps_max=float(cs_config.get('eps_max', 1e-2)),
                    p_eps=float(cs_config.get('p_eps', 1.0)),
                    I_min=int(cs_config.get('I_min', 1)),
                    I_max=int(cs_config.get('I_max', 10)),
                    # Terminal hard region
                    t_hard=float(cs_config.get('t_hard', 0.8)),
                    backend=backend_name,
                )
                constraint_schedulers.append(scheduler)
            elif cs_type == 'emerging_barrier':
                from enerdynamics.core.constraints.schedulers.ConstraintScheduler.emergingbarrier import (
                    EmergingBarrierConstraintScheduler,
                )

                scheduler = EmergingBarrierConstraintScheduler(
                    mu=float(cs_config.get("mu", 10.0)),
                    alpha=float(cs_config.get("alpha", 1.0)),
                    bound=float(cs_config.get("bound", 0.8)),
                    use_min_over_time=bool(cs_config.get("use_min_over_time", True)),
                    terminal_energy_weight=float(cs_config.get("terminal_energy_weight", 0.0)),
                    margin=float(cs_config.get("margin", 0.0)),
                    backend=backend_name,
                )
                constraint_schedulers.append(scheduler)
            else:
                raise ValueError(f"Unknown constraint scheduler type: {cs_type}")
        
        # Create diffusion schedulers
        diffusion_schedulers = []
        diffusion_configs = config.get('diffusion_schedulers', [])
        for ds_config in diffusion_configs:
            ds_type = ds_config.get('type', 'fixed')
            if ds_type == 'fixed':
                scheduler = FixedDiffusionScheduler(
                    M_k=int(ds_config.get('M_k', 64)),
                    T_k=float(ds_config.get('T_k', 0.5)),
                    s_k=ds_config.get('s_k'),
                    beta0=ds_config.get('beta0'),
                    betaT=ds_config.get('betaT'),
                    Ndiffuse=ds_config.get('Ndiffuse'),
                    backend=backend_name,
                )
                diffusion_schedulers.append(scheduler)
            else:
                raise ValueError(f"Unknown diffusion scheduler type: {ds_type}")
        
        # Create composite scheduler
        constraint_strategy_name = config.get('constraint_merge_strategy', 'merge')
        diffusion_strategy_name = config.get('diffusion_merge_strategy', 'merge')
        
        constraint_strategy = MergeStrategy[constraint_strategy_name.upper()] if hasattr(MergeStrategy, constraint_strategy_name.upper()) else MergeStrategy.MERGE
        diffusion_strategy = MergeStrategy[diffusion_strategy_name.upper()] if hasattr(MergeStrategy, diffusion_strategy_name.upper()) else MergeStrategy.MERGE
        
        composite = CompositeScheduler(
            constraint_schedulers=constraint_schedulers if constraint_schedulers else None,
            diffusion_schedulers=diffusion_schedulers if diffusion_schedulers else None,
            constraint_merge_strategy=constraint_strategy,
            diffusion_merge_strategy=diffusion_strategy,
            constraint_weights=config.get('constraint_weights'),
            diffusion_weights=config.get('diffusion_weights'),
        )
        
        return composite
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

