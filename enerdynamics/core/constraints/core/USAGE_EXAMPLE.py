"""
Usage example for the high-performance constraint system.

This demonstrates how to use the new constraint system architecture:
1. Create convexifiers, operators, and schedulers
2. Use the pipeline for end-to-end constraint enforcement
3. Leverage caching and batch processing
"""

import numpy as np
from enerdynamics.core.types import Trajectory, State, Action
from enerdynamics.core.constraints.core import (
    HighPerformanceConstraintPipeline,
    PipelineConfig,
    ScheduleState,
    ScheduleParams,
)
from enerdynamics.core.constraints.convexify import CFSConvexifier, CBFConvexifier
from enerdynamics.core.constraints.operators import PerStepQPFilter
from enerdynamics.core.constraints.schedulers import CosineAnnealScheduler


def example_cfs_convexifier():
    """Example: Using CFS convexifier."""
    from enerdynamics.envs.obstacles.base import ObstacleManager
    
    # Create obstacles
    obstacles = ObstacleManager()  # Initialize with your obstacles
    
    # Create CFS convexifier
    convexifier = CFSConvexifier(
        obstacles=obstacles,
        backend="numpy",
        max_constraints_per_point=8
    )
    
    # Create reference trajectory
    ref = Trajectory(
        states=[np.array([0.0, 0.0, 0.0, 0.0]) for _ in range(10)],
        actions=[np.array([0.0, 0.0]) for _ in range(9)]
    )
    
    # Build constraints
    params = ScheduleParams(margin=0.1)
    state = ScheduleState(k=10, K=100)
    constraints = convexifier.build_constraints(ref, params, state)
    
    print(f"CFS constraints: A.shape={constraints.A.shape}, b.shape={constraints.b.shape}")
    return constraints


def example_cbf_convexifier():
    """Example: Using CBF convexifier."""
    from enerdynamics.envs.obstacles.base import ObstacleManager
    
    # Create obstacles and dynamics
    obstacles = ObstacleManager()
    dynamics = None  # Your dynamics model
    
    # Create CBF convexifier
    convexifier = CBFConvexifier(
        obstacles=obstacles,
        dynamics=dynamics,
        backend="numpy",
        robot_radius=0.05,
        dt=0.1
    )
    
    # Create reference (state, action) pair
    state = np.array([0.0, 0.0, 0.0, 0.0])  # [px, py, vx, vy]
    action = np.array([0.0, 0.0])  # [ax, ay]
    
    # Build constraints
    params = ScheduleParams(margin=0.05)
    schedule_state = ScheduleState(k=10, K=100)
    constraints = convexifier.build_constraints((state, action), params, schedule_state)
    
    print(f"CBF constraints: A.shape={constraints.A.shape}, b.shape={constraints.b.shape}")
    print(f"Per-step: {constraints.is_per_step()}")
    return constraints


def example_per_step_qp_operator():
    """Example: Using per-step QP operator."""
    # Create operator
    operator = PerStepQPFilter(
        use_slack=True,
        solver_backend="numpy"
    )
    
    # Create nominal trajectory
    nominal = Trajectory(
        states=[np.array([0.0, 0.0, 0.0, 0.0]) for _ in range(10)],
        actions=[np.array([0.5, 0.5]) for _ in range(9)]
    )
    
    # Create constraints (from convexifier)
    from enerdynamics.core.constraints.core.types import ConvexConstraint
    A = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    b = np.array([0.0, 0.0], dtype=np.float32)
    constraints = ConvexConstraint(A=A, b=b, meta={"per_step": True})
    
    # Apply operator
    params = ScheduleParams(rho=2.0)
    state = ScheduleState(k=10, K=100)
    repaired, info = operator.apply(nominal, constraints, params, state)
    
    print(f"Repaired trajectory: {len(repaired.actions)} actions")
    print(f"Operator info: {info}")
    return repaired, info


def example_cosine_anneal_scheduler():
    """Example: Using cosine annealing scheduler."""
    # Create scheduler
    scheduler = CosineAnnealScheduler(
        margin_start=0.5,
        margin_end=0.1,
        rho_start=0.1,
        rho_end=10.0
    )
    
    # Get parameters at different steps
    for k in [0, 50, 100]:
        state = ScheduleState(k=k, K=100)
        params = scheduler.params(state)
        print(f"Step {k}: margin={params.margin:.3f}, rho={params.rho:.3f}")


def example_pipeline():
    """Example: Using the high-performance pipeline."""
    from enerdynamics.envs.obstacles.base import ObstacleManager
    
    # Create pipeline
    pipeline = HighPerformanceConstraintPipeline(
        convexifier_name="cfs",
        operator_name="per_step_qp",
        scheduler_name="cosine_anneal",
        config=PipelineConfig(
            backend="numpy",
            use_jit=False,  # JAX not available in this example
            use_batch=True,
            cache_constraints=True,
            cache_params=True
        ),
        obstacles=ObstacleManager(),  # Pass obstacles as kwargs
        use_slack=True
    )
    
    # Create trajectories
    nominal = Trajectory(
        states=[np.array([0.0, 0.0, 0.0, 0.0]) for _ in range(10)],
        actions=[np.array([0.5, 0.5]) for _ in range(9)]
    )
    ref = nominal  # Use same as reference
    
    # Apply pipeline
    state = ScheduleState(k=10, K=100)
    repaired, info = pipeline.apply(nominal, ref, state)
    
    print(f"Pipeline repaired trajectory: {len(repaired.actions)} actions")
    print(f"Pipeline info: {info}")
    return repaired, info


def example_batch_processing():
    """Example: Batch processing with pipeline."""
    from enerdynamics.envs.obstacles.base import ObstacleManager
    
    # Create pipeline
    pipeline = HighPerformanceConstraintPipeline(
        convexifier_name="cfs",
        operator_name="per_step_qp",
        scheduler_name="cosine_anneal",
        config=PipelineConfig(
            backend="numpy",
            use_batch=True
        ),
        obstacles=ObstacleManager()
    )
    
    # Create batch of trajectories
    nominals = [
        Trajectory(
            states=[np.array([0.0, 0.0, 0.0, 0.0]) for _ in range(10)],
            actions=[np.array([0.5, 0.5]) for _ in range(9)]
        )
        for _ in range(5)
    ]
    refs = nominals
    
    # Apply to batch
    state = ScheduleState(k=10, K=100)
    repaired_list, info = pipeline.apply_batch(nominals, refs, state)
    
    print(f"Batch processed: {len(repaired_list)} trajectories")
    print(f"Batch info: {info}")
    return repaired_list, info


if __name__ == "__main__":
    print("=== CFS Convexifier Example ===")
    example_cfs_convexifier()
    
    print("\n=== CBF Convexifier Example ===")
    example_cbf_convexifier()
    
    print("\n=== Per-Step QP Operator Example ===")
    example_per_step_qp_operator()
    
    print("\n=== Cosine Anneal Scheduler Example ===")
    example_cosine_anneal_scheduler()
    
    print("\n=== Pipeline Example ===")
    example_pipeline()
    
    print("\n=== Batch Processing Example ===")
    example_batch_processing()


