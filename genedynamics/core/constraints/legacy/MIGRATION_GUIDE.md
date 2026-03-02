# Migration Guide: Legacy to New Architecture

This guide helps migrate from the legacy constraint system to the new high-performance architecture.

## Architecture Comparison

### Legacy Architecture
```
ConstraintManager
├── SoftConstraint (energy evaluation)
├── HardConstraint (feasibility checking)
├── FeasibilityOperator (CFSProjection - does both convexify + QP)
└── ActionFilterOperator (CBF - does both convexify + QP)
```

### New Architecture
```
HighPerformanceConstraintPipeline
├── Convexifier (generates constraints: A, b)
├── Operator (enforces constraints: QP, projection)
└── Scheduler (schedules parameters: margin, rho, etc.)
```

## Migration Steps

### Step 1: Identify Your Components

**Legacy Components:**
- `CFSProjection`: Both convexifies AND projects (coupled)
- `CBFDoubleIntegrator2DActionFilter`: Both linearizes AND filters (coupled)
- `ConstraintScheduleManager`: Parameter scheduling

**New Components:**
- `CFSConvexifier`: Only convexifies (generates A, b)
- `CBFConvexifier`: Only convexifies (generates A, b)
- `PerStepQPFilter`: Only enforces (solves QP)
- `CosineAnnealScheduler`: Only schedules (generates params)

### Step 2: Replace ConstraintManager with Pipeline

**Before (Legacy):**
```python
from genedynamics.core.constraints import ConstraintManager, CFSProjection

# Create constraint manager
constraint_manager = ConstraintManager(
    feasibility_operator=CFSProjection(obstacles),
    schedule_manager=schedule_manager
)

# Use in diffusion loop
projected = constraint_manager.project_hard(trajectory, step=k, total_steps=K)
```

**After (New):**
```python
from genedynamics.core.constraints.core import (
    HighPerformanceConstraintPipeline,
    PipelineConfig,
    ScheduleState
)

# Create pipeline
pipeline = HighPerformanceConstraintPipeline(
    convexifier_name="cfs",
    operator_name="per_step_qp",
    scheduler_name="cosine_anneal",
    config=PipelineConfig(backend="numpy", use_batch=True),
    obstacles=obstacles
)

# Use in diffusion loop
repaired, info = pipeline.apply(
    nominal=trajectory,
    ref=trajectory,
    state=ScheduleState(k=k, K=K)
)
```

### Step 3: Replace CBF Action Filter

**Before (Legacy):**
```python
from genedynamics.core.constraints import CBFDoubleIntegrator2DActionFilter

# Create filter
cbf_filter = CBFDoubleIntegrator2DActionFilter(obstacles)

# Use during rollout
filtered_actions = cbf_filter.filter_actions_numpy(x0, actions, step=k, total_steps=K)
```

**After (New):**
```python
from genedynamics.core.constraints.convexify import CBFConvexifier
from genedynamics.core.constraints.operators import PerStepQPFilter
from genedynamics.core.constraints.core import ScheduleState, ScheduleParams

# Create components
cbf_convexifier = CBFConvexifier(obstacles, dynamics, backend="numpy")
qp_operator = PerStepQPFilter(use_slack=True)

# Use during rollout
state = ScheduleState(k=k, K=K)
params = ScheduleParams(margin=0.05)

# For each time step
for t in range(len(actions)):
    x_t = states[t]
    u_t = actions[t]
    
    # Build constraints
    constraints = cbf_convexifier.build_constraints(
        (x_t, u_t), params, state
    )
    
    # Apply QP
    repaired_traj, _ = qp_operator.apply(
        Trajectory(states=[x_t], actions=[u_t]),
        constraints, params, state
    )
    actions[t] = repaired_traj.actions[0]
```

**Or use pipeline (simpler):**
```python
# Pipeline handles everything
repaired, info = pipeline.apply(nominal, ref, state)
```

### Step 4: Replace Schedule Manager

**Before (Legacy):**
```python
from genedynamics.core.constraints import ConstraintScheduleManager

schedule_manager = ConstraintScheduleManager.create_soft_to_hard(
    soft_alpha_start=1.0,
    soft_alpha_end=0.0,
    hard_clearance_start=0.5,
    hard_clearance_end=0.1
)

alpha = schedule_manager.get_soft_alpha(step=k, total_steps=K)
clearance = schedule_manager.get_hard_clearance(step=k, total_steps=K)
```

**After (New):**
```python
from genedynamics.core.constraints.schedulers import CosineAnnealScheduler
from genedynamics.core.constraints.core import ScheduleState

scheduler = CosineAnnealScheduler(
    margin_start=0.5,
    margin_end=0.1,
    rho_start=0.1,
    rho_end=10.0
)

state = ScheduleState(k=k, K=K)
params = scheduler.params(state)

# Access parameters
margin = params.margin
rho = params.rho
```

## Performance Improvements

### Expected Improvements

| Feature | Legacy | New | Speedup |
|---------|--------|-----|---------|
| Single trajectory | Baseline | 1x | - |
| Batch (10 traj) | Sequential | Vectorized | 5-10x |
| Batch (100 traj, JAX) | Sequential | GPU + JIT | 50-200x |
| With caching | No cache | LRU cache | 1.5-3x |
| JIT compilation | No | Yes (JAX) | 2-5x |

### Migration Checklist

- [ ] Replace `ConstraintManager` with `HighPerformanceConstraintPipeline`
- [ ] Replace `CFSProjection` with `CFSConvexifier` + `PerStepQPFilter`
- [ ] Replace `CBFDoubleIntegrator2DActionFilter` with `CBFConvexifier` + `PerStepQPFilter`
- [ ] Replace `ConstraintScheduleManager` with `CosineAnnealScheduler`
- [ ] Update diffusion loop to use `ScheduleState`
- [ ] Enable batch processing for multiple trajectories
- [ ] Enable JIT compilation (if using JAX backend)
- [ ] Enable caching for repeated constraints
- [ ] Test and validate results match legacy implementation

## Backward Compatibility

The legacy components are still available and functional:

```python
# Legacy imports still work
from genedynamics.core.constraints.legacy import (
    CFSProjection,
    CBFDoubleIntegrator2DActionFilter,
    ConstraintScheduleManager
)
```

However, new code should use the new architecture for:
- Better performance
- Clearer separation of concerns
- Multi-backend support
- Easier extension

## Common Issues

### Issue 1: Component Not Found

**Error**: `Convexifier 'cfs' with backend 'jax' not found`

**Solution**: Register the component or use numpy backend:
```python
# Use numpy backend (always available)
pipeline = HighPerformanceConstraintPipeline(
    convexifier_name="cfs",
    operator_name="per_step_qp",
    scheduler_name="cosine_anneal",
    config=PipelineConfig(backend="numpy")  # Use numpy
)
```

### Issue 2: JIT Compilation Fails

**Error**: JIT compilation errors

**Solution**: Disable JIT or fix function signatures:
```python
config = PipelineConfig(backend="jax", use_jit=False)  # Disable JIT
```

### Issue 3: Batch Processing Slower

**Error**: Batch processing is slower than single

**Solution**: Ensure batch size is large enough, or use JAX backend:
```python
# Use JAX backend for true vectorization
config = PipelineConfig(backend="jax", use_batch=True, use_jit=True)
```

## Support

For questions or issues during migration:
1. Check this guide
2. Review architecture documentation
3. Check usage examples
4. Review test cases

