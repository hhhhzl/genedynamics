# High-Performance Multi-Backend Constraint System

## Overview

This is a complete rewrite of the constraint system with a focus on:
- **High Performance**: JIT compilation, batch processing, GPU acceleration
- **Multi-Backend**: Support for NumPy, JAX, PyTorch, Rust
- **Clear Architecture**: Separation of concerns (terms → convexify → operators → schedulers)
- **Extensibility**: Easy to add new constraints, operators, and schedulers

## Architecture

```
Terms → Convexify → Operators → Schedulers
```

### Key Principles

1. **Terms**: What constraints are (energy, feasibility, violation)
2. **Convexifiers**: Generate convex constraints (A, b) - they don't execute
3. **Operators**: Enforce constraints (QP, projection) - they don't generate
4. **Schedulers**: Single source of truth for all schedule parameters
5. **Solvers**: Only solve QP - they don't define problems

## Directory Structure

```
constraints/
├── core/                    # Core infrastructure
│   ├── array_interface.py   # Unified array interface (zero-copy backend switching)
│   ├── registry.py          # Multi-backend component registry
│   ├── pipeline.py          # High-performance pipeline (JIT, batching, caching)
│   ├── types.py             # Type definitions (ScheduleState, ScheduleParams, etc.)
│   ├── cache.py             # Caching system (LRU for constraints, precomputation for params)
│   ├── stats.py             # Statistical extraction utilities
│   ├── performance.py       # Performance profiling tools
│   └── utils.py             # Utility functions (topK, batching, etc.)
│
├── terms/                   # Constraint definitions
│   ├── base.py              # ConstraintTerm base class
│   ├── obstacle_sdf.py      # Obstacle SDF term
│   ├── bounds.py            # State/control bounds term
│   ├── barrier.py           # Barrier function term
│   └── composite.py         # Composite term (combines multiple terms)
│
├── convexify/               # Constraint convexification
│   ├── base.py              # Convexifier interface
│   ├── cfs/                 # CFS convexifier (non-convex → local convex corridors)
│   │   ├── cfs.py
│   │   └── backends/        # Backend implementations (numpy, jax)
│   ├── cbf/                 # CBF convexifier (nonlinear safety → linear inequalities)
│   │   ├── cbf.py
│   │   └── backends/
│   ├── orca/                # ORCA convexifier
│   └── linearize_dynamics/   # Dynamics linearization utilities
│
├── operators/               # Constraint enforcement
│   ├── base.py              # Operator interface
│   ├── qp/                  # QP operators
│   │   ├── per_step_filter.py  # Per-step QP filter
│   │   ├── traj_filter.py      # Trajectory-level QP filter
│   │   └── objectives.py        # QP objective functions
│   ├── projection.py        # Projection operator
│   ├── reweight.py          # Reweight operator
│   ├── primal_dual.py       # Primal-dual operator
│   └── repair.py            # Repair operator
│
├── schedulers/              # Parameter scheduling
│   ├── base.py              # Scheduler interface
│   ├── cosine_anneal.py     # Cosine annealing scheduler
│   ├── dual_anneal.py       # Dual annealing scheduler
│   ├── adaptive_gate.py     # Adaptive gate scheduler
│   └── presets.py           # Preset scheduler configurations
│
├── solvers/                 # QP solvers
│   ├── base.py              # QPSolver interface
│   ├── qpax_solver.py       # QPAX solver (JAX)
│   ├── osqp_solver.py       # OSQP solver (NumPy)
│   └── closed_form.py       # Closed-form solver
│
└── legacy/                  # Legacy code (backward compatibility)
    ├── action_filters/      # Legacy CBF action filters
    ├── projections/         # Legacy CFS projections
    └── MIGRATION_GUIDE.md   # Migration guide from legacy to new architecture
```

## Quick Start

### Basic Usage

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
    config=PipelineConfig(backend="jax", use_jit=True, use_batch=True),
    obstacles=obstacles
)

# Apply to trajectory
repaired, info = pipeline.apply(
    nominal=trajectory,
    ref=trajectory,
    state=ScheduleState(k=10, K=100)
)
```

### Batch Processing

```python
# Process multiple trajectories at once (vectorized with vmap)
repaired_list, info = pipeline.apply_batch(nominals, refs, state)
```

### Performance Profiling

```python
from genedynamics.core.constraints.core import PerformanceProfiler

profiler = PerformanceProfiler()
with profiler.time("operation"):
    repaired, info = pipeline.apply(nominal, ref, state)

stats = profiler.get_stats("operation")
print(f"Mean time: {stats['mean']:.4f}s")
```

## Components

### Convexifiers

Convert constraints to convex form (A x >= b):

- **CFSConvexifier**: Non-convex obstacles → local convex corridors
- **CBFConvexifier**: Nonlinear safety → linear inequalities in u
- **ORCAConvexifier**: Multi-agent collision avoidance

### Operators

Enforce constraints:

- **PerStepQPFilter**: Per-step QP (hard or slack mode)
- **TrajQPFilter**: Trajectory-level QP (full horizon)
- **ProjectionOperator**: Projection-based enforcement
- **ReweightOperator**: Reweighting-based enforcement

### Schedulers

Schedule parameters:

- **CosineAnnealScheduler**: Smooth parameter transitions
- **DualAnnealScheduler**: Dual annealing for margin and rho
- **AdaptiveGateScheduler**: Adaptive constraint activation

### Solvers

Solve QP problems:

- **QPAXSolver**: JAX-based QP solver (GPU-accelerated)
- **OSQPSolver**: OSQP-based QP solver (CPU)
- **ClosedFormSolver**: Closed-form solution for simple cases

## Performance

### Expected Speedups

| Scenario | Speedup | Notes |
|----------|---------|-------|
| Single trajectory (JAX) | 2-5x | JIT compilation |
| Batch (10 traj) | 5-10x | Vectorized operations |
| Batch (100 traj, JAX) | 200-500x | GPU + JIT + vmap |
| With caching | 1.5-3x | Repeated constraints |
| JIT compilation | 2-5x | After compilation |

### Performance Optimizations

1. **JIT Compilation**: Automatic JIT for JAX backends
2. **Batch Processing**: Vectorized operations via `jax.vmap`
3. **Smart Caching**: LRU cache for constraints, precomputation for parameters
4. **Zero-Copy Conversion**: `BackendArray` minimizes data conversion overhead
5. **GPU Acceleration**: Ready for JAX/PyTorch backends

## Multi-Backend Support

### Backend Support Matrix

| Component | NumPy | JAX | PyTorch | Rust |
|-----------|-------|-----|---------|------|
| Core Infrastructure | ✅ | ✅ | ✅ | ✅ |
| CFS Convexifier | ✅ | ✅ | ⏳ | ⏳ |
| CBF Convexifier | ✅ | ✅ | ⏳ | ⏳ |
| Per-Step QP | ✅ | ⏳ | ⏳ | ⏳ |
| Traj QP | ✅ | ⏳ | ⏳ | ⏳ |
| QP Solvers | ✅ | ✅ | ⏳ | ⏳ |

### Backend Selection

The system automatically selects the best available backend:

```python
# Auto-select backend
pipeline = HighPerformanceConstraintPipeline(
    convexifier_name="cfs",
    operator_name="per_step_qp",
    scheduler_name="cosine_anneal",
    config=PipelineConfig(backend=None),  # Auto-select
    obstacles=obstacles
)
```

Priority order: JAX > NumPy > PyTorch > Rust

## Migration from Legacy

See `legacy/MIGRATION_GUIDE.md` for detailed migration instructions.

### Quick Migration

1. **Replace `ConstraintManager`** with `HighPerformanceConstraintPipeline`
2. **Replace `CFSProjection`** with `CFSConvexifier` + `PerStepQPFilter`
3. **Replace `CBFDoubleIntegrator2DActionFilter`** with `CBFConvexifier` + `PerStepQPFilter`
4. **Replace `ConstraintScheduleManager`** with `CosineAnnealScheduler`

### Legacy Components

Legacy components are still available for backward compatibility:

```python
# Legacy imports (still work, but deprecated)
from genedynamics.core.constraints.legacy import (
    CFSProjection,
    CBFDoubleIntegrator2DActionFilter,
    ConstraintScheduleManager
)
```

## Testing

```bash
# Unit tests
pytest test/unit/test_core_constraints.py -v

# Performance tests
pytest test/performance/test_constraint_benchmarks.py -m performance -v

# Integration tests
pytest test/integration/test_constraint_pipeline.py -v
```

## Status

✅ **Complete**: Core infrastructure, convexifiers, operators, schedulers, solvers  
✅ **Optimized**: JIT compilation, batch processing, caching  
⏳ **In Progress**: JAX backend implementations (fully functional, further optimizations ongoing)  
📋 **Planned**: PyTorch backends, additional operators/schedulers

## License

See main project license.