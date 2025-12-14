# Architecture Documentation

## Overview

The enerdynamics framework provides a unified interface for energy-driven control across multiple environments, backends, and physics engines.

## Design Principles

1. **Modularity**: Components are loosely coupled and can be used independently
2. **Backend Agnostic**: Support for multiple computational backends (JAX, PyTorch, NumPy)
3. **Environment Agnostic**: Unified interface for different environment types
4. **Extensibility**: Easy to add new backends, environments, and obstacles

## Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│              (Solvers, Experiments, etc.)                │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                   Environment Layer                      │
│  BaseEnv │ Adapters │ Obstacles │ Robots │ Factories    │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                    Backend Layer                          │
│  Computational │ Physics │ Rendering │ Protocols         │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                  External Libraries                      │
│  JAX │ PyTorch │ MuJoCo │ Isaac Sim │ Gymnasium │ Brax  │
└─────────────────────────────────────────────────────────┘
```

## Component Overview

### Core Layer

- **Backend Protocol**: Defines computational backend interface
- **Types**: Common types (State, Action, etc.)
- **Utils**: Utility functions

### Environment Layer

- **BaseEnv**: Unified environment interface
- **Adapters**: Bridge to external environments (Gymnasium, Brax)
- **Obstacles**: Collision detection and SDF computation
- **Robots**: Robot kinematics and dynamics
- **Factories**: Environment creation utilities

### Backend Layer

- **Computational Backends**: JAX, PyTorch, NumPy
- **Physics Backends**: MuJoCo, Isaac Sim, Dummy
- **Render Backends**: Matplotlib, Gymnasium, MuJoCo, Isaac Sim

## Data Flow

```
User Code
    │
    ├─> Create Environment (BaseEnv)
    │       │
    │       ├─> Add Obstacles
    │       ├─> Set Physics Backend
    │       ├─> Set Render Backend
    │       └─> Set Computational Backend
    │
    ├─> Reset Environment
    │       └─> Returns: (state, info)
    │
    ├─> Step Environment
    │       ├─> Apply Action
    │       ├─> Check Collisions (Obstacles)
    │       ├─> Compute Dynamics (Physics Backend or Custom)
    │       ├─> Compute Cost
    │       └─> Returns: (next_state, cost, done, info)
    │
    └─> Render Environment
            └─> Uses Render Backend
```

## Extension Points

### Adding a New Backend

1. Implement the `Backend` protocol
2. Register in `enerdynamics/core/backend.py`
3. Add tests in `test/unit/test_backends.py`

### Adding a New Environment

1. Implement the `BaseEnv` protocol or inherit from `BaseEnvMixin`
2. Add adapter if needed (for external environments)
3. Add tests in `test/unit/test_base_env.py` or `test/unit/test_adapters.py`

### Adding a New Obstacle

1. Implement the `Obstacle` protocol
2. Add to appropriate module (`convex.py` or `nonconvex.py`)
3. Add tests in `test/unit/test_obstacles.py`

## Testing Strategy

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test end-to-end workflows
- **Performance Tests**: Benchmark critical operations
- **Example Scripts**: Demonstrate usage patterns

## Future Enhancements

- Rust backend integration (via FFI)
- More physics engines (PyBullet, Brax native)
- More robot models
- Advanced obstacle types
- GPU acceleration support
