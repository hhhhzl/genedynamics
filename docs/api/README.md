# API Documentation

This directory contains API documentation for the enerdynamics framework.

## Structure

- `core.md`: Core API (backends, types, protocols)
- `envs.md`: Environment API (BaseEnv, adapters, obstacles)
- `robots.md`: Robot model API
- `solvers.md`: Solver API (if applicable)

## Generating API Documentation

To generate API documentation from docstrings:

```bash
# Install sphinx
pip install sphinx sphinx-rtd-theme

# Generate documentation
cd docs/api
sphinx-build -b html . _build/html
```

## Manual Documentation

For now, API documentation is maintained manually. Key interfaces:

### Core Backends

- `Backend`: Computational backend protocol (JAX, PyTorch, NumPy)
- `PhysicsBackend`: Physics simulation backend protocol (MuJoCo, Isaac Sim)
- `RenderBackend`: Rendering backend protocol (Matplotlib, Gymnasium, MuJoCo, Isaac Sim)

### Environments

- `BaseEnv`: Environment protocol
- `BaseEnvMixin`: Default environment implementation
- `GymnasiumEnvAdapter`: Gymnasium environment adapter
- `BraxEnvAdapter`: Brax environment adapter
- `UnifiedEnvAdapter`: Unified environment adapter

### Obstacles

- `Obstacle`: Obstacle protocol
- `ObstacleManager`: Obstacle manager
- Convex obstacles: `BoxObstacle`, `SphereObstacle`, `CylinderObstacle`, `CapsuleObstacle`
- Non-convex obstacles: `MeshObstacle`, `UnionObstacle`, `DifferenceObstacle`, `IntersectionObstacle`

### Robots

- `RobotModel`: Robot model protocol
- `ManipulatorModel`: Robotic arm model
- `DroneModel`: Quadrotor drone model
