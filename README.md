# GeneDynamics Framework

A unified framework for energy-driven control across multiple environments, backends, and physics engines.

## Features

- **Multi-Backend Support**: JAX, PyTorch, NumPy
- **Multi-Environment Support**: Gymnasium, Brax, custom environments
- **Physics Engines**: MuJoCo, Isaac Sim, custom dynamics
- **Obstacle System**: Convex and non-convex obstacles with CSG operations
- **Robot Models**: Manipulators and drones
- **Rendering**: Matplotlib, Gymnasium, MuJoCo, Isaac Sim

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from genedynamics.envs.base_env import BaseEnvMixin
from genedynamics.envs.obstacles.convex import BoxObstacle

# Create environment with obstacles
obstacles = [
    BoxObstacle(center=[1.0, 0.0, 0.0], half_extents=[0.2, 0.2, 0.2])
]

class MyEnv(BaseEnvMixin):
    def reset(self, rng=None, **kwargs):
        return np.zeros(6), {}
    
    def step(self, state, action, t=None, info=None):
        next_state = state + action * self.dt
        cost = float(np.sum(next_state ** 2))
        return next_state, cost, False, {}
    
    def transition(self, state, action):
        return state + action * self.dt
    
    def cost(self, state):
        return float(np.sum(state ** 2))

env = MyEnv(obstacles=obstacles, dt=0.1)
state, _ = env.reset()
next_state, cost, done, _ = env.step(state, action)
```

## Documentation

- **API Documentation**: `docs/api/`
- **Architecture**: `docs/architecture/`
- **Migration Guide**: `docs/migration_guide.md`
- **Mac CPU Compatibility**: `docs/mac_cpu_compatibility.md`
- **Examples**: `test/examples/`

## Testing

Run all tests:

```bash
pytest test/ -v
```

Run specific test categories:

```bash
pytest test/unit/ -v          # Unit tests
pytest test/integration/ -v   # Integration tests
pytest test/ -m "not slow"    # Skip slow tests
```

## Examples

See `test/examples/` for usage examples:

- `basic_usage.py`: Basic environment usage
- `obstacle_configuration.py`: Obstacle configuration
- `multi_environment_comparison.py`: Comparing environments

## Contributing

1. Follow the architecture principles in `docs/architecture/`
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass: `pytest test/ -v`

## License

[Your License Here]
