# Migration Guide

This guide helps you migrate from existing environment implementations to the enerdynamics framework.

## Overview

The enerdynamics framework provides a unified interface for working with different environments, backends, and physics engines. If you're currently using:

- Gymnasium environments
- Brax environments
- Custom environments
- MuJoCo directly
- Isaac Sim directly

This guide will help you migrate to enerdynamics.

## Migration from Gymnasium

### Before

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
state, info = env.reset()
action = env.action_space.sample()
next_state, reward, done, truncated, info = env.step(action)
```

### After

```python
import gymnasium as gym
from enerdynamics.envs.adapters.gymnasium_adapter import GymnasiumEnvAdapter

gym_env = gym.make("CartPole-v1")
env = GymnasiumEnvAdapter(gym_env)

state, info = env.reset()
action = env.action_space.sample()
next_state, cost, done, info = env.step(state, action)  # Note: cost = -reward
```

### Key Changes

1. Wrap Gymnasium environment with `GymnasiumEnvAdapter`
2. `step()` now takes `state` as first argument
3. Returns `cost` instead of `reward` (cost = -reward)
4. Unified interface allows adding obstacles, physics backends, etc.

## Migration from Brax

### Before

```python
import brax
import jax

env = brax.envs.create("ant")
state = env.reset(jax.random.PRNGKey(42))
action = jax.random.uniform(jax.random.PRNGKey(43), (8,))
next_state = env.step(state, action)
```

### After

```python
import brax
import jax
from enerdynamics.envs.adapters.brax_adapter import BraxEnvAdapter

brax_env = brax.envs.create("ant")
env = BraxEnvAdapter(brax_env)

rng = jax.random.PRNGKey(42)
state, info = env.reset(rng=rng)
action = jax.random.uniform(jax.random.PRNGKey(43), (env.act_dim,))
next_state, cost, done, info = env.step(state, action)
```

### Key Changes

1. Wrap Brax environment with `BraxEnvAdapter`
2. `reset()` returns `(state, info)` tuple
3. `step()` returns `(next_state, cost, done, info)` tuple
4. Can now add obstacles and use unified interface

## Migration from Custom Environments

### Before

```python
class MyEnv:
    def __init__(self):
        self.state = np.zeros(6)
    
    def reset(self):
        self.state = np.zeros(6)
        return self.state
    
    def step(self, action):
        self.state = self.state + action * 0.1
        reward = -np.sum(self.state ** 2)
        done = False
        return self.state, reward, done, {}
```

### After

```python
from enerdynamics.envs.base_env import BaseEnvMixin
import numpy as np

class MyEnv(BaseEnvMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state = np.zeros(6, dtype=np.float32)
    
    def reset(self, rng=None, **kwargs):
        self.state = np.zeros(6, dtype=np.float32)
        return self.state, {}
    
    def step(self, state, action, t=None, info=None):
        next_state = state + action * self.dt
        cost = float(np.sum(next_state ** 2))
        done = False
        return next_state, cost, done, {}
    
    def transition(self, state, action):
        return state + action * self.dt
    
    def cost(self, state):
        return float(np.sum(state ** 2))
```

### Key Changes

1. Inherit from `BaseEnvMixin`
2. Implement required methods: `reset()`, `step()`, `transition()`, `cost()`
3. `step()` takes `state` as first argument
4. Returns `cost` instead of `reward`
5. Can now add obstacles, physics backends, render backends

## Migration from MuJoCo

### Before

```python
import mujoco

model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)
mujoco.mj_step(model, data)
```

### After

```python
from enerdynamics.core.backends.adapters.mujoco_adapter import MujocoPhysicsBackend

backend = MujocoPhysicsBackend(model_path="robot.xml", dt=0.01)
state = backend.get_state()  # {'qpos': ..., 'qvel': ...}
next_state = backend.step(action)
```

### Key Changes

1. Use `MujocoPhysicsBackend` instead of direct MuJoCo API
2. Unified interface with other physics backends
3. Can be integrated into environments via `BaseEnvMixin`

## Migration from Isaac Sim

### Before

```python
from omni.isaac.core import World

world = World()
# ... setup ...
world.step()
```

### After

```python
from enerdynamics.core.backends.adapters.isaac_adapter import IsaacSimBackend

backend = IsaacSimBackend(model_path="robot.usd", dt=0.01, use_gpu=True)
state = backend.get_state()
next_state = backend.step(action)
```

### Key Changes

1. Use `IsaacSimBackend` instead of direct Isaac Sim API
2. Unified interface with other physics backends
3. Can be integrated into environments via `BaseEnvMixin`

## Adding Obstacles

Once migrated, you can easily add obstacles:

```python
from enerdynamics.envs.obstacles.convex import BoxObstacle, SphereObstacle

obstacles = [
    BoxObstacle(center=[1.0, 0.0, 0.0], half_extents=[0.2, 0.2, 0.2]),
    SphereObstacle(center=[-1.0, 0.0, 0.0], radius=0.3),
]

env = MyEnv(obstacles=obstacles)
collision = env.check_collision(state)
```

## Adding Physics Backends

```python
from enerdynamics.core.backends.adapters import MujocoPhysicsBackend

physics_backend = MujocoPhysicsBackend(model_path="robot.xml", dt=0.01)
env = MyEnv(physics_backend=physics_backend)
```

## Common Issues and Solutions

### Issue: `step()` signature mismatch

**Solution**: Update `step()` to take `state` as first argument:
```python
def step(self, state, action, t=None, info=None):
    # ...
```

### Issue: Reward vs Cost

**Solution**: Convert reward to cost (cost = -reward):
```python
cost = -reward
return next_state, cost, done, info
```

### Issue: State format

**Solution**: Ensure state is numpy array with float32 dtype:
```python
state = np.asarray(state, dtype=np.float32)
```

## Testing Migration

After migration, run tests to ensure compatibility:

```bash
pytest test/unit/test_adapters.py -v
pytest test/integration/test_end_to_end.py -v
```

## Getting Help

If you encounter issues during migration:

1. Check the API documentation: `docs/api/`
2. Review examples: `test/examples/`
3. Run tests: `pytest test/ -v`
4. Check architecture docs: `docs/architecture/`
