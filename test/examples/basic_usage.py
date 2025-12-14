"""
Basic usage example.

This example demonstrates:
- Creating an environment
- Adding obstacles
- Running a simple simulation
"""

import numpy as np
from enerdynamics.envs.base_env import BaseEnvMixin
from enerdynamics.envs.obstacles.convex import BoxObstacle, SphereObstacle


class SimpleEnv(BaseEnvMixin):
    """Simple environment for demonstration."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    def reset(self, rng=None, **kwargs):
        self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
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


def main():
    """Run basic usage example."""
    print("="*60)
    print("Basic Usage Example")
    print("="*60)
    
    # Create obstacles
    obstacles = [
        BoxObstacle(
            center=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            half_extents=np.array([0.2, 0.2, 0.2], dtype=np.float32)
        ),
        SphereObstacle(
            center=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            radius=0.3
        ),
    ]
    
    # Create environment
    env = SimpleEnv(obstacles=obstacles, dt=0.1, horizon=100)
    
    # Reset
    state, info = env.reset()
    print(f"Initial state: {state}")
    
    # Run simulation
    for t in range(10):
        action = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        next_state, cost, done, info = env.step(state, action)
        
        # Check collision
        collision = env.check_collision(next_state)
        
        print(f"Step {t}: state={next_state}, cost={cost:.4f}, collision={collision}")
        
        state = next_state
        
        if done:
            break
    
    print("="*60)
    print("Example completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
