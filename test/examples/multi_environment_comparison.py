"""
Multi-environment comparison example.

This example demonstrates:
- Using different environment adapters
- Comparing behavior across environments
- Unified interface usage
"""

import numpy as np


def main():
    """Run multi-environment comparison example."""
    print("="*60)
    print("Multi-Environment Comparison Example")
    print("="*60)
    
    # Example 1: Native environment
    try:
        from enerdynamics.envs.double_integrator_box import DoubleIntegratorBoxEnv
        
        native_env = DoubleIntegratorBoxEnv()
        state, info = native_env.reset()
        print(f"✓ Native environment: state_dim={native_env.state_dim}, act_dim={native_env.act_dim}")
    except ImportError:
        print("⚠ Native environment not available")
    
    # Example 2: Gymnasium adapter
    try:
        import gymnasium as gym
        from enerdynamics.envs.adapters.gymnasium_adapter import GymnasiumEnvAdapter
        
        gym_env = gym.make("CartPole-v1")
        gym_adapter = GymnasiumEnvAdapter(gym_env)
        state, info = gym_adapter.reset()
        print(f"✓ Gymnasium adapter: state_dim={gym_adapter.state_dim}, act_dim={gym_adapter.act_dim}")
    except ImportError:
        print("⚠ Gymnasium adapter not available")
    
    # Example 3: Unified adapter
    try:
        from enerdynamics.envs.adapters.unified_adapter import UnifiedEnvAdapter
        from enerdynamics.envs.double_integrator_box import DoubleIntegratorBoxEnv
        
        native_env = DoubleIntegratorBoxEnv()
        unified_adapter = UnifiedEnvAdapter(native_env)
        state, info = unified_adapter.reset()
        print(f"✓ Unified adapter: state_dim={unified_adapter.state_dim}, act_dim={unified_adapter.act_dim}")
    except ImportError:
        print("⚠ Unified adapter not available")
    
    print("="*60)
    print("All environments use the same BaseEnv interface!")
    print("="*60)


if __name__ == "__main__":
    main()
