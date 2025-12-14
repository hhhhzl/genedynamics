"""
Physics backend protocol for physical simulation engines.

This module defines the PhysicsBackend protocol that physical simulation
engines (MuJoCo, Isaac Sim, PyBullet, etc.) should implement.
"""

from typing import Protocol, Optional, Any, runtime_checkable
import numpy as np

from enerdynamics.core.types import State, Action


@runtime_checkable
class PhysicsBackend(Protocol):
    """
    Protocol for physics simulation backends.
    
    Physics backends handle physical simulation, including:
    - Loading robot and environment models
    - State management
    - Dynamics computation
    - Collision detection
    - Obstacle injection
    
    This is separate from computational backends (JAX, PyTorch) which handle
    tensor operations. A physics backend can work with any computational backend.
    
    Examples:
        - MuJoCo: High-performance physics engine
        - Isaac Sim: NVIDIA's physics engine with GPU acceleration
        - PyBullet: Bullet physics engine
        - Brax: JAX-native physics engine
        - Rust: Custom Rust physics implementation (via FFI)
    """
    
    name: str  # "mujoco", "isaac", "pybullet", "brax", "rust", etc.
    
    def load_model(self, model_path: str, **kwargs) -> None:
        """
        Load physics model (e.g., URDF, MJCF, SDF).
        
        Args:
            model_path: Path to model file
            **kwargs: Additional model loading parameters
        """
        ...
    
    def set_state(self, state: State) -> None:
        """
        Set the physics engine state.
        
        Args:
            state: State to set (can be numpy array, dict, or backend-specific)
        """
        ...
    
    def get_state(self) -> State:
        """
        Get current physics engine state.
        
        Returns:
            Current state (numpy array, dict, or backend-specific)
        """
        ...
    
    def step(self, action: Action) -> State:
        """
        Step physics simulation forward.
        
        Args:
            action: Action to apply
            
        Returns:
            Next state after simulation step
        """
        ...
    
    def add_obstacle(self, obstacle: Any, **kwargs) -> None:
        """
        Add obstacle to physics scene.
        
        Args:
            obstacle: Obstacle object (should implement Obstacle protocol)
            **kwargs: Additional obstacle parameters
        """
        ...
    
    def remove_obstacle(self, obstacle_id: Any) -> None:
        """
        Remove obstacle from physics scene.
        
        Args:
            obstacle_id: Identifier of obstacle to remove
        """
        ...
    
    def check_collision(self, state: Optional[State] = None) -> bool:
        """
        Check for collisions in the physics scene.
        
        Args:
            state: Optional state to check (if None, uses current state)
            
        Returns:
            True if collision detected, False otherwise
        """
        ...
    
    def get_collision_pairs(self) -> list:
        """
        Get list of colliding object pairs.
        
        Returns:
            List of (object1_id, object2_id) tuples
        """
        ...
    
    def reset(self, **kwargs) -> State:
        """
        Reset physics simulation to initial state.
        
        Args:
            **kwargs: Additional reset parameters
            
        Returns:
            Initial state
        """
        ...
    
    def close(self) -> None:
        """Close physics engine and free resources."""
        ...


class DummyPhysicsBackend:
    """
    Dummy physics backend for testing or environments without physics engine.
    
    This backend does nothing and is useful for:
    - Testing environments that don't need physics
    - Environments with custom dynamics (e.g., double integrator)
    - Development before integrating actual physics engine
    """
    
    name = "dummy"
    
    def load_model(self, model_path: str, **kwargs) -> None:
        """No-op."""
        pass
    
    def set_state(self, state: State) -> None:
        """No-op."""
        pass
    
    def get_state(self) -> State:
        """Return empty state in dict format for consistency."""
        return {'qpos': np.array([]), 'qvel': np.array([])}
    
    def step(self, action: Action) -> State:
        """Return empty state in dict format for consistency."""
        return {'qpos': np.array([]), 'qvel': np.array([])}
    
    def add_obstacle(self, obstacle: Any, **kwargs) -> None:
        """No-op."""
        pass
    
    def remove_obstacle(self, obstacle_id: Any) -> None:
        """No-op."""
        pass
    
    def check_collision(self, state: Optional[State] = None) -> bool:
        """Always return False."""
        return False
    
    def get_collision_pairs(self) -> list:
        """Return empty list."""
        return []
    
    def reset(self, **kwargs) -> State:
        """Return empty state in dict format for consistency."""
        return {'qpos': np.array([]), 'qvel': np.array([])}
    
    def close(self) -> None:
        """No-op."""
        pass
