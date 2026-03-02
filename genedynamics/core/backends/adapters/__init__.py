"""
Physics backend adapters for MuJoCo, Isaac Sim, etc.

This module provides concrete implementations of PhysicsBackend for
various physics engines.
"""

from genedynamics.core.backends.physics import PhysicsBackend, DummyPhysicsBackend

# MuJoCo adapter (optional; skip if mujoco not installed)
try:
    from genedynamics.core.backends.adapters.mujoco_adapter import MujocoPhysicsBackend
    __all__ = [
        "PhysicsBackend",
        "DummyPhysicsBackend",
        "MujocoPhysicsBackend",
    ]
except (ImportError, ModuleNotFoundError):
    __all__ = [
        "PhysicsBackend",
        "DummyPhysicsBackend",
    ]

# Isaac Sim adapter (optional)
try:
    from genedynamics.core.backends.adapters.isaac_adapter import IsaacSimBackend
    __all__.append("IsaacSimBackend")
except ImportError:
    pass
