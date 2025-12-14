"""
Backend extensions for physics engines and rendering.

This module provides protocols for:
- Backend: Computational backends (JAX, PyTorch, NumPy) - in base.py
- PhysicsBackend: Physical simulation engines (MuJoCo, Isaac Sim, etc.)
- RenderBackend: Rendering systems (MuJoCo, Isaac Sim, Matplotlib, etc.)

The computational Backend (in base.py) handles tensor operations, while
physics and rendering backends handle environment-specific operations.
"""

# Computational backend (base abstraction)
from enerdynamics.core.backends.base import (
    Backend,
    JaxBackend,
    NumpyBackend,
    TorchBackend,
    get_backend,
)

# Physics and rendering backends
from enerdynamics.core.backends.physics import PhysicsBackend
from enerdynamics.core.backends.render import RenderBackend

# Import concrete implementations
from enerdynamics.core.backends.render_impl import (
    MatplotlibRenderer,
    GymnasiumRenderer,
    NullRenderer,
)

# Base __all__ with required exports
__all__ = [
    # Computational backends
    "Backend",
    "JaxBackend",
    "NumpyBackend",
    "TorchBackend",
    "get_backend",
    # Physics and rendering
    "PhysicsBackend",
    "RenderBackend",
    "MatplotlibRenderer",
    "GymnasiumRenderer",
    "NullRenderer",
]

# Optional renderers (may not be available)
try:
    from enerdynamics.core.backends.render_impl import BraxRenderer
    __all__.append("BraxRenderer")
except ImportError:
    pass

try:
    from enerdynamics.core.backends.render_impl import MujocoRenderer
    __all__.append("MujocoRenderer")
except ImportError:
    pass

try:
    from enerdynamics.core.backends.render_impl import IsaacSimRenderer
    __all__.append("IsaacSimRenderer")
except ImportError:
    pass

# Import physics backend adapters (optional)
try:
    from enerdynamics.core.backends.adapters import MujocoPhysicsBackend
    __all__.append("MujocoPhysicsBackend")
except ImportError:
    pass

try:
    from enerdynamics.core.backends.adapters import IsaacSimBackend
    __all__.append("IsaacSimBackend")
except ImportError:
    pass
