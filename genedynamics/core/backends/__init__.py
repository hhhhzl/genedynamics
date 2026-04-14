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
from genedynamics.core.backends.base import (
    Backend,
    get_backend,
)

# Runtime backends (concrete implementations)
from genedynamics.core.backends.runtime import (
    JaxBackend,
    NumpyBackend,
    TorchBackend,
    RuntimeBackendManager,
)

# Physics and rendering backends
from genedynamics.core.backends.physics import PhysicsBackend
from genedynamics.core.backends.render import RenderBackend

# Import concrete implementations
from genedynamics.core.backends.render_impl import (
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
    "RuntimeBackendManager",
    # Physics and rendering
    "PhysicsBackend",
    "RenderBackend",
    "MatplotlibRenderer",
    "GymnasiumRenderer",
    "NullRenderer",
]

# Optional renderers (may not be available)
try:
    from genedynamics.core.backends.render_impl import BraxRenderer
    __all__.append("BraxRenderer")
except ImportError:
    pass

try:
    from genedynamics.core.backends.render_impl import MujocoRenderer
    __all__.append("MujocoRenderer")
except (ImportError, AttributeError):
    pass

try:
    from genedynamics.core.backends.render_impl import IsaacSimRenderer
    __all__.append("IsaacSimRenderer")
except ImportError:
    pass

# Import physics backend adapters (optional)
try:
    from genedynamics.core.backends.adapters import MujocoPhysicsBackend
    __all__.append("MujocoPhysicsBackend")
except (ImportError, AttributeError):
    pass

try:
    from genedynamics.core.backends.adapters import IsaacSimBackend
    __all__.append("IsaacSimBackend")
except ImportError:
    pass
