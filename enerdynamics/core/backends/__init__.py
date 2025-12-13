"""
Backend extensions for physics engines and rendering.

This module provides protocols for:
- PhysicsBackend: Physical simulation engines (MuJoCo, Isaac Sim, etc.)
- RenderBackend: Rendering systems (MuJoCo, Isaac Sim, Matplotlib, etc.)

These are separate from the computational Backend (JAX, PyTorch) which handles
tensor operations. Physics and rendering backends handle environment-specific
operations.
"""

from enerdynamics.core.backends.physics import PhysicsBackend
from enerdynamics.core.backends.render import RenderBackend

# Import concrete implementations
from enerdynamics.core.backends.render_impl import (
    MatplotlibRenderer,
    GymnasiumRenderer,
    NullRenderer,
)

# Optional renderers (may not be available)
try:
    from enerdynamics.core.backends.render_impl import BraxRenderer
    __all__ = [
        "PhysicsBackend",
        "RenderBackend",
        "MatplotlibRenderer",
        "GymnasiumRenderer",
        "BraxRenderer",
        "NullRenderer",
    ]
except ImportError:
    __all__ = [
        "PhysicsBackend",
        "RenderBackend",
        "MatplotlibRenderer",
        "GymnasiumRenderer",
        "NullRenderer",
    ]

try:
    from enerdynamics.core.backends.render_impl import MujocoRenderer
    if "MujocoRenderer" not in __all__:
        __all__.append("MujocoRenderer")
except ImportError:
    pass

try:
    from enerdynamics.core.backends.render_impl import IsaacSimRenderer
    if "IsaacSimRenderer" not in __all__:
        __all__.append("IsaacSimRenderer")
except ImportError:
    pass

# Import physics backend adapters (optional)
try:
    from enerdynamics.core.backends.adapters import MujocoPhysicsBackend
    if "MujocoPhysicsBackend" not in __all__:
        __all__.append("MujocoPhysicsBackend")
except ImportError:
    pass

try:
    from enerdynamics.core.backends.adapters import IsaacSimBackend
    if "IsaacSimBackend" not in __all__:
        __all__.append("IsaacSimBackend")
except ImportError:
    pass
