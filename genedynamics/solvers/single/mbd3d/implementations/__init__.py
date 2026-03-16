"""
Reference implementations for MBD3D protocols.

- GaussianSplatScene: simple 3DGS scene representation
- GaussianObservationLikelihood: L2 / low-rank noise likelihood
"""

from .scene import GaussianSplatScene
from .likelihood import GaussianObservationLikelihood

try:
    from .jax_splat_renderer import JaxSplatRenderer
    JAX_SPLAT_AVAILABLE = True
except ImportError:
    JaxSplatRenderer = None
    JAX_SPLAT_AVAILABLE = False

try:
    from .jaxsplat_renderer import JaxsplatRenderer, JAXSPLAT_AVAILABLE
except ImportError:
    JaxsplatRenderer = None
    JAXSPLAT_AVAILABLE = False

try:
    from .gsplat_renderer import GsplatRenderer, GSPLAT_AVAILABLE
except ImportError:
    GsplatRenderer = None
    GSPLAT_AVAILABLE = False

__all__ = [
    "GaussianSplatScene",
    "JaxSplatRenderer",
    "JAX_SPLAT_AVAILABLE",
    "JaxsplatRenderer",
    "JAXSPLAT_AVAILABLE",
    "GsplatRenderer",
    "GSPLAT_AVAILABLE",
    "GaussianObservationLikelihood",
]
