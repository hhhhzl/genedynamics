"""
Runtime backends for computational operations.

This module provides concrete implementations of the Backend protocol
for different computational frameworks (JAX, NumPy, PyTorch, etc.)
with support for device selection (CPU, GPU, WebGPU).
"""

from genedynamics.core.backends.runtime.jax_backend import JaxBackend
from genedynamics.core.backends.runtime.numpy_backend import NumpyBackend
from genedynamics.core.backends.runtime.torch_backend import TorchBackend
from genedynamics.core.backends.runtime.manager import RuntimeBackendManager

__all__ = [
    "JaxBackend",
    "NumpyBackend",
    "TorchBackend",
    "RuntimeBackendManager",
]

