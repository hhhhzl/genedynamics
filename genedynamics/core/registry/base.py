"""
Back-compat shim — :class:`BaseRegistry` now lives in
:mod:`genedynamics.registry_base`.

It was relocated to a dependency-light **top-level** module so importing a
registry base does NOT trigger ``genedynamics.core/__init__`` (which eagerly
loads the JAX-backed stack). Robot/deploy code imports from
``genedynamics.registry_base`` directly and stays JAX-free; everything that
historically did ``from genedynamics.core.registry.base import BaseRegistry``
keeps working via this re-export.
"""

from genedynamics.registry_base import BaseRegistry, T

__all__ = ["BaseRegistry", "T"]