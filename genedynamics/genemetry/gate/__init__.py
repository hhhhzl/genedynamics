"""
Gate policies for geometry activation control.

Auto-registers available backends on import.
"""

from genedynamics.genemetry.gate.multimodal import MultimodalGate

try:
    from genedynamics.genemetry.gate.backends import multimodal_jax  
except ImportError:
    pass

__all__ = ["MultimodalGate"]
