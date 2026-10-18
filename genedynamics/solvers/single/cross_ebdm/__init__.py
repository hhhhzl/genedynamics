"""Cross-EBDM solver frontends.

Task controllers live under :mod:`genedynamics.tasks.transfer.controllers`;
this package owns only the method, its numerical backends, and its learning
implementation.
"""

from .contact_planner import ContactPlanner
from .cross_ebdm import CrossEBDMSolver, PolicyBackend, PolicyChunk

__all__ = [
    "ContactPlanner",
    "CrossEBDMSolver",
    "PolicyBackend",
    "PolicyChunk",
]
