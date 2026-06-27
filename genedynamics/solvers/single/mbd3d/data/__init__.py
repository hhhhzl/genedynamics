"""
Backward-compatibility re-exports.

Data adapters have moved to `genedynamics.data`. Prefer importing from there.
This module is kept only so existing `from genedynamics.solvers.single.mbd3d.data`
imports continue to work.
"""

from genedynamics.data import *
from genedynamics.data import __all__  
