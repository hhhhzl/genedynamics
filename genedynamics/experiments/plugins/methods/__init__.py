"""
Method plugin implementations.
"""

from .ebmbd import EBMBDMethodPlugin
from .mbd import MBDMethodPlugin
from .mbd3d import MBD3DMethodPlugin
from .mbd3d_active import MBD3DActiveMethodPlugin
from .mrmfmbd import MRMFMBDMethodPlugin
from .d3il_unified import D3ILUnifiedMethodPlugin
from .mdoc import MDOCMethodPlugin
from .cfsmbd import CFSMBDMethodPlugin
from .cfsmbd_full import CFSMBDFullMethodPlugin
from .mppi import MPPIMethodPlugin
from .twogo import TwoGOMethodPlugin
from .contact_receding import (
    ATACOMContactMethodPlugin,
    DIALContactMethodPlugin,
    MGAMethodPlugin,
    ISSAContactMethodPlugin,
    ModelBasedOnlyMethodPlugin,
    MPPIContactMethodPlugin,
    PegasusFlowContactMethodPlugin,
    StandaloneRLMethodPlugin,
)
try:
    from .dpcc import DPCCMethodPlugin
except Exception as e:
    DPCCMethodPlugin = None
try:
    from .safediffuser import SafeDiffuserMethodPlugin
except Exception:
    SafeDiffuserMethodPlugin = None

__all__ = [
    'EBMBDMethodPlugin',
    'MBDMethodPlugin',
    'MBD3DMethodPlugin',
    'MBD3DActiveMethodPlugin',
    'MRMFMBDMethodPlugin',
    'D3ILUnifiedMethodPlugin',
    'MDOCMethodPlugin',
    'CFSMBDMethodPlugin',
    'CFSMBDFullMethodPlugin',
    'MPPIMethodPlugin',
    'TwoGOMethodPlugin',
    'DPCCMethodPlugin',
    'SafeDiffuserMethodPlugin',
    'MGAMethodPlugin',
    'ModelBasedOnlyMethodPlugin',
    'DIALContactMethodPlugin',
    'MPPIContactMethodPlugin',
    'PegasusFlowContactMethodPlugin',
    'ISSAContactMethodPlugin',
    'ATACOMContactMethodPlugin',
    'StandaloneRLMethodPlugin',
]
