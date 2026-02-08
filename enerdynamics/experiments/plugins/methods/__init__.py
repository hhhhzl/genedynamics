"""
Method plugin implementations.
"""

from .edoc import EDOCMethodPlugin
from .ebmbd import EBMBDMethodPlugin
from .mbd import MBDMethodPlugin
from .edoc_mpc import EDOCMPCMethodPlugin
from .d3il_unified import D3ILUnifiedMethodPlugin
from .mdoc import MDOCMethodPlugin
from .cfsmbd import CFSMBDMethodPlugin
from .cfsmbd_full import CFSMBDFullMethodPlugin
try:
    from .dpcc import DPCCMethodPlugin
except Exception as e:
    DPCCMethodPlugin = None

__all__ = [
    'EDOCMethodPlugin',
    'EBMBDMethodPlugin',
    'MBDMethodPlugin',
    'EDOCMPCMethodPlugin',
    'D3ILUnifiedMethodPlugin',
    'MDOCMethodPlugin',
    'CFSMBDMethodPlugin',
    'CFSMBDFullMethodPlugin',
    'DPCCMethodPlugin',
]

