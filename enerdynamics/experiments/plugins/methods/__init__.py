"""
Method plugin implementations.
"""

from .edoc import EDOCMethodPlugin
from .ebmbd import EBMBDMethodPlugin
from .mbd import MBDMethodPlugin
from .edoc_mpc import EDOCMPCMethodPlugin
from .mbd_mpc import MBDMPCMethodPlugin
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
    'MBDMPCMethodPlugin',
    'MDOCMethodPlugin',
    'CFSMBDMethodPlugin',
    'CFSMBDFullMethodPlugin',
    'DPCCMethodPlugin',
]

