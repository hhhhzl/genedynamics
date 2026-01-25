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

try:
    from .dpcc import DPCCMethodPlugin
except (ImportError, ModuleNotFoundError):
    DPCCMethodPlugin = None  # diffuser optional; skip DPCC registration when missing

__all__ = [
    'EDOCMethodPlugin',
    'EBMBDMethodPlugin',
    'MBDMethodPlugin',
    'EDOCMPCMethodPlugin',
    'MBDMPCMethodPlugin',
    'MDOCMethodPlugin',
    'CFSMBDMethodPlugin',
    'DPCCMethodPlugin',
]

