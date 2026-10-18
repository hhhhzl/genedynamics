"""
Single solver implementations.

This package contains individual solver implementations organized by solver type.
"""

__all__ = [
    "MBDSolver",
    "EBMBDSolver",
    "MDOCSolver",
    "CFSMBDSolver",
    "TwoGOSolver",
]


_EXPORTS = {
    "MBDSolver": ("genedynamics.solvers.single.mbd.mbd", "MBDSolver"),
    "EBMBDSolver": ("genedynamics.solvers.single.ebmbd.ebmbd", "EBMBDSolver"),
    "MDOCSolver": ("genedynamics.solvers.single.mdoc.mdoc", "MDOCSolver"),
    "CFSMBDSolver": ("genedynamics.solvers.single.cfsmbd.cfsmbd", "CFSMBDSolver"),
    "TwoGOSolver": ("genedynamics.solvers.single.twogo.twogo", "TwoGOSolver"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


