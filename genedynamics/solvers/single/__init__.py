"""Single-agent solvers; optional numerical libraries load only when selected."""

from importlib import import_module

__all__ = ["CFSMBDSolver", "EBMBDSolver", "MBDSolver", "MDOCSolver", "TwoGOSolver"]
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
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value
