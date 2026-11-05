"""Public exports loaded on demand so optional backends remain optional."""

from importlib import import_module

__all__ = [
    "CEMSolver",
    "CFSMBDSolver",
    "DIALMPCSolver",
    "EBMBDSolver",
    "MBDSolver",
    "MDOCSolver",
    "MPPISolver",
    "OptimizationSolver",
    "SamplingSolver",
    "Solver",
    "TwoGOSolver",
    "run_cem",
    "run_mbd",
    "run_mppi",
]

_EXPORTS = {
    "Solver": ("genedynamics.core.solvers", "Solver"),
    "SamplingSolver": ("genedynamics.core.solvers", "SamplingSolver"),
    "OptimizationSolver": ("genedynamics.core.solvers", "OptimizationSolver"),
    "MBDSolver": ("genedynamics.solvers.single.mbd.mbd", "MBDSolver"),
    "run_mbd": ("genedynamics.solvers.single.mbd.mbd", "run_mbd"),
    "EBMBDSolver": ("genedynamics.solvers.single.ebmbd.ebmbd", "EBMBDSolver"),
    "MDOCSolver": ("genedynamics.solvers.single.mdoc.mdoc", "MDOCSolver"),
    "CFSMBDSolver": ("genedynamics.solvers.single.cfsmbd.cfsmbd", "CFSMBDSolver"),
    "TwoGOSolver": ("genedynamics.solvers.single.twogo.twogo", "TwoGOSolver"),
    "MPPISolver": ("genedynamics.solvers.single.mppi.mppi", "MPPISolver"),
    "run_mppi": ("genedynamics.solvers.single.mppi.mppi", "run_mppi"),
    "CEMSolver": ("genedynamics.solvers.single.cem.cem", "CEMSolver"),
    "run_cem": ("genedynamics.solvers.single.cem.cem", "run_cem"),
    "DIALMPCSolver": ("genedynamics.solvers.single.dial.dial", "DIALMPCSolver"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
