"""Native PyTorch ACP solver and source-derived production diffusion policy.

Torch and native encoder dependencies load only when these exports are used.
"""

__all__ = ["ACPSolver", "ACPDiffusionPolicy", "VirtualTargetEstimator", "load_native_policy"]


def __getattr__(name):
    if name == "ACPSolver":
        from .acp import ACPSolver
        return ACPSolver
    if name == "ACPDiffusionPolicy":
        from .policy import ACPDiffusionPolicy
        return ACPDiffusionPolicy
    if name == "VirtualTargetEstimator":
        from .compliance import VirtualTargetEstimator
        return VirtualTargetEstimator
    if name == "load_native_policy":
        from .native import load_native_policy
        return load_native_policy
    raise AttributeError(name)
