"""ACP solver frontend with lazy Torch backend imports."""

__all__ = ["ACPSolver", "ACPDiffusionPolicy", "VirtualTargetEstimator", "load_native_policy"]


def __getattr__(name):
    if name == "ACPSolver":
        from .acp import ACPSolver
        return ACPSolver
    if name == "ACPDiffusionPolicy":
        from genedynamics.solvers.single.acp.learning.models.diffusion_policy import ACPDiffusionPolicy
        return ACPDiffusionPolicy
    if name == "VirtualTargetEstimator":
        from .backends.compliance import VirtualTargetEstimator
        return VirtualTargetEstimator
    if name == "load_native_policy":
        from .backends.native_source import load_native_policy
        return load_native_policy
    raise AttributeError(name)
