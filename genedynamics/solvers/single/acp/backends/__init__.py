"""ACP numerical backends."""

__all__ = ["ACPBackendTorch"]


def __getattr__(name):
    if name == "ACPBackendTorch":
        from .acp_torch import ACPBackendTorch

        return ACPBackendTorch
    raise AttributeError(name)
