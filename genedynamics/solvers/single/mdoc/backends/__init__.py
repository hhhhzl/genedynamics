"""MDOC backend implementations loaded only when selected."""

from importlib import import_module

__all__ = ["MDOCBackendJax", "MDOCBackendNumpy", "MDOCBackendTorch"]


def __getattr__(name):
    modules = {
        "MDOCBackendJax": ".mdoc_jax",
        "MDOCBackendNumpy": ".mdoc_numpy",
        "MDOCBackendTorch": ".mdoc_torch",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value


