"""2GO backend implementations loaded only when selected."""

from importlib import import_module

__all__ = ["TwoGOBackendJax", "TwoGOBackendTorch"]


def __getattr__(name):
    modules = {
        "TwoGOBackendJax": ".twogo_jax",
        "TwoGOBackendTorch": ".twogo_torch",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value

