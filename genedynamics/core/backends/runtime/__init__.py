"""Public exports loaded on demand so optional backends remain optional."""

from importlib import import_module

__all__ = ['JaxBackend', 'NumpyBackend', 'TorchBackend', 'RuntimeBackendManager']

_EXPORTS = {'JaxBackend': ('genedynamics.core.backends.runtime.jax_backend', 'JaxBackend'), 'NumpyBackend': ('genedynamics.core.backends.runtime.numpy_backend', 'NumpyBackend'), 'TorchBackend': ('genedynamics.core.backends.runtime.torch_backend', 'TorchBackend'), 'RuntimeBackendManager': ('genedynamics.core.backends.runtime.manager', 'RuntimeBackendManager')}

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
