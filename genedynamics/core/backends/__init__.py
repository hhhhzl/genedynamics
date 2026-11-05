"""Public exports loaded on demand so optional backends remain optional."""

from importlib import import_module

__all__ = [
    "Backend",
    "GymnasiumRenderer",
    "JaxBackend",
    "MatplotlibRenderer",
    "NullRenderer",
    "NumpyBackend",
    "PhysicsBackend",
    "RenderBackend",
    "RuntimeBackendManager",
    "TorchBackend",
    "get_backend",
]

_EXPORTS = {
    "Backend": ("genedynamics.core.backends.base", "Backend"),
    "get_backend": ("genedynamics.core.backends.base", "get_backend"),
    "JaxBackend": ("genedynamics.core.backends.runtime", "JaxBackend"),
    "NumpyBackend": ("genedynamics.core.backends.runtime", "NumpyBackend"),
    "TorchBackend": ("genedynamics.core.backends.runtime", "TorchBackend"),
    "RuntimeBackendManager": (
        "genedynamics.core.backends.runtime",
        "RuntimeBackendManager",
    ),
    "PhysicsBackend": ("genedynamics.core.backends.physics", "PhysicsBackend"),
    "RenderBackend": ("genedynamics.core.backends.render", "RenderBackend"),
    "MatplotlibRenderer": (
        "genedynamics.core.backends.render_impl",
        "MatplotlibRenderer",
    ),
    "GymnasiumRenderer": (
        "genedynamics.core.backends.render_impl",
        "GymnasiumRenderer",
    ),
    "NullRenderer": ("genedynamics.core.backends.render_impl", "NullRenderer"),
    "BraxRenderer": ("genedynamics.core.backends.render_impl", "BraxRenderer"),
    "MujocoRenderer": ("genedynamics.core.backends.render_impl", "MujocoRenderer"),
    "IsaacSimRenderer": ("genedynamics.core.backends.render_impl", "IsaacSimRenderer"),
    "MujocoPhysicsBackend": (
        "genedynamics.core.backends.adapters",
        "MujocoPhysicsBackend",
    ),
    "IsaacSimBackend": ("genedynamics.core.backends.adapters", "IsaacSimBackend"),
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
