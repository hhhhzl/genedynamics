# Registry System

Unified registry system for all enerdynamics components with support for both entry points (for installed packages) and programmatic registration (for user code).

## Architecture

All registries follow a consistent pattern:

- **BaseRegistry**: Generic base class with lazy-loading entry points
- **Component-specific registries**: Specialized registries for each component type
- **Dual registration**: Support both entry points and programmatic registration
- **Priority**: Programmatic registrations take precedence over entry points

## Registry Types

### 1. Backend Registry
Manages computational backend implementations (JAX, NumPy, PyTorch, etc.)

```python
from enerdynamics.core.registry.backends import register_backend, get_backend_registry

register_backend("my_backend", MyBackendClass)
registry = get_backend_registry()
backend = registry.create("my_backend", device="cpu")
```

### 2. Projection Registry
Manages projection implementations with two-level structure: `{projection_type: {backend: impl}}`

```python
from enerdynamics.core.registry.projections import register_projection, get_projection_registry

@register_projection("cfs", "numpy")
class CFSProjectionNumpy:
    pass

registry = get_projection_registry()
impl_class = registry.get("cfs", "numpy")
```

### 3. Action Filter Registry
Manages action filter implementations with two-level structure

```python
from enerdynamics.core.registry.action_filters import register_action_filter

@register_action_filter("cbf_double_integrator_2d", "numpy")
class CBFActionFilterNumpy:
    pass
```

### 4. Environment Registry
Manages environment class registrations

```python
from enerdynamics.core.registry.environments import register_env

register_env("my_env", MyEnvClass)
```

### 5. Energy Registry
Manages energy functional factory functions

```python
from enerdynamics.core.registry.energy import register_energy

def make_my_energy():
    return LegacyEnergyFunctional({...})

register_energy("my_energy", make_my_energy)
```

## Performance Optimizations

1. **Lazy loading**: Entry points are only loaded on first access
2. **Caching**: Loaded entry points are cached to avoid repeated loading
3. **Fast lookups**: Dictionary-based lookups for O(1) access
4. **Priority system**: Programmatic registrations checked first (no entry point loading if found)

## Entry Points Integration

Entry points are configured in `setup.py`:

```python
entry_points={
    "enerdynamics.backends": [
        "numpy = enerdynamics.core.backends.runtime.numpy_backend:NumpyBackend",
    ],
    "enerdynamics.constraints.projections": [
        "cfs.numpy = enerdynamics.core.constraints.projections.backends.cfs_numpy:CFSProjectionNumpy",
    ],
}
```

## Usage Examples

See the main documentation for component-specific usage examples.

