"""
Unified registry system for all genedynamics components.

This package provides a centralized registry system for managing component
implementations (backends, projections, environments, etc.) with support for
both entry points (for installed packages) and programmatic registration
(for user code).
"""

# Base registry
from genedynamics.core.registry.base import BaseRegistry

# Backend registry
from genedynamics.core.registry.backends import (
    get_backend_registry,
    register_backend,
)

# Projection registry
from genedynamics.core.registry.projections import (
    get_projection_registry,
    register_projection,
    ProjectionTypeRegistry,
)

# Action filter registry
from genedynamics.core.registry.action_filters import (
    get_action_filter_registry,
    register_action_filter,
    ActionFilterRegistry,
)

# Environment registry
from genedynamics.core.registry.environments import (
    get_environment_registry,
    register_env,
)

# Energy registry
from genedynamics.core.registry.energy import (
    get_energy_registry,
    register_energy,
)

# Solver registry
from genedynamics.core.registry.solvers import (
    get_solver_registry,
    register_solver,
)

# EDOC backend registry
from genedynamics.core.registry.edoc_backends import (
    get_edoc_backend_registry,
    register_edoc_backend,
)

__all__ = [
    # Base
    "BaseRegistry",
    
    # Backends
    "get_backend_registry",
    "register_backend",
    
    # Projections
    "get_projection_registry",
    "register_projection",
    "ProjectionTypeRegistry",
    
    # Action Filters
    "get_action_filter_registry",
    "register_action_filter",
    "ActionFilterRegistry",
    
    # Environments
    "get_environment_registry",
    "register_env",
    
    # Energy
    "get_energy_registry",
    "register_energy",
    
    # Solvers
    "get_solver_registry",
    "register_solver",
    
    # EDOC Backends
    "get_edoc_backend_registry",
    "register_edoc_backend",
]

