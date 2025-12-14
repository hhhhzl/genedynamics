"""
Dynamics model abstraction for the energy-driven control framework.

This module defines the DynamicsModel interface, which represents system dynamics
that can be either:
- True physics models (e.g., rigid body dynamics, double integrator)
- Learned world models (e.g., neural network dynamics, flow matching models)

All dynamics models must implement the step() method for single-step transitions,
and can optionally implement rollout() for efficient batch rollouts.
"""

from enerdynamics.core.dynamics.base import (
    DynamicsModel,
    DeterministicDynamicsModel,
    StochasticDynamicsModel,
)
from enerdynamics.core.dynamics.adapters import (
    EnvDynamicsAdapter,
    DynamicsToEnvAdapter,
)

__all__ = [
    "DynamicsModel",
    "DeterministicDynamicsModel",
    "StochasticDynamicsModel",
    "EnvDynamicsAdapter",
    "DynamicsToEnvAdapter",
]
