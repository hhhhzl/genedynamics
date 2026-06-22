"""Toy integrator environments (numpy/jax, no physics backend).

Importing the modules runs their registry registration, exposing the env names
``double_integrator_box`` / ``double_integrator_box_2d`` /
``single_integrator_box_2d`` via the environment registry.
"""

from . import double_integrator_box  # noqa: F401
from . import double_integrator_box_2d  # noqa: F401
from . import single_integrator_box_2d  # noqa: F401
