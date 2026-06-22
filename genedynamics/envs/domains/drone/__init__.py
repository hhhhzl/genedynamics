"""Drone domain envs — registered into the environment registry on import.

Each backend import is guarded: optional backends (mujoco / isaac) may not be
installed. On failure we register an ImportError-raising placeholder so callers
that ``try: make_env(...) except ImportError: skip`` keep skipping (preserving
the former make_env if/elif behaviour) rather than getting a ValueError.
"""

from genedynamics.core.registry.environments import (
    register_environment,
    register_unavailable_environment,
)

try:
    from . import simple as _simple
    register_environment("drone_box_3d")(_simple.DroneBox3DEnv)
except Exception as _e:
    register_unavailable_environment("drone_box_3d", _e)

try:
    from . import full_3d as _full
    register_environment("drone_full_3d")(_full.DroneFull3DEnv)
except Exception as _e:
    register_unavailable_environment("drone_full_3d", _e)

try:
    from . import physics as _phys
    register_environment("drone_full_3d_physics")(_phys.DroneFull3DPhysicsEnv)
except Exception as _e:
    register_unavailable_environment("drone_full_3d_physics", _e)

try:
    from . import mjx as _mjx
    register_environment("drone_full_3d_mjx")(_mjx.DroneFull3DMjxEnv)
except Exception as _e:
    register_unavailable_environment("drone_full_3d_mjx", _e)

try:
    from . import isaac as _isaac
    register_environment("drone_full_3d_isaac")(_isaac.DroneFull3DIsaacEnv)
except Exception as _e:
    register_unavailable_environment("drone_full_3d_isaac", _e)

try:
    from . import mujoco as _mujoco
    register_environment("drone_full_3d_mujoco")(_mujoco.DroneFull3DMujocoEnv)
except Exception as _e:
    register_unavailable_environment("drone_full_3d_mujoco", _e)
