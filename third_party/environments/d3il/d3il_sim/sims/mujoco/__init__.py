import logging

try:
    import mujoco_py

    if mujoco_py.__file__ is None:
        raise ImportError(
            f"mujoco_py has been uninstalled but the generated mujoco files still exist in {list(mujoco_py.__path__)}"
        )

    from .MujocoFactory import *
except ImportError as e:
    logging.getLogger(__name__).info(e)
    logging.getLogger(__name__).info(
        "No mujoco py installed. Mujoco simulation is not available."
    )
    pass
