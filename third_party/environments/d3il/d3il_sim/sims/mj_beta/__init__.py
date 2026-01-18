import logging

try:
    import mujoco

    from .MjFactory import *
except ImportError as e:
    logging.getLogger(__name__).info(e)
    logging.getLogger(__name__).info(
        "No MuJoCo bindings are(new bindings from DeepMind) installed.",
        "Install MuJoCo >= 2.1.5 and load bindings via pip, conda or mamba",
    )
    pass
