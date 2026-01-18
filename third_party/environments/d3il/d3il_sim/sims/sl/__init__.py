import logging

try:
    import py_at_broker as pab

    from .SlFactory import *
except ImportError as e:
    logging.getLogger(__name__).info(e)
    logging.getLogger(__name__).info(
        "No SL installed. SL simulation and control is not available."
    )
    pass
