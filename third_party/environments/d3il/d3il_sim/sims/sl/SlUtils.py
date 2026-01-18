import ctypes
import logging
import os
from os import chdir, getcwd, path

import numpy as np

# __here = path.abspath(path.dirname(main.__file__))
# __root = __here + '/../../'
pathSLpanda = os.environ["SL_PANDA_BUILD_DIR"]
lib = ctypes.CDLL(pathSLpanda + "/libslrobot.so")
logging.getLogger(__name__).debug("LIB")
logging.getLogger(__name__).debug(lib)

n_dofs = ctypes.c_int.in_dll(lib, "n_dofs").value
n_links = ctypes.c_int.in_dll(lib, "n_links").value
# TODO n_endeff?

jNames = [
    str(bName.value, "utf-8")
    for bName in ((ctypes.c_char * 20) * 8).in_dll(lib, "joint_names")
]
jNames = jNames[1:]

lib.PyWrapFK.argtypes = (
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
)

lib.PyWrapJacobian.argtypes = (
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
)

lib.PyWrapIDyn.argtypes = (
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
)

__curr_path = path.abspath(getcwd())
chdir(pathSLpanda)
lib.initPyWrap()
chdir(__curr_path)


def FK(q: np.array) -> np.array:
    if q.ndim != 2:
        logging.getLogger(__name__).warning(
            "FK: two dim. array, #eval x #NDOF, expected"
        )
        return

    if q.shape[1] != n_dofs:
        logging.getLogger(__name__).warning(
            "FK: two dim. array, #eval x #NDOF, expected"
        )
        logging.getLogger(__name__).warning("FK: #NDOF should be " + str(n_dofs))
        return

    # output is #eval x (#links + #N_CART + #N_ORIENT + #N_QUAT
    res_w = n_links * (3 + 3 + 4)
    res = np.zeros((q.shape[0], res_w), dtype=np.float64)

    qt = q.astype(np.float64)  # TODO
    ret = lib.PyWrapFK(
        ctypes.c_int(q.shape[0]),
        qt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        res.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    return ret, res


def IDyn(q: np.array, q_des: np.array) -> np.array:
    res = np.zeros((q.shape[0], n_dofs), dtype=np.float64)

    qt = q.astype(np.float64)  # TODO
    qt_des = q_des.astype(np.float64)  # TODO
    # TODO C_CONTIGUOUS
    ret = lib.PyWrapIDyn(
        ctypes.c_int(q.shape[0]),
        qt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        qt_des.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        res.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    return ret, res


def GravCmp(q: np.array) -> np.array:
    res = np.zeros((q.shape[0], n_dofs), dtype=np.float64)

    qt = q.astype(np.float64)  # TODO
    # TODO C_CONTIGUOUS
    ret = lib.PyWrapGravCmp(
        ctypes.c_int(q.shape[0]),
        qt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        res.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    return ret, res


def Jacobian(q: np.array, link_id) -> np.array:
    if q.ndim != 2:
        logging.getLogger(__name__).warning(
            "FK: two dim. array, #eval x #NDOF, expected"
        )
        return

    if q.shape[1] != n_dofs:
        logging.getLogger(__name__).warning(
            "FK: two dim. array, #eval x #NDOF, expected"
        )
        logging.getLogger(__name__).warning("FK: #NDOF should be " + str(n_dofs))
        return

    res = np.zeros((q.shape[0] * 6, n_dofs), dtype=np.float64)

    qt = q.astype(np.float64)  # TODO
    ret = lib.PyWrapJacobian(
        ctypes.c_int(q.shape[0]),
        qt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(link_id),
        res.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    return ret, res
